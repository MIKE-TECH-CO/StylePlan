"""Opt-in, bounded v0.1.1 GPU validation; no paper-result reproduction claim.

Outputs/checkpoints refuse overwrite. Evaluation always uses sparse reward.
Unqualified pools stop before ego training unless diagnostic continuation is
explicitly requested. Historical stage checkpoints remain independently usable.
Full/NoPM have matched PPO sampling budgets. Full additionally collects the
declared auxiliary-pretraining batches; total physical budgets are unequal.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
from styleplan_paper import model
from styleplan_paper.config import ModelConfig, TrainConfig
from styleplan_paper.cli import SelfPlaySpecification
from styleplan_paper.envs import OvercookedV2
from styleplan_paper.checkpoint import save_checkpoint, load_checkpoint, source_manifest
from styleplan_paper.partners import ModelPool, assert_disjoint, tree_hash
from styleplan_paper.training import init_train_state, init_runner, make_rollout, make_update, replay, tree_norm
from styleplan_paper.losses import auxiliary, future_targets, generalized_advantage
from styleplan_paper.evaluation import evaluate_switch


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2)


def parse_stages(value):
    stages = tuple(int(x) for x in value.split(','))
    if not stages or stages[0] < 1 or tuple(sorted(set(stages))) != stages:
        raise argparse.ArgumentTypeError('Stages must be strictly increasing positive cumulative update counts')
    return stages


def training_budgets(num_envs, steps, pool_updates, ego_updates, pretrain_updates, epochs):
    """Keep physical world steps distinct from shared-policy role samples."""
    if min(num_envs, steps, pool_updates, ego_updates, epochs) < 1 or pretrain_updates < 0:
        raise ValueError('Invalid training budgets')
    batch_steps = num_envs*steps
    partner_physical = pool_updates*batch_steps
    ppo_physical = ego_updates*batch_steps
    pretrain_physical = pretrain_updates*batch_steps
    return dict(comparison_protocol='matched PPO sampling budgets; Full has extra auxiliary pretraining',
        matched_ppo_sampling_budgets=True, matched_total_physical_budgets=pretrain_updates == 0,
        environment_steps_per_batch=batch_steps,
        per_partner=dict(physical_environment_steps=partner_physical,
            collected_role_transitions=2*partner_physical, optimizer_role_sample_uses=2*partner_physical*epochs),
        full=dict(ppo_updates=ego_updates, pretrain_updates=pretrain_updates,
            ppo_environment_steps=ppo_physical, pretrain_environment_steps=pretrain_physical,
            total_physical_environment_steps=ppo_physical+pretrain_physical),
        no_partner_model=dict(ppo_updates=ego_updates, pretrain_updates=0,
            ppo_environment_steps=ppo_physical, pretrain_environment_steps=0,
            total_physical_environment_steps=ppo_physical))


def qualification(returns, random_returns, min_success_rate=.25, bootstrap_samples=2000):
    """Predeclared screening rule, not a multi-seed method superiority test.

    Equal rollout seeds supply a paired screening comparison, though sampled
    trajectories diverge as actions diverge. The bootstrap is descriptive and
    not a correction for repeated checkpoint selection.
    """
    values, control = np.asarray(returns, float), np.asarray(random_returns, float)
    if (values.ndim != 1 or values.shape != control.shape or len(values) < 2
            or not np.isfinite(values).all() or not np.isfinite(control).all()
            or not 0 <= min_success_rate <= 1 or bootstrap_samples < 1):
        raise ValueError('Invalid qualification inputs')
    difference = values-control
    indices = np.random.default_rng(1701).integers(0, len(values), (bootstrap_samples, len(values)))
    lower = float(np.quantile(difference[indices].mean(1), .025))
    success = float((values > 0).mean())
    return dict(qualified=bool(success >= min_success_rate and lower > 0),
        episodes=len(values), mean_return=float(values.mean()), random_mean_return=float(control.mean()),
        success_rate=success, minimum_success_rate=min_success_rate,
        paired_mean_improvement=float(difference.mean()), bootstrap_lower_95=lower,
        bootstrap_seed=1701, bootstrap_samples=bootstrap_samples,
        rule='success fraction >= floor AND paired bootstrap 2.5% improvement quantile > 0',
        scope='pool screening only; repeated stage selection is not significance evidence')


def rollout_returns(data, env_batch, self_play=False):
    reward = np.asarray(data.transitions.reward)
    expected = env_batch*(2 if self_play else 1)
    if reward.shape[1] != expected:
        raise AssertionError('Unexpected rollout batch width')
    if self_play:
        # Shared team rewards are replicated for PPO, not two environment events.
        np.testing.assert_array_equal(reward[:, :env_batch], reward[:, env_batch:])
        reward = reward[:, :env_batch]
    done = np.asarray(data.transitions.terminated | data.transitions.truncated)[:, :env_batch]
    if done[:-1].any() or not done[-1].all():
        raise AssertionError('Evaluation must contain exactly one complete configured-horizon episode per lane')
    return reward.sum(0)


def trace(params, norm, data, config):
    def step(memory, transition):
        out, memory = model.apply(params, norm, memory, transition.obs,
                                  transition.context, transition.global_state, config)
        logp = jax.nn.log_softmax(out['logits'])
        stats = dict(hidden_norm=jnp.linalg.norm(memory.hidden, axis=-1),
                     hidden_max=jnp.abs(memory.hidden).max(-1),
                     entropy=-(jnp.exp(logp)*logp).sum(-1),
                     logit_range=jnp.ptp(out['logits'], axis=-1))
        memory = model.observe(memory, transition.obs, transition.observed_partner_action,
                               transition.position, transition.position_visible)
        return model.reset_state(memory, transition.terminated | transition.truncated), stats
    return jax.lax.scan(step, data.initial_state, data.transitions)[1]


def trace_summary(values):
    result = {}
    for name, value in values.items():
        a = np.asarray(value)
        if not np.isfinite(a).all():
            raise FloatingPointError('Nonfinite recurrent trace: '+name)
        result[name] = dict(mean=float(a.mean()), maximum=float(a.max()),
                           windows=[float(a[i:i+50].mean()) for i in range(0, len(a), 50)])
    return result


def stability_control(pilot, output, episodes, horizon):
    """Same legacy weights and recorded inputs: isolate bounded recurrence only."""
    item = load_checkpoint(Path(pilot)/'full/checkpoint')
    old = item['config']
    if old.style_injection != 'legacy':
        raise ValueError('Stability input must be a legacy v0.1.0 checkpoint')
    env = OvercookedV2(horizon=horizon, public_actions=True, task_context_mode='legacy')
    if old.task_dim != env.task_dim:
        raise ValueError('Legacy task context does not match checkpoint')
    pool = ModelPool([Path(pilot)/'pool_seed201', Path(pilot)/'pool_seed202'])
    tc = TrainConfig(num_envs=episodes, num_steps=horizon, epochs=1)
    state = item['state']
    before = tree_hash(state)
    runner = init_runner(jax.random.PRNGKey(44000), env, pool, old, tc)
    runner = runner._replace(partner_ids=jnp.arange(episodes, dtype=jnp.int32)%pool.size)
    _, data = make_rollout(env, pool, old, tc)(state, runner)
    results = {}
    for name, config in [('legacy', old), ('bounded_same_weights', replace(old, style_injection='bounded'))]:
        results[name] = trace_summary(jax.jit(lambda d: trace(state.params, state.norm, d, config))(data))
        # Full-length derivative of the replayed actor objective, not an optimizer step.
        def objective(params):
            return jnp.mean(replay(params, state.norm, data, config)['logits']**2)
        loss, gradient = jax.jit(jax.value_and_grad(objective))(state.params)
        results[name]['gradient_probe'] = dict(loss=float(loss), gradient_norm=float(tree_norm(gradient)),
            finite=bool(all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(gradient))),
            objective='mean squared actor logits over the same recorded history; not PPO')
    if not results['bounded_same_weights']['gradient_probe']['finite']:
        raise FloatingPointError('Bounded recurrence gradient is nonfinite')
    if results['bounded_same_weights']['hidden_max']['maximum'] > 1.00001:
        raise AssertionError('Bounded GRU hidden state escaped [-1,1]')
    if before != tree_hash(state):
        raise AssertionError('Stability diagnostic changed legacy weights')
    report = dict(results=results, horizon=horizon, episodes=episodes,
        same_weights=True, same_recorded_history=True, legacy_weights_unchanged=True,
        control='Only style_injection configuration changes; original injection matrix stays unchanged',
        warning='Inference/gradient stability evidence, not a retrained reward ablation')
    write_json(output/'legacy_stability.json', report)
    return report


def train_batches(name, out, state, runner, collect, update, start_update, stop_update, stage, env_batch, steps, self_play=False):
    for index in range(start_update, stop_update):
        start = time.perf_counter()
        runner, batch = collect(state, runner)
        state, metrics = update(state, batch)
        jax.block_until_ready(metrics['loss'])
        row = {k: float(v) for k, v in metrics.items()}
        if row['finite_gradient'] != 1 or not np.isfinite(list(row.values())).all():
            raise FloatingPointError(name+' produced nonfinite gradients/metrics')
        row.update(name=name, stage=stage, cumulative_update=index+1,
            seconds=time.perf_counter()-start, environment_steps=env_batch*steps,
            ppo_role_transitions=env_batch*steps*(2 if self_play else 1), includes_compile=index == start_update)
        with (out/'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        if index == start_update or (index+1)%32 == 0 or index+1 == stop_update:
            print(json.dumps({k: row[k] for k in ('name', 'stage', 'cumulative_update', 'seconds', 'reward_mean', 'loss')}), flush=True)
    return state, runner


def selfplay_eval(state, env, config, episodes, seed):
    if getattr(env, 'training_shaping_coef', 0.) != 0:
        raise ValueError('Qualification evaluation must use sparse reward only')
    before = tree_hash(state)
    specification = SelfPlaySpecification()
    tc = TrainConfig(num_envs=episodes, num_steps=env.horizon, epochs=1)
    runner = init_runner(jax.random.PRNGKey(seed), env, specification, config, tc, self_play=True)
    _, data = make_rollout(env, specification, config, tc, self_play=True)(state, runner)
    returns = rollout_returns(data, episodes, self_play=True)
    if before != tree_hash(state):
        raise AssertionError('Self-play evaluation changed checkpoint')
    return returns


def static_eval(state, env, pool, config, episodes_per_partner, seed):
    if getattr(env, 'training_shaping_coef', 0.) != 0:
        raise ValueError('XP evaluation must use sparse reward only')
    before = tree_hash(state)
    count = episodes_per_partner*pool.size
    tc = TrainConfig(num_envs=count, num_steps=env.horizon, epochs=1)
    runner = init_runner(jax.random.PRNGKey(seed), env, pool, config, tc)
    ids = jnp.repeat(jnp.arange(pool.size, dtype=jnp.int32), episodes_per_partner)
    runner = runner._replace(partner_ids=ids)
    _, data = make_rollout(env, pool, config, tc)(state, runner)
    returns = rollout_returns(data, count)
    records = [dict(partner=pool.identities[int(ids[i])], partner_hash=pool.content_hashes[int(ids[i])],
        episode=i%episodes_per_partner, rollout_lane=i, rollout_seed=seed, episode_return=float(value))
        for i, value in enumerate(returns)]
    if before != tree_hash(state):
        raise AssertionError('XP evaluation changed checkpoint')
    return dict(records=records, mean_return=float(returns.mean()), success_rate=float((returns > 0).mean()),
        parameters_unchanged=True, normalization_unchanged=True, sparse_reward_only=True,
        published_metrics_reproduced=False, benchmark_claim=False)


def gradient_diagnostics(state, data, config, tc):
    t = data.transitions
    advantage, returns = generalized_advantage(t.reward, t.value, t.next_value,
        t.terminated, t.truncated, tc.gamma, tc.gae_lambda)
    advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
    targets, mask = future_targets(t.partner_action, t.terminated | t.truncated, config.prediction_horizon)
    def losses(params, aux_only):
        out = replay(params, state.norm, data, config)
        lp = jax.nn.log_softmax(out['logits'])
        if aux_only:
            return auxiliary(out, targets, mask, t.partner_id, config, tc)['auxiliary_loss']
        chosen = jnp.take_along_axis(lp, t.action[..., None], -1).squeeze(-1)
        ratio = jnp.exp(chosen-t.log_prob)
        policy = -jnp.minimum(ratio*advantage, jnp.clip(ratio, 1-tc.clip_epsilon, 1+tc.clip_epsilon)*advantage).mean()
        clipped = t.value+jnp.clip(out['value']-t.value, -tc.clip_epsilon, tc.clip_epsilon)
        value = .5*jnp.maximum((out['value']-returns)**2, (clipped-returns)**2).mean()
        entropy = -(jnp.exp(lp)*lp).sum(-1).mean()
        return policy+tc.value_coef*value-tc.entropy_coef*entropy
    rl, rg = jax.jit(jax.value_and_grad(lambda p: losses(p, False)))(state.params)
    aux, ag = jax.jit(jax.value_and_grad(lambda p: losses(p, True)))(state.params)
    groups = {}
    for name in rg:
        rn, an = float(tree_norm(rg[name])), float(tree_norm(ag[name]))
        groups[name] = dict(rl_norm=rn, auxiliary_norm=an, auxiliary_to_rl=an/rn if rn > 1e-12 else None)
    out = replay(state.params, state.norm, data, config)
    ce = -jnp.take_along_axis(jax.nn.log_softmax(out['future_action_logits']), targets[..., None], -1).squeeze(-1)
    total_ce, count = float((ce*mask).sum()), int(mask.sum())
    return dict(rl_loss=float(rl), auxiliary_loss=float(aux), groups=groups,
        gradients_finite=bool(all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves((rg, ag)))),
        intent_cross_entropy_sum=total_ce, valid_targets=count,
        intent_ce_per_target=total_ce/count if count else None, uniform_ce=float(np.log(config.num_actions)),
        context='Fresh frozen held-out batch; no optimizer/normalizer update')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--legacy-pilot', required=True)
    parser.add_argument('--pool-stages', type=parse_stages, default=parse_stages('128,256'))
    parser.add_argument('--ego-updates', type=int, default=128,
        help='PPO batches per ego variant; Full additionally collects --pretrain-updates batches')
    parser.add_argument('--pretrain-updates', type=int, default=8)
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--horizon', type=int, default=400)
    parser.add_argument('--qualification-episodes', type=int, default=64)
    parser.add_argument('--eval-episodes-per-partner', type=int, default=32)
    parser.add_argument('--stability-episodes', type=int, default=16)
    parser.add_argument('--diagnostic-batches', type=int, default=3)
    parser.add_argument('--min-success-rate', type=float, default=.25)
    parser.add_argument('--intent-coef', type=float, default=.1,
        help='Intent coefficient in joint PPO updates')
    parser.add_argument('--pretrain-intent-coef', type=float, default=1.,
        help='Intent coefficient in auxiliary-only pretraining updates')
    parser.add_argument('--training-shaping-coef', type=float, default=0.,
        help='Explicit train-only protocol: sparse + coefficient * sum backend shaped rewards; evaluations stay sparse')
    parser.add_argument('--run-unqualified-diagnostic', action='store_true')
    args = parser.parse_args()
    if (min(args.ego_updates, args.num_envs, args.steps, args.epochs, args.eval_episodes_per_partner,
            args.stability_episodes, args.diagnostic_batches) < 1 or args.qualification_episodes < 2
            or args.pretrain_updates < 0 or args.horizon < 400 or args.intent_coef < 0
            or not np.isfinite(args.intent_coef)
            or args.pretrain_intent_coef < 0 or not np.isfinite(args.pretrain_intent_coef)
            or args.training_shaping_coef < 0 or not np.isfinite(args.training_shaping_coef)
            or not 0 <= args.min_success_rate <= 1):
        parser.error('Invalid budget; long-horizon validation requires horizon >= 400')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    env = OvercookedV2(horizon=args.horizon, public_actions=True)
    train_env = OvercookedV2(horizon=args.horizon, public_actions=True,
                           training_shaping_coef=args.training_shaping_coef)
    config = ModelConfig(env.obs_dim, env.task_dim, global_state_dim=env.global_state_dim)
    tc = TrainConfig(num_envs=args.num_envs, num_steps=args.steps, epochs=args.epochs,
        intent_coef=args.intent_coef, pretrain_intent_coef=args.pretrain_intent_coef)
    metadata = dict(version='v0.1.1', source_hashes=source_manifest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model=asdict(config), training=asdict(tc),
        budget=vars(args), environment=env.name, sparse_reward_only=args.training_shaping_coef == 0,
        evaluation_sparse_reward_only=True, training_shaping_coef=args.training_shaping_coef,
        reward_protocol='sparse_only' if args.training_shaping_coef == 0 else 'train_only_constant_backend_shaping',
        shaping_schedule='constant; no annealing; optional protocol differs from sparse baseline', task_context='spatial',
        jax_version=jax.__version__, devices=[str(x) for x in jax.devices()],
        matmul_precision=str(jax.config.jax_default_matmul_precision), pool_seeds=[101, 102, 201, 202],
        ego_seed=7, published_metrics_reproduced=False, statistical_superiority_claim=False,
        max_pool_environment_steps=4*args.pool_stages[-1]*args.num_envs*args.steps,
        max_ego_environment_steps=(2*args.ego_updates+args.pretrain_updates)*args.num_envs*args.steps,
        training_budget_accounting=training_budgets(args.num_envs, args.steps, args.pool_stages[-1],
            args.ego_updates, args.pretrain_updates, args.epochs),
        self_play_ppo_roles=2, freeze_selection='First passing predeclared stage; final stage if none pass')
    write_json(output/'protocol.json', metadata)
    stability = stability_control(args.legacy_pilot, output, args.stability_episodes, args.horizon)
    baseline = replace(config, ablation='no_partner_model')
    specification = SelfPlaySpecification()
    collect_pool = make_rollout(train_env, specification, baseline, tc, self_play=True)
    update_pool = make_update(baseline, tc)
    random_state = init_train_state(jax.random.PRNGKey(9001), baseline)
    random_params = dict(random_state.params)
    random_params['actor'] = jax.tree_util.tree_map(jnp.zeros_like, random_params['actor'])
    random_state = random_state._replace(params=random_params)
    random_returns = selfplay_eval(random_state, env, baseline, args.qualification_episodes, 33000)
    write_json(output/'uniform_selfplay.json', dict(returns=random_returns.tolist(), mean_return=float(random_returns.mean()),
        seed=33000, episodes=args.qualification_episodes, sparse_reward_only=True))
    pool_results, selected_paths = [], []
    for seed in metadata['pool_seeds']:
        state = init_train_state(jax.random.PRNGKey(seed), baseline)
        runner = init_runner(jax.random.PRNGKey(seed+1000), train_env, specification, baseline, tc, self_play=True)
        start_update, stages, selected_stage = 0, [], None
        for stop_update in args.pool_stages:
            state, runner = train_batches(f'pool_seed{seed}', output, state, runner, collect_pool, update_pool,
                start_update, stop_update, 'self_play', args.num_envs, args.steps, self_play=True)
            path = output/f'pool_seed{seed}'/f'update_{stop_update}'
            returns = selfplay_eval(state, env, baseline, args.qualification_episodes, 33000)
            gate = qualification(returns, random_returns, args.min_success_rate)
            save_checkpoint(path, state, baseline, dict(metadata, pool_seed=seed, cumulative_updates=stop_update,
                qualification=gate, self_play_both_roles=True))
            stage = dict(checkpoint=str(path), cumulative_updates=stop_update,
                environment_steps=stop_update*args.num_envs*args.steps, qualification=gate, returns=returns.tolist())
            stages.append(stage)
            print(json.dumps(dict(stage='pool_qualification', seed=seed, **gate)), flush=True)
            start_update = stop_update
            if gate['qualified'] and selected_stage is None:
                selected_stage = stage
        # Keep/report every predeclared stage, even after a screening pass.
        selected_stage = selected_stage or stages[-1]
        selected_paths.append(Path(selected_stage['checkpoint']))
        pool_results.append(dict(seed=seed, selected_checkpoint=selected_stage['checkpoint'],
            qualified=selected_stage['qualification']['qualified'], stages=stages))
    pool_qualified = all(x['qualified'] for x in pool_results)
    write_json(output/'pool_qualification.json', dict(pool_qualified=pool_qualified, partners=pool_results,
        main_comparison_eligible=pool_qualified, random_returns=random_returns.tolist()))
    results = []
    frozen_pool_unchanged = None
    if pool_qualified or args.run_unqualified_diagnostic:
        train_pool, test_pool = ModelPool(selected_paths[:2]), ModelPool(selected_paths[2:])
        assert_disjoint(train_pool, test_pool)
        frozen = tree_hash({'train': train_pool.params, 'test': test_pool.params, 'train_norms': train_pool.norms, 'test_norms': test_pool.norms})
        random_xp = static_eval(random_state, env, test_pool, baseline, args.eval_episodes_per_partner, 10000)
        write_json(output/'uniform_heldout.json', random_xp)
        for variant in ('full', 'no_partner_model'):
            vc = replace(config, ablation=variant)
            state = init_train_state(jax.random.PRNGKey(7), vc)
            runner = init_runner(jax.random.PRNGKey(8), train_env, train_pool, vc, tc)
            collect = make_rollout(train_env, train_pool, vc, tc)
            if variant == 'full' and args.pretrain_updates:
                state, runner = train_batches(variant, output, state, runner, collect, make_update(vc, tc, pretrain=True),
                    0, args.pretrain_updates, 'pretrain', args.num_envs, args.steps)
            state, runner = train_batches(variant, output, state, runner, collect, make_update(vc, tc),
                0, args.ego_updates, 'ppo', args.num_envs, args.steps)
            path = output/variant/'checkpoint'
            save_checkpoint(path, state, vc, dict(metadata, pool_qualified=pool_qualified,
                diagnostic_only=not pool_qualified, train_partners=train_pool.manifest, heldout_partners=test_pool.manifest))
            xp = static_eval(state, env, test_pool, vc, args.eval_episodes_per_partner, 10000)
            xp.update(pool_qualified=pool_qualified, main_comparison_eligible=pool_qualified)
            write_json(output/variant/'xp.json', xp)
            switch = evaluate_switch(state, env, test_pool, vc, args.horizon//2, 20,
                args.eval_episodes_per_partner, seed=20000)
            switch.update(pool_qualified=pool_qualified, main_comparison_eligible=pool_qualified)
            write_json(output/variant/'switch.json', switch)
            if variant == 'full':
                diag_runner = init_runner(jax.random.PRNGKey(55000), env, test_pool, vc, tc)
                diag_collect = make_rollout(env, test_pool, vc, tc)
                diagnostics = []
                before = tree_hash(state)
                for _ in range(args.diagnostic_batches):
                    diag_runner, data = diag_collect(state, diag_runner)
                    diagnostics.append(gradient_diagnostics(state, data, vc, tc))
                if before != tree_hash(state) or not all(x['gradients_finite'] for x in diagnostics):
                    raise AssertionError('Frozen multi-batch gradient diagnostic failed')
                count = sum(x['valid_targets'] for x in diagnostics)
                write_json(output/variant/'gradient_diagnostics.json', dict(batches=diagnostics,
                    weighted_intent_ce_per_target=sum(x['intent_cross_entropy_sum'] for x in diagnostics)/count if count else None,
                    uniform_ce=float(np.log(vc.num_actions)), coefficients=asdict(tc),
                    warning='Limited held-out batches; not a whole-training gradient balance claim'))
            result = dict(variant=variant, checkpoint=str(path), xp_mean_return=xp['mean_return'],
                xp_success_rate=xp['success_rate'], post_switch_mean_return=switch['mean_post_switch_return'],
                normalized_drop=switch['mean_normalized_drop'], recovery_steps=switch['mean_recovery_steps_observed'],
                pool_qualified=pool_qualified, main_comparison_eligible=pool_qualified, published_metrics_reproduced=False)
            results.append(result)
            print(json.dumps(dict(stage='evaluation', **result)), flush=True)
        if frozen != tree_hash({'train': train_pool.params, 'test': test_pool.params, 'train_norms': train_pool.norms, 'test_norms': test_pool.norms}):
            raise AssertionError('Frozen partner pool changed')
        frozen_pool_unchanged = True
    write_json(output/'summary.json', dict(status='complete', protocol=metadata, pool_results=pool_results,
        pool_qualified=pool_qualified, main_comparison_eligible=pool_qualified,
        ego_training_skipped=not results, results=results, legacy_weights_unchanged=stability['legacy_weights_unchanged'],
        frozen_pool_unchanged=frozen_pool_unchanged,
        published_metrics_reproduced=False, statistical_superiority_claim=False,
        note='One ego seed and staged pool screening; preserve zeros/null recovery; no comparative superiority conclusion'))
    print(json.dumps(dict(status='complete', output=str(output.resolve()), pool_qualified=pool_qualified)), flush=True)


if __name__ == '__main__':
    main()
