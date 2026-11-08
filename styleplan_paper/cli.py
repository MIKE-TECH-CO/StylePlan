"""Training/evaluation entrypoints; output paths are always new and isolated."""
import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import time
import numpy as np
import jax
from .config import ModelConfig, TrainConfig, ABLATIONS
from .envs import SyntheticCoordination, OvercookedV2
from .partners import ScriptedPool, ModelPool, assert_disjoint, validate_pool
from .training import init_train_state, init_runner, make_rollout, make_update, evaluate
from .checkpoint import save_checkpoint, load_checkpoint, source_manifest
from .evaluation import evaluate_switch


def environment(args, loaded=None, sparse=False):
    if args.env == 'synthetic':
        if not sparse and getattr(args, 'training_shaping_coef', 0.):
            raise ValueError('--training-shaping-coef is only supported for Overcooked')
        return SyntheticCoordination(args.horizon)
    mode = getattr(args, 'task_context_mode', None)
    if mode is None and loaded is not None:
        meta = loaded['metadata']
        mode = ('legacy' if meta['schema_version'] == 2 else
            meta.get('metadata', {}).get('environment_config', {}).get('task_context_mode', 'spatial'))
    return OvercookedV2(args.layout, args.horizon, public_actions=args.public_actions,
        task_context_mode=mode or 'spatial',
        training_shaping_coef=0. if sparse else getattr(args, 'training_shaping_coef', 0.))


def model_config(args, env, loaded=None):
    # Unspecified architecture options inherit the resumed checkpoint. Explicit
    # overrides are checked against it below, never silently accepted.
    fields = asdict(loaded['config']) if loaded is not None else {}
    fields.update(obs_dim=env.obs_dim, task_dim=env.task_dim, global_state_dim=env.global_state_dim)
    for name in ('hidden_dim', 'history_window', 'intent_window', 'ablation',
                 'style_injection', 'slow_injection_scale'):
        value = getattr(args, name, None)
        if value is not None:
            fields[name] = value
    if loaded is None:
        fields['intent_window'] = min(fields.get('intent_window', 10), fields.get('history_window', 50))
    c = ModelConfig(**fields)
    if loaded is not None and c != loaded['config']:
        raise ValueError('Resume architecture/ablation must match exactly')
    return c


def partner_pool(args, env, heldout=False):
    paths = args.heldout_checkpoints if heldout else args.partner_checkpoints
    if paths:
        pool = ModelPool(paths)
    elif args.partner_factory:
        module, name = args.partner_factory.split(':', 1)
        pool = getattr(importlib.import_module(module), name)(env=env, heldout=heldout)
    elif args.env == 'synthetic':
        pool = ScriptedPool((3, 4) if heldout else (0, 1, 2))
    else:
        raise ValueError('Overcooked training/eval requires real frozen --partner-checkpoints or --partner-factory')
    validate_pool(pool, env.obs_dim, env.num_actions)
    return pool


class SelfPlaySpecification:
    size = 1
    identities = ['self-play/current-policy']
    content_hashes = ['dynamic-self-play-not-a-frozen-checkpoint']
    manifest = [dict(kind='SELF_PLAY', updated_between_batches=True)]


def run_train(args):
    if args.updates < 1 or args.pretrain_updates < 0:
        raise ValueError('updates must be positive and pretrain_updates nonnegative')
    loaded = load_checkpoint(args.resume) if args.resume else None
    env = environment(args, loaded)
    c = model_config(args, env, loaded)
    if args.self_play and c.ablation != 'no_partner_model':
        raise ValueError('--self-play is partner-pool training; set --ablation no_partner_model')
    pool = SelfPlaySpecification() if args.self_play else partner_pool(args, env)
    heldout = None
    if not args.self_play and (args.env == 'synthetic' or args.heldout_checkpoints or args.partner_factory):
        heldout = partner_pool(args, env, heldout=True)
        assert_disjoint(pool, heldout)
    tc = TrainConfig(num_envs=args.num_envs, num_steps=args.steps, epochs=args.epochs,
                     learning_rate=args.learning_rate, style_coef=args.style_coef,
                     intent_coef=args.intent_coef, pretrain_intent_coef=args.pretrain_intent_coef,
                     prior_coef=args.prior_coef,
                     smoothness_coef=args.smoothness_coef)
    if loaded is not None:
        state = loaded['state']
    else:
        state = init_train_state(jax.random.PRNGKey(args.seed), c)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(model=asdict(c), training=asdict(tc), environment=env.name,
        environment_config=dict(horizon=env.horizon, layout=args.layout if args.env == 'overcooked' else None,
            public_actions=env.public_actions, time_limit_is_terminal=env.time_limit_is_terminal,
            task_context_mode=getattr(env, 'task_context_mode', None),
            training_shaping_coef=getattr(env, 'training_shaping_coef', 0.),
            reward='sparse_team + training_shaping_coef * sum_both_agent_shaped_rewards',
            evaluation_training_shaping_coef=0.),
        requested_updates=args.updates,
        effective_pretrain_updates=0 if c.ablation == 'no_partner_model' else args.pretrain_updates,
        seed=args.seed, train_partners=pool.manifest,
        heldout_partners=[] if heldout is None else heldout.manifest,
        source_hashes=source_manifest(), devices=[str(x) for x in jax.devices()],
        self_play=args.self_play, resume=args.resume, resume_resets_environment=bool(args.resume),
        observation_contract='ego observation + public completed partner actions + visible position',
        benchmark_claim=False, note='Implementation validation only; published metrics not reproduced.')
    (output/'run.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    runner = init_runner(jax.random.PRNGKey(args.seed+1), env, pool, c, tc, self_play=args.self_play)
    rollout_fn = make_rollout(env, pool, c, tc, self_play=args.self_play)
    train_fn = make_update(c, tc)
    pretrain_fn = make_update(c, tc, pretrain=True)
    records = []
    pretrain_updates = 0 if c.ablation == 'no_partner_model' else args.pretrain_updates
    stages = [('pretrain', pretrain_updates, pretrain_fn), ('ppo', args.updates, train_fn)]
    for stage, count, update in stages:
        for step in range(count):
            start = time.perf_counter()
            runner, data = rollout_fn(state, runner)
            state, metrics = update(state, data)
            jax.block_until_ready(metrics['loss'])
            elapsed = time.perf_counter()-start
            row = {k: float(v) for k, v in metrics.items()}
            if row['finite_gradient'] != 1. or not all(np.isfinite(list(row.values()))):
                raise FloatingPointError('Nonfinite training loss/gradient; update was rejected')
            physical_steps = tc.num_envs*tc.num_steps
            row.update(stage=stage, update=step, seconds=elapsed,
                       physical_environment_steps=physical_steps,
                       policy_samples=physical_steps*(2 if args.self_play else 1),
                       physical_environment_steps_per_second=physical_steps/max(elapsed, 1e-9),
                       transitions=physical_steps,
                       transitions_per_second=physical_steps/max(elapsed, 1e-9),
                       includes_first_compile=step == 0)
            records.append(row)
            with (output/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            print(json.dumps({k: row[k] for k in ('stage', 'update', 'loss', 'reward_mean',
                'parameter_delta', 'physical_environment_steps', 'policy_samples', 'seconds')}), flush=True)
            if args.snapshot_every and stage == 'ppo' and (step+1)%args.snapshot_every == 0:
                save_checkpoint(output/f'snapshot_{step+1:06d}', state, c, dict(metadata, update=step+1))
    save_checkpoint(output/'checkpoint', state, c, dict(metadata, completed_updates=args.updates))
    if heldout is not None and args.eval_episodes:
        eval_env = environment(args, loaded, sparse=True)
        result = evaluate(state, eval_env, heldout, c, args.eval_episodes, seed=args.seed+10000)
        result['evaluation_training_shaping_coef'] = 0.
        result['training_pool_disjoint'] = True
        (output/'heldout_evaluation.json').write_text(json.dumps(result, indent=2))
    (output/'summary.json').write_text(json.dumps(dict(status='complete', environment=env.name,
        updates=args.updates, pretrain_updates=pretrain_updates, output=str(output.resolve()),
        published_metrics_reproduced=False, final_metrics=records[-1]), indent=2))
    return output


def run_eval(args):
    loaded = load_checkpoint(args.checkpoint)
    env = environment(args, loaded, sparse=True)
    pool = partner_pool(args, env, heldout=not bool(args.partner_checkpoints))
    c = loaded['config']
    if c.obs_dim != env.obs_dim or c.task_dim != env.task_dim or c.global_state_dim != env.global_state_dim:
        raise ValueError('Environment and checkpoint shapes differ')
    train = loaded['metadata']['metadata'].get('train_partners', [])
    known = {x['content_hash'] for x in train if 'content_hash' in x}
    if known & set(pool.content_hashes):
        raise ValueError('Evaluation pool overlaps checkpoint training pool')
    if args.switch_step is None:
        result = evaluate(loaded['state'], env, pool, c, args.eval_episodes, args.seed, args.deterministic)
    else:
        result = evaluate_switch(loaded['state'], env, pool, c, args.switch_step,
            args.score_window, args.eval_episodes, args.seed, args.deterministic)
    result['training_pool_disjoint'] = bool(known) and not (known & set(pool.content_hashes))
    result['evaluation_training_shaping_coef'] = 0.
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(dict(output=str(path.resolve()), mean_return=result['mean_return'])))


def build_parser():
    parser = argparse.ArgumentParser(description='Independent paper-faithful StylePlan in native JAX')
    sub = parser.add_subparsers(dest='command', required=True)
    train, ev = sub.add_parser('train'), sub.add_parser('evaluate')
    for p in (train, ev):
        p.add_argument('--env', choices=['synthetic', 'overcooked'], default='synthetic')
        p.add_argument('--layout', default='cramped_room')
        p.add_argument('--horizon', type=int, default=32)
        p.add_argument('--public-actions', action='store_true')
        p.add_argument('--task-context-mode', choices=['spatial', 'legacy'], default=None,
                       help='Defaults to spatial; checkpoint metadata selects evaluation/resume context')
        p.add_argument('--partner-checkpoints', nargs='+')
        p.add_argument('--heldout-checkpoints', nargs='+')
        p.add_argument('--partner-factory')
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--output', required=True)
        p.add_argument('--eval-episodes', type=int, default=2)
    train.add_argument('--updates', type=int, default=3)
    train.add_argument('--pretrain-updates', type=int, default=1)
    train.add_argument('--num-envs', type=int, default=4)
    train.add_argument('--steps', type=int, default=32)
    train.add_argument('--epochs', type=int, default=2)
    train.add_argument('--hidden-dim', type=int, default=None, help='New runs: 64; resume: checkpoint value')
    train.add_argument('--history-window', type=int, default=None, help='New runs: 50; resume: checkpoint value')
    train.add_argument('--intent-window', type=int, default=None, help='New runs: 10; resume: checkpoint value')
    train.add_argument('--learning-rate', type=float, default=3e-4)
    train.add_argument('--style-coef', type=float, default=0.)
    train.add_argument('--intent-coef', type=float, default=.1)
    train.add_argument('--pretrain-intent-coef', type=float, default=1.)
    train.add_argument('--style-injection', choices=['bounded', 'legacy'], default=None,
                       help='New runs: bounded; resume: checkpoint setting')
    train.add_argument('--slow-injection-scale', type=float, default=None,
                       help='New runs: 0.03; resume: checkpoint setting')
    train.add_argument('--training-shaping-coef', type=float, default=0.,
                       help='Explicit training-only shared shaped reward; evaluation always uses sparse reward')
    train.add_argument('--prior-coef', type=float, default=.01)
    train.add_argument('--smoothness-coef', type=float, default=.01)
    train.add_argument('--ablation', choices=ABLATIONS, default=None, help='New runs: full; resume: checkpoint value')
    train.add_argument('--self-play', action='store_true')
    train.add_argument('--snapshot-every', type=int, default=0)
    train.add_argument('--resume')
    ev.add_argument('--checkpoint', required=True)
    ev.add_argument('--deterministic', action='store_true')
    ev.add_argument('--switch-step', type=int)
    ev.add_argument('--score-window', type=int, default=10)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.eval_episodes < (1 if args.command == 'evaluate' else 0) or args.horizon <= 0:
        parser.error('Invalid episode count/horizon')
    run_train(args) if args.command == 'train' else run_eval(args)

if __name__ == '__main__':
    main()
