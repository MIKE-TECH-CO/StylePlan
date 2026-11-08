"""Opt-in v0.1.2 warm-start experiment with an immutable v0.1.1 partner pool.

Only training-partner validation selects a configuration. Held-out evaluation
starts after selection.json is written. Runner seeds share a trained starting
checkpoint: these are continuation replicates, not independent training seeds.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
import time
from threading import Lock

import jax
import jax.numpy as jnp
import numpy as np

from styleplan_paper.checkpoint import load_checkpoint, save_checkpoint, source_manifest
from styleplan_paper.config import TrainConfig
from styleplan_paper.envs import OvercookedV2
from styleplan_paper.partners import ModelPool, assert_disjoint, tree_hash
from styleplan_paper.training import init_runner, make_rollout, make_update


POOL_STAGES = {101: 1024, 102: 1024, 201: 256, 202: 1024}
DEFAULT_TRIALS = [
    dict(name='control', overrides={}, shaping='constant'),
    dict(name='more_epochs', overrides={'epochs': 8}, shaping='constant'),
    dict(name='anneal_shaping', overrides={}, shaping='linear_to_zero'),
]


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2)


def validate_trials(trials):
    if not isinstance(trials, list) or not 1 <= len(trials) <= 3:
        raise ValueError('Specify a control and at most two candidate configurations')
    names = set()
    allowed = set(asdict(TrainConfig()))-{'num_envs', 'num_steps'}
    cleaned = []
    for trial in trials:
        if not isinstance(trial, dict) or set(trial) != {'name', 'overrides', 'shaping'}:
            raise ValueError('A trial requires exactly name, overrides and shaping')
        name = trial['name']
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', name) or name in names:
            raise ValueError('Trial names must be unique safe directory components')
        overrides = trial['overrides']
        if not isinstance(overrides, dict) or not set(overrides) <= allowed:
            raise ValueError('Only TrainConfig loss/optimizer overrides are allowed; sampling budgets are fixed')
        try:
            replace(TrainConfig(), **overrides)
            if any(not np.isfinite(value) for value in overrides.values()):
                raise ValueError('All override values must be finite')
            if 'epochs' in overrides and (not isinstance(overrides['epochs'], int) or isinstance(overrides['epochs'], bool)):
                raise ValueError('epochs must be an integer')
        except TypeError as exc:
            raise ValueError('Training overrides must be numeric') from exc
        if trial['shaping'] not in ('constant', 'linear_to_zero'):
            raise ValueError('Unknown training reward schedule')
        names.add(name)
        cleaned.append(dict(name=name, overrides=dict(overrides), shaping=trial['shaping']))
    if cleaned[0] != dict(name='control', overrides={}, shaping='constant'):
        raise ValueError('The first trial must be the unchanged continuation control')
    return cleaned


def shaping_coefficient(schedule, update, updates, start=1.):
    if updates < 1 or not 0 <= update < updates or not np.isfinite(start) or start < 0:
        raise ValueError('Invalid shaping schedule input')
    if schedule == 'constant':
        return float(start)
    if schedule == 'linear_to_zero':
        # Reach zero during the first 75%, then train the last 25% on sparse reward.
        anneal_updates = int(np.ceil(.75*updates))
        return float(start*max(1-update/(anneal_updates-1), 0.)) if anneal_updates > 1 else 0.
    raise ValueError('Unknown shaping schedule')


def select_trial(validation_rows):
    """Stable declaration-order tie break; cannot inspect test metrics."""
    if not validation_rows or validation_rows[0]['name'] != 'control':
        raise ValueError('Validation rows must start with control')
    if any(not np.isfinite(row['validation_mean_return']) for row in validation_rows):
        raise ValueError('Nonfinite validation score')
    return max(validation_rows, key=lambda row: row['validation_mean_return'])['name']


def checkpoint_files(paths):
    return [Path(path)/name for path in paths for name in ('metadata.json', 'arrays.npz')]


def file_hashes(paths):
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def verify_hashes(expected):
    actual = file_hashes([Path(path) for path in expected])
    if actual != expected:
        raise AssertionError('A reused checkpoint was changed: '+', '.join(k for k in expected if actual[k] != expected[k]))


def first_episode_returns(data, episodes):
    t = data.transitions
    rewards = np.asarray(t.reward)
    done = np.asarray(t.terminated | t.truncated)
    if rewards.ndim != 2 or rewards.shape[1] != episodes or done.shape != rewards.shape:
        raise AssertionError('Unexpected evaluation batch shape')
    if done[:-1].any() or not done[-1].all():
        raise AssertionError('Evaluation must contain exactly one complete episode per lane')
    if not np.isfinite(rewards).all():
        raise FloatingPointError('Nonfinite evaluation reward')
    return rewards.sum(0)


def paired_summary(left, right, *, stratum_sizes, bootstrap_seed=912):
    """Resample paired episodes inside fixed contiguous partner strata.

    Each stratum keeps its observed sample count and weight. If several runner
    seeds are concatenated, the caller must pass separate seed-by-partner strata;
    this measures evaluation noise conditional on those frozen policies.
    """
    left, right = np.asarray(left, float), np.asarray(right, float)
    if left.ndim != 1 or left.shape != right.shape or len(left) < 2 or not np.isfinite(left-right).all():
        raise ValueError('Paired comparisons require matching finite episode arrays')
    sizes = tuple(stratum_sizes)
    if (not sizes or any(not isinstance(size, (int, np.integer)) or isinstance(size, (bool, np.bool_))
                         or size < 1 for size in sizes) or sum(sizes) != len(left)):
        raise ValueError('Positive integer stratum sizes must cover the paired episode array exactly')
    difference = left-right
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_totals = np.zeros(2000)
    offset = 0
    for size in sizes:
        indices = rng.integers(offset, offset+size, (2000, size))
        bootstrap_totals += difference[indices].sum(1)
        offset += size
    interval = np.quantile(bootstrap_totals/len(left), [.025, .975]).tolist()
    return dict(mean_difference=float(difference.mean()), episode_bootstrap_95=interval,
                paired_episodes=len(left), bootstrap_seed=bootstrap_seed,
                stratum_sizes=[int(size) for size in sizes], bootstrap_samples=2000,
                stratification='Fixed contiguous partner blocks; separate seed-by-partner blocks if runner seeds are pooled',
                scope='Stratified paired episode bootstrap of evaluation noise conditional on these frozen policies; not independent-seed significance')


class CompiledOperations:
    """Keep identical rollout/update callables alive across continuation trials."""
    def __init__(self, train_env, eval_env, train_pool):
        self.train_env, self.eval_env, self.train_pool = train_env, eval_env, train_pool
        self.collectors, self.updates, self.evaluators = {}, {}, {}
        self.cache_lock = Lock()

    def training(self, config, tc, dynamic_shaping=False):
        # Collection only uses num_envs and num_steps from TrainConfig.
        collect_key = (config, tc.num_envs, tc.num_steps, dynamic_shaping)
        with self.cache_lock:
            if collect_key not in self.collectors:
                self.collectors[collect_key] = make_rollout(self.train_env, self.train_pool, config, tc,
                                                           dynamic_shaping=dynamic_shaping)
            update_key = (config, tc)
            if update_key not in self.updates:
                self.updates[update_key] = make_update(config, tc)
        return self.collectors[collect_key], self.updates[update_key]

    def evaluate(self, state, pool, config, episodes_per_partner, seed):
        if getattr(self.eval_env, 'training_shaping_coef', 0.) != 0.:
            raise ValueError('All reported evaluations must use sparse rewards')
        count = episodes_per_partner*pool.size
        key = (id(pool), config, episodes_per_partner)
        tc = TrainConfig(num_envs=count, num_steps=self.eval_env.horizon, epochs=1)
        if key not in self.evaluators:
            self.evaluators[key] = make_rollout(self.eval_env, pool, config, tc)
        before = tree_hash(state)
        runner = init_runner(jax.random.PRNGKey(seed), self.eval_env, pool, config, tc)
        ids = jnp.repeat(jnp.arange(pool.size, dtype=jnp.int32), episodes_per_partner)
        runner = runner._replace(partner_ids=ids)
        _, data = self.evaluators[key](state, runner)
        returns = first_episode_returns(data, count)
        if before != tree_hash(state):
            raise AssertionError('Frozen evaluation changed parameters, optimizer or normalizer')
        return dict(mean_return=float(returns.mean()), success_rate=float((returns > 0).mean()),
            returns=returns.tolist(), episodes=count, episodes_per_partner=episodes_per_partner,
            rollout_seed=seed, partner_hashes=list(pool.content_hashes),
            per_partner=[dict(identity=pool.identities[index], content_hash=pool.content_hashes[index],
                mean_return=float(returns[index*episodes_per_partner:(index+1)*episodes_per_partner].mean()),
                successes=int((returns[index*episodes_per_partner:(index+1)*episodes_per_partner] > 0).sum()))
                for index in range(pool.size)], sparse_reward_only=True, frozen_state_unchanged=True)


def train_trial(output, trial, seed, source, base_tc, updates, operations, shaping_start, provenance):
    """Start every trial from the same immutable TrainState, with a fresh runner."""
    output.mkdir(parents=True, exist_ok=False)
    config, state = source['config'], source['state']
    tc = replace(base_tc, **trial['overrides'])
    before = tree_hash(state)
    collect, update = operations.training(config, tc, dynamic_shaping=True)
    runner = init_runner(jax.random.PRNGKey(seed), operations.train_env,
                         operations.train_pool, config, tc)
    metadata = dict(version='v0.1.2', trial=trial, continuation_runner_seed=seed,
        resume=dict(initial_state_hash=before, optimizer_count=int(state.optimizer.count),
            normalizer_count=float(state.norm.count), preserves_parameters_optimizer_normalizer=True,
            environment_and_recurrent_state='reset at continuation start; not restored from v0.1.1'),
        training=asdict(tc), model=asdict(config), source=provenance,
        source_hashes=source_manifest(), updates=updates,
        physical_environment_steps=updates*tc.num_envs*tc.num_steps,
        optimizer_transition_uses=updates*tc.num_envs*tc.num_steps*tc.epochs,
        training_shaping_start=shaping_start, training_shaping_schedule=trial['shaping'],
        training_reward_protocol='reward_mean in metrics is sparse team reward plus coefficient times the sum of backend shaping rewards',
        evaluation_sparse_reward_only=True, independent_initialization_seed=False)
    write_json(output/'trial.json', metadata)
    trial_started = time.perf_counter()
    for index in range(updates):
        start = time.perf_counter()
        coefficient = shaping_coefficient(trial['shaping'], index, updates, shaping_start)
        runner, batch = collect(state, runner, jnp.asarray(coefficient, jnp.float32))
        state, metrics = update(state, batch)
        jax.block_until_ready(metrics['loss'])
        row = {key: float(value) for key, value in metrics.items()}
        if row['finite_gradient'] != 1. or not np.isfinite(list(row.values())).all():
            raise FloatingPointError('Nonfinite training metrics in '+trial['name'])
        row.update(update=index+1, seconds=time.perf_counter()-start, shaping_coefficient=coefficient,
                   physical_environment_steps=tc.num_envs*tc.num_steps)
        with (output/'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        if index == 0 or (index+1)%64 == 0 or index+1 == updates:
            print(json.dumps(dict(stage='continuation', name=trial['name'], seed=seed,
                **{key: row[key] for key in ('update', 'seconds', 'reward_mean', 'loss', 'shaping_coefficient')})), flush=True)
    if tree_hash(source['state']) != before:
        raise AssertionError('Trial mutated the shared starting checkpoint')
    metadata.update(final_state_hash=tree_hash(state), final_optimizer_count=int(state.optimizer.count),
                    elapsed_seconds=time.perf_counter()-trial_started)
    save_checkpoint(output/'checkpoint', state, config, metadata)
    write_json(output/'complete.json', metadata)
    return dict(state=state, config=config, path=output, metadata=metadata)


def parse_seeds(value):
    try:
        seeds = tuple(int(part) for part in value.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Seeds must be integers') from exc
    if not seeds or seeds[0] != 7 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError('Unique nonnegative continuation seeds must start with screen seed 7')
    return seeds


def execute_jobs(jobs, workers):
    """Run independent trials, return in declaration order, propagate failures."""
    if workers < 1:
        raise ValueError('Parallel worker count must be positive')
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {name: executor.submit(train_trial, **kwargs) for name, kwargs in jobs.items()}
        return {name: future.result() for name, future in futures.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True, help='Immutable v0.1.1 extended_shaping run')
    parser.add_argument('--output', required=True)
    parser.add_argument('--trial-json', help='Predeclared JSON list of control and up to two candidates')
    parser.add_argument('--updates', type=int, default=512)
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--horizon', type=int, default=400)
    parser.add_argument('--validation-episodes-per-partner', type=int, default=64)
    parser.add_argument('--test-episodes-per-partner', type=int, default=128)
    parser.add_argument('--replicate-seeds', type=parse_seeds, default=parse_seeds('7,17,27'))
    parser.add_argument('--training-shaping-start', type=float, default=1.)
    parser.add_argument('--parallel-trials', type=int, default=2)
    args = parser.parse_args()
    if min(args.updates, args.num_envs, args.steps, args.validation_episodes_per_partner,
           args.test_episodes_per_partner, args.parallel_trials) < 1 or args.horizon < 400:
        parser.error('Positive budgets and a complete 400-step-or-longer horizon are required')
    if not np.isfinite(args.training_shaping_start) or args.training_shaping_start < 0:
        parser.error('Invalid training shaping coefficient')
    trials = validate_trials(json.loads(Path(args.trial_json).read_text()) if args.trial_json else DEFAULT_TRIALS)
    source_run, output = Path(args.source_run).resolve(), Path(args.output).resolve()
    if source_run == output or source_run in output.parents:
        parser.error('New results must be outside the immutable v0.1.1 source run')
    if output.exists():
        raise FileExistsError('Refusing to overwrite '+str(output))
    paths = {seed: source_run/f'pool_seed{seed}'/f'update_{stage}' for seed, stage in POOL_STAGES.items()}
    source_paths = {variant: source_run/variant/'checkpoint' for variant in ('full', 'no_partner_model')}
    protected = file_hashes(checkpoint_files([*paths.values(), *source_paths.values()]))
    for path in paths.values():
        meta = json.loads((path/'metadata.json').read_text())
        if not meta.get('metadata', {}).get('qualification', {}).get('qualified'):
            raise ValueError('The fixed reused partner checkpoint was not qualified: '+str(path))
    sources = {name: load_checkpoint(path) for name, path in source_paths.items()}
    for name, source in sources.items():
        if source['config'].ablation != name:
            raise ValueError('Resume checkpoint has the wrong ablation')
    if replace(sources['full']['config'], ablation='no_partner_model') != sources['no_partner_model']['config']:
        raise ValueError('Starting Full/NoPM architectures must differ only in ablation')
    train_pool = ModelPool([paths[101], paths[102]])
    original_train_pool_hash = tree_hash((train_pool.params, train_pool.norms))
    env = OvercookedV2(horizon=args.horizon, public_actions=True)
    train_env = OvercookedV2(horizon=args.horizon, public_actions=True,
                           training_shaping_coef=args.training_shaping_start)
    config = sources['full']['config']
    if (config.obs_dim, config.task_dim, config.global_state_dim) != (env.obs_dim, env.task_dim, env.global_state_dim):
        raise ValueError('Resume checkpoint observation contract does not match the environment')
    tc_data = sources['full']['metadata']['metadata'].get('training')
    if not tc_data or tc_data != sources['no_partner_model']['metadata']['metadata'].get('training'):
        raise ValueError('Starting checkpoints must declare the same training configuration')
    base_tc = replace(TrainConfig(**tc_data), num_envs=args.num_envs, num_steps=args.steps)
    if base_tc.epochs != 2 or base_tc.intent_coef != .1:
        raise ValueError('The declared v0.1.2 control requires v0.1.1 epochs=2 and intent_coef=0.1')
    output.mkdir(parents=True, exist_ok=False)
    experiment_started = time.perf_counter()
    protocol = dict(version='v0.1.2', arguments=vars(args), trials=trials, training=asdict(base_tc),
        source_hashes=source_manifest(), harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reused_checkpoint_file_hashes=protected, fixed_pool_stages=POOL_STAGES,
        original_train_pool_hash=original_train_pool_hash,
        train_partner_seeds=[101, 102], heldout_partner_seeds=[201, 202],
        selection='Maximum sparse mean return on training partners; ties use declaration order, control first',
        annealing_schedule='Linearly reduce shaping coefficient to zero over the first 75% of updates; zero in the final 25%',
        no_pm_comparison='Apply the selected Full training factors to NoPM; same additional sampling and optimization budgets',
        validation_rollout_seed=71007, screen_continuation_seed=7,
        heldout_access='No new heldout rollout before writing selection.json; previous v0.1.1 published results exist',
        replicate_scope='Same v0.1.1 checkpoint with different continuation runner RNGs; not independent initializations',
        pool_training_steps=0, continuation_steps_per_trial=args.updates*args.num_envs*args.steps,
        jax_version=jax.__version__, devices=[str(device) for device in jax.devices()],
        matmul_precision=str(jax.config.jax_default_matmul_precision),
        published_metrics_reproduced=False, statistical_superiority_claim=False)
    write_json(output/'protocol.json', protocol)
    operations = CompiledOperations(train_env, env, train_pool)
    provenance = {name: dict(path=str(path), state_hash=tree_hash(sources[name]['state']),
        array_sha256=protected[str((path/'arrays.npz').resolve())]) for name, path in source_paths.items()}
    common = dict(base_tc=base_tc, updates=args.updates, operations=operations,
                  shaping_start=args.training_shaping_start)
    jobs = {trial['name']: dict(common, output=output/'screen'/trial['name'], trial=trial, seed=7,
                source=sources['full'], provenance=provenance['full']) for trial in trials}
    screen = execute_jobs(jobs, args.parallel_trials)
    validation_rows = []
    completed_trials = [item['metadata'] for item in screen.values()]
    for trial in trials:
        item = screen[trial['name']]
        validation = operations.evaluate(item['state'], train_pool, item['config'],
            args.validation_episodes_per_partner, 71007)
        write_json(item['path']/'validation.json', validation)
        row = dict(name=trial['name'], validation_mean_return=validation['mean_return'],
                   validation_success_rate=validation['success_rate'], path=str(item['path']))
        validation_rows.append(row)
        print(json.dumps(dict(stage='validation', **row)), flush=True)
    selected = select_trial(validation_rows)
    selection = dict(selected=selected, validation=validation_rows, heldout_evaluations_so_far=0,
                     rule=protocol['selection'], trials=trials)
    if tree_hash((train_pool.params, train_pool.norms)) != original_train_pool_hash:
        raise AssertionError('Screening changed the frozen training partner pool')
    write_json(output/'selection.json', selection)
    print(json.dumps(dict(stage='selection', selected=selected)), flush=True)
    # Held-out parameters enter the process only after candidate selection.
    test_pool = ModelPool([paths[201], paths[202]])
    assert_disjoint(train_pool, test_pool)
    frozen_pool = tree_hash((train_pool.params, train_pool.norms, test_pool.params, test_pool.norms))
    chosen = next(trial for trial in trials if trial['name'] == selected)
    control = trials[0]
    results = []
    for seed in args.replicate_seeds:
        full_trials = {'control': control}
        if selected != 'control':
            full_trials['selected'] = chosen
        trained = {}
        jobs = {}
        for label, trial in full_trials.items():
            if seed == 7:
                trained[label] = screen[trial['name']]
            else:
                jobs[label] = dict(common, output=output/'replicates'/f'seed{seed}'/label,
                    trial=trial, seed=seed, source=sources['full'], provenance=provenance['full'])
        no_pm = dict(chosen, name='no_partner_model')
        jobs['no_partner_model'] = dict(common, output=output/'replicates'/f'seed{seed}'/'no_partner_model',
            trial=no_pm, seed=seed, source=sources['no_partner_model'], provenance=provenance['no_partner_model'])
        new_items = execute_jobs(jobs, args.parallel_trials)
        completed_trials.extend(item['metadata'] for item in new_items.values())
        trained.update(new_items)
        if selected == 'control':
            trained['selected'] = trained['control']
        evaluations = {}
        for label in ('control', 'selected', 'no_partner_model'):
            item = trained[label]
            if label == 'selected' and selected == 'control':
                evaluations[label] = evaluations['control']
                continue
            evaluations[label] = operations.evaluate(item['state'], test_pool, item['config'],
                args.test_episodes_per_partner, 91000+seed)
        for name, source in sources.items():
            evaluations['v011_'+name] = operations.evaluate(source['state'], test_pool, source['config'],
                args.test_episodes_per_partner, 91000+seed)
        stratum_sizes = [args.test_episodes_per_partner]*test_pool.size
        row = dict(continuation_seed=seed, checkpoints={name: str(item['path']/'checkpoint') for name, item in trained.items()},
            evaluations=evaluations,
            selected_minus_control=paired_summary(evaluations['selected']['returns'], evaluations['control']['returns'],
                stratum_sizes=stratum_sizes),
            selected_minus_no_pm=paired_summary(evaluations['selected']['returns'], evaluations['no_partner_model']['returns'],
                stratum_sizes=stratum_sizes),
            selected_minus_v011_full=paired_summary(evaluations['selected']['returns'], evaluations['v011_full']['returns'],
                stratum_sizes=stratum_sizes))
        write_json(output/f'heldout_seed{seed}.json', row)
        results.append(row)
        print(json.dumps(dict(stage='heldout', seed=seed, means={name: value['mean_return'] for name, value in evaluations.items()})), flush=True)
    verify_hashes(protected)
    if tree_hash((train_pool.params, train_pool.norms, test_pool.params, test_pool.norms)) != frozen_pool:
        raise AssertionError('Frozen partner parameters or normalization changed')
    means = {name: [row['evaluations'][name]['mean_return'] for row in results] for name in results[0]['evaluations']}
    summary = dict(status='complete', selected=selected, validation=validation_rows,
        heldout_means={name: dict(mean=float(np.mean(values)), per_continuation_seed=values) for name, values in means.items()},
        continuation_runner_seeds=list(args.replicate_seeds), independent_initialization_seeds=False,
        reused_checkpoint_files_unchanged=True, frozen_pool_unchanged=True,
        actual_training_trials=len(completed_trials),
        pool_training_steps=0, physical_environment_steps_per_trial=args.updates*args.num_envs*args.steps,
        selected_checkpoint_paths=[row['checkpoints']['selected'] for row in results],
        evaluations_sparse_reward_only=True, statistical_superiority_claim=False,
        elapsed_seconds=time.perf_counter()-experiment_started,
        actual_optimizer_transition_uses=sum(item['optimizer_transition_uses'] for item in completed_trials),
        note='Training-partner validation selected configuration; new heldout scores were never used for selection. '
             'Episode bootstraps and continuation replicates do not establish across-initialization method superiority.')
    summary['actual_physical_environment_steps'] = summary['actual_training_trials']*summary['physical_environment_steps_per_trial']
    write_json(output/'summary.json', summary)
    print(json.dumps(dict(status='complete', output=str(output), **summary['heldout_means'])), flush=True)


if __name__ == '__main__':
    main()
