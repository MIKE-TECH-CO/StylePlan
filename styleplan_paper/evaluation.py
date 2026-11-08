"""Within-episode switches with explicit, independently inspectable metrics."""
import itertools
import numpy as np
import jax
import jax.numpy as jnp
from .config import TrainConfig
from .partners import tree_hash
from .training import init_runner, make_rollout, replay


def switch_metrics(reward, switch_step, window):
    """Use full non-crossing windows; never invent recovery for a censored run.

    A recovery at W means the first fully post-switch window reached the mean
    of the W steps immediately before the switch. Nonpositive pre-scores have
    undefined relative drop/recovery and are reported as null, not zero.
    """
    reward = np.asarray(reward, dtype=float)
    if reward.ndim != 1 or not np.isfinite(reward).all():
        raise ValueError('Expected a finite one-dimensional reward trajectory')
    if window < 1 or switch_step < window or len(reward)-switch_step < window:
        raise ValueError('Need a full score window on both sides of the switch')
    before = float(reward[switch_step-window:switch_step].mean())
    post = reward[switch_step:]
    rolling = np.convolve(post, np.ones(window)/window, mode='valid')
    recovery = np.flatnonzero(rolling >= before) if before > 0 else []
    return dict(pre_switch_score=before, post_switch_return=float(post.sum()),
        worst_post_switch_score=float(rolling.min()),
        normalized_drop=float(max(0., (before-rolling.min())/abs(before))) if before > 0 else None,
        recovery_steps=int(recovery[0]+window) if len(recovery) else None,
        recovery_censored=bool(not len(recovery)) if before > 0 else None,
        relative_metrics_defined=before > 0, post_switch_steps=len(post))


def evaluate_switch(state, env, pool, c, switch_step, window=10,
                    episodes_per_pair=4, seed=0, deterministic=False):
    """Evaluate every ordered pair of distinct frozen partners in parallel."""
    if getattr(env, 'training_shaping_coef', 0.) != 0:
        raise ValueError('Switch evaluation requires sparse reward only')
    if pool.size < 2 or episodes_per_pair < 1:
        raise ValueError('Switch evaluation requires >=2 partners and >=1 episode')
    # Validate the metric protocol before compiling/running anything.
    switch_metrics(np.ones(env.horizon), switch_step, window)
    pairs = list(itertools.permutations(range(pool.size), 2))
    pair_array = np.asarray(pairs, dtype=np.int32)
    schedule = np.where(np.arange(env.horizon)[:, None] < switch_step,
                        pair_array[None, :, 0], pair_array[None, :, 1])
    tc = TrainConfig(num_envs=len(pairs), num_steps=env.horizon, epochs=1)
    collect = make_rollout(env, pool, c, tc, deterministic=deterministic, partner_schedule=schedule)
    replay_fn = jax.jit(lambda data: replay(state.params, state.norm, data, c))
    before = tree_hash(state)
    records = []
    for episode in range(episodes_per_pair):
        runner = init_runner(jax.random.PRNGKey(seed+episode), env, pool, c, tc)
        runner = runner._replace(partner_ids=jnp.asarray(pair_array[:, 0]))
        _, data = collect(state, runner)
        outputs = replay_fn(data)
        rewards = np.asarray(data.transitions.reward)
        done = np.asarray(data.transitions.terminated | data.transitions.truncated)
        if done[:-1].any():
            raise ValueError('Switch protocol requires a single episode lasting the configured horizon')
        style, intent = np.asarray(outputs['style']), np.asarray(outputs['intent'])
        for i, (a, b) in enumerate(pairs):
            metrics = switch_metrics(rewards[:, i], switch_step, window)
            records.append(dict(episode=episode, seed=seed+episode,
                pre_partner=pool.identities[a], post_partner=pool.identities[b],
                pre_partner_hash=pool.content_hashes[a], post_partner_hash=pool.content_hashes[b],
                episode_return=float(rewards[:, i].sum()), **metrics,
                reward_trace=rewards[:, i].tolist(), gate_trace=np.asarray(outputs['gate'])[:, i].tolist(),
                conflict_trace=np.asarray(outputs['conflict'])[:, i].tolist(),
                style_change_trace=np.linalg.norm(np.diff(style[:, i], axis=0), axis=-1).tolist(),
                intent_change_trace=np.linalg.norm(np.diff(intent[:, i], axis=0), axis=-1).tolist()))
    if before != tree_hash(state):
        raise AssertionError('Switch evaluation changed train state')
    recovered = [r['recovery_steps'] for r in records if r['recovery_steps'] is not None]
    valid = [r for r in records if r['relative_metrics_defined']]
    return dict(environment=env.name, records=records,
        mean_return=float(np.mean([r['episode_return'] for r in records])),
        mean_post_switch_return=float(np.mean([r['post_switch_return'] for r in records])),
        mean_normalized_drop=float(np.mean([r['normalized_drop'] for r in valid])) if valid else None,
        mean_recovery_steps_observed=float(np.mean(recovered)) if recovered else None,
        recovery_censored_fraction=float(np.mean([r['recovery_censored'] for r in valid])) if valid else None,
        relative_metric_valid_records=len(valid), deterministic=deterministic,
        protocol=dict(switch_step=switch_step, score_window=window,
            recovery='first fully post-switch window reaching last pre-switch window mean',
            undefined='null if pre-switch score is nonpositive; unrecovered trajectories remain censored',
            ego_state_reset_at_switch=False, partner_state_reset_at_switch=True,
            pairs='all ordered distinct pairs', paper_identical_metric_protocol_verified=False),
        parameters_unchanged=True, normalization_unchanged=True,
        benchmark_claim=False, published_metrics_reproduced=False,
        is_benchmark_environment=not env.name.startswith('synthetic'))
