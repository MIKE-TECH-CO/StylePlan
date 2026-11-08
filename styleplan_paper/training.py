"""Compiled vectorized rollout, causal recurrent replay, auxiliary pretraining/PPO."""
from dataclasses import replace
import jax
import jax.numpy as jnp
from . import model
from .types import AdamState, TrainState, RunnerState, Transition, Rollout, initial_state, initial_norm
from .losses import future_targets, auxiliary, generalized_advantage


def select(mask, yes, no):
    return jax.tree_util.tree_map(lambda a, b: jnp.where(
        mask.reshape(mask.shape+(1,)*(a.ndim-mask.ndim)), a, b), yes, no)


def stop(tree):
    return jax.tree_util.tree_map(jax.lax.stop_gradient, tree)


def reciprocal_inputs(frame):
    """The second seat's public context and seat-ordered critic observation."""
    context = frame.task_context if frame.partner_task_context is None else frame.partner_task_context
    global_state = frame.global_state
    if global_state.shape[-1]:
        if global_state.shape[-1] % 2:
            raise ValueError('Self-play critic input must contain two equal seat-ordered halves')
        first, second = jnp.split(global_state, 2, axis=-1)
        global_state = jnp.concatenate((second, first), axis=-1)
    position = jnp.zeros_like(frame.partner_position) if frame.ego_position is None else frame.ego_position
    visible = jnp.zeros_like(frame.position_visible) if frame.ego_position_visible is None else frame.ego_position_visible
    return context, global_state, position, visible


def init_train_state(key, c):
    params = model.init_params(key, c)
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return TrainState(params, initial_norm(c), AdamState(jnp.array(0), zeros, zeros))


def init_runner(key, env, pool, c, tc, self_play=False):
    key, reset_key, pool_key = jax.random.split(key, 3)
    env_state, frame = jax.vmap(env.reset)(jax.random.split(reset_key, tc.num_envs))
    ids = jax.random.randint(pool_key, (tc.num_envs,), 0, pool.size)
    partner_state = initial_state(c, tc.num_envs) if self_play else pool.initial_state(tc.num_envs)
    return RunnerState(key, env_state, frame, initial_state(c, tc.num_envs), partner_state, ids)


def make_rollout(env, pool, c, tc, deterministic=False, self_play=False, partner_schedule=None,
                 dynamic_shaping=False):
    if dynamic_shaping and not hasattr(env, 'step_with_shaping'):
        raise ValueError('Dynamic shaping requires an explicit environment adapter')
    if self_play and c.ablation != 'no_partner_model':
        raise ValueError('Self-play pool training requires NoPM baseline architecture')
    if partner_schedule is not None:
        import numpy as np
        schedule = np.asarray(partner_schedule)
        if (self_play or schedule.shape != (tc.num_steps, tc.num_envs)
                or not np.issubdtype(schedule.dtype, np.integer)
                or np.any(schedule < 0) or np.any(schedule >= pool.size)):
            raise ValueError('Invalid frozen-partner schedule')
        partner_schedule = jnp.asarray(schedule, dtype=jnp.int32)

    def collect(train_state, runner, shaping_coef=None):
        if dynamic_shaping:
            if shaping_coef is None or jnp.ndim(shaping_coef) != 0:
                raise ValueError('Dynamic shaping requires a scalar coefficient')
        elif shaping_coef is not None:
            raise ValueError('Enable dynamic_shaping before passing a coefficient')
        initial = runner.model_state
        if self_play:
            initial = jax.tree_util.tree_map(lambda a, b: jnp.concatenate((a, b), axis=0),
                                            initial, runner.partner_state)
        def step(r, scheduled_ids):
            if partner_schedule is not None:
                # The ego receives no switch signal. Only the replaced partner's
                # recurrent state is cleared; ego memory and world continue.
                changed = scheduled_ids != r.partner_ids
                partner_state = select(changed, pool.initial_state(tc.num_envs), r.partner_state)
                r = r._replace(partner_ids=scheduled_ids, partner_state=partner_state)
            keys = jax.random.split(r.key, 7)
            out, memory = model.apply(train_state.params, train_state.norm, r.model_state,
                r.frame.ego_obs, r.frame.task_context, r.frame.global_state, c)
            action = jnp.argmax(out['logits'], -1) if deterministic else jax.random.categorical(keys[1], out['logits'])
            action = action.astype(jnp.int32)
            logp = jnp.take_along_axis(jax.nn.log_softmax(out['logits']), action[:, None], -1).squeeze(-1)
            if self_play:
                partner_context, partner_global, ego_position, ego_visible = reciprocal_inputs(r.frame)
                partner_out, partner_memory = model.apply(train_state.params, train_state.norm,
                    r.partner_state, r.frame.partner_obs, partner_context, partner_global, c)
                partner_action = (jnp.argmax(partner_out['logits'], -1) if deterministic else
                    jax.random.categorical(keys[2], partner_out['logits'])).astype(jnp.int32)
                partner_logp = jnp.take_along_axis(jax.nn.log_softmax(partner_out['logits']),
                    partner_action[:, None], -1).squeeze(-1)
            else:
                partner_action, partner_memory = pool.act(keys[2], r.partner_ids, r.frame.partner_obs, r.partner_state)
            step_keys = jax.random.split(keys[3], tc.num_envs)
            if dynamic_shaping:
                env_state, final_frame, reward, terminal, truncated = jax.vmap(
                    env.step_with_shaping, in_axes=(0, 0, 0, 0, None))(
                    step_keys, r.env_state, action, partner_action, shaping_coef)
            else:
                env_state, final_frame, reward, terminal, truncated = jax.vmap(env.step)(
                    step_keys, r.env_state, action, partner_action)
            observed = partner_action if env.public_actions else jnp.full_like(partner_action, -1)
            memory = model.observe(memory, r.frame.ego_obs, observed, r.frame.partner_position, r.frame.position_visible)
            if self_play:
                observed_ego = action if env.public_actions else jnp.full_like(action, -1)
                partner_memory = model.observe(partner_memory, r.frame.partner_obs,
                    observed_ego, ego_position, ego_visible)
            # Value of the real final observation, BEFORE autoreset, for truncation.
            bootstrap, _ = model.apply(train_state.params, train_state.norm, memory,
                final_frame.ego_obs, final_frame.task_context, final_frame.global_state, c)
            next_value = jnp.where(terminal, 0., bootstrap['value'])
            if self_play:
                final_partner_context, final_partner_global, _, _ = reciprocal_inputs(final_frame)
                partner_bootstrap, _ = model.apply(train_state.params, train_state.norm, partner_memory,
                    final_frame.partner_obs, final_partner_context, final_partner_global, c)
                partner_next_value = jnp.where(terminal, 0., partner_bootstrap['value'])
            done = terminal | truncated
            def reset_some(args):
                es, fr = args
                reset_env, reset_frame = jax.vmap(env.reset)(jax.random.split(keys[4], tc.num_envs))
                return select(done, reset_env, es), select(done, reset_frame, fr)
            env_state, frame = jax.lax.cond(done.any(), reset_some, lambda x: x, (env_state, final_frame))
            fresh_partner = initial_state(c, tc.num_envs) if self_play else pool.initial_state(tc.num_envs)
            partner_memory = select(done, fresh_partner, partner_memory)
            new_ids = jax.random.randint(keys[5], r.partner_ids.shape, 0, pool.size)
            ids = jnp.where(done, new_ids, r.partner_ids)
            transition = Transition(r.frame.ego_obs, r.frame.task_context, r.frame.global_state,
                r.frame.partner_position, r.frame.position_visible, action, partner_action, observed,
                reward, terminal, truncated, logp, out['value'], next_value, r.partner_ids)
            if self_play:
                second_transition = Transition(r.frame.partner_obs, partner_context, partner_global,
                    ego_position, ego_visible, partner_action, action, observed_ego,
                    reward, terminal, truncated, partner_logp, partner_out['value'],
                    partner_next_value, r.partner_ids)
                # Each physical trajectory contributes both score-function terms.
                # The runner still carries B worlds; PPO replays 2B seat memories.
                transition = jax.tree_util.tree_map(lambda a, b: jnp.concatenate((a, b), axis=0),
                                                    transition, second_transition)
            return RunnerState(keys[0], env_state, frame, model.reset_state(memory, done), partner_memory, ids), transition
        runner, transitions = jax.lax.scan(step, runner, partner_schedule, length=tc.num_steps)
        return stop(runner), stop(Rollout(initial, transitions))
    return jax.jit(collect)


def replay(params, norm, rollout, c):
    """Start from the SAVED rollout-start state, never the ending GRU carry."""
    def step(memory, transition):
        out, memory = model.apply(params, norm, memory, transition.obs,
                                   transition.context, transition.global_state, c)
        memory = model.observe(memory, transition.obs, transition.observed_partner_action,
                               transition.position, transition.position_visible)
        memory = model.reset_state(memory, transition.terminated | transition.truncated)
        return memory, out
    _, outputs = jax.lax.scan(step, stop(rollout.initial_state), rollout.transitions)
    return outputs


def tree_norm(tree):
    return jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree_util.tree_leaves(tree)))


def adam(params, optimizer, gradient, tc):
    scale = jnp.minimum(1., tc.max_grad_norm/(tree_norm(gradient)+1e-8))
    gradient = jax.tree_util.tree_map(lambda x: x*scale, gradient)
    m = jax.tree_util.tree_map(lambda old, g: .9*old+.1*g, optimizer.m, gradient)
    v = jax.tree_util.tree_map(lambda old, g: .999*old+.001*g*g, optimizer.v, gradient)
    count = optimizer.count+1
    params = jax.tree_util.tree_map(lambda p, mt, vt: p-tc.learning_rate*(mt/(1-.9**count))/
        (jnp.sqrt(vt/(1-.999**count))+1e-8), params, m, v)
    return params, AdamState(count, m, v)


def make_update(c, tc, pretrain=False):
    # Pretraining retains the prediction objective's full coefficient. During
    # joint PPO, use the separately configured auxiliary weight for RL balance.
    loss_tc = replace(tc, intent_coef=tc.pretrain_intent_coef) if pretrain else tc
    def update(state, rollout):
        transitions = stop(rollout.transitions)
        adv, returns = generalized_advantage(transitions.reward, transitions.value, transitions.next_value,
            transitions.terminated, transitions.truncated, tc.gamma, tc.gae_lambda)
        adv = stop((adv-adv.mean())/(adv.std()+1e-8))
        returns = stop(returns)
        targets, target_mask = future_targets(transitions.partner_action,
            transitions.terminated | transitions.truncated, c.prediction_horizon)
        frozen_norm = state.norm

        def objective(params):
            outputs = replay(params, frozen_norm, rollout, c)
            log_probs = jax.nn.log_softmax(outputs['logits'])
            logp = jnp.take_along_axis(log_probs, transitions.action[..., None], -1).squeeze(-1)
            ratio = jnp.exp(logp-transitions.log_prob)
            clipped = jnp.clip(ratio, 1-tc.clip_epsilon, 1+tc.clip_epsilon)
            policy_loss = -jnp.minimum(ratio*adv, clipped*adv).mean()
            value_clipped = transitions.value+jnp.clip(outputs['value']-transitions.value, -tc.clip_epsilon, tc.clip_epsilon)
            value_loss = .5*jnp.maximum((outputs['value']-returns)**2, (value_clipped-returns)**2).mean()
            entropy = -(jnp.exp(log_probs)*log_probs).sum(-1).mean()
            aux = auxiliary(outputs, targets, target_mask, transitions.partner_id, c, loss_tc)
            rl = policy_loss+tc.value_coef*value_loss-tc.entropy_coef*entropy
            total = aux['auxiliary_loss'] if pretrain else rl+aux['auxiliary_loss']
            metrics = dict(aux, loss=total, policy_loss=policy_loss, value_loss=value_loss, entropy=entropy,
                clip_fraction=(jnp.abs(ratio-1) > tc.clip_epsilon).mean(),
                approx_kl=((ratio-1)-(logp-transitions.log_prob)).mean(),
                gate_mean=outputs['gate'].mean(), conflict_mean=outputs['conflict'].mean(),
                context_temporal_std=outputs['context'].std(axis=0).mean(),
                valid_future_targets=target_mask.sum(), reward_mean=transitions.reward.mean())
            metrics.update(policy_samples=jnp.asarray(transitions.action.size, dtype=jnp.float32),
                collected_agent_samples=jnp.asarray(transitions.action.size, dtype=jnp.float32),
                physical_environment_steps=jnp.asarray(tc.num_steps*tc.num_envs, dtype=jnp.float32),
                trained_roles=jnp.asarray(transitions.action.shape[1]/tc.num_envs, dtype=jnp.float32))
            target_logp = jnp.take_along_axis(jax.nn.log_softmax(outputs['future_action_logits']),
                targets[..., None], -1).squeeze(-1)
            metrics['intent_ce_per_valid_target'] = -(target_logp*target_mask).sum()/jnp.maximum(target_mask.sum(), 1)
            metrics.update(mean_hidden_norm=outputs['hidden_norm'].mean(),
                max_hidden_abs=outputs['hidden_abs_max'].max(),
                effective_intent_coef=jnp.asarray(loss_tc.intent_coef, dtype=jnp.float32))
            return total, metrics

        def epoch(carry, _):
            (loss, metrics), gradient = jax.value_and_grad(objective, has_aux=True)(carry.params)
            finite = jnp.isfinite(loss) & jnp.stack([jnp.isfinite(x).all() for x in jax.tree_util.tree_leaves(gradient)]).all()
            params, optimizer = adam(carry.params, carry.optimizer, gradient, tc)
            params = jax.tree_util.tree_map(lambda new, old: jnp.where(finite, new, old), params, carry.params)
            optimizer = jax.tree_util.tree_map(lambda new, old: jnp.where(finite, new, old), optimizer, carry.optimizer)
            metrics.update(gradient_norm=tree_norm(gradient), finite_gradient=finite.astype(jnp.float32))
            metrics.update({'grad/'+name: tree_norm(g) for name, g in gradient.items()})
            return TrainState(params, frozen_norm, optimizer), metrics
        new, metrics = jax.lax.scan(epoch, state, None, length=tc.epochs)
        outputs = replay(new.params, frozen_norm, rollout, c)
        final_logp = jnp.take_along_axis(jax.nn.log_softmax(outputs['logits']),
            transitions.action[..., None], -1).squeeze(-1)
        final_log_ratio = final_logp-transitions.log_prob
        final_ratio = jnp.exp(final_log_ratio)
        norm = model.update_normalizer(frozen_norm, outputs['raw_fingerprint'], outputs['history_valid'])
        metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
        # Existing KL/clip metrics average pre-update epochs. These report the
        # actual final policy relative to the behavior policy on the same batch.
        metrics['final_approx_kl'] = (final_ratio-1-final_log_ratio).mean()
        metrics['final_clip_fraction'] = (jnp.abs(final_ratio-1) > tc.clip_epsilon).mean()
        metrics['parameter_delta'] = tree_norm(jax.tree_util.tree_map(lambda a, b: a-b, new.params, state.params))
        return new._replace(norm=norm), metrics
    return jax.jit(update)


def evaluate(state, env, pool, c, episodes_per_partner=4, seed=0, deterministic=False):
    """Frozen batched evaluation; one lane per partner, no optimizer/stat updates."""
    import numpy as np
    from .config import TrainConfig
    from .partners import tree_hash
    if episodes_per_partner < 1:
        raise ValueError('Evaluation requires at least one episode per partner')
    if getattr(env, 'training_shaping_coef', 0.) != 0.:
        raise ValueError('Evaluation requires a sparse-reward environment; training reward shaping must be disabled')
    before = tree_hash({'params': state.params, 'norm': state.norm})
    tc = TrainConfig(num_envs=pool.size, num_steps=env.horizon, epochs=1)
    run = make_rollout(env, pool, c, tc, deterministic=deterministic)
    rows = []
    for episode in range(episodes_per_partner):
        runner = init_runner(jax.random.PRNGKey(seed+episode), env, pool, c, tc)
        runner = runner._replace(partner_ids=jnp.arange(pool.size, dtype=jnp.int32))
        _, batch = run(state, runner)
        # Autoreset is useful during training, but later episodes must never be
        # assigned to the initial partner in a static evaluation record.
        done = np.asarray(batch.transitions.terminated | batch.transitions.truncated)
        completed_before = np.concatenate((np.zeros((1, pool.size), dtype=bool),
            np.logical_or.accumulate(done, axis=0)[:-1]), axis=0)
        included = ~completed_before
        reward = np.where(included, np.asarray(batch.transitions.reward), 0.).sum(0)
        length = included.sum(0)
        for i, value in enumerate(reward):
            rows.append(dict(partner=pool.identities[i], partner_hash=pool.content_hashes[i],
                             episode=episode, seed=seed+episode, episode_return=float(value),
                             episode_length=int(length[i]),
                             terminated=bool(np.any(done[:, i] & included[:, i] &
                                 np.asarray(batch.transitions.terminated)[:, i])),
                             truncated=bool(np.any(done[:, i] & included[:, i] &
                                 np.asarray(batch.transitions.truncated)[:, i]))))
    after = tree_hash({'params': state.params, 'norm': state.norm})
    if before != after:
        raise AssertionError('Evaluation changed parameters or normalization')
    return dict(environment=env.name, deterministic=deterministic, records=rows,
                mean_return=float(np.mean([r['episode_return'] for r in rows])),
                parameters_unchanged=True, normalization_unchanged=True,
                benchmark_claim=False, published_metrics_reproduced=False,
                is_benchmark_environment=not env.name.startswith('synthetic'))
