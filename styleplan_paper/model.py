"""Native JAX paper model, with explicit legacy or bounded slow conditioning.

All history arrays hold completed transitions. apply() has no future-label input,
and never mutates running statistics. No Flax/Optax/PyTorch dependency is used.
"""
import jax
import jax.numpy as jnp
from .types import ModelState, RunningNorm


def init_params(key, c):
    keys = iter(jax.random.split(key, 80))
    def linear(n, m, scale=1.0, bias=True):
        result = {'w': jax.random.normal(next(keys), (n, m)) * (scale / n**0.5)}
        if bias:
            result['b'] = jnp.zeros(m)
        return result
    def mlp(n, m):
        return {'in': linear(n, c.hidden_dim), 'out': linear(c.hidden_dim, m)}
    def gru(n):
        return {'x': linear(n, 3*c.hidden_dim), 'h': linear(c.hidden_dim, 3*c.hidden_dim)}
    history_features = c.obs_dim + c.num_actions + 4
    q_features = 2*c.hidden_dim + c.style_dim + c.task_dim
    return {
        'style_encoder': mlp(c.fingerprint_dim, c.style_dim),
        'history_encoder': gru(history_features),
        'current_encoder': mlp(c.obs_dim, c.hidden_dim),
        'posterior': mlp(q_features, 2*c.intent_dim),
        'intent_decoder': linear(c.intent_dim, c.prediction_steps*c.num_actions),
        'prior': mlp(c.style_dim+c.task_dim, 2*c.intent_dim),
        'prior_projection': linear(c.intent_dim, c.intent_dim, bias=False),
        'intent_projection': linear(c.intent_dim, c.intent_dim, bias=False),
        'gate': linear(c.style_dim+c.intent_dim+1+c.task_dim, 1),
        'fusion': mlp(c.style_dim+2*c.intent_dim+1, c.context_dim),
        'observation_encoder': mlp(c.obs_dim, c.hidden_dim),
        'slow_injection': linear(c.style_dim, c.hidden_dim,
                                 scale=c.slow_injection_scale, bias=False),
        'actor_gru': gru(c.hidden_dim),
        'film': linear(c.context_dim, 2*c.hidden_dim, scale=0.03),
        'actor': linear(c.hidden_dim, c.num_actions, scale=0.1),
        'critic': mlp(c.hidden_dim+c.context_dim+(c.global_state_dim or c.obs_dim), 1),
    }


def dense(p, x):
    return x @ p['w'] + p.get('b', 0.)


def mlp(p, x):
    return dense(p['out'], jax.nn.tanh(dense(p['in'], x)))


def gru(p, h, x):
    xr, xz, xn = jnp.split(dense(p['x'], x), 3, axis=-1)
    hr, hz, hn = jnp.split(dense(p['h'], h), 3, axis=-1)
    reset, update = jax.nn.sigmoid(xr+hr), jax.nn.sigmoid(xz+hz)
    candidate = jnp.tanh(xn+reset*hn)
    return update*h+(1-update)*candidate


def reset_state(state, done):
    def reset(x, zero):
        mask = done.reshape(done.shape + (1,)*(x.ndim-done.ndim))
        return jnp.where(mask, jnp.full_like(x, zero), x)
    return ModelState(*(reset(x, -1 if i == 3 else 0) for i, x in enumerate(state)))


def observe(state, obs, action, position, position_visible):
    """Append a just-completed transition, AFTER the ego action was selected."""
    def append(xs, x):
        return jnp.concatenate((xs[:, 1:], x[:, None]), axis=1)
    return state._replace(observations=append(state.observations, obs),
        actions=append(state.actions, action), positions=append(state.positions, position),
        valid=append(state.valid, jnp.ones(obs.shape[0], dtype=bool)),
        position_valid=append(state.position_valid, position_visible))


def fingerprint(state, c):
    valid = state.valid
    seen = valid & (state.actions >= 0) & (state.actions < c.num_actions)
    n = jnp.maximum(seen.sum(1), 1)
    freq = (jax.nn.one_hot(state.actions, c.num_actions)*seen[..., None]).sum(1)/n[:, None]
    entropy = -(freq*jnp.log(jnp.maximum(freq, 1e-8))).sum(-1)/jnp.log(float(c.num_actions))
    pair = seen[:, 1:] & seen[:, :-1]
    switches = ((state.actions[:, 1:] != state.actions[:, :-1])*pair).sum(1)/jnp.maximum(pair.sum(1), 1)
    posmask = valid & state.position_valid
    posn = jnp.maximum(posmask.sum(1), 1)
    mean = (state.positions*posmask[..., None]).sum(1)/posn[:, None]
    variance = (((state.positions-mean[:, None])**2)*posmask[..., None]).sum(1)/posn[:, None]
    ppair = posmask[:, 1:] & posmask[:, :-1]
    delta = state.positions[:, 1:]-state.positions[:, :-1]
    speed = (jnp.sqrt(jnp.sum(delta**2, -1)+1e-8)*ppair).sum(1)/jnp.maximum(ppair.sum(1), 1)
    # Distinct visible cells, using fixed-size comparisons rather than dynamic unique().
    same = jnp.all(state.positions[:, :, None] == state.positions[:, None, :], axis=-1)
    earlier = jnp.tril(jnp.ones((c.history_window, c.history_window), dtype=bool), k=-1)
    duplicate = (same & earlier[None] & posmask[:, None, :]).any(-1)
    coverage = (posmask & ~duplicate).sum(1)/posn
    extras = jnp.stack((entropy, switches, freq[:, :min(4, c.num_actions)].sum(-1),
        freq[:, min(5, c.num_actions-1)], coverage, speed, valid.mean(1), seen.mean(1),
        posmask.mean(1), mean[:, 0], mean[:, 1], jnp.sqrt(variance[:, 0]+1e-8),
        jnp.sqrt(variance[:, 1]+1e-8)), axis=-1)
    return jnp.concatenate((freq, extras), axis=-1)


def update_normalizer(norm, samples, valid=None):
    """Parallel Welford update; runner calls this ONLY between PPO batches."""
    x = samples.reshape((-1, samples.shape[-1]))
    weights = jnp.ones(x.shape[0]) if valid is None else valid.reshape(-1).astype(jnp.float32)
    n = weights.sum()
    batch_mean = (x*weights[:, None]).sum(0)/jnp.maximum(n, 1)
    batch_m2 = ((x-batch_mean)**2*weights[:, None]).sum(0)
    total = norm.count+n
    delta = batch_mean-norm.mean
    mean = norm.mean+delta*n/jnp.maximum(total, 1)
    m2 = norm.m2+batch_m2+delta**2*norm.count*n/jnp.maximum(total, 1)
    return RunningNorm(total, jnp.where(n > 0, mean, norm.mean), jnp.where(n > 0, m2, norm.m2))


def normalize(norm, x):
    variance = jnp.where(norm.count > 1, norm.m2/jnp.maximum(norm.count, 1), jnp.ones_like(norm.mean))
    return jnp.clip((x-norm.mean)/jnp.sqrt(jnp.maximum(variance, 1e-4)), -10, 10)


def gaussian_kl(mean_q, logvar_q, mean_p, logvar_p):
    return 0.5*jnp.sum(logvar_p-logvar_q +
        (jnp.exp(logvar_q)+(mean_q-mean_p)**2)*jnp.exp(-logvar_p)-1, axis=-1)


def projected_conflict(prior, intent):
    # Exact epsilon-outside-product form of Eq.11, safe at zero vectors.
    dot = jnp.sum(prior*intent, -1)
    pn = jnp.sqrt(jnp.sum(prior**2, -1)+1e-12)
    gn = jnp.sqrt(jnp.sum(intent**2, -1)+1e-12)
    return jnp.clip(1-dot/(pn*gn+1e-8), 0, 2)


def apply(params, norm, state, obs, task_context, global_state, c):
    """Batched causal inference. All choices in c are static under jax.jit."""
    b = obs.shape[0]
    use_style = c.ablation not in ('no_partner_model', 'intent_only', 'no_style')
    use_intent = c.ablation not in ('no_partner_model', 'style_only', 'no_intent')
    use_prior = c.ablation in ('full', 'no_arbitration', 'no_style', 'no_intent')
    raw = fingerprint(state, c)
    instant = jnp.tanh(mlp(params['style_encoder'], normalize(norm, raw)))
    instant = jnp.where(state.valid.any(1)[:, None], instant, 0)
    instant = instant if use_style else jnp.zeros_like(instant)
    style = c.ema_decay*state.style+(1-c.ema_decay)*instant
    style = style if use_style else jnp.zeros_like(style)

    hobs = state.observations[:, -c.intent_window:]
    hact = state.actions[:, -c.intent_window:]
    hpos = state.positions[:, -c.intent_window:]
    hvalid = state.valid[:, -c.intent_window:]
    pvalid = state.position_valid[:, -c.intent_window:] & hvalid
    features = jnp.concatenate((hobs, jax.nn.one_hot(hact, c.num_actions),
        jnp.where(pvalid[..., None], hpos, 0), (hact >= 0)[..., None], pvalid[..., None]), -1)
    def encode(h, inputs):
        x, valid = inputs
        new = gru(params['history_encoder'], h, x)
        h = jnp.where(valid[:, None], new, h)
        return h, None
    history, _ = jax.lax.scan(encode, jnp.zeros((b, c.hidden_dim)),
        (jnp.swapaxes(features, 0, 1), jnp.swapaxes(hvalid, 0, 1)))
    current = mlp(params['current_encoder'], obs)
    posterior = mlp(params['posterior'], jnp.concatenate((history, current, style, task_context), -1))
    qm, qlv = jnp.split(posterior, 2, -1)
    qlv = jnp.clip(qlv, -8, 4)
    intent = qm if use_intent else jnp.zeros_like(qm)
    future_logits = dense(params['intent_decoder'], intent).reshape(b, c.prediction_steps, c.num_actions)
    pm, plv = jnp.split(mlp(params['prior'], jnp.concatenate((style, task_context), -1)), 2, -1)
    plv = jnp.clip(plv, -8, 4)
    pm = pm if use_prior else jnp.zeros_like(pm)
    conflict = projected_conflict(dense(params['prior_projection'], pm),
                                  dense(params['intent_projection'], intent))
    conflict = conflict if use_prior and use_intent else jnp.zeros(b)
    gate_input = jnp.concatenate((style, intent, conflict[:, None], task_context), -1)
    gate = jax.nn.sigmoid(dense(params['gate'], gate_input)).squeeze(-1)
    if c.ablation == 'no_arbitration':
        gate = jnp.full_like(gate, 0.5)
    elif not use_prior:
        gate = jnp.ones_like(gate)
    elif not use_intent:
        gate = jnp.zeros_like(gate)
    fused = jnp.concatenate((style, gate[:, None]*intent, (1-gate[:, None])*pm, conflict[:, None]), -1)
    context = mlp(params['fusion'], fused)
    if c.ablation == 'no_partner_model':
        context = jnp.zeros_like(context)
    previous_hidden = state.hidden
    if use_style:
        previous_hidden = previous_hidden+dense(params['slow_injection'], style)
        if c.style_injection == 'bounded':
            # Explicit stabilization of Eq.14: the GRU's carried input remains
            # in [-1, 1]. Its candidate is also bounded, so the convex update
            # cannot repeatedly accumulate an unbounded style injection.
            previous_hidden = jnp.tanh(previous_hidden)
    hidden = gru(params['actor_gru'], previous_hidden, mlp(params['observation_encoder'], obs))
    if c.ablation == 'no_partner_model':
        policy_features = hidden
    else:
        scale, shift = jnp.split(dense(params['film'], context), 2, -1)
        policy_features = (1+scale)*hidden+shift
    logits = dense(params['actor'], policy_features)
    critic_input = global_state if c.global_state_dim else obs
    value = mlp(params['critic'], jnp.concatenate((hidden, context, critic_input), -1)).squeeze(-1)
    output = dict(logits=logits, value=value, style=style, instant_style=instant,
        previous_style=state.style, intent=intent, posterior_mean=qm, posterior_logvar=qlv,
        prior_mean=pm, prior_logvar=plv, future_action_logits=future_logits,
        conflict=conflict, gate=gate, context=context, raw_fingerprint=raw,
        history_valid=state.valid.any(1), hidden_norm=jnp.linalg.norm(hidden, axis=-1),
        hidden_abs_max=jnp.abs(hidden).max(-1))
    return output, state._replace(hidden=hidden, style=style)
