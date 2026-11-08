"""Masked future supervision and the paper's Eq.18--20 objectives."""
import jax
import jax.numpy as jnp
from .model import gaussian_kl


def future_targets(actions, done, horizon):
    """actions/done [T,B]. Targets t+k never cross an episode boundary."""
    t, b = actions.shape
    offsets = jnp.arange(horizon+1)
    index = jnp.arange(t)[:, None]+offsets[None, :]
    safe_index = jnp.minimum(index, t-1)
    targets = jnp.transpose(actions[safe_index], (0, 2, 1))
    # Episode ID BEFORE each action. done at target itself does not invalidate it.
    episode = jnp.cumsum(done.astype(jnp.int32), axis=0)-done.astype(jnp.int32)
    same_episode = episode[safe_index] == episode[:, None, :]
    valid = (index < t)[..., None] & same_episode
    return targets, jnp.transpose(valid, (0, 2, 1))


def contrastive(styles, partner_ids, valid, temperature):
    x = styles.reshape((-1, styles.shape[-1]))
    x = x/jnp.sqrt(jnp.sum(x*x, -1, keepdims=True)+1e-8)
    ids, valid = partner_ids.reshape(-1), valid.reshape(-1)
    n = len(ids)
    pairs = valid[:, None] & valid[None, :] & ~jnp.eye(n, dtype=bool)
    positive = pairs & (ids[:, None] == ids[None, :])
    negative = pairs & (ids[:, None] != ids[None, :])
    logits = x @ x.T/temperature
    log_den = jax.nn.logsumexp(jnp.where(pairs, logits, -1e9), axis=1)
    per = -(jnp.where(positive, logits-log_den[:, None], 0).sum(1))/jnp.maximum(positive.sum(1), 1)
    anchors = positive.any(1) & negative.any(1)
    return jnp.where(anchors, per, 0).sum()/jnp.maximum(anchors.sum(), 1)


def auxiliary(outputs, targets, mask, ids, c, tc):
    zero = jnp.sum(outputs['logits'])*0.
    active_style = c.ablation not in ('no_partner_model', 'intent_only', 'no_style')
    active_intent = c.ablation not in ('no_partner_model', 'style_only', 'no_intent')
    active_prior = c.ablation in ('full', 'no_style', 'no_arbitration')
    log_probs = jax.nn.log_softmax(outputs['future_action_logits'], axis=-1)
    ce = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    # Eq.9 sums horizon offsets, then averages over prediction origins.
    intent = (ce*mask).sum(-1).mean() if active_intent else zero
    prior = gaussian_kl(outputs['posterior_mean'], outputs['posterior_logvar'],
                        outputs['prior_mean'], outputs['prior_logvar']).mean() if active_prior else zero
    smooth = jnp.sum((outputs['style']-outputs['previous_style'])**2, -1).mean() if active_style else zero
    style = contrastive(outputs['instant_style'], ids, outputs['history_valid'], tc.contrastive_temperature) if active_style and tc.style_coef else zero
    total = tc.intent_coef*intent+tc.prior_coef*prior+tc.smoothness_coef*smooth+tc.style_coef*style
    return dict(intent_loss=intent, prior_loss=prior, smoothness_loss=smooth, style_loss=style, auxiliary_loss=total)


def generalized_advantage(reward, value, next_value, terminated, truncated, gamma, lam):
    # Truncation bootstraps its FINAL observation but does not propagate next episode GAE.
    delta = reward+gamma*(1-terminated.astype(jnp.float32))*next_value-value
    continuation = ~(terminated | truncated)
    def step(carry, xs):
        d, cont = xs
        adv = d+gamma*lam*cont*carry
        return adv, adv
    _, advantage = jax.lax.scan(step, jnp.zeros_like(value[0]), (delta, continuation), reverse=True)
    return advantage, advantage+value
