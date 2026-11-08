"""JAX PyTrees: actor history never contains the current action's future labels."""
from typing import NamedTuple
import jax.numpy as jnp

class ModelState(NamedTuple):
    hidden: object
    style: object
    observations: object
    actions: object
    positions: object
    valid: object
    position_valid: object

class RunningNorm(NamedTuple):
    count: object
    mean: object
    m2: object

class Frame(NamedTuple):
    ego_obs: object
    partner_obs: object  # Runner-only: NEVER passed to the ego actor.
    task_context: object
    global_state: object
    partner_position: object
    position_visible: object
    # Optional reciprocal view for shared-policy self-play. Defaults preserve
    # existing six-field frame constructors and frozen-partner consumers.
    partner_task_context: object = None
    ego_position: object = None
    ego_position_visible: object = None

class AdamState(NamedTuple):
    count: object
    m: object
    v: object

class TrainState(NamedTuple):
    params: object
    norm: RunningNorm
    optimizer: AdamState

class RunnerState(NamedTuple):
    key: object
    env_state: object
    frame: Frame
    model_state: ModelState
    partner_state: object
    partner_ids: object

class Transition(NamedTuple):
    obs: object
    context: object
    global_state: object
    position: object
    position_visible: object
    action: object
    partner_action: object  # Supervision only, observed action below may be -1.
    observed_partner_action: object
    reward: object
    terminated: object
    truncated: object
    log_prob: object
    value: object
    next_value: object
    partner_id: object

class Rollout(NamedTuple):
    initial_state: ModelState
    transitions: Transition


def initial_state(config, batch_size):
    b, w = batch_size, config.history_window
    return ModelState(jnp.zeros((b, config.hidden_dim)), jnp.zeros((b, config.style_dim)),
        jnp.zeros((b, w, config.obs_dim)), jnp.full((b, w), -1, dtype=jnp.int32),
        jnp.zeros((b, w, 2)), jnp.zeros((b, w), dtype=bool),
        jnp.zeros((b, w), dtype=bool))


def initial_norm(config):
    return RunningNorm(jnp.array(0., dtype=jnp.float32),
        jnp.zeros(config.fingerprint_dim), jnp.zeros(config.fingerprint_dim))
