"""Native JAX environments. Synthetic results are never benchmark evidence."""
from typing import NamedTuple
import math
import jax
import jax.numpy as jnp
from .types import Frame

class SyntheticState(NamedTuple):
    time: object
    cue: object
    last_ego: object
    last_partner: object

class SyntheticCoordination:
    """Six-action public-action coordination test, not an Overcooked substitute."""
    num_actions = 6
    obs_dim = 19
    task_dim = 7
    global_state_dim = 38
    public_actions = True
    name = 'synthetic_coordination_TEST_ONLY'

    def __init__(self, horizon=32, time_limit_is_terminal=True):
        if horizon <= 0:
            raise ValueError('horizon must be positive')
        self.horizon = horizon
        self.time_limit_is_terminal = time_limit_is_terminal

    def frame(self, s):
        cue = jax.nn.one_hot(s.cue, 6)
        ego = jnp.concatenate((cue, jax.nn.one_hot(s.last_partner, 6),
            jax.nn.one_hot(s.last_ego, 6), jnp.array([s.time/self.horizon])))
        partner = jnp.concatenate((cue, jax.nn.one_hot(s.last_ego, 6),
            jax.nn.one_hot(s.last_partner, 6), jnp.array([s.time/self.horizon])))
        pos = jnp.array([jnp.maximum(s.last_partner, 0)%3/2,
                         jnp.maximum(s.last_partner, 0)//3], dtype=jnp.float32)
        ego_pos = jnp.array([jnp.maximum(s.last_ego, 0)%3/2,
                             jnp.maximum(s.last_ego, 0)//3], dtype=jnp.float32)
        task = jnp.concatenate((cue, jnp.array([s.time/self.horizon])))
        return Frame(ego, partner, task, jnp.concatenate((ego, partner)),
                     pos, s.last_partner >= 0, partner_task_context=task,
                     ego_position=ego_pos, ego_position_visible=s.last_ego >= 0)

    def reset(self, key):
        s = SyntheticState(jnp.array(0), jax.random.randint(key, (), 0, 6),
                           jnp.array(-1), jnp.array(-1))
        return s, self.frame(s)

    def step(self, key, s, ego_action, partner_action):
        cue = jnp.where((s.time+1)%4 == 0, jax.random.randint(key, (), 0, 6), s.cue)
        new = SyntheticState(s.time+1, cue, ego_action, partner_action)
        done = new.time >= self.horizon
        reward = (ego_action == partner_action).astype(jnp.float32)
        return new, self.frame(new), reward, done & self.time_limit_is_terminal, done & ~jnp.asarray(self.time_limit_is_terminal)


class OvercookedV2:
    """JAXMarl adapter, using step_env to preserve terminal observations.

    Full-grid observations are explicitly required. Partner positions then appear
    in ego's view. Public past partner actions are a declared observation-contract
    assumption, not silently read from a hidden channel. Sparse team reward is
    the default; training-only shaping requires an explicit coefficient. Other
    partial-observation contracts require a visibility adapter.
    """
    num_actions = 6
    public_actions = True

    def __init__(self, layout='cramped_room', horizon=400, public_actions=False,
                 time_limit_is_terminal=True, task_context_mode='spatial',
                 training_shaping_coef=0.0):
        if not math.isfinite(training_shaping_coef) or training_shaping_coef < 0:
            raise ValueError('training_shaping_coef must be finite and nonnegative')
        if task_context_mode not in ('spatial', 'legacy'):
            raise ValueError("task_context_mode must be 'spatial' or 'legacy'")
        if not public_actions:
            raise ValueError('Explicit public_actions=True is required: confirm the paper observation contract')
        try:
            from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2 as Backend
        except ImportError as exc:
            raise ImportError('JAXMarl is required only for real Overcooked runs; use the existing server environment') from exc
        self.backend = Backend(layout=layout, max_steps=horizon, agent_view_size=None)
        if len(self.backend.agents) != 2:
            raise ValueError('This implementation requires exactly two agents')
        self.agents = self.backend.agents
        self.horizon = horizon
        self.time_limit_is_terminal = time_limit_is_terminal
        self.task_context_mode = task_context_mode
        self.training_shaping_coef = float(training_shaping_coef)
        obs, state = self.backend.reset(jax.random.PRNGKey(0))
        # Released/server versions use `time`; other versions use `step`.
        self.time_field = 'time' if hasattr(state, 'time') else 'step'
        if not hasattr(state, self.time_field):
            raise ValueError('Unsupported Overcooked state: no time/step field')
        shape = obs[self.agents[0]].shape
        if len(shape) != 3 or obs[self.agents[1]].shape != shape:
            raise ValueError('Expected equal HWC default observations for two agents')
        self.obs_dim = int(obs[self.agents[0]].size)
        self.task_dim = self.obs_dim+5 if task_context_mode == 'spatial' else int(shape[-1])+1
        self.global_state_dim = 2*self.obs_dim
        self.name = 'OvercookedV2/'+layout

    def frame(self, obs, state):
        ego_grid = obs[self.agents[0]].astype(jnp.float32)
        partner_grid = obs[self.agents[1]].astype(jnp.float32)
        ego = ego_grid.reshape(-1)
        partner = partner_grid.reshape(-1)
        clock = jnp.array([getattr(state, self.time_field)/self.horizon])
        # Reading these coordinates is valid only under the full-grid contract
        # enforced in __init__: both positions appear in each actor's view.
        positions = jnp.stack((state.agents.pos.x/max(self.backend.width-1, 1),
                               state.agents.pos.y/max(self.backend.height-1, 1)), axis=-1)
        if getattr(self, 'task_context_mode', 'spatial') == 'legacy':
            task = jnp.concatenate((ego_grid.mean((0, 1)), clock))
            partner_task = jnp.concatenate((partner_grid.mean((0, 1)), clock))
        else:
            task = jnp.concatenate((ego, positions[0], positions[1], clock))
            partner_task = jnp.concatenate((partner, positions[1], positions[0], clock))
        return Frame(ego, partner, task, jnp.concatenate((ego, partner)),
                     positions[1], jnp.array(True), partner_task_context=partner_task,
                     ego_position=positions[0], ego_position_visible=jnp.array(True))

    def reset(self, key):
        obs, state = self.backend.reset(key)
        return state, self.frame(obs, state)

    def step(self, key, state, ego_action, partner_action):
        coefficient = getattr(self, 'training_shaping_coef', 0.0)
        return self._step(key, state, ego_action, partner_action, coefficient,
                          include_shaping=coefficient > 0)

    def step_with_shaping(self, key, state, ego_action, partner_action, coefficient):
        """Training-only scalar coefficient, dynamic under JIT for schedules.

        The caller validates its schedule before compilation. Frozen evaluation
        continues to use step() on an environment with coefficient zero.
        """
        return self._step(key, state, ego_action, partner_action, coefficient,
                          include_shaping=True)

    def _step(self, key, state, ego_action, partner_action, coefficient, include_shaping):
        obs, state, rewards, dones, info = self.backend.step_env(key, state,
            {self.agents[0]: ego_action, self.agents[1]: partner_action})
        done = dones['__all__']
        reward = rewards[self.agents[0]]
        if include_shaping:
            # Both actors receive one shared team reward. This is explicitly a
            # training aid; benchmark evaluation must use coefficient zero.
            shaped = info['shaped_reward']
            reward = reward + coefficient * sum(shaped[a] for a in self.agents)
        return (state, self.frame(obs, state), reward,
                done & self.time_limit_is_terminal,
                done & ~jnp.asarray(self.time_limit_is_terminal))
