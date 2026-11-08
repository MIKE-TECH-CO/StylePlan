"""Deterministic adapter counterexample; no server or training required."""
import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from styleplan_paper.envs import OvercookedV2


def build_frame(mode, ego_x=1, object_x=2):
    env = object.__new__(OvercookedV2)
    env.agents = ['agent_0', 'agent_1']
    env.backend = SimpleNamespace(width=5, height=4)
    env.horizon, env.time_field = 400, 'time'
    env.task_context_mode = mode
    grid = jnp.zeros((4, 5, 3))
    grid = grid.at[1, ego_x, 0].set(1).at[2, 3, 1].set(1).at[0, object_x, 2].set(1)
    state = SimpleNamespace(
        time=jnp.array(200),
        agents=SimpleNamespace(pos=SimpleNamespace(
            x=jnp.array([ego_x, 3]), y=jnp.array([1, 2])
        )),
    )
    return env.frame({agent: grid for agent in env.agents}, state)


def main():
    results = {}
    for name, changed in (('agent_position', dict(ego_x=4)), ('object_position', dict(object_x=4))):
        old_a, old_b = build_frame('legacy'), build_frame('legacy', **changed)
        new_a, new_b = build_frame('spatial'), build_frame('spatial', **changed)
        different_obs = bool(np.any(np.asarray(old_a.ego_obs) != np.asarray(old_b.ego_obs)))
        old_alias = bool(np.array_equal(old_a.task_context, old_b.task_context))
        new_distinction = bool(not np.array_equal(new_a.task_context, new_b.task_context))
        assert different_obs and old_alias and new_distinction
        results[name] = dict(different_ego_observations=different_obs,
            legacy_context_identical=old_alias, spatial_context_distinct=new_distinction)
    print(json.dumps(dict(counterexamples=results, legacy_task_dim=4, spatial_task_dim=65,
        scope='constructed adapter inputs, not a real environment trajectory'), indent=2))


if __name__ == '__main__':
    main()
