"""Spatial prior-context regressions using constructed observable grids."""
import sys
import unittest
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from styleplan_paper.envs import OvercookedV2, SyntheticCoordination


def adapter(mode='spatial', clock='time'):
    env = object.__new__(OvercookedV2)
    env.agents = ['agent_0', 'agent_1']
    env.backend = SimpleNamespace(width=5, height=4)
    env.horizon, env.time_field = 400, clock
    env.task_context_mode = mode
    return env


def inputs(ego_x=1, object_x=2, clock='time'):
    ego = jnp.zeros((4, 5, 3)).at[1, ego_x, 0].set(1).at[2, 3, 1].set(1)
    ego = ego.at[0, object_x, 2].set(1)
    partner = ego.at[3, 0, 2].set(2)
    state = SimpleNamespace(agents=SimpleNamespace(pos=SimpleNamespace(
        x=jnp.array([ego_x, 3]), y=jnp.array([1, 2]))), **{clock: jnp.array(200)})
    return {'agent_0': ego, 'agent_1': partner}, state


class SpatialContextTests(unittest.TestCase):
    def test_position_and_object_moves_are_distinguished(self):
        for changed in (dict(ego_x=4), dict(object_x=4)):
            obs_a, state_a = inputs()
            obs_b, state_b = inputs(**changed)
            spatial_a = adapter().frame(obs_a, state_a)
            spatial_b = adapter().frame(obs_b, state_b)
            legacy_a = adapter('legacy').frame(obs_a, state_a)
            legacy_b = adapter('legacy').frame(obs_b, state_b)
            self.assertFalse(np.array_equal(spatial_a.task_context, spatial_b.task_context))
            np.testing.assert_array_equal(legacy_a.task_context, legacy_b.task_context)

    def test_legacy_exact_old_values_and_shape(self):
        obs, state = inputs()
        f = adapter('legacy').frame(obs, state)
        expected = jnp.concatenate((obs['agent_0'].mean((0, 1)), jnp.array([.5])))
        np.testing.assert_array_equal(f.task_context, expected)
        self.assertEqual(f.task_context.shape, (4,))
        np.testing.assert_array_equal(f.partner_task_context,
            jnp.concatenate((obs['agent_1'].mean((0, 1)), jnp.array([.5]))))

    def test_role_contexts_use_own_grid_and_swapped_positions(self):
        obs, state = inputs()
        f = adapter().frame(obs, state)
        self.assertEqual(f.task_context.shape, (65,))
        np.testing.assert_array_equal(f.task_context[:60], obs['agent_0'].reshape(-1))
        np.testing.assert_array_equal(f.partner_task_context[:60], obs['agent_1'].reshape(-1))
        np.testing.assert_allclose(f.task_context[-5:], [.25, 1/3, .75, 2/3, .5])
        np.testing.assert_allclose(f.partner_task_context[-5:], [.75, 2/3, .25, 1/3, .5])
        np.testing.assert_allclose(f.ego_position, [.25, 1/3])
        self.assertTrue(bool(f.ego_position_visible))

    def test_context_does_not_read_other_actor_observation_or_hidden_fields(self):
        obs, state = inputs()
        f = adapter().frame(obs, state)
        changed_obs = dict(obs, agent_1=jnp.full_like(obs['agent_1'], 17))
        state.hidden_inventory = jnp.array([123, 456])
        state.partner_private_memory = jnp.ones(100)
        changed = adapter().frame(changed_obs, state)
        np.testing.assert_array_equal(f.task_context, changed.task_context)
        np.testing.assert_array_equal(f.ego_obs, changed.ego_obs)
        self.assertFalse(np.array_equal(f.partner_task_context, changed.partner_task_context))

    def test_time_and_step_clocks(self):
        for mode in ('spatial', 'legacy'):
            frames = []
            for clock in ('time', 'step'):
                obs, state = inputs(clock=clock)
                frames.append(adapter(mode, clock).frame(obs, state))
            np.testing.assert_array_equal(frames[0].task_context, frames[1].task_context)
            self.assertEqual(float(frames[0].task_context[-1]), .5)

    def test_synthetic_preserves_dimensions_and_populates_reciprocal_fields(self):
        env = SyntheticCoordination()
        import jax
        _, f = env.reset(jax.random.PRNGKey(0))
        self.assertEqual(f.ego_obs.shape, (19,))
        self.assertEqual(f.partner_obs.shape, (19,))
        self.assertEqual(f.task_context.shape, (7,))
        np.testing.assert_array_equal(f.partner_task_context, f.task_context)
        self.assertEqual(f.ego_position.shape, (2,))
        self.assertFalse(bool(f.ego_position_visible))

    def test_constructor_dimensions_modes_and_backend_clock(self):
        for clock in ('time', 'step'):
            class Backend:
                agents = ['agent_0', 'agent_1']
                width, height = 5, 4
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                def reset(self, key):
                    return inputs(clock=clock)
            module = ModuleType('jaxmarl.environments.overcooked_v2.overcooked')
            module.OvercookedV2 = Backend
            with patch.dict(sys.modules, {module.__name__: module}):
                for mode, size in (('spatial', 65), ('legacy', 4)):
                    env = OvercookedV2(public_actions=True, task_context_mode=mode)
                    self.assertEqual(env.task_dim, size)
                    self.assertEqual(env.time_field, clock)
                    self.assertIsNone(env.backend.kwargs['agent_view_size'])
                self.assertEqual(OvercookedV2(public_actions=True).task_dim, 65)
        with self.assertRaises(ValueError):
            OvercookedV2(public_actions=True, task_context_mode='invalid')
        for invalid in (-1., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                OvercookedV2(public_actions=True, training_shaping_coef=invalid)

    def test_shaping_is_explicit_shared_training_reward(self):
        obs, state = inputs()
        def step_env(key, old_state, actions):
            return obs, state, {'agent_0': jnp.array(20.), 'agent_1': jnp.array(20.)}, \
                {'__all__': jnp.array(False)}, \
                {'shaped_reward': {'agent_0': jnp.array(3.), 'agent_1': jnp.array(7.)}}
        contexts = []
        for coef, expected in ((0., 20.), (.2, 22.)):
            env = adapter()
            env.training_shaping_coef = coef
            env.time_limit_is_terminal = True
            env.backend.step_env = step_env
            _, f, reward, terminated, truncated = env.step(None, state, 1, 2)
            self.assertEqual(float(reward), expected)
            self.assertFalse(bool(terminated))
            self.assertFalse(bool(truncated))
            contexts.append(f.task_context)
        np.testing.assert_array_equal(contexts[0], contexts[1])

    def test_sparse_default_does_not_require_shaping_info(self):
        obs, state = inputs()
        env = adapter()
        env.time_limit_is_terminal = True
        env.backend.step_env = lambda *args: (obs, state,
            {'agent_0': jnp.array(20.), 'agent_1': jnp.array(20.)},
            {'__all__': jnp.array(False)}, {})
        self.assertEqual(float(env.step(None, state, 1, 2)[2]), 20.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
