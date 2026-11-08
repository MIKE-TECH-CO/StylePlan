"""Dynamic reward curricula and post-update PPO diagnostics regressions."""
from typing import NamedTuple
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from styleplan_paper.config import ModelConfig, TrainConfig
from styleplan_paper.envs import OvercookedV2, SyntheticCoordination
from styleplan_paper.partners import ScriptedPool
from styleplan_paper.training import (init_runner, init_train_state,
    make_rollout, make_update, replay)


class ShapedSynthetic(SyntheticCoordination):
    """Known action-dependent bonus, independent of the world's transition."""
    def __init__(self, coefficient=0.):
        super().__init__(horizon=4)
        self.training_shaping_coef = coefficient
        self.dynamic_traces = 0

    @staticmethod
    def bonus(ego, partner):
        return .25 + .5*(ego == 0) + .75*(partner == 2)

    def step(self, key, state, ego, partner):
        new, frame, reward, terminal, truncated = super().step(key, state, ego, partner)
        return (new, frame, reward+self.training_shaping_coef*self.bonus(ego, partner),
                terminal, truncated)

    def step_with_shaping(self, key, state, ego, partner, coefficient):
        # A Python side effect counts tracing, not compiled runtime execution.
        self.dynamic_traces += 1
        new, frame, reward, terminal, truncated = super().step(key, state, ego, partner)
        return new, frame, reward+coefficient*self.bonus(ego, partner), terminal, truncated


class Position(NamedTuple):
    x: object
    y: object


class Agents(NamedTuple):
    pos: Position


class BackendState(NamedTuple):
    time: object
    agents: Agents


class FakeBackend:
    agents = ('agent_0', 'agent_1')
    width, height = 3, 2

    def step_env(self, key, state, actions):
        new = state._replace(time=state.time+1)
        obs = {'agent_0': jnp.ones((2, 3, 1)), 'agent_1': jnp.zeros((2, 3, 1))}
        shaped = {'agent_0': jnp.array(3.), 'agent_1': jnp.array(7.)}
        return (obs, new, {'agent_0': jnp.array(20.), 'agent_1': jnp.array(20.)},
                {'__all__': new.time >= 2}, {'shaped_reward': shaped})


def fake_adapter(coefficient, terminal=True):
    env = object.__new__(OvercookedV2)
    env.backend = FakeBackend()
    env.agents = env.backend.agents
    env.horizon, env.time_field = 2, 'time'
    env.task_context_mode = 'spatial'
    env.training_shaping_coef = coefficient
    env.time_limit_is_terminal = terminal
    return env


class DynamicShapingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = ModelConfig(19, 7, global_state_dim=38, hidden_dim=8,
            history_window=4, intent_window=2)
        cls.tc = TrainConfig(num_envs=2, num_steps=6, epochs=1)
        cls.pool = ScriptedPool((0, 2))
        cls.state = init_train_state(jax.random.PRNGKey(812), cls.c)

    def assertTreesEqual(self, actual, expected):
        a, structure_a = jax.tree_util.tree_flatten(actual)
        b, structure_b = jax.tree_util.tree_flatten(expected)
        self.assertEqual(structure_a, structure_b)
        for left, right in zip(a, b):
            np.testing.assert_allclose(left, right, atol=2e-6, rtol=1e-5)

    def test_dynamic_coefficients_match_fixed_rollouts_without_changing_actions(self):
        env = ShapedSynthetic()
        runner = init_runner(jax.random.PRNGKey(813), env, self.pool, self.c, self.tc)
        collect = make_rollout(env, self.pool, self.c, self.tc, dynamic_shaping=True)
        baseline = None
        trace_count = None
        for coefficient in (0., .35, 1., 0.):
            with self.subTest(coefficient=coefficient):
                after, batch = collect(self.state, runner, jnp.asarray(coefficient, jnp.float32))
                jax.block_until_ready(batch.transitions.reward)
                if trace_count is None:
                    trace_count = env.dynamic_traces
                    self.assertGreater(trace_count, 0)
                else:
                    self.assertEqual(env.dynamic_traces, trace_count,
                        'Changing only a scalar shaping coefficient retraced the rollout')
                fixed = ShapedSynthetic(coefficient)
                old_after, old_batch = make_rollout(fixed, self.pool, self.c, self.tc)(self.state, runner)
                self.assertTreesEqual((after, batch), (old_after, old_batch))
                if baseline is None:
                    baseline = (after, batch)
                self.assertTreesEqual(after, baseline[0])
                t, sparse = batch.transitions, baseline[1].transitions
                self.assertTreesEqual(t._replace(reward=sparse.reward), sparse)
                expected = coefficient*ShapedSynthetic.bonus(t.action, t.partner_action)
                np.testing.assert_allclose(t.reward-sparse.reward, expected, atol=2e-6)
                self.assertTrue(np.asarray(t.terminated[3]).all())

    def test_adapter_dynamic_reward_counts_team_sparse_reward_once(self):
        state = BackendState(jnp.array(1), Agents(Position(jnp.array([0, 2]), jnp.array([0, 1]))))
        for terminal in (True, False):
            dynamic = fake_adapter(0., terminal)
            step = jax.jit(dynamic.step_with_shaping)
            for coefficient in (0., .35, 1.):
                with self.subTest(terminal=terminal, coefficient=coefficient):
                    key = jax.random.PRNGKey(814)
                    actual = step(key, state, jnp.array(1), jnp.array(2),
                                  jnp.asarray(coefficient, jnp.float32))
                    fixed = fake_adapter(coefficient, terminal)
                    expected = fixed.step(key, state, jnp.array(1), jnp.array(2))
                    self.assertTreesEqual(actual, expected)
                    self.assertAlmostEqual(float(actual[2]), 20.+10.*coefficient, places=5)
                    self.assertEqual(bool(actual[3]), terminal)
                    self.assertEqual(bool(actual[4]), not terminal)


class FinalPolicyDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = ModelConfig(19, 7, global_state_dim=38, hidden_dim=8,
            history_window=4, intent_window=2)
        cls.tc = TrainConfig(num_envs=2, num_steps=6, epochs=1, learning_rate=.05)
        cls.env, cls.pool = SyntheticCoordination(horizon=4), ScriptedPool((0, 2))
        cls.before = init_train_state(jax.random.PRNGKey(821), cls.c)
        runner = init_runner(jax.random.PRNGKey(822), cls.env, cls.pool, cls.c, cls.tc)
        _, cls.data = make_rollout(cls.env, cls.pool, cls.c, cls.tc)(cls.before, runner)
        cls.after, cls.metrics = make_update(cls.c, cls.tc)(cls.before, cls.data)

    def test_final_diagnostics_match_updated_policy_with_collection_normalizer(self):
        # The normalizer changes after PPO; that change must not enter this
        # diagnostic of the optimizer's effect on the recorded behavior policy.
        self.assertGreater(float(self.after.norm.count), float(self.before.norm.count))
        outputs = replay(self.after.params, self.before.norm, self.data, self.c)
        selected = jnp.take_along_axis(jax.nn.log_softmax(outputs['logits']),
            self.data.transitions.action[..., None], axis=-1).squeeze(-1)
        delta = selected-self.data.transitions.log_prob
        ratio = jnp.exp(delta)
        expected_kl = ((ratio-1)-delta).mean()
        expected_clip = (jnp.abs(ratio-1) > self.tc.clip_epsilon).mean()
        np.testing.assert_allclose(self.metrics['final_approx_kl'], expected_kl, atol=2e-6, rtol=1e-5)
        np.testing.assert_allclose(self.metrics['final_clip_fraction'], expected_clip, atol=2e-6)
        self.assertGreater(float(expected_kl), 1e-6)

    def test_final_metrics_are_not_the_pre_update_epoch_average(self):
        # One epoch's old metrics describe the untouched behavior policy. The
        # final diagnostics must instead see the actual nonzero optimizer step.
        self.assertLess(abs(float(self.metrics['approx_kl'])), 1e-6)
        self.assertEqual(float(self.metrics['clip_fraction']), 0.)
        self.assertGreater(float(self.metrics['final_approx_kl']),
                           float(self.metrics['approx_kl'])+1e-6)
        self.assertGreater(float(self.metrics['parameter_delta']), 0.)
        self.assertTrue(np.isfinite([float(value) for value in self.metrics.values()]).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
