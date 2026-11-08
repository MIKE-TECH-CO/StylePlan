"""Regression checks for honest episode accounting and qualification screening."""
import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from styleplan_paper.config import ModelConfig, TrainConfig
from styleplan_paper.envs import SyntheticCoordination
from styleplan_paper.partners import ScriptedPool
from styleplan_paper.training import init_train_state, init_runner, make_rollout

spec = importlib.util.spec_from_file_location('v011_validation', Path(__file__).with_name('real_v011_validation.py'))
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def fake_rollout(rewards, done):
    return SimpleNamespace(transitions=SimpleNamespace(reward=np.asarray(rewards),
        terminated=np.asarray(done), truncated=np.zeros_like(done, dtype=bool)))


class ValidationProtocolTests(unittest.TestCase):
    def test_replicated_selfplay_reward_is_counted_once(self):
        data = fake_rollout([[0, 0, 0, 0], [20, 0, 20, 0]],
                            [[False]*4, [True]*4])
        result = validation.rollout_returns(data, 2, self_play=True)
        np.testing.assert_array_equal(result, [20, 0])

    def test_selfplay_reward_role_mismatch_is_detected(self):
        data = fake_rollout([[0, 0], [20, 0]], [[False]*2, [True]*2])
        with self.assertRaises(AssertionError):
            validation.rollout_returns(data, 1, self_play=True)

    def test_early_reset_is_not_reported_as_single_episode(self):
        data = fake_rollout([[20], [20]], [[True], [True]])
        with self.assertRaises(AssertionError):
            validation.rollout_returns(data, 1)

    def test_incomplete_horizon_is_rejected(self):
        data = fake_rollout([[0], [0]], [[False], [False]])
        with self.assertRaises(AssertionError):
            validation.rollout_returns(data, 1)

    def test_zero_partner_and_equal_random_partner_fail_quality_gate(self):
        zeros = np.zeros(64)
        self.assertFalse(validation.qualification(zeros, zeros)['qualified'])
        sparse = zeros.copy(); sparse[:3] = 20
        self.assertFalse(validation.qualification(sparse, sparse)['qualified'])

    def test_success_floor_cannot_be_bypassed_by_one_large_return(self):
        returns = np.zeros(64); returns[0] = 10000
        self.assertFalse(validation.qualification(returns, np.zeros(64), .25)['qualified'])

    def test_reliably_successful_partner_passes_predeclared_gate(self):
        gate = validation.qualification(np.ones(64)*20, np.zeros(64))
        self.assertTrue(gate['qualified'])
        self.assertEqual(gate['bootstrap_lower_95'], 20)
        self.assertEqual(gate['success_rate'], 1.)

    def test_invalid_qualification_input_is_rejected(self):
        for values, control in (([1], [0]), ([1, 2], [0]), ([float('nan'), 0], [0, 0])):
            with self.assertRaises(ValueError):
                validation.qualification(values, control)

    def test_stage_budgets_are_cumulative_and_unambiguous(self):
        self.assertEqual(validation.parse_stages('64,128,256'), (64, 128, 256))
        for value in ('0', '64,64', '128,64'):
            with self.assertRaises(argparse.ArgumentTypeError):
                validation.parse_stages(value)

    def test_physical_and_dual_role_sample_budgets_are_distinct(self):
        budget = validation.training_budgets(16, 64, 64, 64, 8, 2)
        self.assertEqual(budget['per_partner']['physical_environment_steps'], 65536)
        self.assertEqual(budget['per_partner']['collected_role_transitions'], 131072)
        self.assertEqual(budget['per_partner']['optimizer_role_sample_uses'], 262144)
        self.assertEqual(budget['full']['total_physical_environment_steps'], 73728)
        self.assertEqual(budget['no_partner_model']['total_physical_environment_steps'], 65536)
        self.assertTrue(budget['matched_ppo_sampling_budgets'])
        self.assertFalse(budget['matched_total_physical_budgets'])
        self.assertTrue(validation.training_budgets(16, 64, 64, 64, 0, 2)['matched_total_physical_budgets'])

    def test_shaped_reward_evaluation_is_rejected(self):
        env = SimpleNamespace(training_shaping_coef=.1)
        with self.assertRaises(ValueError):
            validation.selfplay_eval(None, env, None, 64, 0)
        with self.assertRaises(ValueError):
            validation.static_eval(None, env, None, None, 32, 0)

    def test_gradient_probe_reports_per_target_ce_and_preserves_state(self):
        config = ModelConfig(19, 7, global_state_dim=38, hidden_dim=8,
                             history_window=4, intent_window=2)
        tc = TrainConfig(num_envs=2, num_steps=8, epochs=1)
        env, pool = SyntheticCoordination(horizon=8), ScriptedPool((0, 1))
        state = init_train_state(jax.random.PRNGKey(91), config)
        params = dict(state.params)
        params['intent_decoder'] = jax.tree_util.tree_map(jnp.zeros_like, params['intent_decoder'])
        state = state._replace(params=params)
        before = validation.tree_hash(state)
        runner = init_runner(jax.random.PRNGKey(92), env, pool, config, tc)
        _, data = make_rollout(env, pool, config, tc)(state, runner)
        report = validation.gradient_diagnostics(state, data, config, tc)
        self.assertTrue(report['gradients_finite'])
        self.assertGreater(report['valid_targets'], 0)
        self.assertAlmostEqual(report['intent_ce_per_target'], np.log(6), places=5)
        self.assertIn('current_encoder', report['groups'])
        self.assertEqual(before, validation.tree_hash(state))


if __name__ == '__main__':
    unittest.main()
