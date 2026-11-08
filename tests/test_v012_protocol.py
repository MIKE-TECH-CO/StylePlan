"""Protect continuation provenance, validation-only selection and matched trials."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from styleplan_paper.config import ModelConfig, TrainConfig
from styleplan_paper.envs import SyntheticCoordination
from styleplan_paper.partners import ScriptedPool, tree_hash
from styleplan_paper.training import init_train_state, init_runner

spec = importlib.util.spec_from_file_location('v012_optimization', Path(__file__).with_name('real_v012_optimization.py'))
optimization = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optimization)


class SyntheticShaping(SyntheticCoordination):
    training_shaping_coef = 1.

    def step_with_shaping(self, key, state, action, partner, coefficient):
        state, frame, reward, terminal, truncated = super().step(key, state, action, partner)
        return state, frame, reward+2*coefficient, terminal, truncated


class V012ProtocolTests(unittest.TestCase):
    def test_fixed_partner_stages_preserve_v011_selected_pool(self):
        self.assertEqual(optimization.POOL_STAGES, {101: 1024, 102: 1024, 201: 256, 202: 1024})

    def test_declared_candidates_change_one_factor(self):
        trials = optimization.validate_trials(optimization.DEFAULT_TRIALS)
        self.assertEqual(trials[0], dict(name='control', overrides={}, shaping='constant'))
        self.assertEqual(trials[1]['overrides'], {'epochs': 8})
        self.assertEqual(trials[1]['shaping'], 'constant')
        self.assertEqual(trials[2]['overrides'], {})
        self.assertEqual(trials[2]['shaping'], 'linear_to_zero')

    def test_invalid_or_budget_changing_trials_are_rejected(self):
        control = optimization.DEFAULT_TRIALS[0]
        candidates = [
            dict(name='../overwrite', overrides={}, shaping='constant'),
            dict(name='candidate', overrides={'num_envs': 32}, shaping='constant'),
            dict(name='candidate', overrides={'epochs': 2.5}, shaping='constant'),
            dict(name='candidate', overrides={'learning_rate': float('nan')}, shaping='constant'),
            dict(name='candidate', overrides={}, shaping='unknown'),
            dict(name='candidate', overrides={'intent_coef': 'bad'}, shaping='constant'),
        ]
        for candidate in candidates:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                optimization.validate_trials([control, candidate])
        for trials in ([], [dict(control, overrides={'epochs': 8})], [control, control], [control]*4):
            with self.assertRaises(ValueError):
                optimization.validate_trials(trials)

    def test_shaping_reaches_sparse_objective_before_final_quarter(self):
        values = np.array([optimization.shaping_coefficient('linear_to_zero', i, 512) for i in range(512)])
        self.assertEqual(values[0], 1.)
        self.assertEqual(values[383], 0.)
        self.assertTrue((values[384:] == 0.).all())
        self.assertTrue((np.diff(values) <= 0).all())
        self.assertEqual(optimization.shaping_coefficient('constant', 511, 512), 1.)
        self.assertEqual(optimization.shaping_coefficient('linear_to_zero', 0, 1), 0.)
        for args in [('constant', 0, 0), ('constant', -1, 4), ('constant', 4, 4)]:
            with self.assertRaises(ValueError):
                optimization.shaping_coefficient(*args)

    def test_selection_uses_validation_only_with_control_tie_break(self):
        rows = [dict(name='control', validation_mean_return=10., heldout_mean_return=-100.),
                dict(name='candidate', validation_mean_return=10., heldout_mean_return=100.)]
        self.assertEqual(optimization.select_trial(rows), 'control')
        rows[1]['validation_mean_return'] = 10.01
        rows[1]['heldout_mean_return'] = -1000.
        self.assertEqual(optimization.select_trial(rows), 'candidate')
        rows[1]['validation_mean_return'] = float('nan')
        with self.assertRaises(ValueError):
            optimization.select_trial(rows)

    def test_checkpoint_hash_guard_detects_a_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'checkpoint.bin'
            path.write_bytes(b'original')
            hashes = optimization.file_hashes([path])
            optimization.verify_hashes(hashes)
            path.write_bytes(b'changed')
            with self.assertRaises(AssertionError):
                optimization.verify_hashes(hashes)

    def test_fresh_evaluation_counts_only_one_complete_episode(self):
        def rollout(done):
            return SimpleNamespace(transitions=SimpleNamespace(reward=np.array([[0.], [20.]]),
                terminated=np.asarray(done), truncated=np.zeros((2, 1), bool)))
        np.testing.assert_array_equal(optimization.first_episode_returns(rollout([[False], [True]]), 1), [20.])
        for done in ([[True], [True]], [[False], [False]]):
            with self.assertRaises(AssertionError):
                optimization.first_episode_returns(rollout(done), 1)

    def test_paired_bootstrap_keeps_zero_and_negative_differences(self):
        result = optimization.paired_summary([0., 0.], [20., 20.], stratum_sizes=[2])
        self.assertEqual(result['mean_difference'], -20.)
        self.assertEqual(result['episode_bootstrap_95'], [-20., -20.])
        with self.assertRaises(ValueError):
            optimization.paired_summary([0.], [0., 20.], stratum_sizes=[1])

    def test_partner_stratification_preserves_fixed_partner_weight(self):
        # Resampling the partner mixture would create artificial uncertainty:
        # each partner's within-stratum paired differences are constant here.
        result = optimization.paired_summary([10., 10., 0., 0., 0., 0., 0., 0.],
            [0., 0., 10., 10., 10., 10., 10., 10.], stratum_sizes=[2, 6])
        self.assertEqual(result['mean_difference'], -5.)
        self.assertEqual(result['episode_bootstrap_95'], [-5., -5.])
        self.assertEqual(result['stratum_sizes'], [2, 6])

    def test_pooled_continuations_keep_seed_by_partner_strata(self):
        differences = np.repeat([10., -10., 20., -20., 30., -30.], 4)
        result = optimization.paired_summary(differences, np.zeros(24), stratum_sizes=[4]*6)
        self.assertEqual(result['mean_difference'], 0.)
        self.assertEqual(result['episode_bootstrap_95'], [0., 0.])
        for sizes in ([], [3], [1, 0, 1], [1.5, .5], [True, 1]):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                optimization.paired_summary([0., 0.], [0., 0.], stratum_sizes=sizes)

    def test_independent_jobs_keep_declaration_order_and_propagate_failures(self):
        with mock.patch.object(optimization, 'train_trial', side_effect=lambda number: number*2):
            results = optimization.execute_jobs({'first': {'number': 1}, 'second': {'number': 2}}, 2)
        self.assertEqual(list(results), ['first', 'second'])
        self.assertEqual(results, {'first': 2, 'second': 4})
        with mock.patch.object(optimization, 'train_trial', side_effect=RuntimeError('failed')):
            with self.assertRaises(RuntimeError):
                optimization.execute_jobs({'first': {}}, 2)

    def test_continuation_preserves_initial_state_and_refuses_overwrite(self):
        config = ModelConfig(19, 7, global_state_dim=38, hidden_dim=4,
                             history_window=4, intent_window=2)
        tc = TrainConfig(num_envs=2, num_steps=4, epochs=1)
        pool = ScriptedPool((0, 1))
        train_env, eval_env = SyntheticShaping(horizon=4), SyntheticCoordination(horizon=4)
        state = init_train_state(jax.random.PRNGKey(31), config)
        state = state._replace(optimizer=state.optimizer._replace(count=jnp.asarray(5)),
                              norm=state.norm._replace(count=jnp.asarray(10.)))
        before = tree_hash(state)
        source = dict(config=config, state=state)
        operations = optimization.CompiledOperations(train_env, eval_env, pool)
        trial = optimization.DEFAULT_TRIALS[0]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'trial'
            result = optimization.train_trial(output, trial, 7, source, tc, 2, operations, 1., {})
            self.assertEqual(tree_hash(state), before)
            self.assertEqual(result['metadata']['resume']['optimizer_count'], 5)
            self.assertEqual(result['metadata']['resume']['normalizer_count'], 10.)
            self.assertEqual(int(result['state'].optimizer.count), 7)
            self.assertEqual(result['metadata']['physical_environment_steps'], 16)
            self.assertGreaterEqual(float(result['state'].norm.count), 10.)
            # Same retained Adam/norm and reset RNG must replay the continuation exactly.
            collect, update = operations.training(config, tc, dynamic_shaping=True)
            runner = init_runner(jax.random.PRNGKey(7), train_env, pool, config, tc)
            expected = state
            for _ in range(2):
                runner, batch = collect(expected, runner, jnp.asarray(1., jnp.float32))
                expected, _ = update(expected, batch)
            self.assertEqual(tree_hash(expected), tree_hash(result['state']))
            with self.assertRaises(FileExistsError):
                optimization.train_trial(output, trial, 7, source, tc, 2, operations, 1., {})
            saved = json.loads((output/'checkpoint'/'metadata.json').read_text())
            self.assertEqual(saved['metadata']['source'], {})
            self.assertFalse(saved['metadata']['independent_initialization_seed'])
        first = operations.training(config, tc, dynamic_shaping=True)
        second = operations.training(config, replace(tc, epochs=8), dynamic_shaping=True)
        self.assertIs(first[0], second[0])
        self.assertIsNot(first[1], second[1])


if __name__ == '__main__':
    unittest.main()
