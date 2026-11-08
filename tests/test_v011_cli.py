"""CLI compatibility, provenance and sparse evaluation regressions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from styleplan_paper import cli
from styleplan_paper.config import ModelConfig
from styleplan_paper.envs import SyntheticCoordination
from styleplan_paper.partners import ScriptedPool


def args(command='train', *extra):
    base = [command, '--output', 'unused']
    if command == 'evaluate':
        base += ['--checkpoint', 'fake-checkpoint']
    return cli.build_parser().parse_args(base+list(extra))


def loaded(config, schema=3, mode=None):
    environment_config = {} if mode is None else dict(task_context_mode=mode)
    return dict(config=config, state=None, metadata=dict(schema_version=schema,
        metadata=dict(environment_config=environment_config)))


class CliCompatibilityTests(unittest.TestCase):
    def test_new_defaults_and_separate_intent_coefficients(self):
        a = args()
        env = SyntheticCoordination()
        c = cli.model_config(a, env)
        self.assertEqual(c.style_injection, 'bounded')
        self.assertEqual(c.slow_injection_scale, .03)
        self.assertEqual(a.intent_coef, .1)
        self.assertEqual(a.pretrain_intent_coef, 1.)
        self.assertEqual(a.training_shaping_coef, 0.)

    def test_resume_inherits_architecture_and_rejects_explicit_changes(self):
        env = SyntheticCoordination()
        c = ModelConfig(env.obs_dim, env.task_dim, env.global_state_dim,
            hidden_dim=8, history_window=4, intent_window=2,
            ablation='no_partner_model', style_injection='legacy', slow_injection_scale=1.)
        saved = loaded(c, schema=2)
        self.assertEqual(cli.model_config(args(), env, saved), c)
        self.assertEqual(cli.model_config(args('train', '--style-injection', 'legacy',
            '--slow-injection-scale', '1'), env, saved), c)
        for options in (('--style-injection', 'bounded'), ('--slow-injection-scale', '.03'),
                        ('--hidden-dim', '16'), ('--ablation', 'full')):
            with self.assertRaisesRegex(ValueError, 'Resume architecture'):
                cli.model_config(args('train', *options), env, saved)

    def test_checkpoint_schema_and_metadata_select_context(self):
        a = args('train', '--env', 'overcooked', '--public-actions')
        c = ModelConfig(60, 4, 120, style_injection='legacy', slow_injection_scale=1.)
        cases = ((loaded(c, 2), 'legacy'), (loaded(c, 3), 'spatial'),
                 (loaded(c, 3, 'legacy'), 'legacy'))
        with patch.object(cli, 'OvercookedV2') as constructor:
            for saved, expected in cases:
                cli.environment(a, saved)
                self.assertEqual(constructor.call_args.kwargs['task_context_mode'], expected)
            cli.environment(args('train', '--env', 'overcooked', '--task-context-mode', 'spatial'), loaded(c, 2))
            self.assertEqual(constructor.call_args.kwargs['task_context_mode'], 'spatial')

    def test_environment_shaping_is_training_only(self):
        a = args('train', '--env', 'overcooked', '--training-shaping-coef', '.5')
        with patch.object(cli, 'OvercookedV2') as constructor:
            cli.environment(a)
            self.assertEqual(constructor.call_args.kwargs['training_shaping_coef'], .5)
            cli.environment(a, sparse=True)
            self.assertEqual(constructor.call_args.kwargs['training_shaping_coef'], 0.)
        with self.assertRaisesRegex(ValueError, 'only supported for Overcooked'):
            cli.environment(args('train', '--training-shaping-coef', '.5'))

    def test_frozen_eval_uses_checkpoint_context_and_sparse_reward(self):
        env = SyntheticCoordination()
        env.task_context_mode, env.training_shaping_coef = 'legacy', 0.
        c = ModelConfig(env.obs_dim, env.task_dim, env.global_state_dim,
            style_injection='legacy', slow_injection_scale=1.)
        saved = loaded(c, 2)
        with tempfile.TemporaryDirectory() as temp:
            a = args('evaluate', '--env', 'overcooked', '--output', str(Path(temp)/'xp.json'))
            with patch.object(cli, 'load_checkpoint', return_value=saved), \
                 patch.object(cli, 'OvercookedV2', return_value=env) as constructor, \
                 patch.object(cli, 'partner_pool', return_value=ScriptedPool((3, 4))), \
                 patch.object(cli, 'evaluate', return_value=dict(mean_return=0.)) as evaluate:
                cli.run_eval(a)
                self.assertEqual(constructor.call_args.kwargs['task_context_mode'], 'legacy')
                self.assertEqual(constructor.call_args.kwargs['training_shaping_coef'], 0.)
                self.assertIs(evaluate.call_args.args[1], env)
                self.assertTrue((Path(temp)/'xp.json').exists())

    def test_real_synthetic_selfplay_cli_records_two_roles(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'selfplay'
            a = args('train', '--output', str(output), '--ablation', 'no_partner_model',
                '--self-play', '--updates', '1', '--pretrain-updates', '0',
                '--num-envs', '2', '--steps', '4', '--epochs', '1', '--horizon', '4',
                '--history-window', '4', '--intent-window', '2', '--hidden-dim', '8',
                '--eval-episodes', '0')
            cli.run_train(a)
            run = json.loads((output/'run.json').read_text())
            row = json.loads((output/'metrics.jsonl').read_text())
            checkpoint = json.loads((output/'checkpoint'/'metadata.json').read_text())
            self.assertEqual(row['physical_environment_steps'], 8)
            self.assertEqual(row['policy_samples'], 16)
            self.assertEqual(row['trained_roles'], 2)
            self.assertEqual(run['model']['style_injection'], 'bounded')
            self.assertEqual(run['training']['intent_coef'], .1)
            self.assertEqual(run['training']['pretrain_intent_coef'], 1.)
            self.assertEqual(run['environment_config']['evaluation_training_shaping_coef'], 0.)
            self.assertEqual(checkpoint['schema_version'], 3)

    def test_train_heldout_creates_separate_sparse_environment(self):
        constructed = []
        def factory(layout, horizon, **kwargs):
            env = SyntheticCoordination(horizon)
            env.task_context_mode = kwargs['task_context_mode']
            env.training_shaping_coef = kwargs['training_shaping_coef']
            constructed.append(env)
            return env
        def pool(a, env, heldout=False):
            return ScriptedPool((3, 4) if heldout else (0, 1))
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'training'
            a = args('train', '--env', 'overcooked', '--public-actions', '--partner-factory', 'unused:unused',
                '--training-shaping-coef', '.2', '--output', str(output), '--updates', '1',
                '--pretrain-updates', '0', '--num-envs', '2', '--steps', '4', '--epochs', '1',
                '--horizon', '4', '--history-window', '4', '--intent-window', '2', '--hidden-dim', '8',
                '--eval-episodes', '1')
            with patch.object(cli, 'OvercookedV2', side_effect=factory), \
                 patch.object(cli, 'partner_pool', side_effect=pool), \
                 patch.object(cli, 'evaluate', return_value=dict(mean_return=0.)) as evaluate:
                cli.run_train(a)
            self.assertEqual(len(constructed), 2)
            self.assertEqual(constructed[0].training_shaping_coef, .2)
            self.assertEqual(constructed[1].training_shaping_coef, 0.)
            self.assertIs(evaluate.call_args.args[1], constructed[1])
            run = json.loads((output/'run.json').read_text())
            self.assertEqual(run['environment_config']['task_context_mode'], 'spatial')
            self.assertEqual(run['environment_config']['training_shaping_coef'], .2)
            heldout = json.loads((output/'heldout_evaluation.json').read_text())
            self.assertEqual(heldout['evaluation_training_shaping_coef'], 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
