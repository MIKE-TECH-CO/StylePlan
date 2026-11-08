"""Behavioral/gradient/causality tests, not published-result reproduction."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
from styleplan_paper.config import ModelConfig, TrainConfig, ABLATIONS
from styleplan_paper import model
from styleplan_paper.types import initial_state, initial_norm
from styleplan_paper.envs import SyntheticCoordination, OvercookedV2
from styleplan_paper.partners import ScriptedPool, ModelPool, assert_disjoint, validate_pool, tree_hash
from styleplan_paper.training import init_train_state, init_runner, make_rollout, make_update, replay, evaluate
from styleplan_paper.losses import future_targets, generalized_advantage, contrastive, auxiliary
from styleplan_paper.checkpoint import save_checkpoint, load_checkpoint
from styleplan_paper.evaluation import switch_metrics, evaluate_switch


class PaperImplementationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = ModelConfig(19, 7, global_state_dim=38, hidden_dim=16, history_window=6, intent_window=3)
        cls.tc = TrainConfig(num_envs=2, num_steps=8, epochs=1)
        cls.state = init_train_state(jax.random.PRNGKey(42), cls.c)
        cls.env = SyntheticCoordination(horizon=4)
        cls.pool = ScriptedPool((0, 1))
        cls.runner = init_runner(jax.random.PRNGKey(3), cls.env, cls.pool, cls.c, cls.tc)
        cls.rollout_fn = make_rollout(cls.env, cls.pool, cls.c, cls.tc)
        cls.after, cls.data = cls.rollout_fn(cls.state, cls.runner)
        cls.outputs = replay(cls.state.params, cls.state.norm, cls.data, cls.c)
        jax.block_until_ready(cls.outputs['logits'])

    def assertClose(self, a, b, atol=1e-6):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=1e-5)

    def active_memory(self):
        s = initial_state(self.c, 2)
        for t in range(6):
            obs = jax.nn.one_hot(jnp.array([t, t+2]), 19)
            s = model.observe(s, obs, jnp.array([t%6, (t+2)%6]),
                jnp.array([[t/6, .2], [.6, t/6]]), jnp.ones(2, dtype=bool))
        return s

    def forward(self, memory, c=None, global_state=None):
        c = c or self.c
        return model.apply(self.state.params, self.state.norm, memory,
            jnp.arange(38, dtype=jnp.float32).reshape(2, 19)/38,
            jnp.ones((2, 7))*.3,
            jnp.zeros((2, 38)) if global_state is None else global_state, c)

    def test_01_default_paper_dimensions_and_literal_horizon(self):
        c = ModelConfig(19, 7)
        self.assertEqual((c.history_window, c.style_dim, c.intent_dim, c.prediction_steps), (50, 32, 16, 6))
        self.assertEqual(self.outputs['future_action_logits'].shape, (8, 2, 6, 6))

    def test_02_gaussian_kl_analytical_and_conflict(self):
        zeros = jnp.zeros((2, 3))
        self.assertClose(model.gaussian_kl(zeros, zeros, zeros, zeros), jnp.zeros(2))
        self.assertClose(model.gaussian_kl(jnp.ones((2, 3)), zeros, zeros, zeros), jnp.ones(2)*1.5)
        same = model.projected_conflict(jnp.array([[1., 0.]]), jnp.array([[2., 0.]]))
        opposite = model.projected_conflict(jnp.array([[1., 0.]]), jnp.array([[-1., 0.]]))
        self.assertClose(same, [0.]); self.assertClose(opposite, [2.])
        self.assertTrue(np.isfinite(np.asarray(model.projected_conflict(zeros, zeros))).all())

    def test_03_ema_bounded_gru_then_film_equations(self):
        for name in ('prior_projection', 'intent_projection', 'slow_injection'):
            self.assertNotIn('b', self.state.params[name])
        memory = self.active_memory()._replace(style=jnp.ones((2, 32))*.1)
        out, new = self.forward(memory)
        self.assertClose(new.style, .99*memory.style+.01*out['instant_style'])
        obs = jnp.arange(38, dtype=jnp.float32).reshape(2, 19)/38
        hidden = model.gru(self.state.params['actor_gru'],
            jnp.tanh(memory.hidden+model.dense(self.state.params['slow_injection'], out['style'])),
            model.mlp(self.state.params['observation_encoder'], obs))
        self.assertClose(new.hidden, hidden)
        gamma, beta = jnp.split(model.dense(self.state.params['film'], out['context']), 2, -1)
        expected = model.dense(self.state.params['actor'], (1+gamma)*hidden+beta)
        self.assertClose(out['logits'], expected)

    def test_04_privileged_critic_input_cannot_change_actor(self):
        memory = self.active_memory()
        a, _ = self.forward(memory)
        b, _ = self.forward(memory, global_state=jnp.ones((2, 38))*3)
        self.assertClose(a['logits'], b['logits'])
        self.assertClose(a['context'], b['context'])
        self.assertGreater(float(jnp.max(jnp.abs(a['value']-b['value']))), 1e-5)

    def test_05_future_targets_and_episode_boundaries(self):
        actions = jnp.arange(5)[:, None]
        done = jnp.array([False, True, False, False, True])[:, None]
        targets, valid = future_targets(actions, done, 5)
        expected = [[1,1,0,0,0,0], [1,0,0,0,0,0], [1,1,1,0,0,0], [1,1,0,0,0,0], [1,0,0,0,0,0]]
        self.assertClose(valid[:, 0], expected)
        self.assertEqual(int(targets[0, 0, 1]), 1)

    def test_06_replay_matches_behavior_across_resets(self):
        logp = jax.nn.log_softmax(self.outputs['logits'])
        actual = jnp.take_along_axis(logp, self.data.transitions.action[..., None], -1).squeeze(-1)
        self.assertClose(actual, self.data.transitions.log_prob, 2e-6)
        self.assertClose(self.outputs['value'], self.data.transitions.value, 2e-6)
        self.assertEqual(self.after.model_state.valid.dtype, jnp.bool_)
        self.assertTrue(bool(jnp.all(self.after.model_state.actions == -1)))

    def test_07_future_labels_never_enter_actor(self):
        t = self.data.transitions
        changed = self.data._replace(transitions=t._replace(partner_action=(t.partner_action+2)%6,
                                                             reward=t.reward+99))
        out = replay(self.state.params, self.state.norm, changed, self.c)
        self.assertClose(out['logits'], self.outputs['logits'])
        self.assertClose(out['context'], self.outputs['context'])

    def test_08_future_observations_cannot_change_prefix(self):
        t = self.data.transitions
        changed = self.data._replace(transitions=t._replace(obs=t.obs.at[5:].add(7),
            observed_partner_action=t.observed_partner_action.at[5:].set(5)))
        out = replay(self.state.params, self.state.norm, changed, self.c)
        self.assertClose(out['logits'][:5], self.outputs['logits'][:5])
        # A completed action at t only enters the model at t+1.
        changed = self.data._replace(transitions=t._replace(observed_partner_action=t.observed_partner_action.at[1].set(5)))
        out = replay(self.state.params, self.state.norm, changed, self.c)
        self.assertClose(out['logits'][:2], self.outputs['logits'][:2])

    def test_09_two_timescales_and_observable_context(self):
        memory = self.active_memory()
        a, _ = self.forward(memory)
        b, _ = self.forward(memory._replace(actions=memory.actions.at[:, -1].set(4)))
        self.assertGreater(float(jnp.linalg.norm(a['context']-b['context'])), 1e-7)
        old = memory._replace(actions=memory.actions.at[:, :3].set(5),
                              observations=memory.observations.at[:, :3].add(8))
        c = replace(self.c, ablation='no_style')
        a, _ = self.forward(memory, c); b, _ = self.forward(old, c)
        self.assertClose(a['intent'], b['intent'])

    def test_10_structural_ablations(self):
        memory = self.active_memory()
        outs = {name: self.forward(memory, replace(self.c, ablation=name))[0] for name in ABLATIONS}
        for name in ('no_partner_model', 'intent_only', 'no_style'):
            self.assertClose(outs[name]['style'], jnp.zeros((2, 32)))
        for name in ('no_partner_model', 'style_only', 'no_intent'):
            self.assertClose(outs[name]['intent'], jnp.zeros((2, 16)))
        self.assertClose(outs['no_arbitration']['gate'], jnp.ones(2)*.5)
        self.assertClose(outs['no_intent']['gate'], jnp.zeros(2))
        self.assertClose(outs['concat']['prior_mean'], jnp.zeros((2, 16)))
        self.assertClose(outs['no_partner_model']['context'], jnp.zeros((2, 64)))
        nopm, _ = self.forward(initial_state(self.c, 2), replace(self.c, ablation='no_partner_model'))
        self.assertClose(nopm['logits'], outs['no_partner_model']['logits'])
        self.assertGreater(float(jnp.linalg.norm(outs['no_arbitration']['context']-outs['concat']['context'])), 1e-6)

    def test_11_every_main_module_receives_training_gradient(self):
        before_pool = tree_hash(self.pool.weights)
        updated, metrics = make_update(self.c, self.tc)(self.state, self.data)
        jax.block_until_ready(metrics['loss'])
        self.assertEqual(float(metrics['finite_gradient']), 1.)
        for name in ('style_encoder', 'history_encoder', 'posterior', 'intent_decoder', 'prior',
                     'prior_projection', 'intent_projection', 'gate', 'fusion', 'film', 'actor_gru'):
            self.assertGreater(float(metrics['grad/'+name]), 1e-12, name)
        self.assertGreater(float(metrics['parameter_delta']), 0.)
        self.assertEqual(float(self.state.norm.count), 0.)
        self.assertGreater(float(updated.norm.count), 0.)
        self.assertEqual(before_pool, tree_hash(self.pool.weights))
        self.assertTrue(np.isfinite([float(x) for x in metrics.values()]).all())

    def test_12_truncation_bootstraps_but_termination_does_not(self):
        reward, value, next_value = jnp.ones((2, 1)), jnp.zeros((2, 1)), jnp.ones((2, 1))*10
        adv, _ = generalized_advantage(reward, value, next_value,
            jnp.array([[True], [False]]), jnp.array([[False], [True]]), .9, .95)
        self.assertClose(adv, [[1.], [10.]])

    def test_13_normalization_eval_is_read_only(self):
        before = tree_hash(self.state)
        result = evaluate(self.state, SyntheticCoordination(4), ScriptedPool((3, 4)), self.c, 1, seed=5)
        self.assertTrue(result['parameters_unchanged'])
        self.assertEqual(len(result['records']), 2)
        self.assertEqual(before, tree_hash(self.state))
        self.assertFalse(result['benchmark_claim'])

    def test_14_checkpoint_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp)/'model'
            save_checkpoint(dest, self.state, self.c, {'purpose': 'test'})
            loaded = load_checkpoint(dest)
            self.assertEqual(self.c, loaded['config'])
            self.assertEqual(tree_hash(self.state), tree_hash(loaded['state']))
            with self.assertRaises(FileExistsError):
                save_checkpoint(dest, self.state, self.c)
            with self.assertRaises(ValueError):
                ModelPool([dest])  # Full policy cannot be silently used without causal partner history.
            baseline_c = replace(self.c, ablation='no_partner_model')
            baseline = init_train_state(jax.random.PRNGKey(17), baseline_c)
            a, b = Path(temp)/'a', Path(temp)/'alias'
            save_checkpoint(a, baseline, baseline_c); save_checkpoint(b, baseline, baseline_c)
            pool_a, pool_b = ModelPool([a]), ModelPool([b])
            with self.assertRaises(ValueError):
                assert_disjoint(pool_a, pool_b)
            runner = init_runner(jax.random.PRNGKey(2), self.env, pool_a, self.c, self.tc)
            _, data = make_rollout(self.env, pool_a, self.c, self.tc)(self.state, runner)
            self.assertEqual(data.transitions.action.shape, (8, 2))

    def test_15_pool_hash_validation_and_explicit_observation_contract(self):
        assert_disjoint(ScriptedPool((0, 1)), ScriptedPool((3, 4)))
        with self.assertRaises(ValueError):
            assert_disjoint(ScriptedPool((0, 1)), ScriptedPool((1, 2)))
        with self.assertRaises(ValueError):
            validate_pool(self.pool, 600, 6)
        with self.assertRaises(ValueError):
            OvercookedV2(public_actions=False)

    def test_16_contrastive_no_positive_and_normalizer_empty_batch(self):
        value = contrastive(jnp.eye(3), jnp.arange(3), jnp.ones(3, bool), .1)
        self.assertClose(value, 0.)
        norm = model.update_normalizer(initial_norm(self.c), jnp.ones((2, self.c.fingerprint_dim)), jnp.zeros(2, bool))
        self.assertEqual(tree_hash(norm), tree_hash(initial_norm(self.c)))

    def test_17_self_play_pool_training_path(self):
        c = replace(self.c, ablation='no_partner_model')
        state = init_train_state(jax.random.PRNGKey(11), c)
        runner = init_runner(jax.random.PRNGKey(2), self.env, self.pool, c, self.tc, self_play=True)
        _, data = make_rollout(self.env, self.pool, c, self.tc, self_play=True)(state, runner)
        out = replay(state.params, state.norm, data, c)
        logp = jnp.take_along_axis(jax.nn.log_softmax(out['logits']), data.transitions.action[..., None], -1).squeeze(-1)
        self.assertClose(logp, data.transitions.log_prob, 2e-6)

    def test_18_switch_metrics_analytical_and_censoring(self):
        metrics = switch_metrics([2,2,2,2,0,0,2,2,2,2], 4, 2)
        self.assertEqual(metrics['normalized_drop'], 1.)
        self.assertEqual(metrics['recovery_steps'], 4)
        self.assertEqual(metrics['post_switch_return'], 8.)
        censored = switch_metrics([2,2,2,2,0,0,0,0], 4, 2)
        self.assertIsNone(censored['recovery_steps'])
        self.assertTrue(censored['recovery_censored'])
        undefined = switch_metrics([0,0,0,0,1,1,1,1], 4, 2)
        self.assertIsNone(undefined['normalized_drop'])
        self.assertFalse(undefined['relative_metrics_defined'])

    def test_19_switch_preserves_ego_memory_and_hides_switch_signal(self):
        env = SyntheticCoordination(8)
        schedule = np.zeros((8, 2), dtype=np.int32)
        schedule[4:] = 1
        runner = init_runner(jax.random.PRNGKey(4), env, self.pool, self.c, self.tc)
        runner = runner._replace(partner_ids=jnp.zeros(2, dtype=jnp.int32))
        _, changed = make_rollout(env, self.pool, self.c, self.tc, partner_schedule=schedule)(self.state, runner)
        _, fixed = make_rollout(env, self.pool, self.c, self.tc,
            partner_schedule=np.zeros_like(schedule))(self.state, runner)
        self.assertClose(changed.transitions.partner_id, schedule)
        a = replay(self.state.params, self.state.norm, changed, self.c)
        b = replay(self.state.params, self.state.norm, fixed, self.c)
        # The action at the switch cannot know that a new partner has arrived.
        self.assertClose(a['logits'][:5], b['logits'][:5])
        self.assertTrue(bool(a['history_valid'][4].all()))
        self.assertGreater(float(jnp.linalg.norm(a['style'][4])), 0.)

    def test_20_switch_evaluation_frozen_and_complete(self):
        before = tree_hash(self.state)
        result = evaluate_switch(self.state, SyntheticCoordination(8), self.pool,
                                 self.c, 4, 2, episodes_per_pair=1)
        self.assertEqual(len(result['records']), 2)
        self.assertEqual(len(result['records'][0]['gate_trace']), 8)
        self.assertFalse(result['protocol']['ego_state_reset_at_switch'])
        self.assertEqual(before, tree_hash(self.state))
        self.assertFalse(result['published_metrics_reproduced'])

    def test_21_reject_invalid_training_or_evaluation_config(self):
        with self.assertRaises(ValueError):
            TrainConfig(contrastive_temperature=0.)
        with self.assertRaises(ValueError):
            TrainConfig(prior_coef=-.1)
        with self.assertRaises(ValueError):
            evaluate(self.state, self.env, self.pool, self.c, episodes_per_partner=0)
        with self.assertRaises(ValueError):
            make_rollout(self.env, self.pool, self.c, self.tc, partner_schedule=np.zeros((2, 2)))

    def test_22_intent_loss_sums_horizon_as_equation_nine(self):
        outputs = dict(self.outputs, future_action_logits=jnp.zeros((8, 2, 6, 6)))
        result = auxiliary(outputs, jnp.zeros((8, 2, 6), dtype=jnp.int32),
            jnp.ones((8, 2, 6), dtype=bool), self.data.transitions.partner_id, self.c, self.tc)
        self.assertClose(result['intent_loss'], 6*jnp.log(6.))

    def test_23_overcooked_clock_versions(self):
        env = object.__new__(OvercookedV2)
        env.agents = ['agent_0', 'agent_1']
        env.backend = SimpleNamespace(width=5, height=4)
        env.horizon = 400
        obs = {a: jnp.ones((4, 5, 3)) for a in env.agents}
        agents = SimpleNamespace(pos=SimpleNamespace(x=jnp.array([1,3]), y=jnp.array([1,2])))
        for field in ('time', 'step'):
            env.time_field = field
            frame = env.frame(obs, SimpleNamespace(agents=agents, **{field: jnp.array(200)}))
            self.assertClose(frame.task_context[-1], .5)
            self.assertClose(frame.partner_position, [.75, 2/3])


if __name__ == '__main__':
    unittest.main(verbosity=2)
