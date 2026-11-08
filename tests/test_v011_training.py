"""Regression coverage for both-seat PPO and first-episode static evaluation."""
from dataclasses import replace
import unittest
import numpy as np
import jax
import jax.numpy as jnp

from styleplan_paper import model
from styleplan_paper.config import ModelConfig, TrainConfig
from styleplan_paper.envs import SyntheticCoordination
from styleplan_paper.partners import ScriptedPool
from styleplan_paper.types import Frame
from styleplan_paper.training import (init_train_state, init_runner, make_rollout,
    make_update, replay, reciprocal_inputs, evaluate, tree_norm)


class EarlyDoneEnvironment(SyntheticCoordination):
    """Advertised horizon 8, actual episode ends at 2; autoreset can follow."""
    def __init__(self, terminal=True):
        super().__init__(horizon=8, time_limit_is_terminal=terminal)

    def step(self, key, state, ego_action, partner_action):
        new, frame, _, _, _ = super().step(key, state, ego_action, partner_action)
        done = new.time >= 2
        return new, frame, jnp.array(1.), done & self.time_limit_is_terminal, done & ~jnp.asarray(self.time_limit_is_terminal)


class DualRoleTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = ModelConfig(19, 7, global_state_dim=38, hidden_dim=8,
            history_window=4, intent_window=2, ablation='no_partner_model')
        cls.tc = TrainConfig(num_envs=2, num_steps=6, epochs=1)
        cls.env = SyntheticCoordination(horizon=4)
        cls.pool = ScriptedPool((0, 1))
        cls.state = init_train_state(jax.random.PRNGKey(22), cls.c)
        cls.runner = init_runner(jax.random.PRNGKey(23), cls.env, cls.pool,
            cls.c, cls.tc, self_play=True)
        cls.after, cls.data = make_rollout(cls.env, cls.pool, cls.c,
            cls.tc, self_play=True)(cls.state, cls.runner)
        cls.out = replay(cls.state.params, cls.state.norm, cls.data, cls.c)

    def assertClose(self, actual, expected, atol=2e-6):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=atol, rtol=1e-5)

    def test_both_seat_data_and_physical_runner_dimensions(self):
        t = self.data.transitions
        self.assertEqual(t.action.shape, (6, 4))
        self.assertEqual(self.data.initial_state.hidden.shape, (4, 8))
        self.assertEqual(self.after.model_state.hidden.shape, (2, 8))
        self.assertEqual(self.after.partner_state.hidden.shape, (2, 8))
        self.assertClose(t.partner_action[:, :2], t.action[:, 2:])
        self.assertClose(t.partner_action[:, 2:], t.action[:, :2])
        self.assertClose(t.reward[:, :2], t.reward[:, 2:])
        self.assertClose(t.global_state[:, 2:, :19], t.global_state[:, :2, 19:])
        self.assertClose(t.global_state[:, 2:, 19:], t.global_state[:, :2, :19])
        self.assertClose(t.obs[0, 2:], self.runner.frame.partner_obs)
        self.assertClose(t.context[0, 2:], self.runner.frame.task_context)

    def test_replay_matches_both_seats_through_reset_and_saved_carry(self):
        logp = jnp.take_along_axis(jax.nn.log_softmax(self.out['logits']),
            self.data.transitions.action[..., None], -1).squeeze(-1)
        self.assertClose(logp, self.data.transitions.log_prob)
        self.assertClose(self.out['value'], self.data.transitions.value)
        self.assertTrue(np.asarray(self.data.transitions.terminated[3]).all())
        # Continue without resetting: both initial memories must be retained.
        _, following = make_rollout(self.env, self.pool, self.c, self.tc,
            self_play=True)(self.state, self.after)
        self.assertClose(following.initial_state.hidden[:2], self.after.model_state.hidden)
        self.assertClose(following.initial_state.hidden[2:], self.after.partner_state.hidden)
        output = replay(self.state.params, self.state.norm, following, self.c)
        next_logp = jnp.take_along_axis(jax.nn.log_softmax(output['logits']),
            following.transitions.action[..., None], -1).squeeze(-1)
        self.assertClose(next_logp, following.transitions.log_prob)
        self.assertClose(self.after.partner_state.actions[:, -2:], self.data.transitions.action[4:6, :2].T)
        self.assertClose(self.after.partner_state.observations[:, -2:],
            self.data.transitions.obs[4:6, 2:].transpose(1, 0, 2))
        self.assertClose(self.after.partner_state.positions[:, -2:],
            self.data.transitions.position[4:6, 2:].transpose(1, 0, 2))

    def test_second_seat_contributes_score_gradient_and_update(self):
        t = self.data.transitions
        advantages = jnp.arange(24, dtype=jnp.float32).reshape(6, 4)/24-.5
        def seat_loss(params, seat):
            outputs = replay(params, self.state.norm, self.data, self.c)
            selected = jnp.take_along_axis(jax.nn.log_softmax(outputs['logits']),
                t.action[..., None], -1).squeeze(-1)
            return -(selected[:, seat]*advantages[:, seat]).mean()
        first = jax.grad(lambda p: seat_loss(p, slice(0, 2)))(self.state.params)
        second = jax.grad(lambda p: seat_loss(p, slice(2, 4)))(self.state.params)
        joint = jax.grad(lambda p: seat_loss(p, slice(None)))(self.state.params)
        self.assertGreater(float(tree_norm(second['actor'])), 1e-5)
        for a, b, total in zip(jax.tree_util.tree_leaves(first),
                              jax.tree_util.tree_leaves(second), jax.tree_util.tree_leaves(joint)):
            self.assertClose(total, .5*(a+b))
        updated, metrics = make_update(self.c, self.tc)(self.state, self.data)
        self.assertEqual(float(metrics['trained_roles']), 2.)
        self.assertEqual(float(metrics['policy_samples']), 24.)
        self.assertEqual(float(metrics['physical_environment_steps']), 12.)
        self.assertClose(metrics['reward_mean'], t.reward[:, :2].mean())
        self.assertGreater(float(metrics['parameter_delta']), 0.)
        self.assertEqual(float(metrics['finite_gradient']), 1.)
        self.assertTrue(np.isfinite([float(value) for value in metrics.values()]).all())
        self.assertGreater(float(metrics['mean_hidden_norm']), 0.)
        self.assertGreater(float(metrics['max_hidden_abs']), 0.)

    def test_terminal_and_truncated_final_values_for_both_roles(self):
        tc = replace(self.tc, num_steps=2)
        for terminal in (True, False):
            env = EarlyDoneEnvironment(terminal=terminal)
            runner = init_runner(jax.random.PRNGKey(23), env, self.pool,
                self.c, tc, self_play=True)
            after, data = make_rollout(env, self.pool, self.c, tc,
                self_play=True)(self.state, runner)
            t = data.transitions
            self.assertTrue(np.asarray((t.terminated | t.truncated)[1]).all())
            self.assertFalse(np.asarray(after.partner_state.valid).any())
            self.assertFalse(np.asarray(after.model_state.valid).any())
            if terminal:
                self.assertClose(t.next_value[1], jnp.zeros(4))
                continue
            world = runner.env_state
            memory = data.initial_state
            for step in range(2):
                _, memory = model.apply(self.state.params, self.state.norm, memory,
                    t.obs[step], t.context[step], t.global_state[step], self.c)
                memory = model.observe(memory, t.obs[step], t.observed_partner_action[step],
                    t.position[step], t.position_visible[step])
                world, frame, *_ = jax.vmap(env.step)(jax.random.split(jax.random.PRNGKey(99), 2),
                    world, t.action[step, :2], t.action[step, 2:])
            context, global_state, _, _ = reciprocal_inputs(frame)
            final_obs = jnp.concatenate((frame.ego_obs, frame.partner_obs))
            final_context = jnp.concatenate((frame.task_context, context))
            final_global = jnp.concatenate((frame.global_state, global_state))
            final, _ = model.apply(self.state.params, self.state.norm, memory,
                final_obs, final_context, final_global, self.c)
            self.assertClose(t.next_value[1], final['value'])
            self.assertGreater(float(jnp.max(jnp.abs(t.next_value[1]))), 1e-5)

    def test_static_eval_stops_at_first_terminal_or_truncation(self):
        for terminal in (True, False):
            result = evaluate(self.state, EarlyDoneEnvironment(terminal), self.pool,
                self.c, episodes_per_partner=2)
            self.assertEqual(len(result['records']), 4)
            for row in result['records']:
                self.assertEqual(row['episode_return'], 2.)
                self.assertEqual(row['episode_length'], 2)
                self.assertEqual(row['terminated'], terminal)
                self.assertEqual(row['truncated'], not terminal)

    def test_evaluation_rejects_training_reward_shaping(self):
        env = EarlyDoneEnvironment()
        env.training_shaping_coef = .1
        with self.assertRaisesRegex(ValueError, 'sparse-reward'):
            evaluate(self.state, env, self.pool, self.c)

    def test_pretraining_and_joint_ppo_use_distinct_prediction_weights(self):
        c = replace(self.c, ablation='full')
        tc = replace(self.tc, intent_coef=.2, pretrain_intent_coef=.8,
            prior_coef=.03, smoothness_coef=.04, epochs=1)
        state = init_train_state(jax.random.PRNGKey(71), c)
        runner = init_runner(jax.random.PRNGKey(72), self.env, self.pool, c, tc)
        _, data = make_rollout(self.env, self.pool, c, tc)(state, runner)
        _, pretrain = make_update(c, tc, pretrain=True)(state, data)
        _, joint = make_update(c, tc, pretrain=False)(state, data)
        # Both report the same raw prediction error on the same weights/data;
        # only its contribution to the objective differs. Other auxiliary
        # components cancel, so this also detects accidentally shared weights.
        self.assertClose(pretrain['intent_loss'], joint['intent_loss'])
        self.assertGreater(float(joint['intent_loss']), 0.)
        self.assertClose(pretrain['auxiliary_loss']-joint['auxiliary_loss'],
            .6*joint['intent_loss'])
        self.assertClose(pretrain['loss'], pretrain['auxiliary_loss'])
        self.assertClose(joint['loss'], joint['policy_loss']+tc.value_coef*joint['value_loss']
            -tc.entropy_coef*joint['entropy']+joint['auxiliary_loss'])
        self.assertClose(pretrain['effective_intent_coef'], .8)
        self.assertClose(joint['effective_intent_coef'], .2)

    def test_legacy_frames_and_frozen_partner_rollout_keep_batch_size(self):
        old_frame = Frame(*self.runner.frame[:6])
        self.assertIsNone(old_frame.partner_task_context)
        runner = init_runner(jax.random.PRNGKey(23), self.env, self.pool, self.c, self.tc)
        _, data = make_rollout(self.env, self.pool, self.c, self.tc)(self.state, runner)
        self.assertEqual(data.transitions.action.shape, (6, 2))
        context, global_state, position, visible = reciprocal_inputs(old_frame)
        self.assertClose(context, old_frame.task_context)
        self.assertClose(position, jnp.zeros_like(old_frame.partner_position))
        self.assertFalse(np.asarray(visible).any())


if __name__ == '__main__':
    unittest.main(verbosity=2)
