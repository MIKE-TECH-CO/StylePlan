"""Long-horizon regression and historical checkpoint behavior preservation."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from styleplan_paper import model
from styleplan_paper.config import ModelConfig
from styleplan_paper.types import initial_state
from styleplan_paper.training import init_train_state
from styleplan_paper.checkpoint import save_checkpoint, load_checkpoint
from styleplan_paper.partners import tree_hash, ScriptedPool
from styleplan_paper.envs import SyntheticCoordination
from styleplan_paper.evaluation import evaluate_switch


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(19, 7, global_state_dim=38, hidden_dim=8,
                                  history_window=6, intent_window=3)
        self.state = init_train_state(jax.random.PRNGKey(42), self.config)

    def test_800_step_accumulation_is_bounded_with_finite_gradients(self):
        c = self.config
        params = dict(self.state.params)
        params['slow_injection'] = {'w': jnp.ones_like(params['slow_injection']['w'])*.2}
        params['style_encoder'] = jax.tree_util.tree_map(jnp.zeros_like, params['style_encoder'])
        params['style_encoder']['out']['b'] = jnp.ones(c.style_dim)
        params['actor_gru'] = jax.tree_util.tree_map(jnp.zeros_like, params['actor_gru'])
        params['actor_gru']['x']['b'] = params['actor_gru']['x']['b'].at[c.hidden_dim:2*c.hidden_dim].set(12)
        memory = initial_state(c, 1)._replace(style=jnp.ones((1, c.style_dim))*.7,
            valid=jnp.ones((1, c.history_window), bool), actions=jnp.zeros((1, c.history_window), jnp.int32))
        def run(p, config):
            def step(carry, _):
                out, carry = model.apply(p, self.state.norm, carry, jnp.ones((1, 19))*.1,
                                         jnp.zeros((1, 7)), jnp.zeros((1, 38)), config)
                return carry, (out['logits'], carry.hidden)
            return jax.lax.scan(step, memory, None, length=800)[1]
        bounded_logits, bounded_h = jax.jit(lambda p: run(p, c))(params)
        _, legacy_h = jax.jit(lambda p: run(p, replace(c, style_injection='legacy')))(params)
        self.assertLessEqual(float(jnp.abs(bounded_h).max()), 1.00001)
        self.assertGreater(float(jnp.abs(legacy_h).max()), 100.)
        loss, gradient = jax.jit(jax.value_and_grad(lambda p: jnp.mean(run(p, c)[0]**2)))(params)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(gradient)))
        self.assertTrue(np.isfinite(np.asarray(bounded_logits)).all())

    def test_schema2_preserves_literal_legacy_gru_and_all_arrays(self):
        c = replace(self.config, style_injection='legacy', slow_injection_scale=1.)
        state = init_train_state(jax.random.PRNGKey(4), c)
        with tempfile.TemporaryDirectory() as temp:
            path = save_checkpoint(Path(temp)/'old', state, c)
            meta_path = path/'metadata.json'
            meta = json.loads(meta_path.read_text())
            meta['schema_version'] = 2
            del meta['config']['style_injection']
            del meta['config']['slow_injection_scale']
            meta_path.write_text(json.dumps(meta))
            loaded = load_checkpoint(path)
        self.assertEqual(loaded['config'], c)
        self.assertEqual(tree_hash(loaded['state']), tree_hash(state))
        memory = initial_state(c, 1)._replace(hidden=jnp.ones((1, c.hidden_dim))*2,
            style=jnp.ones((1, c.style_dim))*.8)
        obs = jnp.ones((1, c.obs_dim))*.2
        out, new = model.apply(loaded['params'], loaded['norm'], memory, obs,
            jnp.zeros((1, c.task_dim)), jnp.zeros((1, c.global_state_dim)), c)
        expected = model.gru(state.params['actor_gru'], memory.hidden+
            model.dense(state.params['slow_injection'], out['style']),
            model.mlp(state.params['observation_encoder'], obs))
        np.testing.assert_allclose(new.hidden, expected, atol=1e-6)

    def test_schema3_requires_explicit_conditioning(self):
        with tempfile.TemporaryDirectory() as temp:
            path = save_checkpoint(Path(temp)/'new', self.state, self.config)
            self.assertEqual(load_checkpoint(path)['config'].style_injection, 'bounded')
            metadata = json.loads((path/'metadata.json').read_text())
            del metadata['config']['style_injection']
            (path/'metadata.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'explicit recurrent'):
                load_checkpoint(path)

    def test_switch_rejects_shaped_rewards_before_rollout(self):
        env = SyntheticCoordination(400)
        env.training_shaping_coef = .5
        with self.assertRaisesRegex(ValueError, 'sparse reward'):
            evaluate_switch(self.state, env, ScriptedPool((0, 1)), self.config, 200)


if __name__ == '__main__':
    unittest.main()
