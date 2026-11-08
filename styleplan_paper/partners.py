"""Frozen partner pools with content hashes and explicit benchmark identity."""
import hashlib
import json
import numpy as np
import jax
import jax.numpy as jnp
from . import model
from .types import initial_state


def tree_hash(tree):
    digest = hashlib.sha256()
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        x = np.asarray(value)
        digest.update(str(path).encode())
        digest.update(str(x.shape).encode())
        digest.update(str(x.dtype).encode())
        digest.update(x.tobytes(order='C'))
    return digest.hexdigest()


def assert_disjoint(train_pool, test_pool):
    overlap = set(train_pool.content_hashes) & set(test_pool.content_hashes)
    if overlap:
        raise ValueError('Training/held-out partner content overlaps: '+', '.join(sorted(overlap)))


class ScriptedPool:
    """Deterministic cue permutations solely for synthetic correctness tests."""
    def __init__(self, offsets=(0, 1, 2), reverse=False):
        if not offsets or len(set(offsets)) != len(offsets):
            raise ValueError('Need nonempty distinct offsets')
        weights = []
        for shift in offsets:
            w = np.zeros((19, 6), np.float32)
            for cue in range(6):
                action = ((-cue if reverse else cue)+shift)%6
                w[cue, action] = 8.
            weights.append(w)
        self.weights = jnp.asarray(np.stack(weights))
        self.size = len(weights)
        self.identities = [f'synthetic/{reverse}/{i}' for i in offsets]
        self.content_hashes = [tree_hash({'weights': w}) for w in weights]
        self.manifest = [dict(identity=i, content_hash=h, kind='SCRIPTED_TEST_ONLY')
                         for i, h in zip(self.identities, self.content_hashes)]

    def initial_state(self, batch_size):
        return jnp.zeros((batch_size, 0))

    def act(self, key, ids, obs, state):
        logits = jnp.einsum('bi,bia->ba', obs, self.weights[ids])
        return jnp.argmax(logits, -1).astype(jnp.int32), state


class ModelPool:
    """Frozen NoPM checkpoints from this implementation (e.g. self-play pool).

    NoPM needs no partner history. A full StylePlan checkpoint requires its own
    causal partner-history wrapper and is deliberately rejected here.
    """
    def __init__(self, checkpoints):
        from .checkpoint import load_checkpoint
        loaded = [load_checkpoint(p) for p in checkpoints]
        if not loaded:
            raise ValueError('Empty partner checkpoint list')
        self.config = loaded[0]['config']
        if self.config.ablation != 'no_partner_model':
            raise ValueError('Frozen model partner adapter requires NoPM baseline checkpoints')
        if any(x['config'] != self.config for x in loaded):
            raise ValueError('All partner checkpoint configurations must match')
        self.params = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *(x['params'] for x in loaded))
        self.norms = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *(x['norm'] for x in loaded))
        self.size = len(loaded)
        self.identities = [str(p) for p in checkpoints]
        # Names/paths are NOT included; aliases of the same weights collide.
        self.content_hashes = [tree_hash({'params': x['params'], 'norm': x['norm']}) for x in loaded]
        self.manifest = [dict(identity=i, content_hash=h, kind='FROZEN_NO_PM_CHECKPOINT')
                         for i, h in zip(self.identities, self.content_hashes)]

    def initial_state(self, batch_size):
        return initial_state(self.config, batch_size)

    def act(self, key, ids, obs, state):
        params = jax.tree_util.tree_map(lambda x: x[ids], self.params)
        norms = jax.tree_util.tree_map(lambda x: x[ids], self.norms)
        c = self.config
        def one(p, n, s, x):
            s = jax.tree_util.tree_map(lambda z: z[None], s)
            out, new = model.apply(p, n, s, x[None], jnp.zeros((1, c.task_dim)),
                                   jnp.zeros((1, c.global_state_dim)), c)
            return out['logits'][0], jax.tree_util.tree_map(lambda z: z[0], new)
        logits, state = jax.vmap(one)(params, norms, state, obs)
        action = jax.random.categorical(key, logits).astype(jnp.int32)
        return action, state


def validate_pool(pool, obs_dim, num_actions):
    if pool.size < 1 or len(pool.content_hashes) != pool.size:
        raise ValueError('Invalid pool manifest')
    if len(set(pool.content_hashes)) != pool.size:
        raise ValueError('Duplicate partner content inside pool')
    if isinstance(pool, ScriptedPool) and obs_dim != 19:
        raise ValueError('Scripted test partners cannot be used on Overcooked')
    if isinstance(pool, ModelPool) and (pool.config.obs_dim != obs_dim or pool.config.num_actions != num_actions):
        raise ValueError('Partner architecture does not match environment')
