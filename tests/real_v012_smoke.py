"""Opt-in real-backend check of scheduled versus fixed shaping, no source writes."""
import argparse
from pathlib import Path
import json
import jax
import jax.numpy as jnp
import numpy as np
from styleplan_paper.checkpoint import load_checkpoint, source_manifest
from styleplan_paper.envs import OvercookedV2
from styleplan_paper.partners import ModelPool, tree_hash
from styleplan_paper.config import TrainConfig
from styleplan_paper.training import init_runner, make_rollout, make_update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.source_run)
    item = load_checkpoint(root/'full/checkpoint')
    c, state = item['config'], item['state']
    pool = ModelPool([root/'pool_seed101/update_1024', root/'pool_seed102/update_1024'])
    env = OvercookedV2(horizon=400, public_actions=True, training_shaping_coef=1.)
    tc = TrainConfig(num_envs=4, num_steps=16, epochs=2)
    before = tree_hash(state)
    frozen = tree_hash({'params': pool.params, 'norms': pool.norms})
    runner = init_runner(jax.random.PRNGKey(12345), env, pool, c, tc)
    fixed = make_rollout(env, pool, c, tc)
    dynamic = make_rollout(env, pool, c, tc, dynamic_shaping=True)
    a, old_batch = fixed(state, runner)
    b, new_batch = dynamic(state, runner, jnp.asarray(1., jnp.float32))
    for x, y in zip(jax.tree_util.tree_leaves((a, old_batch)), jax.tree_util.tree_leaves((b, new_batch))):
        np.testing.assert_allclose(x, y, atol=2e-6, rtol=1e-5)
    _, sparse = dynamic(state, runner, jnp.asarray(0., jnp.float32))
    np.testing.assert_array_equal(old_batch.transitions.action, sparse.transitions.action)
    updated, metrics = make_update(c, tc)(state, new_batch)
    rows = {k: float(v) for k, v in metrics.items()}
    assert rows['finite_gradient'] == 1. and rows['parameter_delta'] > 0.
    assert np.isfinite(list(rows.values())).all()
    assert tree_hash(state) == before
    assert frozen == tree_hash({'params': pool.params, 'norms': pool.norms})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(dict(status='PASS', fixed_and_dynamic_rollouts_match=True,
            source_checkpoint_unchanged=True, frozen_pool_unchanged=True,
            source_hashes=source_manifest(), metrics=rows, devices=[str(x) for x in jax.devices()]), handle, indent=2)
    print(json.dumps(dict(status='PASS', output=str(output), final_approx_kl=rows['final_approx_kl'])), flush=True)


if __name__ == '__main__':
    main()
