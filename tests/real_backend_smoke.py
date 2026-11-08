"""Opt-in integration check using an ALREADY INSTALLED JAXMarl environment.

Partners are untrained NoPM checkpoints; outputs are compatibility evidence only.
No dependencies are installed and output paths must not already exist.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
from styleplan_paper.config import ModelConfig, TrainConfig
from styleplan_paper.envs import OvercookedV2
from styleplan_paper.training import init_train_state, init_runner, make_rollout, make_update, replay
from styleplan_paper.partners import ModelPool, tree_hash
from styleplan_paper.checkpoint import save_checkpoint, source_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    env = OvercookedV2(horizon=8, public_actions=True)
    c = ModelConfig(env.obs_dim, env.task_dim, global_state_dim=env.global_state_dim, hidden_dim=16)
    tc = TrainConfig(num_envs=2, num_steps=8, epochs=1)
    baseline = replace(c, ablation='no_partner_model')
    paths = []
    for seed in (101, 102):
        path = output/f'untrained_test_partner_{seed}'
        save_checkpoint(path, init_train_state(jax.random.PRNGKey(seed), baseline), baseline,
                        {'untrained_compatibility_test_only': True})
        paths.append(path)
    pool = ModelPool(paths)
    frozen = tree_hash({'params': pool.params, 'norms': pool.norms})
    state = init_train_state(jax.random.PRNGKey(7), c)
    runner = init_runner(jax.random.PRNGKey(8), env, pool, c, tc)
    collect = make_rollout(env, pool, c, tc)
    rows = []
    for pretrain in (True, False):
        start = time.perf_counter()
        runner, batch = collect(state, runner)
        replayed = replay(state.params, state.norm, batch, c)
        logp = jnp.take_along_axis(jax.nn.log_softmax(replayed['logits']),
                                  batch.transitions.action[..., None], -1).squeeze(-1)
        np.testing.assert_allclose(logp, batch.transitions.log_prob, atol=2e-6)
        state, metrics = make_update(c, tc, pretrain=pretrain)(state, batch)
        jax.block_until_ready(metrics['loss'])
        row = {k: float(v) for k, v in metrics.items()}
        assert row['finite_gradient'] == 1. and row['parameter_delta'] > 0.
        assert np.isfinite(list(row.values())).all()
        row.update(stage='pretrain' if pretrain else 'ppo', seconds=time.perf_counter()-start)
        rows.append(row)
        print(json.dumps({'stage': row['stage'], 'seconds': row['seconds'], 'loss': row['loss']}), flush=True)
    assert frozen == tree_hash({'params': pool.params, 'norms': pool.norms})
    report = dict(status='PASS', environment=env.name, model=asdict(c), training=asdict(tc),
        jax_version=jax.__version__, devices=[str(d) for d in jax.devices()],
        source_hashes=source_manifest(), frozen_partners_unchanged=True, replay_matches_collection=True,
        published_metrics_reproduced=False, untrained_partners_for_compatibility_only=True, updates=rows)
    save_checkpoint(output/'ego_checkpoint', state, c, report)
    (output/'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': 'PASS', 'output': str(output.resolve())}), flush=True)


if __name__ == '__main__':
    main()
