"""Bounded real-environment pilot, not a reproduction of the paper budget.

Train four independently seeded NoPM self-play partners. Two frozen checkpoints
are for ego training, two held out. Compare full vs NoPM at equal PPO budgets.
The reward is sparse delivery reward; zero-return outcomes are retained honestly.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import numpy as np
import jax
from styleplan_paper.config import ModelConfig, TrainConfig
from styleplan_paper.envs import OvercookedV2
from styleplan_paper.partners import ModelPool, assert_disjoint, tree_hash
from styleplan_paper.training import init_train_state, init_runner, make_rollout, make_update, evaluate
from styleplan_paper.evaluation import evaluate_switch
from styleplan_paper.checkpoint import save_checkpoint, source_manifest
from styleplan_paper.cli import SelfPlaySpecification


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2)


def train_batches(name, output, state, runner, collect, update, count, stage):
    records = []
    for i in range(count):
        start = time.perf_counter()
        runner, data = collect(state, runner)
        state, metrics = update(state, data)
        jax.block_until_ready(metrics['loss'])
        elapsed = time.perf_counter()-start
        row = {k: float(v) for k, v in metrics.items()}
        if row['finite_gradient'] != 1. or not np.isfinite(list(row.values())).all():
            raise FloatingPointError(name+' produced nonfinite metrics')
        row.update(name=name, stage=stage, update=i, seconds=elapsed,
                   includes_compile=i == 0)
        with (output/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        records.append(row)
        if i == 0 or (i+1)%16 == 0 or i+1 == count:
            print(json.dumps({k:row[k] for k in ('name','stage','update','seconds','reward_mean','loss','parameter_delta')}), flush=True)
    return state, runner, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--pool-updates', type=int, default=64)
    parser.add_argument('--ego-updates', type=int, default=64)
    parser.add_argument('--pretrain-updates', type=int, default=8)
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--horizon', type=int, default=400)
    parser.add_argument('--eval-episodes', type=int, default=8)
    args = parser.parse_args()
    if min(args.pool_updates,args.ego_updates,args.num_envs,args.steps,args.eval_episodes) < 1 or args.pretrain_updates < 0:
        parser.error('Invalid pilot budget')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    env = OvercookedV2(horizon=args.horizon, public_actions=True)
    c = ModelConfig(env.obs_dim,env.task_dim,global_state_dim=env.global_state_dim)
    tc = TrainConfig(num_envs=args.num_envs,num_steps=args.steps,epochs=2)
    seeds = (101,102,201,202)
    metadata = dict(environment=env.name, horizon=env.horizon, model=asdict(c), training=asdict(tc),
        budget=vars(args), pool_seeds=seeds, ego_seed=7, sparse_reward_only=True,
        devices=[str(x) for x in jax.devices()], jax_version=jax.__version__,
        matmul_precision=str(jax.config.jax_default_matmul_precision), source_hashes=source_manifest(),
        published_metrics_reproduced=False, purpose='bounded GPU pilot, not statistical superiority evidence')
    write_json(out/'protocol.json',metadata)
    baseline_c = replace(c,ablation='no_partner_model')
    spec = SelfPlaySpecification()
    collect_pool = make_rollout(env,spec,baseline_c,tc,self_play=True)
    update_pool = make_update(baseline_c,tc)
    pool_paths = []
    pool_results = []
    for seed in seeds:
        name = f'pool_seed{seed}'
        state = init_train_state(jax.random.PRNGKey(seed),baseline_c)
        runner = init_runner(jax.random.PRNGKey(seed+1000),env,spec,baseline_c,tc,self_play=True)
        state, runner, rows = train_batches(name,out,state,runner,collect_pool,update_pool,
                                            args.pool_updates,'self_play')
        path = out/name
        save_checkpoint(path,state,baseline_c,dict(metadata,pool_seed=seed))
        pool_paths.append(path)
        pool_results.append(dict(seed=seed,checkpoint=str(path),last_reward_mean=rows[-1]['reward_mean'],
            total_sparse_reward=sum(r['reward_mean']*args.num_envs*args.steps for r in rows),
            mean_warm_batch_seconds=float(np.mean([r['seconds'] for r in rows[1:]])) if len(rows)>1 else None))
    train_pool, test_pool = ModelPool(pool_paths[:2]), ModelPool(pool_paths[2:])
    assert_disjoint(train_pool,test_pool)
    frozen = tree_hash({'train_params':train_pool.params,'test_params':test_pool.params})
    results = []
    for variant in ('full','no_partner_model'):
        vc = replace(c,ablation=variant)
        state = init_train_state(jax.random.PRNGKey(7),vc)
        runner = init_runner(jax.random.PRNGKey(8),env,train_pool,vc,tc)
        collect = make_rollout(env,train_pool,vc,tc)
        if variant == 'full' and args.pretrain_updates:
            state,runner,_ = train_batches(variant,out,state,runner,collect,make_update(vc,tc,pretrain=True),
                                           args.pretrain_updates,'pretrain')
        state,runner,rows = train_batches(variant,out,state,runner,collect,make_update(vc,tc),args.ego_updates,'ppo')
        checkpoint = out/variant/'checkpoint'
        save_checkpoint(checkpoint,state,vc,dict(metadata,train_partners=train_pool.manifest,
            heldout_partners=test_pool.manifest,variant=variant))
        xp = evaluate(state,env,test_pool,vc,args.eval_episodes,seed=10000)
        xp['training_pool_disjoint'] = True
        write_json(out/variant/'xp.json',xp)
        switch = evaluate_switch(state,env,test_pool,vc,args.horizon//2,min(20,args.horizon//4),
                                 args.eval_episodes,seed=20000)
        switch['training_pool_disjoint'] = True
        write_json(out/variant/'switch.json',switch)
        warm = [r['seconds'] for r in rows[1:]]
        result = dict(variant=variant,checkpoint=str(checkpoint),xp_mean_return=xp['mean_return'],
            post_switch_mean_return=switch['mean_post_switch_return'],
            normalized_drop=switch['mean_normalized_drop'],
            recovery_observed=switch['mean_recovery_steps_observed'],
            recovery_censored_fraction=switch['recovery_censored_fraction'],
            relative_metric_valid_records=switch['relative_metric_valid_records'],
            xp_records=len(xp['records']),switch_records=len(switch['records']),
            warm_training_transitions_per_second=args.num_envs*args.steps/float(np.mean(warm)) if warm else None,
            final_parameter_delta=rows[-1]['parameter_delta'], final_module_gradients={k:v for k,v in rows[-1].items() if k.startswith('grad/')},
            total_sparse_training_reward=sum(r['reward_mean']*args.num_envs*args.steps for r in rows))
        results.append(result)
        print(json.dumps(dict(stage='evaluation',**result)),flush=True)
    assert frozen == tree_hash({'train_params':train_pool.params,'test_params':test_pool.params})
    write_json(out/'summary.json',dict(status='complete',protocol=metadata,pool_results=pool_results,
        results=results,frozen_pool_unchanged=True,published_metrics_reproduced=False,
        note='Small sparse-reward pilot with one ego seed; do not infer comparative superiority.'))
    print(json.dumps({'status':'complete','output':str(out.resolve())}),flush=True)


if __name__ == '__main__':
    main()
