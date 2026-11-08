"""Read-only frozen-policy diagnostics. Writes evidence, never trained weights."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from styleplan_paper import model
from styleplan_paper.config import TrainConfig
from styleplan_paper.cli import SelfPlaySpecification
from styleplan_paper.checkpoint import load_checkpoint, source_manifest
from styleplan_paper.partners import ModelPool, tree_hash
from styleplan_paper.envs import OvercookedV2
from styleplan_paper.training import init_runner, make_rollout, replay, tree_norm
from styleplan_paper.losses import auxiliary, future_targets, generalized_advantage
from styleplan_paper.types import Rollout


def trace(params, norm, data, c):
    def step(memory, t):
        out, next_memory = model.apply(params,norm,memory,t.obs,t.context,t.global_state,c)
        logp = jax.nn.log_softmax(out['logits'])
        stats = dict(hidden_norm=jnp.linalg.norm(next_memory.hidden,axis=-1),
            hidden_max=jnp.abs(next_memory.hidden).max(-1),
            logit_range=jnp.ptp(out['logits'],axis=-1),
            entropy=-(jnp.exp(logp)*logp).sum(-1),
            style_norm=jnp.linalg.norm(out['style'],axis=-1),
            injection_norm=jnp.linalg.norm(model.dense(params['slow_injection'],out['style']),axis=-1),
            gate=out['gate'],conflict=out['conflict'])
        next_memory=model.observe(next_memory,t.obs,t.observed_partner_action,t.position,t.position_visible)
        return model.reset_state(next_memory,t.terminated|t.truncated),stats
    return jax.lax.scan(step,data.initial_state,data.transitions)[1]


def trace_summary(tr):
    result={}
    for key,val in tr.items():
        a=np.asarray(val)
        result[key]=dict(mean=float(a.mean()),maximum=float(a.max()),
            windows=[float(a[i:i+50].mean()) for i in range(0,len(a),50)])
    return result


def gradient_diagnostics(state,data,c):
    tc=TrainConfig(num_envs=16,num_steps=64,epochs=2)
    d=Rollout(jax.tree_util.tree_map(lambda x:x[:16],data.initial_state),
        jax.tree_util.tree_map(lambda x:x[:64,:16],data.transitions))
    t=d.transitions
    adv,returns=generalized_advantage(t.reward,t.value,t.next_value,t.terminated,t.truncated,tc.gamma,tc.gae_lambda)
    adv=(adv-adv.mean())/(adv.std()+1e-8)
    targets,mask=future_targets(t.partner_action,t.terminated|t.truncated,c.prediction_horizon)
    def loss(params,aux_only):
        out=replay(params,state.norm,d,c)
        if aux_only:return auxiliary(out,targets,mask,t.partner_id,c,tc)['auxiliary_loss']
        lp=jax.nn.log_softmax(out['logits'])
        chosen=jnp.take_along_axis(lp,t.action[...,None],axis=-1).squeeze(-1)
        ratio=jnp.exp(chosen-t.log_prob)
        policy=-jnp.minimum(ratio*adv,jnp.clip(ratio,.8,1.2)*adv).mean()
        clipped=t.value+jnp.clip(out['value']-t.value,-.2,.2)
        value=.5*jnp.maximum((out['value']-returns)**2,(clipped-returns)**2).mean()
        entropy=-(jnp.exp(lp)*lp).sum(-1).mean()
        return policy+tc.value_coef*value-tc.entropy_coef*entropy
    rl,rg=jax.jit(jax.value_and_grad(lambda p:loss(p,False)))(state.params)
    al,ag=jax.jit(jax.value_and_grad(lambda p:loss(p,True)))(state.params)
    groups={}
    for k in rg:
        a,b=rg[k],ag[k]
        an,bn=float(tree_norm(a)),float(tree_norm(b))
        dot=sum(float(jnp.vdot(x,y)) for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)))
        groups[k]=dict(rl_norm=an,aux_norm=bn,cosine=dot/(an*bn) if an*bn>1e-15 else None)
    return dict(rl_loss=float(rl),aux_loss=float(al),rl_gradient_norm=float(tree_norm(rg)),
        aux_gradient_norm=float(tree_norm(ag)),groups=groups,scope='one fresh 64x16 held-out rollout prefix')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pilot',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--episodes',type=int,default=64)
    args=ap.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    pilot=Path(args.pilot)
    env=OvercookedV2(horizon=400,public_actions=True)
    tc=TrainConfig(num_envs=args.episodes,num_steps=400,epochs=1)
    partners=[load_checkpoint(pilot/f'pool_seed{s}') for s in (101,102,201,202)]
    full=load_checkpoint(pilot/'full/checkpoint')
    baseline=load_checkpoint(pilot/'no_partner_model/checkpoint')
    test_pool=ModelPool([pilot/'pool_seed201',pilot/'pool_seed202'])
    frozen_before=tree_hash({'full':full['state'],'baseline':baseline['state'],'pool':test_pool.params})
    self_pool=SelfPlaySpecification()
    c=baseline['config']
    sp_collect=make_rollout(env,self_pool,c,tc,self_play=True)
    results=[]
    def record(name,runner,collector,state,config,inspect=False):
        _,data=collector(state,runner)
        returns=np.asarray(data.transitions.reward).sum(0)
        out=replay(state.params,state.norm,data,config) if inspect else None
        result=dict(name=name,episodes=len(returns),mean_return=float(returns.mean()),
            total_return=float(returns.sum()),success_rate=float((returns>0).mean()),
            per_episode_returns=returns.tolist(),nonzero_reward_values=np.unique(np.asarray(data.transitions.reward)[np.asarray(data.transitions.reward)>0]).tolist())
        if inspect:result['trace']=trace_summary(jax.jit(lambda d:trace(state.params,state.norm,d,config))(data))
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('trace','per_episode_returns')}),flush=True)
        return data
    for item,seed in zip(partners,(101,102,201,202)):
        runner=init_runner(jax.random.PRNGKey(33000),env,self_pool,c,tc,self_play=True)
        record(f'partner{seed}_selfplay',runner,sp_collect,item['state'],c)
    params=dict(baseline['state'].params)
    params['actor']=jax.tree_util.tree_map(jnp.zeros_like,params['actor'])
    uniform=baseline['state']._replace(params=params)
    runner=init_runner(jax.random.PRNGKey(33000),env,self_pool,c,tc,self_play=True)
    record('uniform_selfplay',runner,sp_collect,uniform,c)
    base_collect=make_rollout(env,test_pool,c,tc)
    fc=full['config'];full_collect=make_rollout(env,test_pool,fc,tc)
    def xp_runner(config):
        return init_runner(jax.random.PRNGKey(44000),env,test_pool,config,tc)._replace(
            partner_ids=jnp.arange(args.episodes,dtype=jnp.int32)%test_pool.size)
    record('nopm_heldout',xp_runner(c),base_collect,baseline['state'],c,True)
    record('uniform_heldout',xp_runner(c),base_collect,uniform,c)
    data=record('full_heldout',xp_runner(fc),full_collect,full['state'],fc,True)
    gradients=gradient_diagnostics(full['state'],data,fc)
    print(json.dumps({'gradient_diagnostics_done':True}),flush=True)
    params=dict(full['state'].params)
    params['slow_injection']=jax.tree_util.tree_map(jnp.zeros_like,params['slow_injection'])
    no_slow=full['state']._replace(params=params)
    # Fixed recorded history isolates recurrence from changed environment actions.
    control_trace=trace_summary(jax.jit(lambda d:trace(no_slow.params,no_slow.norm,d,fc))(data))
    record('full_zero_slow_injection_control',xp_runner(fc),full_collect,no_slow,fc,True)
    assert frozen_before==tree_hash({'full':full['state'],'baseline':baseline['state'],'pool':test_pool.params})
    evidence=dict(status='complete',source_hashes=source_manifest(),jax_version=jax.__version__,
        devices=[str(d) for d in jax.devices()],matmul_precision=str(jax.config.jax_default_matmul_precision),
        original_weights_unchanged=True,no_training_performed=True,results=results,
        fixed_history_zero_slow_control=control_trace,gradient_decomposition=gradients,
        warning='Zero-injection is an inference diagnostic on the same weights, not a retrained ablation result')
    (output/'evidence.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps({'status':'complete','output':str(output.resolve())}),flush=True)


if __name__=='__main__':main()
