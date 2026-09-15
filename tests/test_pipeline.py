import json
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.engine import *
from experiments.summarize import paired,exact_sign_flip,summarize
from experiments.run import jobs,roles
from amrl.meta_learner import matching_mlp_hidden
from amrl.meta_ppo import gae

CFG=dict(steps=256,interval=32,warmup=32,rollout=16,window=128,eval_episodes=1,meta_episodes=9,meta_batch=3)

def test_component_actuation():
    c=HPController();c.update([.30,.02,-.05]);assert c.a==pytest.approx(3.9e-4);assert c.g==.999;assert c.b==0
    for _ in range(100):c.update([.3,.02,.05])
    assert c.a==pytest.approx(.003)
    n=HPController(safeguard='no_drift')
    for _ in range(100):n.update([.3,0,0])
    assert n.a==.01

@pytest.mark.parametrize('algorithm',['ppo','a2c','dqn'])
def test_hyperparameters_reach_learner(algorithm):
    model=make_model(algorithm,make_env('CartPole-v1'),2,(3e-4,.99,.01),16)
    ctrl=HPController();ctrl.update([.3,-.01,.03]);ctrl.apply(model)
    assert model.policy.optimizer.param_groups[0]['lr']==ctrl.a
    assert model.gamma==ctrl.g
    if algorithm=='dqn':assert model.exploration_schedule(.1)==ctrl.b
    else:assert model.rollout_buffer.gamma==ctrl.g;assert model.ent_coef==ctrl.b
    model.env.close()

@pytest.mark.parametrize('env',['CartPole-v1','MountainCar-v0','bsuite/memory_len/0','bsuite/catch/0'])
def test_seeded_env(env):
    a=make_env(env,5);b=make_env(env,5)
    x,_=a.reset(seed=5);y,_=b.reset(seed=5);assert np.array_equal(x,y)
    for _ in range(5):
        x,r,t,tr,_=a.step(0);y,s,u,ur,_=b.step(0)
        assert np.array_equal(x,y) and (r,t,tr)==(s,u,ur)
        if t or tr:a.reset();b.reset()
    a.close();b.close()

def test_mlp_parameter_matching():
    a=MetaLearner();b=MetaLearner('mlp',matching_mlp_hidden())
    na=sum(x.numel() for x in a.parameters());nb=sum(x.numel() for x in b.parameters())
    assert abs(na-nb)/na<.001

def test_gae_boundary_and_likelihood():
    a,r=gae([1,2],[0,0],gamma=1,lam=1);assert torch.equal(r,torch.tensor([3.,2.]))
    p=MetaLearner();x=torch.zeros(1,3,6);m,v,_=p(x);z=torch.zeros_like(m)
    lp=p.log_prob(m,z);assert lp.shape==(1,3) and torch.isfinite(lp).all()

def test_real_controller_training_and_checkpoint(tmp_path):
    tasks=['CartPole-v1'];hps={(tasks[0],'ppo'):(3e-4,.99,.01)}
    p,n,c=train_controller(tmp_path,'test',0,dict(CFG,meta_episodes=1,meta_batch=1),tasks,['ppo'],hps)
    assert c['parameter_change_l2']>0;assert c['environment_interactions']==256
    before={k:v.clone() for k,v in p.state_dict().items()};ns=n.state()
    result,_=run_base(tasks[0],'ppo',1,CFG,hps[(tasks[0],'ppo')],p,n,output=tmp_path/'deploy')
    assert result['queries']==7 and result['train_steps']==256
    assert all(torch.equal(before[k],v) for k,v in p.state_dict().items());assert ns==n.state()
    restored=ALGORITHMS['ppo'].load(tmp_path/'deploy/base_model.zip',device='cpu')
    scores,_,_=evaluate_model(restored,tasks[0],[1001000]);assert scores==result['scores']

def test_exact_statistics():
    d=np.array([32.8,31.9,27.4,22.4,28]);assert exact_sign_flip(d)==.0625
    p=paired([468.4,472.8,478.9,480.1,474.8],[501.2,504.7,506.3,502.5,502.8])
    assert p['p_t']==pytest.approx(.00010440675);assert p['mean_difference']==28.5
    assert paired([1,1],[1,1])['p_exact']==1

def test_plan_covers_all_seeds_and_transfer():
    cfg=json.loads(Path('configs/review_pilot.json').read_text());j=jobs(cfg);assert len(set(j))==len(j)
    for seed in cfg['seeds']:
        for alg in ['a2c','dqn']:
            assert any(x[0]==seed and x[2]==alg and x[3]=='specific' for x in j)
    assert roles(cfg)['joint']['algorithms']==['ppo','a2c','dqn']
    assert 'bsuite/catch/0' not in roles(cfg)['ppo_pooled']['tasks']
