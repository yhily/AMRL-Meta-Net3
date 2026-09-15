"""Auditable CPU reference runner. New experiments, not reconstructed historical results."""
from pathlib import Path
from collections import deque
import time,json,hashlib,platform,sys,random
import numpy as np
import torch
from stable_baselines3 import PPO,A2C,DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from envs.make_env import make_env
from amrl.hyperparameter_controller import HPController
from amrl.meta_state import MetaState,Normalizer
from amrl.meta_learner import MetaLearner,matching_mlp_hidden
from amrl.meta_ppo import MetaPPO

ALGORITHMS={'ppo':PPO,'a2c':A2C,'dqn':DQN}

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)

def source_hash():
    root=Path(__file__).resolve().parents[1];h=hashlib.sha256()
    for p in sorted(root.rglob('*.py')):
        if any(x.startswith('.') for x in p.relative_to(root).parts) or 'results' in p.parts:continue
        h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()

def hardware():
    import stable_baselines3,gymnasium
    return dict(platform=platform.platform(),processor=platform.processor(),device='cpu',threads=torch.get_num_threads(),python=sys.version,
                torch=torch.__version__,stable_baselines3=stable_baselines3.__version__,gymnasium=gymnasium.__version__)

class Trace(BaseCallback):
    def __init__(self,window):
        super().__init__();self.diag=MetaState(window);self.ep_return=0.;self.episode_rows=[];self.gradient_rows=[]
    def _on_step(self):
        self.ep_return+=float(self.locals['rewards'][0])
        if self.locals['dones'][0]:
            self.diag.episode(self.num_timesteps,self.ep_return)
            self.episode_rows.append(dict(step=self.num_timesteps,return_value=self.ep_return));self.ep_return=0.
        return True

def make_model(algorithm,env,seed,hp,rollout):
    cls=ALGORITHMS[algorithm];kwargs=dict(learning_rate=hp[0],gamma=hp[1],seed=seed,device='cpu',verbose=0,
        policy_kwargs=dict(net_arch=[64,64]))
    if algorithm in ['ppo','a2c']:
        kwargs.update(n_steps=rollout,ent_coef=hp[2])
        if algorithm=='ppo':kwargs.update(batch_size=rollout,n_epochs=4)
    else:
        kwargs.update(buffer_size=50000,learning_starts=rollout,batch_size=64,train_freq=4,gradient_steps=1,
                      target_update_interval=1000,exploration_initial_eps=hp[2],exploration_final_eps=hp[2])
    model=cls('MlpPolicy',env,**kwargs);model.set_logger(configure(None,[]));return model

def evaluate_model(model,env_name,seeds):
    env=make_env(env_name,int(seeds[0]));scores=[];steps=0
    try:
        for seed in seeds:
            obs,_=env.reset(seed=int(seed));ret=0.;done=False
            while not done:
                action,_=model.predict(obs,deterministic=True)
                obs,r,terminated,truncated,_=env.step(int(action));ret+=r;steps+=1;done=terminated or truncated
            scores.append(float(ret))
        info=env.bsuite_info() if hasattr(env,'bsuite_info') else None
    finally:env.close()
    return scores,steps,info

def run_base(env_name,algorithm,seed,cfg,hp,controller=None,normalizer=None,variant='full',training_meta=False,output=None):
    seed_all(seed);start=time.perf_counter();env=make_env(env_name,seed)
    model=make_model(algorithm,env,seed,hp,cfg['rollout']);trace=Trace(cfg['window'])
    ctrl=HPController(*hp,safeguard=variant);ctrl.apply(model)
    grads=[]
    def capture(opt,args,kwargs):
        # SB3 has already clipped gradients when this hook executes; explicitly recorded as post-clip.
        vals=[p.grad.detach().square().sum() for g in opt.param_groups for p in g['params'] if p.grad is not None]
        grads.append(float(torch.stack(vals).sum().sqrt()) if vals else 0.)
    handle=model.policy.optimizer.register_step_pre_hook(capture)
    interval=cfg['interval'];steps=cfg['steps']
    if steps%interval or interval%cfg['rollout'] or interval%4:
        raise ValueError('steps must divide into intervals; intervals must be divisible by rollout and DQN train_freq=4')
    states=[];actions=[];transitions=[];pending=None;h=None;control_seconds=0.;query_seconds=0.
    online_start=time.perf_counter()
    try:
        for stop in range(interval,steps+1,interval):
            grads.clear()
            model.learn(total_timesteps=interval,reset_num_timesteps=False,callback=trace)
            if model.num_timesteps!=stop:raise RuntimeError('learner exceeded exact interaction budget')
            tick=time.perf_counter()
            logs=model.logger.name_to_value
            pl=float(logs.get('train/policy_gradient_loss',logs.get('train/policy_loss',0.)))
            vl=float(logs.get('train/value_loss',logs.get('train/loss',0.)))
            # Gradients are optimizer-level observations; losses are last-update SB3 aggregates.
            for gn in grads:trace.diag.optimization(stop,pl,vl,gn)
            raw=trace.diag.get(stop);performance=float(raw[0])
            if pending is not None:
                pending['reward']=performance-pending.pop('performance');transitions.append(pending);pending=None
            if controller is not None and stop>=cfg['warmup'] and stop<steps:
                state=normalizer.transform(raw,update=training_meta)
                if variant=='reward_only':state[2:]=0
                if variant=='no_loss':state[2:4]=0
                if variant=='no_gradient':state[4:]=0
                query_start=time.perf_counter()
                action,z,logp,value,h=controller.act(state,h,deterministic=not training_meta)
                query_seconds+=time.perf_counter()-query_start
                if variant=='frozen':action=torch.zeros_like(action)
                ctrl.update(action.numpy());ctrl.apply(model)
                actions.append(dict(step=stop,raw_state=raw.tolist(),normalized_state=state.tolist(),action=action.tolist(),hyperparameters=list(ctrl.get())))
                if training_meta:pending=dict(state=state,z=z,logp=logp,value=value,performance=performance)
            states.append(dict(step=stop,mean_recent_episode_return=performance,policy_loss=pl,value_loss=vl,gradient_norm_mean=float(raw[4])))
            control_seconds+=time.perf_counter()-tick
        online_seconds=time.perf_counter()-online_start
        eval_seconds=0.;scores=[];eval_steps=0;info=None
        if not training_meta:
            tick=time.perf_counter();scores,eval_steps,info=evaluate_model(model,env_name,[seed*1000+1000000+i for i in range(cfg['eval_episodes'])]);eval_seconds=time.perf_counter()-tick
        result=dict(env=env_name,algorithm=algorithm,seed=seed,variant=variant,train_steps=model.num_timesteps,evaluation_steps=eval_steps,
            metric='mean_deterministic_episode_return',scores=scores,mean_return=float(np.mean(scores)) if scores else None,
            online_seconds=online_seconds,evaluation_seconds=eval_seconds,control_instrumentation_seconds=control_seconds,
            query_seconds=query_seconds,queries=len(actions),total_seconds=time.perf_counter()-start,initial_hyperparameters=list(hp),
            final_hyperparameters=list(ctrl.get()),episodes=trace.episode_rows,checkpoints=states,actions=actions,bsuite_info=info,
            diagnostic_semantics='episode returns; last SB3 loss aggregate; post-clip optimizer gradient norms',hardware=hardware())
        if output:
            output=Path(output);output.mkdir(parents=True,exist_ok=True);model.save(output/'base_model')
            result['base_checkpoint_sha256']=digest(output/'base_model.zip');save_json(output/'run.json',result)
        return result,transitions
    finally:handle.remove();env.close()

def train_controller(root,role,seed,cfg,tasks,algorithms,hps,architecture='lstm',variant='full',bound_scale=1.):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);start=time.perf_counter();seed_all(seed)
    hidden=64 if architecture=='lstm' else matching_mlp_hidden()
    policy=MetaLearner(architecture,hidden,bound_scale);norm=Normalizer();opt=MetaPPO(policy)
    before=torch.cat([p.detach().flatten() for p in policy.parameters()]).clone()
    episodes=[];records=[];updates=[];update_seconds=0.
    for i in range(cfg['meta_episodes']):
        # Balanced deterministic schedule; per-episode fresh base learner, never target data unless declared.
        env=tasks[i%len(tasks)];algorithm=algorithms[(i//len(tasks))%len(algorithms)]
        result,traj=run_base(env,algorithm,seed*10000+2000+i,cfg,hps[(env,algorithm)],policy,norm,variant,True)
        episodes.append(traj);result['subsequent_meta_rewards']=[x['reward'] for x in traj];records.append(result)
        if len(episodes)==cfg['meta_batch'] or i==cfg['meta_episodes']-1:
            tick=time.perf_counter();loss=opt.update(episodes);update_seconds+=time.perf_counter()-tick;episodes=[];updates.append(loss)
    after=torch.cat([p.detach().flatten() for p in policy.parameters()]);delta=float((before-after).norm())
    if delta==0:raise RuntimeError('Meta-controller parameters did not change')
    checkpoint=dict(policy=policy.state_dict(),config=policy.config,normalizer=norm.state(),training_tasks=tasks,training_algorithms=algorithms,
                    role=role,seed=seed,variant=variant,protocol=cfg,source_sha256=source_hash())
    torch.save(checkpoint,root/'controller.pt')
    record=dict(role=role,seed=seed,status='completed',elapsed_seconds=time.perf_counter()-start,
        base_rollout_and_training_seconds=sum(r['total_seconds'] for r in records),controller_optimization_seconds=update_seconds,
        environment_interactions=sum(r['train_steps'] for r in records),meta_episodes=len(records),updates=updates,
        parameter_change_l2=delta,parameter_count=sum(p.numel() for p in policy.parameters()),checkpoint_sha256=digest(root/'controller.pt'),
        training_tasks=tasks,training_algorithms=algorithms,variant=variant,episodes=records,hardware=hardware(),source_sha256=checkpoint['source_sha256'])
    save_json(root/'cost.json',record);return policy,norm,record

def load_controller(path):
    data=torch.load(path,map_location='cpu',weights_only=False)
    p=MetaLearner(**data['config']);p.load_state_dict(data['policy']);p.eval()
    return p,Normalizer(data['normalizer']),data
