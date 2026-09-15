"""Run/resume declared experiments; immutable protocol, atomic per-run provenance."""
import argparse,itertools,json,time,math
from pathlib import Path
import numpy as np
import optuna
from experiments.engine import *

TASKS=['CartPole-v1','MountainCar-v0','bsuite/memory_len/0','bsuite/catch/0']
TRAIN=TASKS[:3]

def tune(root,env,alg,cfg):
    path=root/'tuning'/f'{env.replace("/","_")}_{alg}.json'
    if path.exists():return json.loads(path.read_text())
    start=time.perf_counter();trials=[]
    sampler=optuna.samplers.TPESampler(seed=cfg['search_seed'])
    study=optuna.create_study(direction='maximize',sampler=sampler)
    # Fixed, separate tuning seeds: evaluation seeds never influence selection.
    def objective(trial):
        hp=(trial.suggest_float('alpha',1e-5,1e-2,log=True),trial.suggest_float('gamma',.90,.999),trial.suggest_float('beta',.01,1.))
        values=[]
        for seed in cfg['tuning_seeds']:
            r,_=run_base(env,alg,seed,cfg,hp)
            trials.append(dict(trial=trial.number,seed=seed,params=list(hp),mean_return=r['mean_return'],
                train_steps=r['train_steps'],evaluation_steps=r['evaluation_steps'],elapsed_seconds=r['total_seconds']))
            values.append(r['mean_return'])
        return float(np.mean(values))
    study.optimize(objective,n_trials=cfg['tpe_trials'])
    best=study.best_params
    rec=dict(env=env,algorithm=alg,hyperparameters=[best['alpha'],best['gamma'],best['beta']],trials=trials,
        elapsed_seconds=time.perf_counter()-start,train_interactions=sum(x['train_steps'] for x in trials),
        evaluation_interactions=sum(x['evaluation_steps'] for x in trials),search_seed=cfg['search_seed'],
        protocol='fixed-budget TPE; no pruning (explicit departure from historical median-pruner description)')
    save_json(path,rec);return rec

def roles(cfg):
    base=dict(tasks=TRAIN,algorithms=['ppo'],architecture='lstm',variant='full',bound_scale=1.,cfg=cfg)
    r={'ppo_pooled':dict(base),'a2c_specific':dict(base,algorithms=['a2c']),'dqn_specific':dict(base,algorithms=['dqn']),
       'joint':dict(base,algorithms=['ppo','a2c','dqn'])}
    for v in ['reward_only','no_loss','no_gradient','no_drift']:
        r[v]=dict(base,variant=v)
    r['mlp']=dict(base,architecture='mlp')
    for label,scale in [('bounds_half',.5),('bounds_double',2.)]:r[label]=dict(base,bound_scale=scale)
    for label,mult in [('interval_half',.5),('interval_double',2)]:r[label]=dict(base,cfg=dict(cfg,interval=int(cfg['interval']*mult)))
    for i,env in enumerate(TASKS):r[f'task_{i}']=dict(base,tasks=[env])
    return r

def jobs(cfg):
    output=[]
    for seed,env,alg in itertools.product(cfg['seeds'],TASKS,['ppo','a2c','dqn']):
        for condition,role in [('static',None),('amrl','ppo_pooled')]:output.append((seed,env,alg,condition,role))
        if alg!='ppo':
            for condition,role in [('specific',alg+'_specific'),('joint','joint')]:output.append((seed,env,alg,condition,role))
    for seed,env in itertools.product(cfg['seeds'],[TASKS[0],TASKS[2]]):
        for name in ['frozen','mlp','reward_only','no_loss','no_gradient','no_drift','bounds_half','bounds_double','interval_half','interval_double']:
            output.append((seed,env,'ppo',name,'ppo_pooled' if name=='frozen' else name))
    for seed in cfg['seeds']:
        for source,target in [(0,1),(0,2),(1,3)]:
            output.append((seed,TASKS[target],'ppo',f'transfer_from_{source}',f'task_{source}'))
            # Same target-specific reference can be reused, not re-run or counted twice.
            job=(seed,TASKS[target],'ppo','target_trained',f'task_{target}')
            if job not in output:output.append(job)
    return output

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--plan-only',action='store_true');args=ap.parse_args();cfg=json.loads(Path(args.config).read_text())
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True);fingerprint=source_hash()
    for key in ['seeds','steps','interval','warmup','rollout','window','meta_episodes','meta_batch','eval_episodes','tpe_trials','tuning_seeds','search_seed','purpose']:
        if key not in cfg:raise ValueError('Missing protocol key: '+key)
    if len(set(cfg['seeds']))!=len(cfg['seeds']):raise ValueError('Duplicate seeds')
    if cfg['meta_episodes']%9:raise ValueError('Meta episode count must be a multiple of 9 for balanced joint controllers')
    if cfg['warmup']>=cfg['steps']-2*cfg['interval']:raise ValueError('Need an intervention even at doubled interval')
    manifest=dict(config=cfg,source_sha256=fingerprint,expected_jobs=[dict(seed=s,env=e,algorithm=a,condition=c,role=r) for s,e,a,c,r in jobs(cfg)],status='planned')
    existing=root/'manifest.json'
    if existing.exists():
        prev=json.loads(existing.read_text())
        if prev['config']!=cfg or prev['source_sha256']!=fingerprint:raise ValueError('Protocol/source changed: choose a new output directory')
    else:save_json(existing,manifest)
    if args.plan_only:print(f"Planned {len(manifest['expected_jobs'])} seed-level evaluations; no training performed.");return
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    hps={};tuning={}
    for env,alg in itertools.product(TASKS,['ppo','a2c','dqn']):
        print('TPE',env,alg,flush=True);r=tune(root,env,alg,cfg);hps[(env,alg)]=r['hyperparameters'];tuning[(env,alg)]=r
    specs=roles(cfg)
    for seed,env,alg,condition,role in jobs(cfg):
        folder=root/'runs'/f'{env.replace("/","_")}_{alg}_{condition}_{seed}'
        if (folder/'result.json').exists():continue
        policy=norm=None;meta_cost=None;run_cfg=cfg;variant='full';checkpoint=None
        if role:
            spec=specs[role];controller_root=root/'controllers'/f'{role}_{seed}';checkpoint=controller_root/'controller.pt'
            if not (controller_root/'cost.json').exists():
                print('META',role,seed,flush=True)
                policy,norm,meta_cost=train_controller(controller_root,role,seed,**spec,hps=hps)
            else:
                policy,norm,_=load_controller(checkpoint);meta_cost=json.loads((controller_root/'cost.json').read_text())
            variant='frozen' if condition=='frozen' else spec['variant'];run_cfg=spec['cfg']
        print('RUN',env,alg,condition,seed,flush=True)
        begin=time.perf_counter()
        initialization_pairs={(env,alg)}
        if role: initialization_pairs.update(itertools.product(spec['tasks'],spec['algorithms']))
        first_use_tpe_seconds=sum(tuning[pair]['elapsed_seconds'] for pair in initialization_pairs)
        try:
            r,_=run_base(env,alg,seed,run_cfg,hps[(env,alg)],policy,norm,variant,False,folder)
            r.update(condition=condition,controller_role=role,status='completed',purpose=cfg['purpose'],source_sha256=fingerprint,
                controller_sha256=digest(checkpoint) if checkpoint else None,controller_path=str(checkpoint.relative_to(root)) if checkpoint else None,
                meta_seconds=meta_cost['elapsed_seconds'] if meta_cost else 0.,meta_interactions=meta_cost['environment_interactions'] if meta_cost else 0,
                tpe_seconds=tuning[(env,alg)]['elapsed_seconds'],tpe_train_interactions=tuning[(env,alg)]['train_interactions'],
                first_use_all_initialization_seconds=first_use_tpe_seconds,
                first_use_training_seconds=first_use_tpe_seconds+(meta_cost['elapsed_seconds'] if meta_cost else 0)+r['online_seconds'],
                job_elapsed_seconds=time.perf_counter()-begin,protocol=run_cfg)
            save_json(folder/'result.json',r)
        except Exception as exc:
            save_json(folder/'failure.json',dict(status='failed',error=repr(exc),seed=seed,env=env,algorithm=alg,condition=condition));raise
    manifest['status']='completed';save_json(existing,manifest)
    from experiments.summarize import summarize
    summarize(root)

if __name__=='__main__':main()
