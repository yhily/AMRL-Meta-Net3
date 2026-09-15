"""Seed is the replicate, evaluation episodes are not independent training replicates."""
from pathlib import Path
import json,csv,itertools,math,argparse
from collections import defaultdict
import numpy as np
from scipy import stats
from experiments.engine import save_json

def exact_sign_flip(d):
    d=np.asarray(d)
    if len(d)>20:raise ValueError('Exact enumeration limited to 20 pairs; do not silently use Monte Carlo')
    threshold=abs(d.mean())-1e-12
    return sum(abs(np.mean(d*np.array(signs)))>=threshold for signs in itertools.product([-1,1],repeat=len(d)))/2**len(d)

def paired(a,b):
    d=np.array(b)-np.array(a);n=len(d)
    if n<2:return dict(n=n,mean_difference=float(d.mean()),ci95=None,p_t=None,p_exact=None)
    sd=d.std(ddof=1);se=sd/math.sqrt(n)
    t_p=float(stats.ttest_rel(b,a).pvalue) if sd>0 else (1. if np.all(d==0) else None)
    ci=[float(d.mean()-stats.t.ppf(.975,n-1)*se),float(d.mean()+stats.t.ppf(.975,n-1)*se)]
    return dict(n=n,mean_difference=float(d.mean()),ci95=ci,p_t=t_p,p_exact=exact_sign_flip(d),dz=float(d.mean()/sd) if sd>0 else None)

def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());groups=defaultdict(dict);rows=[];completed=set()
    for p in sorted((root/'runs').glob('*/result.json')):
        r=json.loads(p.read_text());key=(r['env'],r['algorithm'],r['condition'])
        if r['seed'] in groups[key]:raise ValueError('Duplicate seed in comparison')
        if r['source_sha256']!=manifest['source_sha256']:raise ValueError('Mixed source versions')
        groups[key][r['seed']]=r;completed.add((r['seed'],*key))
        rows.append({k:r[k] for k in ['env','algorithm','condition','seed','mean_return','train_steps','evaluation_steps','online_seconds','evaluation_seconds',
            'meta_seconds','meta_interactions','tpe_seconds','first_use_training_seconds','controller_sha256','source_sha256','purpose']})
    expected={(r['seed'],r['env'],r['algorithm'],r['condition']) for r in manifest['expected_jobs']}
    missing=sorted(expected-completed)
    write_csv(root/'seed_results.csv',rows)
    summary=[];contrasts=[]
    for (env,alg,condition),values in sorted(groups.items()):
        scores=[r['mean_return'] for r in values.values()];n=len(scores)
        summary.append(dict(env=env,algorithm=alg,condition=condition,n=n,mean=float(np.mean(scores)),sample_sd=float(np.std(scores,ddof=1)) if n>1 else None))
        reference='static' if condition=='amrl' or condition.startswith('transfer_') or condition=='target_trained' else 'amrl'
        if condition=='static':continue
        baseline=groups.get((env,alg,reference),{})
        seeds=sorted(set(values)&set(baseline))
        # No complete-case deletion: incomplete groups produce no inferential result.
        if set(seeds)!=set(manifest['config']['seeds']):continue
        test=paired([baseline[s]['mean_return'] for s in seeds],[values[s]['mean_return'] for s in seeds])
        contrasts.append(dict(env=env,algorithm=alg,condition=condition,reference=reference,**test))
    # Holm correction for the twelve primary AMRL-vs-static comparisons, both test families separately.
    for pkey in ['p_t','p_exact']:
        primary=[x for x in contrasts if x['condition']=='amrl' and x[pkey] is not None]
        if len(primary)==12:
            ordered=sorted(primary,key=lambda x:x[pkey]);running=0.
            for rank,x in enumerate(ordered):running=max(running,min(1.,(12-rank)*x[pkey]));x[pkey+'_holm_12']=running
    write_csv(root/'descriptive_summary.csv',summary);save_json(root/'paired_statistics.json',contrasts)
    # Unique measured stages, not per-run duplicated/amortized charges.
    costs=[json.loads(p.read_text()) for p in (root/'controllers').glob('*/cost.json')]
    tuning=[json.loads(p.read_text()) for p in (root/'tuning').glob('*.json')]
    total=dict(unique_controller_count=len(costs),meta_seconds=sum(x['elapsed_seconds'] for x in costs),
        meta_interactions=sum(x['environment_interactions'] for x in costs),tpe_seconds=sum(x['elapsed_seconds'] for x in tuning),
        tpe_train_interactions=sum(x['train_interactions'] for x in tuning),tpe_evaluation_interactions=sum(x['evaluation_interactions'] for x in tuning),
        online_seconds=sum(x['online_seconds'] for x in rows),evaluation_seconds=sum(x['evaluation_seconds'] for x in rows),
        online_interactions=sum(x['train_steps'] for x in rows),evaluation_interactions=sum(x['evaluation_steps'] for x in rows))
    total['sum_measured_stage_seconds']=sum(total[k] for k in ['meta_seconds','tpe_seconds','online_seconds','evaluation_seconds'])
    save_json(root/'cost_summary.json',total)
    report=dict(purpose=manifest['config']['purpose'],expected=len(expected),completed=len(completed),missing=missing,
        all_planned_seed_rows_present=not missing,can_replace_historical_paper_results=False,
        reason='New implementation and declared task IDs/metric/budget; no historical logs provided.')
    save_json(root/'coverage.json',report)
    lines=['# Experiment report',f"Purpose: **{report['purpose']}**. Completed {len(completed)}/{len(expected)} planned seed rows.",
        'These are new measurements, not recovered historical paper data. bsuite results are raw episode returns for explicit IDs, not published tag-level normalized scores.',
        '\n|Environment|Algorithm|Condition|n seeds|Mean return|Sample SD|','|---|---|---|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['env']}|{r['algorithm']}|{r['condition']}|{r['n']}|{r['mean']:.6g}|{r['sample_sd']}|")
    lines+=['\nStatistics: paired_statistics.json; primary tests are Holm-adjusted only when all 12 comparisons exist. No episode-level pseudoreplication.',
        '\nMeasured costs: cost_summary.json. Controller costs include inner-learner rollouts, outer PPO optimization, and checkpoint writing; evaluation is separate.',
        '\nNo full bsuite sweep, population-based baseline, or historical overhead ratio is claimed. A failure/missing row is never imputed as a score.']
    (root/'REPORT.md').write_text('\n\n'.join(lines));print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');summarize(p.parse_args().output)
