"""Rebuild all publication figures from the accompanying immutable source data.
Usage: python scripts/plot_figures.py
Python / matplotlib only; no fitted or synthetic experimental observations.
"""
from pathlib import Path
import os,json,csv,string
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/amrl-paper-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
D=json.loads((ROOT/'source_data/plot_records.json').read_text());F=ROOT/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'lines.linewidth':1.2,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False,'savefig.facecolor':'white'})
BLUE='#287D9A';ORANGE='#BB7945';GREY='#8B949A';INK='#223440';PURPLE='#7866A1'
TASKS=['CartPole-v1','MountainCar-v0','bsuite/memory_len/0','bsuite/catch/0'];NAMES=['CartPole (1000-step cap)','MountainCar','Memory /0','Catch /0']
def records(e,a,c):return sorted([r for r in D if (r['env'],r['algorithm'],r['condition'])==(e,a,c)],key=lambda r:r['seed'])
def values(e,a,c):return np.array([r['mean_return'] for r in records(e,a,c)])
def label(ax,i,title):ax.set_title(title,loc='left',pad=10);ax.text(-.12,1.06,string.ascii_lowercase[i],transform=ax.transAxes,fontweight='bold',fontsize=11)
def save(fig,name):
 fig.savefig(F/(name+'.pdf'));fig.savefig(F/(name+'.svg'));fig.savefig(F/(name+'.png'),dpi=220);fig.savefig(F/(name+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
 plt.close(fig)
def forest(ax,ds,labels,offset=0):
 for i,d in enumerate(ds):
  i+=offset
  mean=d.mean();half=stats.t.ppf(.975,4)*d.std(ddof=1)/np.sqrt(5)
  ax.scatter(d,i+np.linspace(-.13,.13,5),s=16,color=GREY,alpha=.8,zorder=3)
  ax.errorbar(mean,i,xerr=half,fmt='D',color=BLUE,ms=4,capsize=3,lw=1.5,zorder=5)
 ax.axvline(0,color=INK,lw=.7,ls='--');ax.set_yticks(range(offset,offset+len(labels)),labels);ax.invert_yaxis();ax.set_xlabel('Paired return difference');ax.grid(axis='x',color='#E5E8EA',lw=.5);ax.set_axisbelow(True)
# Main: complete 12-comparison family, task-specific scales, no cross-task normalization.
fig,axs=plt.subplots(2,2,figsize=(7.1,4.65));fig.subplots_adjust(left=.10,right=.975,bottom=.12,top=.91,wspace=.32,hspace=.62)
for i,(ax,e) in enumerate(zip(axs.flat,TASKS)):
 forest(ax,[values(e,a,'amrl')-values(e,a,'static') for a in ['ppo','a2c','dqn']],['PPO','A2C','DQN']);label(ax,i,NAMES[i])
 if i==2:ax.set_xlim(-.1,.1);ax.set_xticks([-.1,0,.1])
 if i==3:ax.set_xlim(-.12,.65)
fig.legend(handles=[Line2D([],[],marker='o',color=GREY,lw=0,label='Individual training seed'),Line2D([],[],marker='D',color=BLUE,label='Mean and 95% paired t interval')],loc='upper center',bbox_to_anchor=(.5,1.005),ncol=2,fontsize=8)
fig.text(.5,.015,'AMRL minus Static(TPE); 5 paired seeds per comparison; raw task-specific return units.',ha='center',fontsize=7)
save(fig,'fig3_main_paired')
# Ablation: show absolute outcomes as well as the paired uncertainty.
conds=['amrl','frozen','mlp','reward_only','bounds_half','bounds_double','interval_half','interval_double']
labels=['AMRL (reference)','No actuation','Matched MLP','Reward only','Half action bounds','Double action bounds','Half interval','Double interval']
fig,axs=plt.subplots(1,2,figsize=(7.1,4.15));fig.subplots_adjust(left=.18,right=.975,bottom=.14,top=.87,wspace=.47)
for i,c in enumerate(conds):
 v=values(TASKS[0],'ppo',c);axs[0].scatter(v,i+np.linspace(-.12,.12,5),s=18,color=BLUE if i==0 else GREY,zorder=3);axs[0].plot([v.mean()]*2,[i-.22,i+.22],color=INK,lw=2)
axs[0].set_yticks(range(8),labels);axs[0].invert_yaxis();axs[0].set_xlim(0,1050);axs[0].set_xlabel('Final evaluation return');axs[0].grid(axis='x',color='#E5E8EA',lw=.5)
forest(axs[1],[values(TASKS[0],'ppo',c)-values(TASKS[0],'ppo','amrl') for c in conds[1:]],['']*7,offset=1)
axs[0].set_ylim(7.6,-.6);axs[1].set_ylim(7.6,-.6)
label(axs[0],0,'Absolute outcomes');label(axs[1],1,'Variant minus AMRL')
fig.text(.5,.97,'CartPole–PPO: component and local sensitivity checks',ha='center',fontsize=10,color=INK)
fig.text(.5,.025,'5 seeds; points show individual seeds. Left: mean ticks. Right: mean and 95% paired t intervals.',ha='center',fontsize=7)
save(fig,'fig4_ablation')
# Task transfer: all directions retained; tied observations deliberately offset horizontally only.
fig,axs=plt.subplots(1,3,figsize=(7.1,2.75));fig.subplots_adjust(left=.075,right=.975,bottom=.25,top=.78,wspace=.45)
for i,(ax,e,source,title) in enumerate(zip(axs,TASKS[1:],['transfer_from_0','transfer_from_0','transfer_from_1'],['CartPole → MountainCar','CartPole → Memory','MountainCar → Catch'])):
 arr=np.array([values(e,'ppo',c) for c in ['static',source,'target_trained']]);offset=np.linspace(-.09,.09,5)
 for s in range(5):ax.plot(np.arange(3)+offset[s],arr[:,s],color=GREY,alpha=.7,marker='o',ms=3,lw=.7)
 ax.plot(range(3),arr.mean(axis=1),color=BLUE,marker='D',ms=4,lw=1.5)
 ax.set_xticks(range(3),['Static','Source','Target']);ax.set_ylabel('Raw episode return');label(ax,i,title)
 if i==0:ax.set_ylim(-205,-195);ax.set_yticks([-200]);ax.text(.5,.08,'All seed means = −200',transform=ax.transAxes,ha='center',fontsize=7)
 else:ax.set_ylim(.4,1.08);ax.set_yticks([.5,.75,1.])
 if i==1:ax.text(.5,.08,'All seed means = 1',transform=ax.transAxes,ha='center',fontsize=7)
fig.text(.5,.04,'PPO, 5 paired seeds; grey lines: seeds; blue diamonds: means. Source and target controllers trained separately.',ha='center',fontsize=7)
save(fig,'fig5_transfer')
# Cost: distinct physical units and explicit evaluation stage; complete retained experiment, not per-run duplicates.
C=json.loads((ROOT/'source_data/cost_summary_deduplicated.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(7.1,2.75));fig.subplots_adjust(left=.15,right=.97,bottom=.18,top=.84,wspace=.60)
for ax,vals,labs,title,unit in [(axs[0],[180,252,26.5],['TPE search','Meta-training','Deployment'],'Training interactions','Million environment steps'),(axs[1],[C[k]/3600 for k in ['tpe_seconds','meta_seconds','online_seconds','evaluation_seconds']],['TPE search*','Meta-training','Deployment','Final evaluation'],'Measured stage time','Sum of stage timers (hours)')]:
 colors=[GREY,BLUE,ORANGE,PURPLE][:len(vals)];ax.barh(range(len(vals)),vals,color=colors,height=.56);ax.set_yticks(range(len(vals)),labs);ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.3);ax.set_xlabel(unit)
 for j,v in enumerate(vals):ax.text(v+max(vals)*.025,j,f'{v:.2f}' if unit.endswith('(hours)') else f'{v:g}',va='center',fontsize=8)
 label(ax,0 if ax==axs[0] else 1,title)
fig.text(.5,.025,'70 unique controllers, 12 TPE studies and 265 deployments. *TPE time includes candidate evaluation.',ha='center',fontsize=7)
save(fig,'fig6_cost')
# Supplement: complete cross-algorithm references.
fig,axs=plt.subplots(2,2,figsize=(7.1,4.7));fig.subplots_adjust(left=.11,right=.98,bottom=.10,top=.88,wspace=.3,hspace=.62)
for i,(ax,e) in enumerate(zip(axs.flat,TASKS)):
 for j,(c,col) in enumerate([('amrl',BLUE),('specific',ORANGE),('joint',PURPLE)]):
  for k,a in enumerate(['a2c','dqn']):
   v=values(e,a,c);x=k+(j-1)*.22;ax.scatter(x+np.linspace(-.04,.04,5),v,s=13,color=col,alpha=.8);ax.plot([x-.065,x+.065],[v.mean()]*2,lw=2,color=INK)
 ax.set_xticks([0,1],['A2C','DQN']);ax.set_ylabel('Raw episode return');label(ax,i,NAMES[i])
 if i>=2:ax.set_ylim(.9,1.03);ax.set_yticks([1]);ax.text(.5,.1,'All means = 1',transform=ax.transAxes,ha='center',fontsize=7)
fig.legend(handles=[Line2D([],[],marker='o',color=c,lw=0,label=l) for l,c in [('PPO-trained',BLUE),('Algorithm-specific',ORANGE),('Jointly trained',PURPLE)]],loc='upper center',bbox_to_anchor=(.5,1),ncol=3,fontsize=8)
save(fig,'figS1_cross_algorithm')
# Supplement: all five seeds; provenance-anchored boundedness, not claimed causal mechanism.
fig,axs=plt.subplots(1,3,figsize=(7.1,2.75));fig.subplots_adjust(left=.08,right=.975,bottom=.22,top=.79,wspace=.42)
colors=[BLUE,ORANGE,PURPLE,'#4E8771',GREY]
for r,col in zip(records(TASKS[0],'ppo','amrl'),colors):
 t=[0]+[x['step']/1000 for x in r['actions']]+[100];hp=np.array([r['initial_hyperparameters']]+[x['hyperparameters'] for x in r['actions']]+[r['actions'][-1]['hyperparameters']])
 for j,ax in enumerate(axs):ax.step(t,hp[:,j]/r['initial_hyperparameters'][0] if j==0 else hp[:,j],where='post',color=col,lw=1,label=f'Seed {r["seed"]}')
for i,ax in enumerate(axs):
 ax.axvspan(0,20,color='#EFF1F2',zorder=0);ax.set_xlabel('Training steps (thousands)');label(ax,i,['Learning-rate ratio','Discount factor','Entropy coefficient'][i])
axs[0].axhline(1,color=INK,lw=.5,ls=':');axs[1].set_ylim(.897,1.003);axs[1].axhline(.999,color=INK,lw=.6,ls=':');axs[1].axhline(.9,color=INK,lw=.6,ls=':')
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1),ncol=5,fontsize=7)
save(fig,'figS2_trajectories')
print('Exported 6 figures: PDF, SVG, PNG, 600 dpi TIFF')
