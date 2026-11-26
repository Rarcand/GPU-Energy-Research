"""Paired workload-family summaries for the four-evaluation feedback study."""
from collections import defaultdict
import gzip
import json

import matplotlib.pyplot as plt
import numpy as np

MODES = ('graph', 'individual')
DEVICES = ('H100','RTX3070-WSL')

def summarize_feedback(root):
    records = defaultdict(dict)
    counts = {}
    for mode in MODES:
        for precision in ('fp16','bf16'):
            with gzip.open(root/'workshop-belief'/f'{mode}-{precision}-predictions.jsonl.gz','rt') as stream:
                for r in map(json.loads,stream):
                    t = np.asarray(r['measured_us'])
                    ids = [r['models'][name]['probe_order'][:4] for name in ('simple_world_gaussian','simple_world_no_feedback')]
                    key = r['family'],r['split'],r['shape'],r['dtype'],r['device']
                    records[key][mode] = np.min(t[ids],axis=1) <= 1.05*t.min()
                    if counts.setdefault(r['family'],len(t)) != len(t):
                        raise ValueError('Inconsistent family grid size')
    families,macro = paired_summary(records)
    return dict(families=families,overall=macro,candidates=counts,
                repeated_device_subset=paired_summary({k:v for k,v in records.items() if k[-1] in DEVICES})[1])

def paired_summary(records, *, compare_ridge=False):
    shapes = defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for (family,split,shape,dtype,_),cell in sorted(records.items()):
        shapes[family,split][shape][dtype].append([cell[m] for m in MODES])
    rng = np.random.default_rng(51163)
    means, draws = defaultdict(list),defaultdict(list)
    for (family,_),group in sorted(shapes.items()):
        block = np.array([np.mean([np.mean(dev,axis=0) for dev in prec.values()],axis=0) for prec in group.values()])
        n=len(block)
        weights=rng.multinomial(n,np.full(n,1/n),size=10000)/n
        means[family].append(block.mean(axis=0))
        draws[family].append(np.tensordot(weights,block,axes=1))
    def summarize(v,b):
        effects=v[:,0]-v[:,1]
        paired=b[:,:,0]-b[:,:,1]
        result = dict(values=(v*100).tolist(),gain=(100*effects).tolist(),
                    ci=(100*np.quantile(paired,[.025,.975],axis=0).T).tolist(),
                    interaction=(100*np.r_[effects[0]-effects[1],np.quantile(paired[:,0]-paired[:,1],[.025,.975])]).tolist())
        if compare_ridge:
            ridge_draws = b[:, :, 0] - b[:, :, 2]
            result['gaussian_minus_geometry_ridge_pp'] = (100 * (v[:, 0] - v[:, 2])).tolist()
            result['gaussian_minus_geometry_ridge_ci95_pp'] = (
                100 * np.quantile(ridge_draws, [.025, .975], axis=0).T).tolist()
        return result
    mean={k:np.mean(v,axis=0) for k,v in means.items()}
    boot={k:np.mean(v,axis=0) for k,v in draws.items()}
    return ({k:summarize(mean[k],boot[k]) for k in mean},summarize(np.mean(list(mean.values()),axis=0),np.mean(list(boot.values()),axis=0)))

def independent_summary(root, filename='independent-winners.jsonl.gz', devices=DEVICES, expected_cases=1592, pass_count=3):
    records=defaultdict(dict)
    rotations=defaultdict(dict)
    with gzip.open(root/filename,'rt') as stream:
        for r in map(json.loads,stream):
            key=tuple(r[k] for k in ('family','split','shape','dtype','device','mode'))
            regret=np.array(r['regret'])
            rotations[key][r['select'],r['score']]=np.r_[regret<=.05+1e-12,regret]
    for (*key,mode),cell in rotations.items():
        if set(cell)!={(p,q) for p in range(pass_count) for q in range(pass_count) if p!=q}:
            raise ValueError('Missing pass pairing')
        records[tuple(key)][mode]=np.mean(list(cell.values()),axis=0)
    if len(records)!=expected_cases or any(len(cell)!=2 for cell in records.values()):
        raise ValueError('Incomplete held-out panel')
    families,overall=paired_summary(records, compare_ridge=True)
    return dict(cases_per_mode=len(records),overall=overall,families=families,
                devices={d:paired_summary({k:v for k,v in records.items() if k[-1]==d},
                                          compare_ridge=True)[1] for d in devices})

def feedback_figure(report, labels, destination):
    fig,axs=plt.subplots(1,4,figsize=(7.15,2.65),sharey=True,
                         gridspec_kw={'width_ratios':[1.35,.95,1.35,.95]})
    y=np.arange(len(labels))
    for i,title in enumerate(('(a) Graph replay','(b) Individual launch')):
        ax,delta=axs[2*i:2*i+2]
        data=[report['families'][f] for f in labels]
        values=np.array([r['values'][i] for r in data])
        for row,(on,off) in enumerate(values):
            ax.plot([on,off],[row,row],color='.65',lw=1)
        ax.scatter(values[:,1],y,marker='s',facecolor='white',edgecolor='#b75b00',s=27,label='Feedback disabled',zorder=3)
        ax.scatter(values[:,0],y,color='#1f77b4',s=14,label='Measured feedback',zorder=4)
        gain=np.array([r['gain'][i] for r in data])
        ci=np.array([r['ci'][i] for r in data])
        delta.hlines(y,ci[:,0],ci[:,1],color='#334960',lw=1.4)
        delta.scatter(gain,y,s=11,color='#334960',zorder=3)
        delta.axvline(0,color='.55',lw=.7)
        ax.set(xlim=(-3,103),xticks=[0,50,100],xlabel='Success (%)',title=title)
        delta.set(xlim=(-20,55),xticks=[0,40],xlabel='Gain (pp)',title='Paired 95% CI')
        for axis in (ax,delta):
            axis.grid(axis='y',color='.91',lw=.55)
            axis.tick_params(axis='y',length=0)
    axs[0].set_yticks(y,[f'{label} ({report["candidates"][k]})' for k,label in labels.items()])
    axs[0].invert_yaxis()
    handles,names=axs[0].get_legend_handles_labels()
    fig.legend(handles,names,ncol=2,loc='upper center',frameon=False)
    fig.subplots_adjust(left=.20,right=.99,bottom=.18,top=.74,wspace=.28)
    fig.savefig(destination)
    plt.close(fig)
