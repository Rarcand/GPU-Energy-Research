#!/usr/bin/env python3
"""Summarize saved comparison runs and produce paper tables and scientific plots."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare_surrogates import summarize, save_json


def paired_interval(a, b, metric, repetitions=2000):
    """Paired hierarchical bootstrap of seed and shape; stratified by family.

    Dtypes stay within a shape. A positive difference means method a has a
    higher metric. Intervals condition on the fixed split and measured devices.
    """
    def organize(runs):
        values = defaultdict(list)
        for run in runs:
            for task in run['tasks']:
                if task[metric] is not None:
                    values[(run['seed'],task['family'],task['group'])].append(task[metric])
        return {k:np.mean(v) for k,v in values.items()}
    av,bv = organize(a),organize(b)
    keys = sorted(set(av)&set(bv))
    seeds = sorted(set(k[0] for k in keys))
    families = sorted(set(k[1] for k in keys))
    differences = []
    weights = []
    for family in families:
        groups = sorted(set(k[2] for k in keys if k[1]==family))
        differences.append(np.array([[av[(s,family,g)]-bv[(s,family,g)]
                                      for g in groups] for s in seeds]))
        # Preserve the reported within-family task weighting while resampling
        # whole shapes. A shape can lack a valid dtype-specific task.
        first = a[0]
        weights.append(np.array([sum(t['group']==g for t in first['tasks']) for g in groups]))
    estimate = float(np.mean([np.average(d.mean(0),weights=w) for d,w in zip(differences,weights)]))
    rng = np.random.default_rng(46113)
    draws = []
    for _ in range(repetitions):
        si = rng.integers(0,len(seeds),len(seeds))
        family_draws=[]
        for d,w in zip(differences,weights):
            gi=rng.integers(0,d.shape[1],d.shape[1])
            family_draws.append(np.average(d[np.ix_(si,gi)].mean(0),weights=w[gi]))
        draws.append(np.mean(family_draws))
    return {'difference':estimate,'ci95':np.quantile(draws,[.025,.975]).tolist(),
            'bootstrap_repetitions':repetitions,'shape_groups':sum(d.shape[1] for d in differences),
            'seeds':len(seeds)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    runs=[json.loads(p.read_text()) for p in sorted((args.directory/'runs').glob('*.json'))]
    summary=summarize(runs)
    for record in summary:
        if record['method']=='random_search':
            for metric in ('mape_pct','mae_us','spearman'):
                record['metrics'].pop(metric,None)
    save_json(args.directory/'summary.json',summary)
    a100=[r for r in runs if r['suite']=='a100_seven']
    comparisons={}
    for baseline in ('flat','separate_families','boosted_tree','random_search'):
        comparisons[baseline]={metric:paired_interval(
            [r for r in a100 if r['method']=='three'],
            [r for r in a100 if r['method']==baseline],metric)
            for metric in (('success5_b1','success5_b4') if baseline=='random_search'
                           else ('success5_b1','success5_b4','mape_pct'))}
    save_json(args.directory/'paired_intervals.json',comparisons)
    labels={'three':'Three encoders','flat':'Flat MLP','separate_families':'Separate families',
            'boosted_tree':'Boosted tree','random_search':'Random search',
            'transfer':'Transfer','target_only':'Target only','zero_shot':'Zero shot'}
    table=[]
    for r in summary:
        if r['suite']!='a100_seven': continue
        m=r['metrics']
        if r['method']=='random_search':
            table.append('Random search & --- & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f \\' % (
                100*m['success5_b1']['mean'],100*m['success5_b1']['seed_sd'],
                100*m['success5_b4']['mean'],100*m['success5_b4']['seed_sd']))
            continue
        table.append('%s & %.2f $\\pm$ %.2f & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f \\' % (
            labels[r['method']],m['mape_pct']['mean'],m['mape_pct']['seed_sd'],
            100*m['success5_b1']['mean'],100*m['success5_b1']['seed_sd'],
            100*m['success5_b4']['mean'],100*m['success5_b4']['seed_sd']))
    (args.directory/'a100_table.tex').write_text(
        r'\begin{tabular}{lrrr}\toprule'+'\n'+
        r'Method & MAPE (\%) & $S_1$ (\%) & $S_4$ (\%)\\\midrule'+'\n'+
        '\n'.join(line.rstrip(chr(92))+chr(92)*2 for line in table)+'\n'+
        r'\bottomrule\end{tabular}'+'\n')
    table=[]
    for r in summary:
        if r['suite']!='pilot_transfer': continue
        m=r['metrics']
        table.append('%s & %s & %d & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f \\' % (
            r['target'],labels[r['method']],r['budget'],m['mape_pct']['mean'],m['mape_pct']['seed_sd'],
            100*m['success5_b1']['mean'],100*m['success5_b1']['seed_sd']))
    (args.directory/'transfer_table.tex').write_text(
        r'\begin{tabular}{llrrr}\toprule'+'\n'+
        r'Target & Method & $M$ & MAPE (\%) & $S_1$ (\%)\\\midrule'+'\n'+
        '\n'.join(line.rstrip(chr(92))+chr(92)*2 for line in table)+'\n'+
        r'\bottomrule\end{tabular}'+'\n')
    by_family=[]
    for method in ('three','flat','separate_families','boosted_tree'):
        for family in sorted(set(t['family'] for r in a100 for t in r['tasks'])):
            selected=[{'tasks':[t for t in r['tasks'] if t['family']==family],
                       **{k:v for k,v in r.items() if k!='tasks'}}
                      for r in a100 if r['method']==method]
            by_family.append({'method':method,'family':family,'result':summarize(selected)[0]})
    save_json(args.directory/'family_summary.json',by_family)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(11,3.2),layout='constrained')
    for r in summary:
        if r['suite']!='a100_seven':continue
        values=[100*r['metrics']['success5_b'+str(b)]['mean'] for b in (1,2,4,8)]
        axes[0].plot((1,2,4,8),values,marker='o',label=labels[r['method']])
    axes[0].set(title='Seven-family A100 selection',xlabel='Candidate measurements B',ylabel='Within 5% of optimum (%)')
    axes[0].legend(fontsize=7)
    for ax,target in zip(axes[1:],('H100','A100')):
        for method in ('transfer','target_only'):
            records=sorted([r for r in summary if r['suite']=='pilot_transfer' and
                            r['target']==target and r['method']==method],key=lambda r:r['budget'])
            ax.errorbar([r['budget'] for r in records],
                [r['metrics']['mape_pct']['mean'] for r in records],
                yerr=[r['metrics']['mape_pct']['seed_sd'] for r in records],marker='o',capsize=3,label=labels[method])
        ax.set(title='Pilot transfer to '+target,xlabel='Target calibration labels M',ylabel='Macro-family MAPE (%)')
        ax.legend(fontsize=8)
    fig.savefig(args.directory/'comparison_results.pdf')
    fig.savefig(args.directory/'comparison_results.png',dpi=180)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
