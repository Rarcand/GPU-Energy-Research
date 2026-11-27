"""Replay four queries on one pass and score the selected winner on another."""
import argparse
import csv
import gzip
import io
import json
from tempfile import TemporaryDirectory
from analysis.surrogate.scripts.experiments.audit_final_choice import load_precision_inputs
from analysis.surrogate.scripts.models.compare_surrogate_architectures import prepare_families
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from analysis.surrogate.scripts.experiments.benchmark_candidate_response import conditional_order
from analysis.surrogate.scripts.experiments.benchmark_simple_belief import probe_order, SCALE_VARIANCE
from analysis.surrogate.scripts.models.compare_single_surrogates import fit
from analysis.surrogate.scripts.models.compare_surrogates import SHAPES, number

def pass_grids(archive, targets=('H100', 'RTX3070-WSL')):
    grids = defaultdict(dict)
    with archive.open('candidate_passes.csv') as stream:
        for row in csv.DictReader(io.TextIOWrapper(stream)):
            if row['device'] not in targets:
                continue
            features = json.loads(row['workload_features_json'])
            shape = {k: number(features.get(k)) for k in SHAPES if number(features.get(k)) is not None}
            if 'locality' in features:
                shape['locality'] = features['locality']
            key = (row['device'], row['dtype'], row['kernel_family'] + ':' + json.dumps(shape, sort_keys=True))
            cell = grids[key]
            candidate = row['config_id']
            if row['kernel_family'] in ('softmax','cross_entropy'):
                config = json.loads(row['config_features_json'])
                candidate = tuple(config[k] for k in ('block_multiplier','num_warps','num_stages'))
            index = int(row['pass_index']), candidate
            values = [float(row[m + '_median_ms']) for m in ('graph','individual')]
            if index in cell or row['correctness_pass'] != '1' or not all(np.isfinite(values)) or min(values) <= 0:
                raise ValueError((key, index))
            cell[index] = values
    return grids

def orders(prior, covariance, labels):
    joint = covariance + SCALE_VARIANCE
    return [probe_order(prior, joint, np.log(labels), False, feedback=f, budget=4) for f in (True, False)] + [
        conditional_order(prior, covariance, labels, int(prior.argmin()), budget=4)]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','results'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--target', choices=('H100', 'RTX3070-WSL', 'T4', 'RTXA5000'))
    parser.add_argument('--passes', choices=(2,3), type=int, default=3)
    parser.add_argument('--timings', type=Path, help='Separate verified archive for independent timing passes')
    args = parser.parse_args()
    targets = (args.target,) if args.target else ('H100','RTX3070-WSL')
    if args.target in ('T4','RTXA5000') and args.timings is None:
        parser.error('Supply the verified timing archive for this target')
    with ZipFile(args.dataset) as archive:
        with TemporaryDirectory() as temp:
            for member in archive.namelist():
                if member.startswith('sources/'):
                    archive.extract(member,temp)
            sources = Path(temp)/'sources'
            inputs = load_precision_inputs(sources,sources/'t4-pass00-colab.zip')
            prepared = {(m,d):(list(prepare_families(rows,d,devices,m)),devices)
                        for d,rows,devices in inputs if any(t in devices for t in targets) for m in ('graph','individual')}
        with ZipFile(args.timings or args.dataset) as timings:
            grids = pass_grids(timings, targets)
        splits = {s: json.loads(archive.read('splits/' + filename))['assignment'] for s,filename in
                  [('largest','split-largest-shapes.json'),('next_largest','capacity_split_upper_middle.json')]}
    filename = f'independent-winners-{args.target}.jsonl.gz' if args.target else 'independent-winners.jsonl.gz'
    if args.passes != 3:
        filename = filename.replace('.jsonl.gz',f'-{args.passes}pass.jsonl.gz')
    with gzip.open(args.results/filename,'wt') as out:
        for (mode,dtype),(groups,devices) in prepared.items():
            suffix = 'fp16' if dtype=='float16' else 'bf16'
            with gzip.open(args.results/f'{mode}-{suffix}-predictions.jsonl.gz','rt') as stream:
                traces = {(r['split'],r['device'],r['shape']):r for r in map(json.loads,stream)
                          if r['device'] in targets}
            references = {}
            for folder,methods in [('workshop-belief',['simple_world_gaussian','simple_world_no_feedback']),
                                   ('policy_world',['geometry_ridge'])]:
                with gzip.open(args.results/folder/f'{mode}-{suffix}-predictions.jsonl.gz','rt') as stream:
                    for row in map(json.loads,stream):
                        key = row['split'],row['device'],row['shape']
                        if key in traces:
                            references.setdefault(key,[]).extend(row['models'][m]['probe_order'][:4] for m in methods)
            for split,assignment in splits.items():
                for target,device in enumerate(devices):
                    if device not in targets:
                        continue
                    for family,shapes,times,_,geometry,_ in groups:
                        train,test = ([i for i,s in enumerate(shapes) if assignment[s]==role] for role in ('train','test'))
                        model = fit(geometry,times,[i for i in range(len(devices)) if i!=target],train)
                        for i,prior in zip(test,model.mean(geometry[target,test])):
                            shape = shapes[i]
                            trace = traces[split,device,shape]
                            keys = [tuple(k) if isinstance(k,list) else k for k in trace['candidate_keys']]
                            grid = grids[device,dtype,shape]
                            assert {(p,k) for p,k in grid if p<args.passes} == {(p,k) for p in range(args.passes) for k in keys}
                            passes = np.array([[grid[p,k][int(mode=='individual')] for k in keys] for p in range(args.passes)])
                            archived = passes[0] if device == 'T4' else np.median(passes,axis=0)
                            if device == 'RTXA5000':
                                # The frozen traces were measured on Cybertron3. The scoring board is Cybertron2.
                                archived = times[target,i]
                            np.testing.assert_allclose(archived*1000,trace['measured_us'],rtol=1e-9)
                            original = orders(prior,model.residual_covariance,times[target,i])
                            if original != references[split,device,shape]:
                                raise ValueError(f'Policy mismatch on archived median labels: {device} {shape}')
                            for p,latency in enumerate(passes):
                                chosen = orders(prior,model.residual_covariance,latency)
                                winners = [o[int(np.argmin(latency[o]))] for o in chosen]
                                for q in range(len(passes)):
                                    if q == p:
                                        continue
                                    regret = passes[q,winners]/passes[q].min()-1
                                    out.write(json.dumps(dict(mode=mode,split=split,family=family,shape=shape,device=device,
                                             dtype=dtype,select=p,score=q,winners=winners,regret=regret.tolist()))+'\n')

if __name__=='__main__':
    main()
