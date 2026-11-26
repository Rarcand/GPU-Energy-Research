"""Compare adaptive and fixed query sequences by selection and scoring on distinct passes.

Each output row contains the selected configuration indices and scoring-pass simple
regrets for Gaussian feedback, feedback disabled, global portfolio, and local
portfolio, in that order. The local portfolio uses the eight nearest training
shapes on each source device. Both portfolios are fixed before target queries.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from audit_final_choice import load_precision_inputs
from benchmark_simple_belief import SCALE_VARIANCE, probe_order
from portable_rank_prior import shape_distances
from compare_single_surrogates import fit
from compare_surrogate_architectures import prepare_families
from validate_pass_winners import pass_grids


BUDGETS = (1, 2, 4, 8)
MODES = ('graph', 'individual')
FIELDS = ('device', 'split', 'shape', 'dtype', 'mode', 'select', 'score')
SPLITS = {'largest': 'split-largest-shapes.json', 'next_largest': 'capacity_split_upper_middle.json'}


def greedy_portfolio(training):
    """Maximize new 5%-covered source tasks; ties minimize mean best log slowdown."""
    actual = training.reshape(-1, training.shape[-1])
    if actual.shape[1] < max(BUDGETS):
        raise ValueError('A requested budget exceeds the family candidate grid')
    if np.any(actual <= 0) or not np.isfinite(actual).all():
        raise ValueError('Expected positive finite source timings')
    loss = np.log(actual / actual.min(axis=1, keepdims=True))
    incumbent = np.full(len(actual), np.inf)
    remaining = list(range(loss.shape[1]))
    order = []
    for _ in range(max(BUDGETS)):
        combined = np.minimum(incumbent[:, None], loss[:, remaining])
        hits = (combined <= np.log1p(.05 + 1e-12)).sum(axis=0)
        mean = combined.mean(axis=0)
        best = min(range(len(remaining)), key=lambda j: (-hits[j], mean[j], remaining[j]))
        index = remaining.pop(best)
        order.append(index)
        incumbent = np.minimum(incumbent, loss[:, index])
    return order


def frozen_checks(results):
    check = {}
    for suffix in ('', '-RTXA5000', '-T4', '-A100'):
        with gzip.open(results / f'independent-winners{suffix}.jsonl.gz', 'rt') as stream:
            for row in map(json.loads, stream):
                key = tuple(row[k] for k in FIELDS)
                if key in check:
                    raise ValueError(f'Duplicate frozen case: {key}')
                check[key] = row['winners'][:2], row['regret'][:2]
    return check


def replay(base, timings, results, stream):
    check = frozen_checks(results)
    count = 0
    with ZipFile(base) as z, TemporaryDirectory() as tmp:
        splits = {k: json.loads(z.read('splits/' + f))['assignment'] for k,f in SPLITS.items()}
        for member in z.namelist():
            if member.startswith('sources/'):
                z.extract(member, tmp)
        sources = Path(tmp) / 'sources'
        inputs = load_precision_inputs(sources, sources / 't4-pass00-colab.zip')
    with ZipFile(timings) as z:
        coverage = json.loads(z.read('coverage.json'))['rows_by_pass']
        if coverage != {d: [15000 if d=='T4' else 30000]*3 for d in
                        ('A100','H100','RTXA5000','T4','RTX3070-WSL')}:
            raise ValueError('Expected the complete five-GPU archive')
        grids = pass_grids(z, tuple(coverage))
    for dtype, rows, devices in inputs:
        suffix = 'fp16' if dtype=='float16' else 'bf16'
        for column, mode in enumerate(MODES):
            with gzip.open(results / f'{mode}-{suffix}-predictions.jsonl.gz', 'rt') as f:
                keys = {(r['split'], r['shape']): [tuple(k) if isinstance(k,list) else k for k in r['candidate_keys']]
                        for r in map(json.loads, f) if r['device']=='H100'}
            for family, shapes, times, _, geometry, _ in prepare_families(rows, dtype, devices, mode):
                # The frozen mean is a workload/configuration geometry model.
                for g in geometry[1:]:
                    np.testing.assert_array_equal(geometry[0], g)
                for split, assignment in splits.items():
                    train = [i for i,s in enumerate(shapes) if assignment[s]=='train']
                    test = [i for i,s in enumerate(shapes) if assignment[s]=='test']
                    distances = shape_distances([shapes[i] for i in train], [shapes[i] for i in test])
                    nearest = np.asarray(train)[np.argsort(distances, axis=1, kind='stable')[:, :8]]
                    for dev in [*devices, 'A100']:
                        source = [i for i,d in enumerate(devices) if d!=dev]
                        order = greedy_portfolio(times[source][:,train])
                        model = fit(geometry, times, source, train)
                        joint = model.residual_covariance + SCALE_VARIANCE
                        for i, prior, neighbors in zip(test, model.mean(geometry[0,test]), nearest):
                            local_order = greedy_portfolio(times[source][:, neighbors])
                            shape = shapes[i]
                            candidates = keys[split,shape]
                            grid = grids[dev,dtype,shape]
                            if set(grid) != {(p,k) for p in range(3) for k in candidates}:
                                raise ValueError(f'Incomplete timing grid for {dev} {shape}')
                            labels = np.array([[grid[p,k][column] for k in candidates] for p in range(3)])
                            fixed = probe_order(prior, joint, prior, False, feedback=False, budget=8)
                            for p, measured in enumerate(labels):
                                adaptive = probe_order(prior, joint, np.log(measured), False, budget=8)
                                for budget in BUDGETS:
                                    prefixes = [o[:budget] for o in (adaptive, fixed, order, local_order)]
                                    winners = [o[int(np.argmin(measured[o]))] for o in prefixes]
                                    for q, actual in enumerate(labels):
                                        if p==q:
                                            continue
                                        regret = (actual[winners] / actual.min() - 1).tolist()
                                        key = (dev,split,shape,dtype,mode,p,q)
                                        if budget==4:
                                            old_winners, old_regret = check.pop(key)
                                            if winners[:2]!=old_winners or regret[:2]!=old_regret:
                                                raise ValueError(f'Frozen four-query parity failed for {key}')
                                        row = dict(zip(FIELDS,key),family=family,budget=budget,winners=winners,regret=regret)
                                        stream.write(json.dumps(row, separators=(',',':'))+'\n')
                                        count += 1
    if check or count != (796*4 + 398)*2*6*len(BUDGETS):
        raise ValueError(f'Unexpected replay coverage: {len(check)} missing frozen cases, {count} rows')
    return count


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('base','timings','results','output'):
        parser.add_argument(arg,type=Path)
    args=parser.parse_args()
    with NamedTemporaryFile(dir=args.output.parent, delete=False) as f:
        pending=Path(f.name)
    try:
        with gzip.open(pending,'wt') as stream:
            count=replay(args.base,args.timings,args.results,stream)
        os.replace(pending,args.output)
        print(f'{count} independent pass comparisons; all four-query results matched frozen traces')
    finally:
        pending.unlink(missing_ok=True)
