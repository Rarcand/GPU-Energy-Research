"""Replay frozen four-query policies on the held-out native A100 and held-out shapes."""
import argparse
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from audit_final_choice import load_precision_inputs
from compare_single_surrogates import fit
from compare_surrogate_architectures import prepare_families
from validate_pass_winners import orders, pass_grids


def score(base, timings, results, passes, output):
    with ZipFile(timings) as archive:
        coverage = json.loads(archive.read('coverage.json'))
        if coverage['native_a100_job'] != '3614578' or coverage['rows_by_pass']['A100'][:passes] != [30000] * passes:
            raise ValueError('Requested native A100 passes are incomplete')
        grids = pass_grids(archive, ('A100',))
    with ZipFile(base) as archive, TemporaryDirectory() as temp:
        splits = {k: json.loads(archive.read('splits/' + f))['assignment'] for k, f in
                  [('largest', 'split-largest-shapes.json'), ('next_largest', 'capacity_split_upper_middle.json')]}
        for member in archive.namelist():
            if member.startswith('sources/'):
                archive.extract(member, temp)
        sources = Path(temp) / 'sources'
        inputs = load_precision_inputs(sources, sources / 't4-pass00-colab.zip')
    cases = []
    for dtype, rows, devices in inputs:
        if 'A100' in devices:
            raise ValueError('A100 is reserved for evaluation')
        suffix = 'fp16' if dtype == 'float16' else 'bf16'
        for column, mode in enumerate(('graph', 'individual')):
            with gzip.open(results / f'{mode}-{suffix}-predictions.jsonl.gz', 'rt') as stream:
                traces = {(r['split'], r['shape']): r for r in map(json.loads, stream) if r['device'] == 'H100'}
            for family, shapes, times, _, geometry, _ in prepare_families(rows, dtype, devices, mode):
                # This predictor uses only workload/configuration geometry. Test its invariance before reuse.
                for g in geometry[1:]:
                    np.testing.assert_array_equal(geometry[0], g)
                for split, assignment in splits.items():
                    train = [i for i, shape in enumerate(shapes) if assignment[shape] == 'train']
                    test = [i for i, shape in enumerate(shapes) if assignment[shape] == 'test']
                    model = fit(geometry, times, list(range(len(devices))), train)
                    for i, prior in zip(test, model.mean(geometry[0, test])):
                        shape = shapes[i]
                        trace = traces[split, shape]
                        keys = [tuple(k) if isinstance(k, list) else k for k in trace['candidate_keys']]
                        np.testing.assert_allclose(times[devices.index('H100'), i] * 1000, trace['measured_us'], rtol=1e-9)
                        grid = grids['A100', dtype, shape]
                        if {(p, k) for p, k in grid if p < passes} != {(p, k) for p in range(passes) for k in keys}:
                            raise ValueError(f'Incomplete A100 grid for {shape}')
                        labels = np.array([[grid[p, k][column] for k in keys] for p in range(passes)])
                        for p, measured in enumerate(labels):
                            queries = orders(prior, model.residual_covariance, measured)
                            if any(len(set(q)) != 4 or len(q) != 4 for q in queries):
                                raise ValueError('Expected four distinct queries per policy')
                            winners = [q[int(np.argmin(measured[q]))] for q in queries]
                            for j, actual in enumerate(labels):
                                if p != j or passes == 1:
                                    cases.append(dict(device='A100', family=family, shape=shape, dtype=dtype,
                                                      split=split, mode=mode, select=p, score=j, winners=winners,
                                                      regret=(actual[winners] / actual.min() - 1).tolist()))
    if len(cases) != 796 * 2 * (passes * (passes - 1) or 1):
        raise ValueError(f'Incomplete evaluation with {len(cases)} cases')
    with gzip.open(output, 'wt') as stream:
        for row in cases:
            stream.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'timings', 'results', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--passes', type=int, choices=(1, 2, 3), required=True,
                        help='1 scores on the selection pass. 2 and 3 use distinct scoring passes.')
    args = parser.parse_args()
    score(args.base, args.timings, args.results, args.passes, args.output)
