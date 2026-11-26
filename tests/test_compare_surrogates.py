"""Protocol checks: grouping, feature coverage, capacity and failure accounting."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis' / 'surrogate' / 'scripts'))
from compare_surrogates import (normalized, split_groups, matrices, scores,
                                 Network, Processor, calibration)


def row(shape=128, dtype='float16', gpu='NVIDIA A100', valid=True, block=32):
    return normalized(dict(kernel_family='softmax', rows='16', reduction_width=str(shape),
        dtype=dtype, gpu_name=gpu, block_size=str(block), num_warps='4',
        is_valid=str(valid), status='ok' if valid else 'error', latency_median_ms='.1'))


class ProtocolTests(unittest.TestCase):
    def test_shape_groups_ignore_dtype_device_and_configuration(self):
        a = row()
        b = row(dtype='bfloat16', gpu='NVIDIA H100', block=64)
        self.assertEqual(a['_group'], b['_group'])
        self.assertNotEqual(a['_key'], b['_key'])
        rows = [row(shape=i, dtype=d) for i in range(100, 110) for d in ('float16','bfloat16')]
        splits = split_groups(rows)
        self.assertEqual(set(splits.values()), {'train','validation','test'})
        self.assertEqual(len(splits), 10)

    def test_family_specific_shapes_and_knobs_are_distinct(self):
        rows = []
        for length, block in [(128,16),(256,16),(128,32)]:
            rows.append(normalized(dict(kernel_family='attention', heads='4', q_len=str(length),
                kv_len='256', head_dim='64', causal='False', block_m=str(block),
                dtype='float16', gpu_name='A100', status='ok', is_valid='True', latency_median_ms='1')))
        a,w,c = matrices(rows)
        self.assertFalse(np.array_equal(w[0],w[1]))
        self.assertFalse(np.array_equal(c[0],c[2]))

    def test_invalid_candidates_consume_budget(self):
        rows = [row(valid=False,block=16),row(block=32),row(block=64)]
        result = scores(rows,np.arange(3),np.array([.01,.02,.03]))[0]
        self.assertEqual(result['success5_b1'],0)
        self.assertIsNone(result['regret_b1'])
        self.assertEqual(result['invalid_b1'],1)
        self.assertEqual(result['success5_b2'],1)
        self.assertEqual(result['effective_b8'],3)

    def test_preprocessing_does_not_fit_test_rows(self):
        arrays = [np.array([[1.],[2.],[10000.]],dtype='float32')]*3
        p = Processor().fit(arrays,np.array([0,1]),np.array([1.,2.,10000.]))
        self.assertEqual(p.stats[0][0][0],1.5)
        self.assertAlmostEqual(p.mean,np.log(2)/2)

    def test_capacity_matching(self):
        sizes = [28,50,20]
        counts = [sum(p.numel() for p in Network(sizes,k).parameters()) for k in ('three','flat')]
        self.assertLess(abs(counts[0]-counts[1])/counts[0], .02)

    def test_nested_calibration(self):
        rows = [row(shape=i) for i in range(20)]
        a = calibration(rows,np.arange(20),5,42)
        b = calibration(rows,np.arange(20),10,42)
        self.assertTrue(set(a)<=set(b))

    def test_paired_interval_preserves_task_weights(self):
        from report_comparisons import paired_interval
        def run(method):
            tasks=[]
            # Unequal dtype coverage: group a has two tasks, b has one.
            for group,value in [('a',1.),('a',1.),('b',0.)]:
                tasks.append({'group':group,'family':'gemm',
                              'success5_b1':value if method=='a' else 0.})
            return [{'seed':1,'tasks':tasks}]
        result=paired_interval(run('a'),run('b'),'success5_b1',100)
        self.assertAlmostEqual(result['difference'],2./3)


if __name__ == '__main__':
    unittest.main()
