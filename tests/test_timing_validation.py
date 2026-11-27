"""CPU tests for the timing protocol used by the five collection targets."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

COLLECTION = Path(__file__).resolve().parents[1] / 'collection'
sys.path.insert(0, str(COLLECTION / 'shared'))
import timing_validation as worker


class TimingTests(unittest.TestCase):
    def test_fixed_seed_matches_historical_workload(self):
        self.assertEqual(worker.seeded('gemm-128x192x320-float16'), 1461970402)
        self.assertNotEqual(worker.seeded('gemm-128x192x320-float16'),
                            worker.seeded('gemm-128x192x320-bfloat16'))

    def test_drift_detects_nonstationary_samples(self):
        result = worker.describe([1., 1., 1., 1., 2., 2., 2., 2.])
        self.assertEqual(result['half_median_drift'], 1.)
        self.assertEqual(result['median_ms'], 1.5)

    def test_individual_sample_brackets_one_launch_with_cuda_events(self):
        calls = []
        start, end = MagicMock(), MagicMock()
        start.record.side_effect = lambda: calls.append('start')
        end.record.side_effect = lambda: calls.append('end')
        end.synchronize.side_effect = lambda: calls.append('sync')
        start.elapsed_time.return_value = .023
        torch = MagicMock()
        torch.cuda.Event.side_effect = [start, end]
        values = worker.samples(lambda: calls.append('launch'), torch, 1)
        self.assertEqual(calls, ['start', 'launch', 'end', 'sync'])
        start.elapsed_time.assert_called_once_with(end)
        self.assertEqual(values, [.023])

    def test_replay_uses_first_calibrated_batch_and_divides_each_sample(self):
        torch = MagicMock()
        launch = MagicMock()
        pilots = [[.20, .21, .22], [.8, .9, 1.0]]
        final = [1., 1.5, 1., 1.5]
        with patch.object(worker, 'samples', side_effect=[*pilots, final]) as probe:
            result = worker.graph_samples(launch, torch, len(final))
        self.assertTrue(result['target_reached'])
        self.assertEqual(result['captured_launches'], 32)
        self.assertEqual(result['samples_ms'], [elapsed / 32 for elapsed in final])
        self.assertEqual([c.args[2] for c in probe.call_args_list], [3, 3, 4])
        self.assertEqual(launch.call_count, 8 + 32)

    def test_replay_cap_records_unmet_target(self):
        with patch.object(worker, 'samples', side_effect=lambda launch, torch, n: [.1] * n):
            result = worker.graph_samples(lambda: None, MagicMock(), 4)
        self.assertFalse(result['target_reached'])
        self.assertEqual(result['captured_launches'], 2048)

    def test_incomplete_registry_rejected_for_each_supported_device(self):
        with tempfile.TemporaryDirectory() as directory:
            for device in ('A100', 'H100', 'RTX3070-WSL', 'RTXA5000', 'T4'):
                with self.subTest(device=device), self.assertRaises(ValueError):
                    worker.registry(Path(directory), device)


if __name__ == '__main__':
    unittest.main()
