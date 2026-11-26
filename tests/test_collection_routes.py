"""Exercise collection routing and durable checkpoints without CUDA or Slurm."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = ROOT / "collection"
sys.path.insert(0, str(COLLECTION / "shared"))
import timing_validation as worker


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeviceRoutes(unittest.TestCase):
    def test_smoke_panels_and_production_precisions(self):
        rows = [{"workload_id": workload, "dtype": dtype,
                 "config_features_json": json.dumps({"block": block})}
                for workload in ("b", "a") for dtype in ("bfloat16", "float16")
                for block in (128, 32)]
        for device in ("A100", "H100", "RTX3070-WSL", "RTXA5000"):
            with self.subTest(device=device):
                self.assertEqual(worker.select_rows(rows, device, False), rows)
                self.assertEqual([r["dtype"] for r in worker.select_rows(rows, device, True)],
                                 ["float16", "bfloat16"])
        fp16 = [r for r in rows if r["dtype"] == "float16"]
        self.assertEqual(worker.select_rows(rows, "T4", False), fp16)
        smoke = worker.select_rows(rows, "T4", True)
        self.assertEqual([r["workload_id"] for r in smoke], ["a", "b"])
        self.assertTrue(all(r in fp16 and json.loads(r["config_features_json"])["block"] == 32 for r in smoke))

    def test_device_commands_resolve_without_gpu_libraries(self):
        for entry in ("shared/timing_validation.py", "rtx-3070/collect.py",
                      "rtx-a5000/collect.py", "t4/collect.py", "t4/repeat.py"):
            with self.subTest(entry=entry):
                result = subprocess.run([sys.executable, "-S", str(COLLECTION / entry), "--help"],
                                        capture_output=True, text=True, cwd=Path.home(), timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_all_five_devices_read_the_bundled_registry_from_a_clean_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            clean = Path(directory)
            (clean / "collection").mkdir()
            (clean / "collection/grid").symlink_to(COLLECTION / "grid", target_is_directory=True)
            for device in worker.GPU_NAMES:
                with self.subTest(device=device):
                    units = worker.registry(clean, device)
                    self.assertEqual(len(units), 70)
                    measured = [r for _, rows in units.values() for r in worker.select_rows(rows, device, False)]
                    self.assertEqual(len(measured), 15000 if device == "T4" else 30000)
                    self.assertEqual({r["kernel_family"] for r in measured}, set(worker.FAMILIES))
                    self.assertEqual(len({(r["workload_id"], r["config_id"]) for r in measured}), len(measured))

    def test_existing_measurements_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unit.jsonl"
            args = SimpleNamespace(root=Path(directory), device="H100", family="gemm", shard=0,
                                   limit=False, output=output)
            with patch.object(worker, "registry", return_value={("gemm", 0): (Path("registry.csv"), [])}):
                for existing in (output, output.with_suffix(".summary.json")):
                    with self.subTest(existing=existing.name):
                        existing.write_text("preserve")
                        with self.assertRaises(FileExistsError):
                            worker.unit(args)
                        self.assertEqual(existing.read_text(), "preserve")
                        existing.unlink()

    def test_native_a100_job_requires_a_passing_pilot_before_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "gpu-energy-research/venv/bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to(sys.executable)
            runner = root / "collection/shared/run_timing_validation.py"
            runner.parent.mkdir(parents=True)
            runner.write_text("from pathlib import Path\nimport sys\nPath('dispatch').write_text(' '.join(sys.argv[1:]))\n")
            summary = root / "pilot.summary.json"
            protocol = root / "protocol.json"
            env = dict(os.environ, SCRATCH=str(root), SLURM_SUBMIT_DIR=str(root),
                       SLURM_JOB_ID="test", A100_PILOT_SUMMARY=str(summary))
            cases = [(False, 168, [], "gpu-a100"), (True, 167, [], "gpu-a100"),
                     (True, 168, ["clock"], "gpu-a100"), (True, 168, [], "gpu-a100-dev"),
                     (True, 168, [], "gpu-a100")]
            for passed, count, failures, partition in cases:
                with self.subTest(passed=passed, count=count, failures=failures, partition=partition):
                    summary.write_text(json.dumps(dict(passed=passed, candidates=count, clock_failures=failures)))
                    protocol.write_text(json.dumps(dict(mode="pilot", partition=partition)))
                    proc = subprocess.run(["bash", str(COLLECTION / "a100/collect.slurm")], env=env,
                                          capture_output=True, text=True, timeout=10)
                    eligible = passed and count == 168 and not failures and partition == "gpu-a100"
                    self.assertEqual(proc.returncode == 0, eligible, proc.stderr)
                    self.assertEqual((root / "dispatch").exists(), eligible)
                    if eligible:
                        self.assertIn("--device A100", (root / "dispatch").read_text())
                        self.assertIn("--passes 0 1 2", (root / "dispatch").read_text())

    def test_scheduler_scripts_parse_and_timing_worker_has_one_implementation(self):
        for entry in ("a100/collect.slurm", "a100/qualify.slurm", "h100/collect.slurm"):
            subprocess.run(["bash", "-n", str(COLLECTION / entry)], check=True)
        implementations = {path.resolve() for path in COLLECTION.rglob("timing_validation.py")}
        self.assertEqual(implementations, {(COLLECTION / "shared/timing_validation.py").resolve()})


class DurableCheckpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repeat = load_file("colab_repeat_driver", COLLECTION / "t4/collect.py")

    def test_t4_roundtrip_and_corrupt_checkpoint_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "sample.jsonl"
            output.write_bytes(b'{"timing_status":"ok"}\n')
            output.with_suffix(".summary.json").write_text(json.dumps(
                {"complete": True, "rows": 1, "output_sha256": worker.digest(output)}))
            output.with_suffix(".protocol.json").write_text('{"device":"T4"}')
            store = root / "drive"
            restored = root / "restore" / output.name
            restored.parent.mkdir()
            self.assertFalse(self.repeat.restore_unit(restored, store, 1))
            self.repeat.checkpoint_unit(output, store, 1)
            self.assertTrue(self.repeat.restore_unit(restored, store, 1))
            self.assertEqual(restored.read_bytes(), output.read_bytes())
            (store / "sample.zip").write_bytes(b"truncated transfer")
            with self.assertRaises(RuntimeError):
                self.repeat.restore_unit(restored, store, 1)

    def test_partial_coverage_and_corrupt_published_data(self):
        report = load_file("h100_timing_report", COLLECTION / "h100/report_timing_validation.py")
        result = report.assess({("H100", 0, "gemm", "w", "c"): {"timing_status": "ok"}}, "H100")
        self.assertFalse(result["complete"])
        self.assertNotIn("families", result)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "pass-00"
            folder.mkdir()
            data = folder / "gemm-00.jsonl"
            data.write_text("{}\n")
            data.with_suffix(".summary.json").write_text(json.dumps(
                {"complete": True, "output_sha256": worker.digest(data)}))
            data.write_text("{}\n{}\n")
            with self.assertRaises(ValueError):
                report.read(folder.parent)


if __name__ == "__main__":
    unittest.main()
