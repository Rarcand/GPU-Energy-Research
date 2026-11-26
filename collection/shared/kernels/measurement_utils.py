#!/usr/bin/env python3
"""Shared metadata, telemetry, timing, and output helpers for GPU benchmarks."""

import csv
import json
import math
import os
import platform
import statistics
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch


class CandidateEvaluationError(RuntimeError):
    """Attach a reproducible evaluation stage to a candidate failure."""

    def __init__(self, stage, cause):
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause


def candidate_failure_record(stage, exc, correctness_enabled=True):
    """Return consistent labels without inventing latency for invalid rows."""
    cause = exc.cause if isinstance(exc, CandidateEvaluationError) else exc
    compile_success = stage not in {"precheck", "compile_or_first_launch"}
    runtime_success = "" if not compile_success else stage != "runtime"
    if not correctness_enabled or not compile_success:
        correctness_pass = ""
    elif stage == "correctness":
        correctness_pass = False
    else:
        correctness_pass = True
    return {
        "status": "incorrect" if stage == "correctness" else "invalid",
        "is_valid": False,
        "validity_label": 0,
        "failure_stage": stage,
        "failure_type": type(cause).__name__,
        "failure_message": str(cause),
        "compile_success": compile_success,
        "runtime_success": runtime_success,
        "correctness_pass": correctness_pass,
        # Backward-compatible aliases used by the pilot notebooks.
        "error_type": type(cause).__name__,
        "error": str(cause),
    }


def candidate_success_record():
    return {
        "status": "ok",
        "is_valid": True,
        "validity_label": 1,
        "failure_stage": "",
        "failure_type": "",
        "failure_message": "",
        "compile_success": True,
        "runtime_success": True,
        "error_type": "",
        "error": "",
    }


STATIC_SMI_FIELDS = [
    "name", "uuid", "compute_cap", "driver_version", "memory.total",
    "multiprocessor_count", "clocks.max.sm", "clocks.max.memory", "power.limit",
]

DYNAMIC_SMI_FIELDS = [
    "pstate", "temperature.gpu", "power.draw", "clocks.sm", "clocks.mem",
    "utilization.gpu", "utilization.memory", "memory.used",
]


def _smi_query(fields):
    command = [
        "nvidia-smi",
        "--query-gpu={}".format(",".join(fields)),
        "--format=csv,noheader,nounits",
        "--id=0",
    ]
    try:
        line = subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT, timeout=10,
        ).strip().splitlines()[0]
        return dict(zip(fields, [item.strip() for item in line.split(",")]))
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError):
        return {}


def _number(value, cast=float):
    try:
        return cast(value)
    except (TypeError, ValueError):
        return ""


def architecture_metadata():
    """Return stable architecture features used by the architecture encoder."""
    props = torch.cuda.get_device_properties(0)
    smi = _smi_query(STATIC_SMI_FIELDS)
    gpu_uuid = smi.get("uuid", "")
    architecture_id = "{}-sm{}-{}".format(
        props.name.replace(" ", "_"),
        props.major * 10 + props.minor,
        gpu_uuid[-8:] if gpu_uuid else "unknown",
    )
    return {
        "architecture_id": architecture_id,
        "gpu_name": props.name,
        "gpu_uuid": gpu_uuid,
        "compute_capability": "{}.{}".format(props.major, props.minor),
        "sm_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "warp_size": getattr(props, "warp_size", 32),
        "max_threads_per_block": getattr(props, "max_threads_per_block", ""),
        "max_threads_per_sm": getattr(props, "max_threads_per_multi_processor", ""),
        "shared_memory_per_block_bytes": getattr(props, "shared_memory_per_block", ""),
        "shared_memory_per_sm_bytes": getattr(props, "shared_memory_per_multiprocessor", ""),
        "registers_per_sm": getattr(props, "regs_per_multiprocessor", ""),
        "l2_cache_bytes": getattr(props, "L2_cache_size", ""),
        "memory_bus_width_bits": getattr(props, "memory_bus_width", ""),
        "driver_version": smi.get("driver_version", ""),
        "max_sm_clock_mhz": _number(smi.get("clocks.max.sm")),
        "max_memory_clock_mhz": _number(smi.get("clocks.max.memory")),
        "power_limit_w": _number(smi.get("power.limit")),
    }


def dynamic_telemetry():
    """Sample state after a timing batch; this is context, not per-kernel energy."""
    smi = _smi_query(DYNAMIC_SMI_FIELDS)
    return {
        "gpu_pstate": smi.get("pstate", ""),
        "temperature_c": _number(smi.get("temperature.gpu")),
        "power_draw_w": _number(smi.get("power.draw")),
        "sm_clock_mhz": _number(smi.get("clocks.sm")),
        "memory_clock_mhz": _number(smi.get("clocks.mem")),
        "gpu_utilization_pct": _number(smi.get("utilization.gpu")),
        "memory_utilization_pct": _number(smi.get("utilization.memory")),
        "device_memory_used_mb": _number(smi.get("memory.used")),
    }


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def timing_summary(timings_ms):
    return {
        "latency_mean_ms": statistics.mean(timings_ms),
        "latency_median_ms": statistics.median(timings_ms),
        "latency_std_ms": statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        "latency_min_ms": min(timings_ms),
        "latency_p10_ms": percentile(timings_ms, 0.10),
        "latency_p90_ms": percentile(timings_ms, 0.90),
        "latency_max_ms": max(timings_ms),
        "timing_sample_count": len(timings_ms),
        "timings_ms_json": json.dumps(timings_ms, separators=(",", ":")),
    }


def power_window(launch_once, duration_ms):
    """Estimate energy with sustained launches and NVML sampling.

    This deliberately runs outside latency timing. It is an approximation for
    candidate comparison, not a claim of isolated single-kernel energy.
    """
    if duration_ms <= 0:
        return {"power_measurement_status": "disabled"}
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as exc:
        return {
            "power_measurement_status": "unavailable",
            "power_measurement_error": "{}: {}".format(type(exc).__name__, exc),
        }

    samples = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                samples.append((time.perf_counter(), pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0))
            except Exception:
                pass
            stop.wait(0.01)

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    start = time.perf_counter()
    launches = 0
    try:
        while (time.perf_counter() - start) * 1e3 < duration_ms:
            launch_once()
            launches += 1
        torch.cuda.synchronize()
    finally:
        stop.set(); worker.join(timeout=2.0)
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    elapsed = time.perf_counter() - start
    powers = [value for _, value in samples]
    if not powers or launches == 0:
        return {
            "power_measurement_status": "insufficient_samples",
            "power_sample_count": len(powers), "power_window_launches": launches,
        }
    average_power = statistics.mean(powers)
    energy_j = average_power * elapsed
    return {
        "power_measurement_status": "ok",
        "power_sample_count": len(powers),
        "power_window_duration_ms": elapsed * 1e3,
        "power_window_launches": launches,
        "power_window_mean_w": average_power,
        "power_window_min_w": min(powers),
        "power_window_max_w": max(powers),
        "power_window_energy_j": energy_j,
        "estimated_energy_per_launch_j": energy_j / launches,
    }


def run_metadata(seed, session_id):
    return {
        "run_id": str(uuid.uuid4()),
        "session_id": session_id or os.environ.get("COLLECTION_SESSION_ID", "session-1"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
    }


def safe_ratio(numerator, denominator):
    try:
        return numerator / denominator if denominator else ""
    except TypeError:
        return ""


def atomic_write_csv(path, rows):
    if not rows:
        raise ValueError("No measurement rows were produced")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

