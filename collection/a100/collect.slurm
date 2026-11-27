#!/bin/bash
#SBATCH -J a100-matched-full
#SBATCH -o a100-matched-full.%j.out
#SBATCH -e a100-matched-full.%j.err
#SBATCH -p gpu-a100
#SBATCH -A ASC25053
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python="$SCRATCH/gpu-energy-research/venv/bin/python"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TRITON_CACHE_DIR="$SCRATCH/gpu-energy-research/timing-validation-cache"
# Pass the completed qualification job's pilot.summary.json when submitting.
: "${A100_PILOT_SUMMARY:?Set A100_PILOT_SUMMARY to the passing native pilot summary}"
"$python" - "$A100_PILOT_SUMMARY" <<'PYTHON'
import json
import sys
from pathlib import Path

summary = Path(sys.argv[1])
pilot = json.loads(summary.read_text())
protocol = json.loads(summary.with_name("protocol.json").read_text())
if not (pilot.get("passed") is True and pilot.get("candidates") == 168
        and pilot.get("clock_failures") == [] and protocol.get("mode") == "pilot"
        and protocol.get("partition") == "gpu-a100"):
    raise SystemExit("Expected a passing native A100 clock qualification")
PYTHON
output="results/timing-validation/collection/a100/$SLURM_JOB_ID"
"$python" -u collection/shared/run_timing_validation.py --root "$PWD" --device A100 \
    --output "$output" --passes 0 1 2 --max-hours 47.8
