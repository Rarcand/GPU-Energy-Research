#!/bin/bash
#SBATCH -J a100-clock-native
#SBATCH -o a100-clock.%j.out
#SBATCH -e a100-clock.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:40:00
#SBATCH -A ASC25053

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python="$SCRATCH/gpu-energy-research/venv/bin/python"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TRITON_CACHE_DIR="$SCRATCH/gpu-energy-research/timing-validation-cache"
root="$SLURM_SUBMIT_DIR/results/timing-validation/collection/a100-qualification"
flags=()
# Comparing to an older allocation is optional and does not alter the pilot gate.
if [[ -n "${A100_PRIOR_PILOT:-}" ]]; then
    flags+=(--prior "$A100_PRIOR_PILOT")
fi
"$python" -u collection/a100/qualify.py \
    --root "$SLURM_SUBMIT_DIR" --output "$root/$SLURM_JOB_ID" \
    "${flags[@]}"
