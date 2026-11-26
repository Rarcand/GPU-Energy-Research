#!/bin/bash
#SBATCH -J h100-recovery
#SBATCH -o h100-recovery.%j.out
#SBATCH -e h100-recovery.%j.err
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -A ASC25053
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python="$SCRATCH/gpu-energy-research/venv/bin/python"
output="${H100_RUN_DIR:-$SLURM_SUBMIT_DIR/results/timing-validation/collection/h100/$SLURM_JOB_ID}"
health="$output/diagnostics-$SLURM_JOB_ID"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TRITON_CACHE_DIR="$SCRATCH/gpu-energy-research/h100-timing-validation-cache"
mkdir -p "$output"
# Preserve inherited Slurm visibility; the diagnostic selects only an allocated,
# working GPU. Never reset a device or override the scheduler's node choice.
python3 -u collection/h100/diagnose_h100_recovery.py --python "$python" --output "$health"
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(cat "$health/selected-uuid.txt")
# Debug synchronization is confined to probes, not production timing.
unset CUDA_LAUNCH_BLOCKING
runner="$SLURM_SUBMIT_DIR/collection/shared/run_timing_validation.py"
"$python" -u "$runner" --root "$SLURM_SUBMIT_DIR" --device H100 \
    --output "$output/preflight-$SLURM_JOB_ID" --smoke-only
if [[ "${H100_DIAGNOSTIC_ONLY:-0}" == 1 ]]; then exit 0; fi
: "${H100_PASS_INDEX:?H100_PASS_INDEX required}"
"$python" -u "$runner" --root "$SLURM_SUBMIT_DIR" --device H100 \
    --output "$output/pass-session-$H100_PASS_INDEX" --passes "$H100_PASS_INDEX" --max-hours 23.5
python3 collection/h100/report_timing_validation.py --input "$output" --output "$output/timing-report-$SLURM_JOB_ID.json"
