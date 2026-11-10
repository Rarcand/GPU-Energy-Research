#!/bin/bash
#SBATCH -J transfer-data-a100
#SBATCH -o transfer-data.%j.out
#SBATCH -e transfer-data.%j.err
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ASC25053

set -euo pipefail
REPO_DIR="$SLURM_SUBMIT_DIR"
ENV_ROOT="$SCRATCH/gpu-energy-research/venv"
RUN_DIR="$SCRATCH/gpu-energy-research/runs/$SLURM_JOB_ID"
GPU_SLUG=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr '[:upper:] ' '[:lower:]_' | tr -cd '[:alnum:]_-')
FINAL_DIR="$REPO_DIR/results/$GPU_SLUG"
[[ -x "$ENV_ROOT/bin/python" ]] || { echo "Missing environment: $ENV_ROOT" >&2; exit 1; }
mkdir -p "$RUN_DIR" "$FINAL_DIR"
source "$ENV_ROOT/bin/activate"

python "$REPO_DIR/scripts/benchmark_matmul.py" \
  --output "$RUN_DIR/matmul-transfer-$SLURM_JOB_ID.csv" \
  --shapes 384x3072x1280,3072x384x1280,640x2560x1536,2560x640x1536,896x1792x2304,1792x896x2304,1536x3072x768,3072x1536x768,3584x3584x1536 \
  --dtypes float16,bfloat16 --warmup 10 --repetitions 50 --max-configs 48 \
  --config-seed 2026 --seed 3026 --session-id "a100-transfer-$SLURM_JOB_ID"

python "$REPO_DIR/scripts/benchmark_reductions.py" \
  --output "$RUN_DIR/reductions-transfer-$SLURM_JOB_ID.csv" \
  --families softmax,cross_entropy \
  --shapes 128x768,384x1280,768x1536,1536x2560,3072x3072,640x5120,1280x6144,2560x7168,384x12288,768x16384 \
  --dtypes float16,bfloat16 --warmup 10 --repetitions 50 --max-configs 12 \
  --config-seed 2026 --seed 3026 --session-id "a100-transfer-$SLURM_JOB_ID"

cp "$RUN_DIR"/*.csv "$FINAL_DIR"/
echo "Expanded A100 transfer data: $FINAL_DIR"
