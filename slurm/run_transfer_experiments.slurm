#!/bin/bash
#SBATCH -J transfer-eval
#SBATCH -o transfer-eval.%j.out
#SBATCH -e transfer-eval.%j.err
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A ASC25053

set -euo pipefail

REPO_DIR="$SLURM_SUBMIT_DIR"
ENV_ROOT="$SCRATCH/gpu-energy-research/venv"
OUTPUT_DIR="$REPO_DIR/results/experiments/transfer-$SLURM_JOB_ID"

[[ -x "$ENV_ROOT/bin/python" ]] || { echo "Missing environment: $ENV_ROOT" >&2; exit 1; }
cd "$REPO_DIR"
source "$ENV_ROOT/bin/activate"
shopt -s nullglob
inputs=(results/*/matmul-*.csv results/*/reductions-*.csv)
(( ${#inputs[@]} > 0 )) || { echo "No measurement CSVs found" >&2; exit 1; }

python scripts/run_transfer_experiments.py "${inputs[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --source-token A100 \
    --target-token H100 \
    --few-shot-fractions 0,0.01,0.05,0.10,0.25 \
    --budgets 1,3,5,10,25,50 \
    --random-repeats 20 \
    --device cpu

echo "Transfer experiment artifacts: $OUTPUT_DIR"
