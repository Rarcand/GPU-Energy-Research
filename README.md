# When Does Timing Feedback Help Budgeted GPU Kernel Autotuning?

When both the GPU and the workload shape are new, how should an autotuner spend
its first few measurements? We train a Gaussian world model on other GPUs and
shapes. Each time the tuner measures a kernel configuration, it updates its
latency estimates for the remaining candidates and chooses what to try next.

The [workshop paper](paper/main.pdf) evaluates this strategy across five GPUs
and seven Triton kernel families. With four CUDA graph measurements, the tuner
selected a configuration within 5% of the fastest measured on a separate timing
pass in 87.9% of cases (macro average). A fixed shortlist learned from similar
shapes reached 80.7%. Timing feedback mattered much less for individual kernel
launches, where enabling it improved the same tuner's success rate by only
0.08 percentage points.

| Task | Start here |
|---|---|
| Measure GPU workloads | [GPU entry points, qualification and checkpoints](collection/README.md) |
| Inspect the timing protocol | [One shared production worker](collection/shared/timing_validation.py) and [Triton kernels](collection/shared/kernels/) |
| Inspect surrogate code | [Analysis scripts](analysis/surrogate/scripts/) |
| Review the data | [405,000 timing rows](results/datasets/gpu-tuning-five-device-current.zip), [comparison tables by workload, GPU, budget, and precision](results/comparisons/README.md) |
| Edit the paper | [LaTeX](paper/main.tex) or [Overleaf source](paper/workshop-draft-overleaf.zip) |
| Inspect the original simulator and transfer setup | [Legacy files](legacy/README.md) |

## Reproduce the portfolio comparison

Extract the [replication package](artifacts/workshop-artifact.zip) in a fresh directory.
The CPU replay uses Python 3.11 and the recorded measurements.

```bash
replay_dir=$(mktemp -d)
unzip -q artifacts/workshop-artifact.zip -d "$replay_dir"
cd "$replay_dir"
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
make portfolio PYTHON=.venv/bin/python
```

The replay writes estimates under `benchmark/` and the table and figure under
`paper/`. Use `make replay PYTHON=.venv/bin/python` for the full set of paper
comparisons.
