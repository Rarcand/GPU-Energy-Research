# When Does Timing Feedback Help Budgeted GPU Kernel Autotuning?

The [workshop paper](paper/main.pdf) compares adaptive and fixed configuration
choices on five GPU environments across seven Triton workload families. After
four graph-replay measurements, the adaptive Gaussian surrogate achieved 87.9%
macro-averaged independent-pass success, compared with 80.7% for the
shape-conditioned fixed portfolio.

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
