# Collect GPU timings

Run from the checkout root in the target GPU environment. All five routes call
[the same production timing worker](shared/timing_validation.py) with the
[same Triton kernels](shared/kernels/). The worker measures both a single
launch and a calibrated CUDA graph replay. The [candidate grid](grid/) supplies
30,000 configurations, including 15,000 FP16 configurations supported on T4.

| GPU | Command or entry point | Device-specific behavior |
|---|---|---|
| A100 | First `sbatch collection/a100/qualify.slurm`, then [production](a100/collect.slurm) | The separate clock qualification pilot must pass. Set `A100_PILOT_SUMMARY` to its `pilot.summary.json` before submitting production. |
| H100 | `H100_PASS_INDEX=0 sbatch collection/h100/collect.slurm` | Runs diagnostics and a smoke test on the Slurm allocation. Use pass indices 0, 1 and 2. |
| RTX 3070 | `python collection/rtx-3070/collect.py --output PATH` | Runs under WSL and checks desktop GPU activity before timing. |
| RTX A5000 | `python collection/rtx-a5000/collect.py --root "$PWD" --uuid GPU_UUID --pass-index 0 --output PATH` | Checks the physical GPU UUID and competing compute processes. Use a separate output path for each pass. |
| Colab T4 | [Initial pass](t4/collect.py) and [repeat passes](t4/repeat.py) | Supply `--output PATH --checkpoint-dir DRIVE_PATH`. Repeats take `--pass-index 1` or `2`. Use separate locations for each pass. |

TACC launchers expect the project’s Slurm allocation and scratch environment.
Use the Python entry points with `--help` to inspect their arguments without
starting a run. Start a fresh output directory when switching from a historical
collector: checkpoint manifests record the code and registry used for each run.
