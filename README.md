# GPU kernel autotuning

Start in [collection/](collection/README.md) to measure kernels on A100, H100,
RTX 3070, RTX A5000, or T4. The five device entry points use a common CUDA-event
timer and a shared grid of workload shapes and configurations.

The original transfer experiment and simulator submodules live in [legacy/](legacy/README.md).
