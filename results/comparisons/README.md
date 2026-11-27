# Five-GPU transfer results

Each tuner chooses configurations for a held-out GPU and workload shape. A choice succeeds if its latency on a separate timing pass is within 5% of the fastest measured candidate on that pass.

| Four measurements | Gaussian surrogate | Feedback disabled | Shape-conditioned shortlist |
| --- | ---: | ---: | ---: |
| CUDA graph replay | 87.9% | 81.2% | 80.7% |
| Individual kernel launch | 61.4% | 61.3% | 59.6% |

These are macro-averaged success rates. The tables retain full precision.

- [By workload and budget](independent-portfolio-summary.csv) includes both fixed shortlists and paired uncertainty intervals.
- [By GPU and workload](independent-portfolio-device-family.csv) breaks down the portfolio comparisons for each device.
- [By precision](independent-policy-by-precision-3pass.csv) compares Gaussian, feedback disabled, and geometry ridge at four measurements. FP16 covers five GPUs and BF16 covers four. Columns ending in `_pp` give differences in percentage points.

The [replication bundle](../../artifacts/workshop-artifact.zip) includes the full JSON reports under `benchmark/`, including the paired FP16–BF16 comparison restricted to the same four GPUs. After extraction, run `make portfolio-summary precision` to regenerate the tables and JSON from the saved traces.
