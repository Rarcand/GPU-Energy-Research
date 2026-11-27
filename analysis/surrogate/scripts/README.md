# Surrogate analysis

Start with **replay** to follow the experiment. Each fold holds out one GPU and a range of workload shapes. The replay reveals saved timings as the tuner chooses configurations, then scores its choice on another pass.

| Directory | What to look for |
| --- | --- |
| [replay](replay/) | `validate_pass_winners.py` scores the adaptive tuner. `replay_source_portfolio.py` evaluates fixed shortlists. |
| [models](models/) | Surrogate fits, neural encoders, and ranking baselines. |
| [features](features/) | Measurement loading and kernel resource features. |
| [experiments](experiments/) | Transfer studies, candidate selection, and few-shot model comparisons used by the replay. |
| [reporting](reporting/) | Aggregate scores, uncertainty intervals, tables, and figures. |

From the extracted workshop artifact, run `make replay` for the full comparison or `make portfolio-summary` for the fixed-shortlist results from saved traces. To inspect a single command from the repository or artifact root, use e.g. `python -m analysis.surrogate.scripts.replay.validate_pass_winners --help`.
