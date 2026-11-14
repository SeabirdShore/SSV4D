# Safe Adaptive Verification Window

This experiment implements the first part of the new direction:

> Predict **when to verify**, not whether to verify.

Every token is still verified eventually. A lightweight model only decides
whether to verify early at checkpoints 16, 32, or 48. The 64-token checkpoint
and finished partial tails are always verified. A model false negative can
therefore affect latency and rollback cost, but cannot skip verification
forever.

## Early-Rollback Model

The model predicts whether the current window's first rollback occurs before a
checkpoint. It uses canonical logprob features:

- minimum, q10, and mean margin
- margin standard deviation
- fraction of low-margin tokens
- position of minimum margin
- window index

Grouped cross-validation:

| Checkpoint | AUC | Early-rollback recall | Trigger rate |
|---:|---:|---:|---:|
| 16 | 0.826 | 100% | 54.5% |
| 32 | 0.771 | 100% | 93.6% |
| 48 | 0.704 | 100% | 94.3% |

The 16-token checkpoint is useful; later checkpoints trigger too frequently.

## Runtime Results

32 extended prompts, three rounds, 130 tokens:

| Threshold | Warm round | Output tok/s | Verification windows | Rollback tokens | Outputs equal baseline |
|---|---:|---:|---:|---:|---:|
| LLM-42 baseline | 2.6610 s | 1563.30 | 171 | 787 | 32/32 |
| Safe adaptive 0.1 | 2.8362 s | 1466.76 | 218 | 170 | 32/32 |
| Safe adaptive 0.2 | 2.8640 s | 1452.50 | 207 | 170 | 32/32 |
| Safe adaptive 0.3 | 3.1547 s | 1318.68 | 181 | 246 | 32/32 |

Safe adaptive threshold 0.1:

- Rollback tokens fall by 78.4%.
- Verification windows rise by 27.5%.
- Throughput falls by 6.2%.

Threshold 0.2 has the same rollback-token count with 21.1% more verification
windows and 7.1% lower throughput.

Threshold 0.3 reduces verification-window growth to 5.8%, but rollback savings
fall to 68.7% and throughput falls by 15.7%. The non-monotonic wall-time result
indicates that verification batch composition and scheduling dominate raw call
count.

## 1024-Token Long-Output Recheck

The same configurations were rerun with 32 extended prompts, three rounds, and
`max_new_tokens=1024`. Round 1:

| Configuration | Total time | Output tok/s | Verification windows | Rollback tokens | Outputs equal native |
|---|---:|---:|---:|---:|---:|
| Native LLM-42 | 12.7363 s | 2572.81 | 716 | 3130 | 32/32 |
| SSV4D always-on | 14.6367 s | 2238.76 | 716 | 3130 | 32/32 |
| Safe adaptive 0.1 | 20.6083 s | 1590.04 | 1786 | 688 | 32/32 |
| Safe adaptive 0.2 | 27.4284 s | 1194.67 | 1799 | 761 | 32/32 |
| Safe adaptive 0.3 | 26.7422 s | 1225.33 | 1707 | 845 | 32/32 |

Safe adaptive 0.1 still cuts rollback tokens by 78.02%, but verification
windows grow by 149.44% and throughput falls by 38.20% relative to native.
The frontier audit reports 96/96 fully verified finished requests and no
monotonicity violations.

The long-output result is a clear negative scaling result. Of 5358 verified
request-window transitions at threshold 0.1, 4845 (90.43%) use checkpoint 16.
The predictor was calibrated on shorter windows, so extra early decisions
dominate the reduced rollback work. The 0.2 and 0.3 absolute timings also show
high run-to-run/scheduling variance and are not a stable ranking.

See `long_output_1024_report.md` for the full experiment.

Independent 16/32/48/64-token fixed pools were also implemented. They preserve
mode-2 outputs but do not improve throughput, because one verifier call costs
about 32-35 ms regardless of 16 or 64 tokens at batch size 8. See
`multi_shape_verification_report.md`.

## Findings

The safe formulation works:

- Model errors do not violate deterministic correctness.
- All 32 outputs match native LLM-42.
- Strict frontier checks show no observed unverified finished request.
- Earlier verification substantially reduces rollback work.

The performance result is negative:

- Verification calls increase faster than rollback work decreases.
- The current verifier remains inline and globally serialized.
- Extra checkpoint decisions and smaller verification batches reduce GPU
  efficiency.

## Next Step

The remaining half of the new direction is required:

- move verification off the global scheduler critical path;
- batch/delay verification work adaptively;
- isolate verification on a separate CUDA stream or SM subset;
- maintain the same verified frontier and commit barrier.

Without non-blocking scheduling, adaptive windows reduce rollback damage but
cannot improve end-to-end throughput.

## Artifacts

- `safe_window.py`
- `train_safe_window_model.py`
- `results/safe_window_model.json`
- `results/safe_window_model_report.json`
- `results/safe_adaptive_window.json`
- `results/safe_adaptive_window_0p2.json`
- `results/safe_adaptive_window_0p3.json`
- `results/safe_adaptive_window_1024_window_slice.json`
- `results/safe_adaptive_window_0p2_1024_window_slice.json`
- `results/safe_adaptive_window_0p3_1024_window_slice.json`
- `logs/safe_adaptive_window.jsonl`
- `logs/safe_adaptive_window_0p2.jsonl`
- `logs/safe_adaptive_window_0p3.jsonl`
- `scripts/run_safe_adaptive_window.sh`
