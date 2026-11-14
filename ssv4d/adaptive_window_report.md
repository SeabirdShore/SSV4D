# Adaptive Verification Window

This experiment replaces the fixed 64-token verification trigger with
per-request adaptive checkpoints while keeping the verifier batch shape fixed.

## Design

- Checkpoints: `16, 32, 48, 64` unverified tokens.
- At each checkpoint, the worker replays a shadow prefix with isolated Req/K-V
  state.
- If the dynamic risk curve triggers, verification occurs at that checkpoint.
- If no risk is found before the next checkpoint, decoding continues.
- Finished requests always verify their partial tail.
- The deterministic verifier still has fixed shape. Actual lengths are padded
  to 64, while `LLM42Info.output_lens` advances only the verified prefix.

The first implementation used `first_risk_position + 1` as the verifier length.
That caused one-token verification thrash because risk was often present at
replay position zero. The final policy verifies at the first risky checkpoint.

## Results

Workload: 16 prompts, 70 output tokens, four warm rounds.

| Configuration | Mean time | Output tok/s | Verification windows | Rollback tokens |
|---|---:|---:|---:|---:|
| Fixed window 64 | 1.4930 s | 750.19 | 47 | 184 |
| Adaptive checkpoints | 1.5530 s | 721.25 | 67 | 55 |

Outputs:

- All 16 output digests match the fixed-window baseline.
- Cross-round outputs are deterministic.

Adaptive verification reduced rollback work by 70.1%, but increased verifier
request-windows by 42.6%. End-to-end time increased by about 4.0%, and throughput
decreased by about 3.9%.

## Actual Window Lengths

The adaptive-audit run produced:

| Predicted length | Count |
|---:|---:|
| 16 | 40 |
| 32 | 10 |
| 48 | 2 |
| Finished partial tails | 15 |
| **Total** | **67** |

Finished tails included lengths from 5 to 15. No full 64-token checkpoint was
needed because either an earlier checkpoint triggered or the request finished.

## Interpretation

Variable-length verification is technically feasible with a fixed-shape
verifier. Early verification reduced rollback damage substantially, but the
extra checkpoint replays and verifier invocations outweighed the saved decode.

The current signal triggers too frequently at 16 tokens. To make adaptive
windows a performance improvement, the predictor must distinguish:

- false-positive early risk that does not produce a rollback;
- true risk where verifying early saves rollback work;
- stable requests that can continue to a later checkpoint.

Until selectivity improves, fixed-window verification remains faster on this
workload.

## Artifacts

- `results/adaptive_fixed_baseline_5round.json`
- `results/adaptive_gated_fixed_5round.json`
- `results/adaptive_audit_fixed.json`
- `results/adaptive_window_compare.json`
- `logs/adaptive_fixed_baseline_5round.jsonl`
- `logs/adaptive_gated_fixed.jsonl`
- `logs/adaptive_audit_fixed.jsonl`
- `scripts/run_runtime_probe.sh adaptive`
- `scripts/run_runtime_probe.sh adaptive-audit`
