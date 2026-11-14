# SSV4D 1024-Token Long-Output Evaluation

## Scope

This run repeats the deterministic verification comparison with
`max_new_tokens=1024`. It tests whether the conclusions from the
130-token workload survive when each request remains active for about eight
verification windows.

Common configuration:

- 32 extended prompts, three rounds.
- Two H100 80 GB GPUs; each configuration uses one GPU.
- `llm42-window-size=64`, `llm42-verify-batch-size=8`.
- Temperature 0, `ignore_eos=true`, seed 42.
- FA3 attention, no radix cache, no chunked prefix cache, no overlap scheduler.
- Every request generated exactly 1024 output tokens.

## Results

Warm round (round 1):

| Configuration | Total time | Output tok/s | Verify windows | Rollback tokens | Outputs equal |
|---|---:|---:|---:|---:|---:|
| Native LLM-42 | 12.7363 s | 2572.81 | 716 | 3130 | 32/32 |
| SSV4D always-on, window-slice fix | 14.6367 s | 2238.76 | 716 | 3130 | 32/32 |
| Safe Adaptive, threshold 0.1 | 20.6083 s | 1590.04 | 1786 | 688 | 32/32 |
| Safe Adaptive, threshold 0.2 | 27.4284 s | 1194.67 | 1799 | 761 | 32/32 |
| Safe Adaptive, threshold 0.3 | 26.7422 s | 1225.33 | 1707 | 845 | 32/32 |

Relative to native LLM-42, Safe Adaptive 0.1:

- reduces rollback tokens by 78.02%;
- increases verification windows by 149.44%;
- reduces output throughput by 38.20%.

Relative to instrumented always-on, Safe Adaptive 0.1 reduces throughput by
28.98%. The regression is therefore caused by the adaptive policy and verifier
scheduling, not solely by SSV4D instrumentation.

The 0.2 and 0.3 runs have unusually high wall-time variance compared with the
0.1 run. Their call counts are internally consistent, but the absolute times
should be treated as observed values rather than a stable threshold ranking.

## Correctness

All configurations are deterministic across the three rounds:

- 32/32 output digests match the native LLM-42 baseline.
- Every configuration finishes all 96 request-rounds at 1024 tokens.
- Frontier audit reports 96/96 fully verified final requests.
- No monotonicity violations were observed.

The detailed audit is in `results/frontier_audit_1024_window_slice.json`.

## Why Safe Adaptive Becomes Expensive

For threshold 0.1, the audit contains 5358 verified request-window transitions:

| Selected checkpoint | Count | Share |
|---:|---:|---:|
| 16 | 4845 | 90.43% |
| 32 | 132 | 2.46% |
| 48 | 69 | 1.29% |
| 64 | 216 | 4.03% |
| finished partial tail | 96 | 1.79% |

At short output lengths, early verification can avoid an expensive rollback.
At 1024 tokens, the predictor selects checkpoint 16 for about 90% of verified
windows. The resulting verifier-call growth dominates the 78% reduction in
rollback tokens. The model was trained on shorter windows and does not transfer
its trigger rate to long contexts.

This is a negative result for the current safe-window model as a throughput
mechanism. It remains valid as a correctness-preserving verification schedule.

## Instrumentation Fix

The first always-on 1024-token run reached 2049.93 tok/s, 20.32% below native.
The runtime recomputed window margins on every decode step by copying the full
logprob history to CPU, even though only the final 64 rows were needed.

`runtime.py` now slices tensors or lists before CPU conversion. The rerun
reached 2238.76 tok/s, reducing the instrumentation gap from 20.32% to 12.98%.
The pre-fix and post-fix artifacts are both retained.

The remaining gap means feature extraction and synchronous JSONL logging are
still on the request critical path. Any throughput claim for SSV4D must use the
post-fix runtime and report instrumentation cost separately.

## Non-Blocking Frontier Probe

An eight-prompt, standard-set probe with concurrency 4 also used 1024 tokens:

| Measurement | Mean |
|---|---:|
| Canonical stream finish | 7.169 s |
| First verifier start | 0.518 s |
| Verifier/canonical overlap | 6.651 s |
| Committed frontier finish | 35.574 s |
| Round time | 71.527 s |
| Output throughput | 114.53 tok/s |
| Fully verified requests | 8/8 |

All requests reached `frontier=1024`. Six of eight canonical outputs differed
from the deterministic committed output, and the verifier repaired those tails
from the common prefix.

This service-level prototype replays the complete prompt prefix for every
verification window. With 1024 tokens, repeated prefill dominates and the
throughput number is not comparable to a scheduler-native verifier. It verifies
the commit mechanism, not a performance advantage.

## Reproduction

```bash
./ssv4d/scripts/run_safe_adaptive_window.sh 1024 window_slice

python -m ssv4d.nonblocking_frontier_probe \
  --prompt-set standard \
  --num-prompts 8 \
  --rounds 1 \
  --prompt-concurrency 4 \
  --window 64 \
  --tail-merge-tokens 8 \
  --max-new-tokens 1024 \
  --output ssv4d/results/nonblocking_frontier_1024_small.json
```

## Artifacts

- `results/native_llm42_1024.json`
- `results/always_on_1024.json`
- `results/always_on_1024_window_slice.json`
- `results/safe_adaptive_window_1024.json`
- `results/safe_adaptive_window_1024_window_slice.json`
- `results/safe_adaptive_window_0p2_1024_window_slice.json`
- `results/safe_adaptive_window_0p3_1024_window_slice.json`
- `results/nonblocking_frontier_1024_small.json`
- `results/frontier_audit_1024_window_slice.json`
- `logs/*_1024*.jsonl`
