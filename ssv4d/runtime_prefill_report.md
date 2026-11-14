# SSV4D Runtime and Prefill Exploration

This report implements and evaluates the four next-stage directions proposed
after the first API-level shadow experiment.

## 1. Runtime Hook Inside LLM42Worker

The implementation is in `runtime.py` and `launch_runtime_server.py`.

The runtime wraps:

- `LLM42Worker.forward_batch_generation` for draft timing.
- `LLM42Worker.check_and_verify_deterministic_requests` for window-level
  accept/verify decisions.
- `LLM42Worker._run_verification` for verification timing and numerical probes.
- `LLM42Info.prepare_verify_batch` to request top-k logprobs for the verifier.

The gate uses a first-window bootstrap:

1. Every request's first verification window is always checked.
2. If that window rolls back, the request is marked risky and remains
   always-on for later windows.
3. If the first window is stable and the current window minimum margin is above
   the threshold, the window is accepted without verification.

This avoids treating high-margin but globally unstable requests as safe.

## 2. Vocabulary-Aligned Distribution Metrics

`features.py` aligns canonical and verifier top-k entries by token ID rather
than rank. Missing leading tokens receive a small probability floor, and the
union support is renormalized before computing:

- canonical-to-verifier KL
- verifier-to-canonical KL
- Jensen-Shannon divergence
- top-1 token agreement
- verifier-minus-canonical margin delta

The runtime collector aggregates these metrics per verification window.

## 3. K/V Column Deviation

Before verification, the runtime snapshots the target request's last written
K/V column at the selected layer. Verification overwrites that column with the
deterministic result. After verification, the runtime records:

- K/V L2 distance
- K/V relative L2 distance
- K/V cosine similarity

The default probe layer is the final transformer layer (`-1`). The layer list
is configurable through `SSV4D_KV_LAYERS`.

## 4. Window/Token Callback and Real Timing

The gate operates at LLM-42 verification-window granularity and records real
wall and CUDA-event timing. Accepted windows advance
`llm42_verified_tokens`, preserving the request's deterministic-set membership
and fixed-shape verification semantics.

## Runtime Environment

Start an instrumented server:

```bash
SSV4D_RUNTIME_ENABLE=1 \
SSV4D_GATE_ENABLE=0 \
SSV4D_VERIFY_TOP_K=8 \
SSV4D_KV_LAYERS=-1 \
SSV4D_RUNTIME_LOG=ssv4d/logs/runtime_always.jsonl \
python -m ssv4d.launch_runtime_server \
  --model-path NousResearch/Meta-Llama-3.1-8B-Instruct \
  --tp 1 --enable-llm42 3 --llm42-window-size 64 \
  --llm42-verify-batch-size 8 --attention-backend fa3 \
  --disable-radix-cache --disable-chunked-prefix-cache \
  --disable-overlap-schedule --chunked-prefill-size -1 --random-seed 42
```

For the gated run:

```bash
SSV4D_GATE_ENABLE=1 SSV4D_MARGIN_THRESHOLD=1.0 \
SSV4D_RUNTIME_LOG=ssv4d/logs/runtime_gated_bootstrap.jsonl \
python -m ssv4d.launch_runtime_server ...
```

Run the identical 16-prompt workload:

```bash
python -m ssv4d.runtime_benchmark \
  --num-prompts 16 --max-new-tokens 130 --rounds 2 \
  --output ssv4d/results/runtime_always.json
```

## Runtime Results

Stable second round, 16 concurrent deterministic requests:

| Configuration | Output tok/s | Total time | Verification windows | Outputs equal to always-on |
|---|---:|---:|---:|---:|
| Always-on LLM-42 | 752.41 | 2.764 s | 77 | 16/16 |
| Bootstrap margin gate | 835.24 | 2.490 s | 68 | 16/16 |

The gate reduced measured total time by about 9.9% and increased output
throughput by about 11.0%, while preserving all 16 output sequences. It skipped
18 request-window verification decisions across two rounds.

Both configurations include the same top-k and K/V instrumentation. Absolute
latency therefore includes probe overhead; the comparison isolates the effect
of the bootstrap gate rather than representing an uninstrumented production
number.

### In-worker prefill-shadow probe

`runtime.py` also supports a real prefill clone probe through
`SSV4D_PREFILL_SHADOW_ENABLE=1`. On the first gate decision for each request it
temporarily allocates isolated req/KV slots and runs:

- canonical prefill batch size 1
- shadow prefill batch size 8 with the protected prompt at position 7

It compares first-token KL/JSD and final-layer K/V columns. Across two rounds:

- 32 probes (16 requests per round)
- 6 prefill probes classified risky
- mean probe cost: 18.36 ms
- canonical/shadow prefill GPU time: 8.48/8.43 ms each
- maximum JSD: 0.0450
- maximum K/V relative L2: 0.90%/1.69%

The serial probe did not discover additional verification savings relative to
the bootstrap gate: it retained all 16 outputs but used the same 68 verification
windows. Its extra cost reduced output throughput from 835.24 to 744.99 tok/s
and increased total time from 2.490 to 2.792 seconds.

This is an important negative systems result. A genuinely useful prefill shadow
path must execute in parallel with canonical prefill, or share scheduler work,
so that its ~18 ms request-level sidecar cost does not sit on the critical path.

Runtime instrumentation observed:

### Verification timing

- Always-on: 28 verify events, 154 verified request-windows,
  mean verify GPU time 39.98 ms.
- Bootstrap gate: 26 verify events, 136 verified request-windows,
  mean verify GPU time 24.84 ms.

### Distribution divergence

- Mean canonical-verifier KL: 12.71
- Mean JSD: 0.479
- Mean top-1 agreement: 0.314

The final layer K/V column moved substantially on risky rollback windows:

- Mean K/V relative L2: 7.2% / 9.8%
- Mean maximum K/V relative L2: 30.9% / 40.6%
- A representative rollback window reached 87.0% K and 117.7% V relative
  deviation.

## Real Prefill Probe

`prefill_probe.py` loads the actual `ModelRunner`, constructs real
`ScheduleBatch` prefill batches, and clones the protected prompt into different
batch sizes and positions. It measures first-token logits and final-layer K/V
for the protected request.

Eight-prompt averages:

```bash
python -m ssv4d.prefill_probe \
  --num-prompts 8 --top-k 8 --kv-layer -1 \
  --output-dir ssv4d/results/prefill_probe
```

| Plan | Batch | Prefill ms | KL | JSD | Margin delta | K rel L2 | V rel L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bs1` | 1 | 27.36 | 0 | 0 | 0 | 0 | 0 |
| `bs4_first` | 4 | 21.91 | 4.72e-5 | 1.19e-5 | -0.0078 | 0.081% | 0.144% |
| `bs8_first` | 8 | 24.39 | 5.11e-4 | 1.28e-4 | -0.0313 | 0.299% | 0.554% |
| `bs8_middle` | 8 | 14.69 | 5.11e-4 | 1.28e-4 | -0.0313 | 0.299% | 0.554% |
| `bs8_last` | 8 | 13.99 | 5.11e-4 | 1.28e-4 | -0.0313 | 0.299% | 0.554% |
| `bs16_last` | 16 | 21.47 | 6.46e-4 | 1.61e-4 | -0.0156 | 0.766% | 1.428% |

Observations:

1. Real prefill batch size changes both next-token distribution and K/V columns.
2. In this workload batch size dominates; first/middle/last position produced
   identical numerical values at batch size 8.
3. Top-1 token remained stable at prefill, but lower-ranked probability mass and
   the K/V column moved measurably. Runtime verification is where top-1
   divergence becomes common.
4. Prefill timing is not monotonic in batch size because total batch tokens and
   kernel utilization differ.

## Interpretation

The strongest working configuration is **bootstrap verification plus margin
gating**:

- Bootstrap catches requests whose first window is already unstable.
- Later high-margin windows can skip verification.
- Vocab-aligned logprobs and K/V deviations are useful diagnostics.
- K/V movement is strongly associated with rollback windows, but it is measured
  during verification and cannot by itself gate that same verification pass.
- A true zero-cost shadow decision still requires a scheduler-native parallel
  prefill/decode clone that produces logits/K/V before the canonical window is
  accepted.

## Artifacts

- `logs/runtime_always.jsonl`
- `logs/runtime_gated_bootstrap.jsonl`
- `results/runtime_always.json`
- `results/runtime_gated_bootstrap.json`
- `results/runtime_compare_bootstrap.json`
- `results/prefill_probe/prefill_metrics.csv`
- `results/prefill_probe/summary.json`
