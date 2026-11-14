# Multi-Shape Fixed Verification Pools

## Motivation

The previous Safe Adaptive run selected checkpoint 16 for most long-output
verification windows, but the fixed verifier still padded every request to
64 tokens. This experiment gives each checkpoint its own fixed-shape pool:

```text
batch=8, window=16
batch=8, window=32
batch=8, window=48
batch=8, window=64
```

Enable it with:

```bash
SSV4D_VERIFY_WINDOW_SIZES=16,32,48,64
```

The worker groups requests by the smallest fixed pool that can hold their
requested verification length. Underfilled batches are completed with
not-ready requests from the same shape when possible, then with pre-allocated
dummies.

## Correctness Constraint

Multi-shape verification is only enabled for LLM-42 modes 1 and 2, where
verification uses batch-invariant matmul. Mode 3 is fixed-shape deterministic
but not cross-shape deterministic.

A mode 3 experiment confirmed this distinction:

- 16/32/48/64-token verification was deterministic across rounds;
- only 7/32 outputs matched the native mode-3 baseline;
- 25/32 outputs changed because the verifier window changed the numerical
  reduction path.

The runtime now detects mode 3 and falls back to the single 64-token pool with
a warning. The raw mode-3 experiment is retained as negative evidence.

## Mode-2 Results

Configuration: 32 extended prompts, one 1024-token round, batch size 8.

| Configuration | Total time | Output tok/s | Verify windows | Rollback tokens | Outputs equal |
|---|---:|---:|---:|---:|---:|
| Fixed 64, always-on | 16.1810 s | 2025.09 | 709 | 2010 | 32/32 |
| Multi-shape Safe Adaptive | 26.6419 s | 1229.94 | 1611 | 388 | 32/32 |

The multi-shape candidate:

- preserves all 32 outputs against the mode-2 fixed-64 baseline;
- reaches the full 1024-token verified frontier;
- reduces rollback tokens by 80.7%;
- but increases verification windows by 127.2%;
- and reduces throughput by 39.3%.

An eight-prompt preliminary run showed that filling underfilled shapes with
not-ready requests improves the multi-shape runtime from 21.8165 s to
14.5534 s. The final 32-prompt run uses this coalescing.

## Why It Does Not Improve Throughput

In the final mode-2 run, mean verifier wall-time was almost independent of the
fixed window:

| Window | Verify calls | Mean requests/call | Mean verify wall-time |
|---:|---:|---:|---:|
| 16 | 66 | 4.82 | 33.92 ms |
| 32 | 11 | 1.18 | 31.60 ms |
| 48 | 10 | 1.00 | 33.65 ms |
| 64 | 20 | 1.65 | 34.71 ms |

For an 8B model with batch size 8, one verification forward is dominated by
weight traffic and per-layer launch overhead. Reducing 64 tokens to 16 only
saves roughly 2% of wall-time in the observed batches. The four-times smaller
window therefore does not pay for the extra calls created by the adaptive
checkpoint policy.

The primary bottleneck is no longer padding alone. Further work must reduce
the number of verification calls or overlap verification with decode, not just
reduce tokens per call.

## Implementation

- `llm42_worker.py` maintains `fixed_pools` keyed by verification window.
- Each pool has independent dummy tokens, KV slots, request slots, padding
  slots, and sampling tensors.
- Ready requests are selected into the smallest pool that can hold their
  requested length.
- Mode 3 automatically disables multi-shape verification.
- Scheduler accounting sums KV and request-pool reservations across all pools.

Files:

- `python/sglang/srt/llm42/llm42_worker.py`
- `python/sglang/srt/managers/scheduler.py`
- `scripts/run_multishape_verify.sh`

Run the 32-prompt mode-2 comparison:

```bash
./ssv4d/scripts/run_multishape_verify.sh 1024 32 1
```

## Artifacts

- `results/mode2_fixed64_1024_32.json`
- `results/mode2_multishape_1024_32.json`
- `results/mode2_multishape_1024_32_compare.json`
- `logs/mode2_fixed64_1024_32.jsonl`
- `logs/mode2_multishape_1024_32.jsonl`
- `results/safe_adaptive_window_1024_multishape.json`
- `logs/safe_adaptive_window_1024_multishape.jsonl`
