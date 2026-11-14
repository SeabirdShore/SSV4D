# Shuffled Shared-Batch Self-Shadow

This experiment interleaves copies of multiple prompts in one fast batch
instead of sending one duplicate-only batch per prompt.

## Batch Layout

Each wave contains eight logical prompts and eight copies of each prompt:

```text
64 entries =
  8 copies of P0
  8 copies of P1
  ...
  8 copies of P7
```

All 64 entries are shuffled with a deterministic seed before being submitted as
one fast `/generate` batch. For each prompt, the first occurrence in the
shuffled layout is treated as canonical and the remaining seven occurrences are
self-shadows.

With prompt concurrency 8, four waves cover 32 prompts. Each round therefore
submits 256 sequence requests in four 64-row mixed batches.

## Results

32 extended prompts, three rounds:

| Configuration | Fast latency | Gate selected | First-window recall | Round time |
|---|---:|---:|---:|---:|
| Shuffled 64-row gated | 0.902 s | 60/96 | — | 6.597 s |
| Shuffled 64-row audit | 0.883 s | 62/96 | 70.3% | 6.629 s |

Shuffled audit confusion matrix:

- True positives: 26
- False positives: 36
- False negatives: 11
- True negatives: 23

For comparison:

| Layout | Fast latency | Recall | Round time |
|---|---:|---:|---:|
| Per-prompt batch 8 | 0.782 s | 77.1% | 6.383 s |
| Per-prompt batch 16 | 1.069 s | 78.8% | 7.338 s |
| Shuffled mixed batch 64 | 0.883 s | 70.3% | 6.629 s |

The shuffled layout is slower than per-prompt batch 8 and has lower recall.
The gate saves almost no wall time because 60 selected verifiers still dominate
the wave.

## Interpretation

Explicitly mixing prompts in one HTTP batch does create a stronger, visibly
different batch composition. However, it does not improve correspondence with
the deterministic rollback labels.

SGLang already groups concurrently submitted requests dynamically. Sending eight
separate per-prompt batches under concurrency eight still gives the scheduler an
opportunity to form large mixed batches. The explicit shuffled layout therefore
adds ordering noise without adding a predictive signal aligned with
deterministic instability.

This negative result suggests that batch-position disagreement is not a
sufficient substitute for the more relevant semantic and logit signals.

## Artifacts

- `shuffled_self_shadow_gate.py`
- `scripts/run_shuffled_self_shadow_gate.sh`
- `results/shuffled_self_shadow_32.json`
- `results/shuffled_self_shadow_32_audit.json`
- `logs/shuffled_self_shadow_32_fast_server.log`
- `logs/shuffled_self_shadow_32_verifier_server.log`
