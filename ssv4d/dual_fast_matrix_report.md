# Large-Scale Dual-Fast Baseline Matrix

This benchmark compares four execution modes on the same prompt set and resident
server pair.

## Workload

- 32 prompts from `prompts_extended.py`
- 3 rounds
- Prompt concurrency: 8
- Canonical stream: 130 tokens
- Perturbed plan: `bs8_middle`
- Verification window: first 64 tokens
- Modes: canonical-only, deterministic verifier-only, gated, verify-all audit

The deterministic verifier-only outputs are the reference for sequence and token
agreement.

## Throughput and Determinism

| Mode | Output tok/s | Full-sequence match | First-64 match | Mean token agreement | Cross-round stable prompts | Verifier calls |
|---|---:|---:|---:|---:|---:|---:|
| Canonical-only | 949.70 | 52.1% | 61.5% | 0.721 | 29/32 | 0 |
| Deterministic verifier-only | 461.55 | 100.0% | 100.0% | 1.000 | 32/32 | 96/96 |
| SSV4D gated | 424.55 | 53.1% | 62.5% | 0.734 | 32/32 | 81/96 |
| Verify-all audit | 502.34 | 53.1% | 65.6% | 0.741 | 24/32 | 96/96 |

Important distinction:

- Cross-round stability means repeated runs produce the same output.
- Deterministic correctness means output matches the deterministic baseline.

Gated output is stable across repeats for all 32 prompts, but only 53.1% of full
sequences match the deterministic reference. The current implementation verifies
and reports the first 64 tokens but does not repair or verify the remaining 66
tokens.

## Baseline Comparison

Canonical-only is about 2.06x faster than the deterministic verifier baseline,
but only half of full sequences are deterministic.

Gated mode is about 8.0% slower than deterministic verifier-only:

```text
424.55 / 461.55 - 1 = -8.0%
```

The gate does not provide a speedup at this scale because it adds a second fast
decode instance and 81/96 requests still enter verification.

## Gate Audit

Across 96 request-rounds:

- Gate selected 81 requests.
- Audit observed 33 first-window mismatches.
- Comparing gated decisions with audit labels gives:
  - true positives: 30
  - false positives: 51
  - false negatives: 3
  - true negatives: 12
  - recall: 90.9%

Gate decisions were stable for 28/32 prompts across all three rounds. Four
prompts changed decisions under different concurrent scheduling.

The larger-scale result is weaker than the 8-prompt pilot, where gate recall was
100%. It also shows the current first-window verification is insufficient for
full-output determinism.

## Interpretation

The baseline matrix establishes two separate bottlenecks:

1. **Correctness:** verifying only the first 64 tokens does not make a
   130-token output deterministic.
2. **Performance:** running canonical and perturbed fast instances costs more
   than direct deterministic verification when most requests still need a
   verifier.

The pipeline needs a verified frontier spanning the full sequence before it can
claim deterministic output. It also needs substantially lower gate traffic to
offset the second fast instance.

## Artifacts

- `dual_fast_matrix.py`
- `scripts/run_dual_fast_matrix.sh`
- `results/dual_fast_matrix.json`
- `logs/dual_fast_matrix_canonical_server.log`
- `logs/dual_fast_matrix_shadow_server.log`
- `logs/dual_fast_matrix_verifier_server.log`
