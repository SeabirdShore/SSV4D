# Self-Shadow Gate Without a Second Fast Server

This experiment tests whether the perturbed fast path can be removed by
duplicating the protected prompt inside the GPU0 fast batch.

## Topology

```text
GPU 0 fast server
  one request containing N identical prompts
  [prompt, prompt, ..., prompt]
      position 0 = canonical
      positions 1..N-1 = self-shadows

GPU 1 deterministic verifier
  launched only if the gate selects the request
```

No perturbed fast server runs on GPU1.

The gate compares position 0 against all other duplicate positions using the
same dynamic threshold:

```text
tau(t) = max_shadow |margin(position_0, t) - margin(position_j, t)|
```

## Small Pilot

Eight standard prompts, batch size 8:

- Fast batch latency: 0.558 s.
- Verifier latency: 0.649 s.
- Gate selected 5/8 requests.
- Both first-window mismatches were selected.
- All three skipped requests were stable in forced audit.
- Gate recall: 100%.

This showed that duplicate requests in one batch create measurable
batch-position numerical differences without a second fast server.

## Large-Scale Results

32 extended prompts, three rounds, prompt concurrency 8:

| Configuration | Fast batch latency | Gate selected | First-window recall | Round time |
|---|---:|---:|---:|---:|
| Batch 8 gated | 0.708 s | 65/96 | — | 5.193 s |
| Batch 8 forced audit | 0.782 s | 67/96 | 77.1% | 6.383 s |
| Batch 16 forced audit | 1.069 s | 64/96 | 78.8% | 7.338 s |

Batch 8 audit confusion matrix:

- True positives: 27
- False positives: 40
- False negatives: 8
- True negatives: 21

Batch 16 audit confusion matrix:

- True positives: 26
- False positives: 38
- False negatives: 7
- True negatives: 25

Increasing self-shadow batch size from 8 to 16 improves recall only from 77.1%
to 78.8%, while fast batch latency increases from 0.782 s to 1.069 s.

## Throughput

For batch size 8:

- Gated throughput: `32 * 64 / 5.193 = 394.4 tok/s`.
- Forced-verifier audit throughput: `32 * 64 / 6.383 = 320.8 tok/s`.

The gate saves about 18.6% wall time relative to verifying every first window,
but it still misses eight rollback windows in the larger test.

Removing the GPU1 perturbed fast server is a real systems improvement: one fast
batch replaces the previous canonical request plus separate perturbed batch.
However, duplicate prompts alone are not a reliable predictor across the larger
prompt suite.

## Interpretation

The idea is technically valid:

- Duplicate prompts change effective batch size and row position.
- Numerically different outputs are observable within one fast batch.
- No second fast server or second model process is required.

The limitation is predictive quality, not implementation cost. Self-shadow
disagreement fails to cover roughly 22% of first-window mismatches. A larger
duplicate batch only gives a marginal recall improvement and adds substantial
latency.

This topology is therefore useful as a low-cost batch-shape diagnostic, but not
yet safe as the sole gate for deterministic output.

## Artifacts

- `self_shadow_gate.py`
- `scripts/run_self_shadow_gate.sh`
- `results/self_shadow_gate.json`
- `results/self_shadow_gate_audit.json`
- `results/self_shadow_32.json`
- `results/self_shadow_32_audit.json`
- `results/self_shadow_32_b16_audit.json`
- `logs/self_shadow_32_fast_server.log`
- `logs/self_shadow_32_verifier_server.log`
