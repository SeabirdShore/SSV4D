# Long Canonical Stream with Commit Barrier

This experiment extends the canonical fast decoder from a 64-token probe to a
130-token stream while keeping verification at the first 64-token window.

## Timeline

```text
t=0
├─ GPU0 canonical stream: 130 tokens, continues in background
├─ GPU1 perturbed fast:   64 tokens, bs8_middle
│
├─ canonical first-64 prefix ready
├─ perturbed 64-token trace ready
├─ SSV4D gate
│
├─ if risky: GPU1 verifier starts
│            GPU0 canonical stream continues
│
├─ canonical stream finishes
└─ commit = max(canonical finish, verifier finish)
```

The canonical request is not blocked at the gate. It continues decoding while
the gate and verifier run. Only the final commit has a logical dependency on the
verifier.

This first experiment still verifies only the first 64 tokens. The remaining
66 canonical tokens are generated speculatively and are not covered by this
verifier pass.

## Workload

- 8 standard prompts
- Canonical output: 130 tokens
- Perturbed branch: `bs8_middle`, 64 tokens
- Verifier: deterministic 64 tokens
- 3 gated rounds and 1 forced-verifier audit

## Results

Gate behavior:

- Gate selected 15/24 request-rounds.
- Forced audit found 6 canonical/verifier first-window mismatches.
- The gate covered all 6 mismatches.
- The 9 skipped requests were stable in the audit.
- Gate decisions were identical in all three rounds.

Timing over three gated rounds:

| Measurement | Mean |
|---|---:|
| Canonical 130-token stream | 0.9172 s |
| Perturbed 64-token fast branch | 0.6107 s |
| Gate-ready point | 0.6108 s |
| Selected verifier | 0.6049 s |
| Verifier/canonical-tail overlap | 0.3238 s |
| Wait after canonical finishes | 0.2814 s |
| Commit, selected requests | 1.2021 s |
| Commit, skipped requests | 0.9357 s |
| Mean round time | 8.8383 s |

Selected requests overlap about 54% of verifier execution with the remaining
canonical stream. Because verification is slightly longer than the canonical
tail, a selected request usually commits about 0.28 s after canonical reaches
130 tokens.

Compared with the earlier 64-token-only dual-fast experiment:

- Mean round time increases from 7.66 s to 8.84 s.
- Output tokens per round increase from 512 to 1040.
- Sequential effective throughput increases from about 66.8 to 117.7 tok/s.

## Interpretation

The requested asynchronous behavior is working:

- GPU0 canonical streaming does not wait for GPU1 verification.
- The verifier overlaps with the canonical tail.
- Commit waits only for the final verifier result.

However, this is still a service-level prototype:

1. Only the first 64 tokens are verified.
2. Full-output commit remains conditional on a verifier that checks only the
   first window.
3. GPU1 contains both the perturbed fast server and verifier; their requests are
   sequential in this probe rather than concurrently resident on the compute
   path.
4. The extra canonical stream and perturbed branch still duplicate fast decode.

The next step for a complete design is multi-window streaming: maintain a
verified frontier, continue canonical decode, verify subsequent windows, and
release only tokens up to the verified frontier.

## Artifacts

- `dual_fast_gate.py`
- `scripts/run_dual_fast_longstream.sh`
- `results/dual_fast_longstream.json`
- `results/dual_fast_longstream_audit.json`
- `results/dual_fast_longstream_3round.json`
- `logs/dual_fast_longstream_3round_canonical_server.log`
- `logs/dual_fast_longstream_3round_shadow_server.log`
- `logs/dual_fast_longstream_3round_verifier_server.log`
