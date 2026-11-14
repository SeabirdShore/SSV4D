# Two Fast-Decode Instances with SSV4D Gate

This experiment runs canonical and perturbed fast-decode instances separately,
then places the SSV4D gate after the perturbed branch.

## Topology

```text
GPU 0, canonical fast server
  original prompt -> 64-token fast trace
                                      \
                                       -> dynamic comparison -> SSV4D gate
                                      /                           |
GPU 1, perturbed fast server          |                            v
  prompt + seven fillers, middle slot |                  GPU 1 verifier
  bs8_middle -> 64-token fast trace   |                  deterministic 64
```

GPU 1 hosts both the perturbed fast server and deterministic verifier. The
servers use `mem-fraction-static=0.28` and are started sequentially to avoid a
simultaneous allocation race.

The gate compares vocabulary-aligned top-1 margins from the two fast traces:

```text
tau(t) = abs(margin(canonical, t) - margin(perturbed, t))
verify if canonical margin <= tau(t), top-1 tokens disagree, or sampled tokens
divergence is observed.
```

## Workload

- Model: `NousResearch/Meta-Llama-3.1-8B-Instruct`
- Prompt: first 8 built-in standard prompts
- Perturbation: batch size 8, protected prompt at middle index 3
- Window: 64 output tokens
- Rounds: 3
- Canonical and perturbed requests start concurrently on separate GPUs.
- Verifier starts after both fast traces return and the gate selects the request.

## Gate Audit

The forced-verifier audit produced:

- 24 request-rounds.
- 6 canonical/verifier mismatches in the first window.
- Gate selected 15/24 requests.
- All 6 mismatching requests were selected: recall 100%.
- 9 selected requests were stable in this sample.
- 9 skipped requests were all stable.

The gate itself takes about 2.5 ms of CPU time.

## Timing

| Configuration | Mean round time | Verifier calls |
|---|---:|---:|
| Gate + selected verifier | 7.6639 s | 15 |
| Audit: verifier for all requests | 9.4464 s | 24 |

Compared with verifying every request:

- Mean round time falls by 18.9%.
- Throughput-equivalent speedup is 1.23x.
- Three rounds produce the same gate decisions.

Mean per-request fast-decoder latency:

- Canonical instance: 0.469 s.
- Perturbed instance: 0.563 s.
- Selected verifier: 0.611 s.

## Important Limitation

The comparison above is against the audit policy that verifies all requests.
It is not faster than directly running the deterministic verifier:

```text
estimated direct verifier per round ~= 8 * 0.611 = 4.89 s
```

The dual-fast path costs about 7.66 s per round because it performs two fast
64-token generations before verifier selection. The current result demonstrates
100% recall and verifier-budget savings, but not an end-to-end win over direct
deterministic verification.

A full-sequence workload may change this conclusion: if the fast instances
generate substantially more tokens than the 64-token verifier window, the
verifier savings can amortize the second fast instance. That requires a
multi-window implementation rather than this first-window probe.

## Artifacts

- `dual_fast_gate.py`
- `scripts/run_dual_fast_gate.sh`
- `results/dual_fast_gate.json`
- `results/dual_fast_gate_audit.json`
- `results/dual_fast_gate_3round.json`
- `results/dual_fast_gate_audit_3round.json`
- `logs/dual_fast_gate_3round_canonical_server.log`
- `logs/dual_fast_gate_3round_shadow_server.log`
- `logs/dual_fast_gate_3round_verifier_server.log`
