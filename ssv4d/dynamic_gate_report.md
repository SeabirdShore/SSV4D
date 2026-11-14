# SSV4D Dynamic Gate Threshold

This report replaces the fixed SSV4D margin threshold with a threshold computed
online from fast draft decodes running in parallel.

## Decision Rule

For decode position `t`:

- Let `m_c(t)` be the canonical fast-draft top-1/top-2 logprob margin.
- Let `m_j(t)` be the margin of perturbed fast draft plan `j`.

The dynamic threshold is

```text
tau(t) = max_j |m_c(t) - m_j(t)|
```

The gate requests verification when any of the following holds:

```text
m_c(t) <= tau(t)
top1_token(canonical, t) != top1_token(shadow_j, t)
sampled_token(canonical, t) != sampled_token(shadow_j, t)
```

A verification window is selected if any position satisfies one of these
conditions. The margin threshold is not configured in advance. It is recomputed
from the canonical and perturbed fast drafts for every token position.

The first predicate has a direct interpretation: if perturbation-induced margin
uncertainty is at least as large as the canonical separation, the canonical
top-1 choice is not demonstrably separated. Top-1 and sampled-token
disagreement are direct instability events.

Implementation: `ssv4d/dynamic_gate.py`.

## Parallel Layout

GPU 0 runs:

- one canonical fast draft stream;
- one or three perturbed fast drafts of the first verification window.

GPU 1 runs the deterministic verifier. It is launched asynchronously only when
the dynamic gate selects the request.

The perturbation plans alter batch size and protected-prompt position:

- single-plan experiment: `bs8_middle`;
- three-plan experiment: `bs4_first`, `bs8_middle`, `bs8_last`.

This is still a service-level prototype. Each shadow plan is an independent
batch request; it is not yet a scheduler-native cloned request sharing K/V.

## Offline Audit Results

Three rounds, eight prompts, 64-token first window:

| Configuration | Window triggers | Actual mismatches | Covered | Missed | Dynamic token rate |
|---|---:|---:|---:|---:|---:|
| Three shadow fast drafts | 15/24 | 6/24 | 6/6 | 0 | 22.27% |
| Three shadow, async audit | 15/24 | 6/24 | 6/6 | 0 | 9.70% |
| Single `bs8_middle`, async audit | 15/24 | 4/24 | 4/4 | 0 | 14.52% |
| Single `bs8_middle`, gated | 15/24 | 6 observed among verified | 6/6 | 0 | 15.56% |

The deterministic verifier confirmed that all skipped requests in the audited
runs had matching first-window tokens. Dynamic threshold values selected at the
first trigger were BF16-quantized values `0`, `0.0625`, and `0.125`, but these
were outputs of the formula rather than configured constants.

On the same workload, a fixed threshold of `1.0` selected all 24 request-
windows. A fixed threshold of `0.125` selected 18/24 with the same observed
recall. The dynamic gate selected 15/24.

## Runtime Cost

Three warm rounds:

| Configuration | Mean round time | Output tok/s | Mean gate latency |
|---|---:|---:|---:|
| Fixed-gate async, no shadow | 1.223 s | 850.59 | 0.545 s |
| Dynamic, one shadow, gated | 1.988 s | 523.25 | 0.720 s |
| Dynamic, one shadow, audit | 1.867 s | 557.15 | 0.687 s |
| Dynamic, three shadows, gated | 2.333 s | 445.74 | 0.966 s |
| Dynamic, three shadows, audit | 2.399 s | 433.54 | 1.134 s |

The dynamic gate detects observed first-window divergence, but the current
implementation does not reduce end-to-end time. Extra fast-draft batch requests
slow GPU 0 and move the gate decision later. Three shadow plans are especially
expensive. One plan produced the same 15/24 trigger decisions on this workload
and reduced gate latency by about 26%, so it is the better prototype default.

A scheduler-native implementation must keep the shadow branch in the same
fast-draft batch, share prefill/K/V allocation, and avoid an additional HTTP
request per perturbation. Until then, dynamic gating is a correctness and
selectivity result rather than a throughput win.

## Artifacts

- `dynamic_gate.py`: dynamic threshold and window decision logic.
- `dynamic_gate_probe.py`: canonical, perturbed fast drafts, and verifier
  launched concurrently.
- `analyze_dynamic_gate.py`: audit and budget summary.
- `results/dynamic_gate_probe_3round.json`
- `results/dynamic_async_gate_3round.json`
- `results/dynamic_async_gate_audit_3round.json`
- `results/dynamic_async_gate_single_3round.json`
- `results/dynamic_async_gate_single_audit_3round.json`
- `results/dynamic_gate_summary.json`
- `logs/dynamic_gate_probe_3round_fast_server.log`
- `logs/dynamic_gate_probe_3round_verifier_server.log`
- `scripts/run_dynamic_gate_probe.sh`
