# SSV4D Research Notes

## Starting Point

The closest recent work is [MarginGate (arXiv:2605.30218)](https://arxiv.org/abs/2605.30218).
Its useful observations for SSV4D are:

1. Batch-shape-induced token flips are sparse, roughly 0.3%-1.3% in the paper's
   reported BF16 regimes.
2. Risky flips are near-tie events and are exposed by a small top-1/top-2
   margin.
3. K/V drift before the first token divergence is small, so repair can target
   the current K/V column.
4. A calibrated selective verifier can preserve sequence determinism while
   invoking verification less often than always-on LLM-42.

SSV4D asks a different question: can a *shadow branch over perturbed prefill
batches* provide an additional signal beyond the current-token margin?

## Hypotheses

- **H1, sparsity:** perturbing batch size, protected-prompt position, and filler
  order exposes a small number of token flips rather than broad trajectory
  drift.
- **H2, locality:** flips concentrate at a few decode positions and should map
  to a small number of verification windows.
- **H3, margin signal:** low top-1/top-2 margins should have high discriminative
  power for token flips.
- **H4, shadow complement:** disagreement among perturbed draft trajectories
  should add recall or reduce verifier budget relative to margin-only gating.

## Implemented Prototype

Code and experiment artifacts are unified in `ssv4d/`.

- `perturbation.py` constructs protected-prompt batches of size 1/4/8/16,
  placing the protected prompt first/middle/last and rotating filler order.
- `client.py` runs one native SGLang `/generate` request per perturbation plan
  and records output IDs, top-k logprobs, top-1/top-2 margins, and latency.
- `metrics.py` computes token sparsity, window features, top-1 vote
  disagreement, rank-aligned top-k JSD, and feature AUC.
- `policy.py` performs leave-one-prompt-out calibration and compares always-on,
  margin-only, shadow-only, and margin-or-shadow gates.
- `llm42_hook.py` installs a fail-open per-request verification gate on a live
  `LLM42Worker` without changing the draft decode path.
- `run_experiment.py` orchestrates the reference/shadow branches and writes all
  logs, raw traces, CSVs, figures, and the generated report.

The current prototype compares shadow fast-path generations against an LLM-42
deterministic reference for measurement. The deployable gate uses only the
canonical draft token margin and cross-shadow disagreement.

## Final Experiment

Configuration:

- Model: public `NousResearch/Meta-Llama-3.1-8B-Instruct` mirror.
- 16 built-in prompts across factual, reasoning, coding, science, language,
  statistics, and planning categories.
- 128 output tokens per request.
- 19 perturbation plans per prompt (size 1 plus 2 repeats of 3 sizes x 3
  positions).
- 304 shadow traces.
- Calibration: leave-one-prompt-out grouped cross-validation.
- Target window/token recall: 100%.

Raw data is in `results/run_20261112_135628/`.

## Results

Sparse errors:

- 24,966 synchronous decode steps.
- 152 token flip events, a 0.61% flip rate.
- 8 of 46 verification windows contain at least one flip.
- Flip positions are clustered: 0, 1, 24, 28, 36, 38, 63, and 100.

Window-level gate:

| Policy | Verification-window budget | Window recall | Prompt determinism |
|---|---:|---:|---:|
| Always-on LLM-42 | 100.00% | 100.00% | 100.00% |
| Margin-only | 73.91% | 100.00% | 100.00% |
| Shadow-only | 97.83% | 87.50% | 93.75% |
| SSV4D | 73.91% | 100.00% | 100.00% |

Token-level selective verification:

| Policy | Verify-token budget | Token recall | Prompt determinism |
|---|---:|---:|---:|
| Always-on | 100.00% | 100.00% | 100.00% |
| Margin-only | 4.16% | 100.00% | 100.00% |
| Shadow-only | 100.00% | 100.00% | 100.00% |
| SSV4D | 4.16% | 100.00% | 100.00% |

Feature AUC:

- Negative min-margin: 0.9855.
- Shadow top-1 vote disagreement: 0.5000.
- Rank-aligned shadow JSD: 0.3525.

## Findings

H1 and H2 are supported on this workload: flips are sparse and localized.

H3 is strongly supported. The margin is close to an oracle ranking, and a token
budget of 4.16% recovers every observed flip under grouped cross-validation.

H4 is not supported by this first experiment. The shadow disagreement features
have near-random or inverted ranking value. The calibrated SSV4D gate therefore
rejects the shadow feature and falls back to margin-only verification. This is
a useful negative result: under the current batch perturbations, cheap logit
margin dominates cross-shadow consensus for deciding where to verify.

## Why Shadow Disagreement May Be Weak Here

1. Many protected prompts flip in all shadow variants together. Cross-shadow
   vote disagreement is zero even though every perturbed draft differs from the
   deterministic verifier. The margin still exposes the near tie.
2. Rank-aligned JSD compares top-k probability vectors by rank because token
   identities are not vocabulary-aligned. Stable trajectories can show small
   float noise while a true flip still has almost identical ranked mass.
3. The public-API prototype does not yet access canonical fast-path logits from
   the exact scheduler batch. The next iteration should capture the canonical
   draft and perturbed path inside the scheduler before tokenization/HTTP.

## Recommended Next Experiments

1. Implement an in-scheduler `ShadowBranch` that clones the protected request
   with altered prefill batch metadata and runs beside the canonical draft.
2. Compare vocabulary-aligned top-set disagreement, KL/JSD over shared top-k
   token IDs, canonical-vs-shadow margin delta, and first-layer K/V deviation.
3. Calibrate two policies: a high-recall margin gate and a shadow gate used only
   when it passes a held-out AUC/coverage test.
4. Add a runtime budget optimizer with explicit `(determinism, verifier rate)`
   constraints and integrate a window-level callback into
   `LLM42Worker.check_and_verify_deterministic_requests`.
5. Move from rollback-window verification to selected-token verification plus
   current-column K/V repair, matching the selective-verification objective.

## Current Recommendation

Use margin as the primary deterministic-verification threshold. Keep the shadow
branch as a diagnostic and as a candidate feature source, but enable it only
when calibration demonstrates incremental coverage over margin-only. Do not
merge shadow disagreement into the production gate based on the current data.

## Runtime Follow-up

The next-stage worker implementation is documented in
`runtime_prefill_report.md`.

- Worker-level top-k instrumentation now reports vocabulary-aligned KL/JSD and
  margin deltas.
- The verifier now records final-layer current-column K/V deviation.
- A bootstrap window plus margin gate preserves all 16 outputs in the runtime
  workload while reducing verification windows from 77 to 68.
- Real `ModelRunner` prefill probes show batch-size-dependent first-token
  distribution and K/V changes before decode starts.

## Dual-GPU Async Follow-up

`async_dual_gpu.py` now runs fast draft and deterministic verification on
separate GPUs. The client starts the verifier as soon as the first 64 streamed
tokens expose a low-margin window and continues consuming the 130-token draft.

Across three warm rounds of eight concurrent prompts:

- Fast draft only: 1.160 s, 896.86 output tok/s.
- Deterministic verifier only, full 130 tokens: 1.334 s, 779.86 tok/s.
- Async 64-token verification: 1.223 s, 850.59 tok/s.

Async verification hides 0.564 s of a 0.673 s verifier call, but is still 5.4%
slower than draft only. It is 8.3% faster than full deterministic generation
while covering only the verified prefix. The two repeated mismatches occur at
positions 31-63 and 63, respectively, and the deterministic verifier is stable
across all rounds.

This confirms service-level GPU overlap, but not end-to-end speculative
decoding. Full verified-prefix commit, repeated window verification, LLM-42
always-on comparison, and scheduler-native K/V sharing remain necessary.

## Dynamic Threshold Update

The gate no longer needs a configured margin threshold. For every token position
it computes

```text
tau(t) = max_shadow |margin(canonical, t) - margin(shadow, t)|
```

and verifies when the canonical margin is no larger than this perturbation-
induced uncertainty, or when top-1/sampled tokens disagree.

Across three audited rounds, three shadow plans selected 15/24 windows and
covered 6/6 observed mismatches. A single `bs8_middle` perturbation produced
the same 15/24 window decisions while reducing mean gate latency from 0.97 s to
0.72 s.

The current service-level shadow path is still slower than fixed-gate async:
one shadow plus dynamic gating took 1.99 s versus 1.22 s for fixed-gate async.
The next step is a scheduler-native shadow request in the same fast-draft batch,
with shared K/V allocation, rather than additional HTTP batch requests.

## Scheduler-Native Replay Shadow

The first in-worker implementation batches all protected requests into one
isolated replay extend at the first 64-token window. It uses fresh request and
K/V slots, so it does not overwrite canonical state.

Over three fresh rounds, the dynamic gate selected 42/48 first windows,
covered all 15 verifier rollback requests, and missed none. Across four warm
rounds, gating reduced total verification windows from 77 to 75 and preserved
all 16 output digests.

The result is still not a throughput win: dynamic gated time was 2.5769 s
versus 2.2865 s for always-on LLM-42. The shadow pass and allocation disrupt the
critical path by more than the removed verifier work. This triggers the planned
go/no-go branch: improve selectivity substantially, or position SSV4D as a
verification predictor rather than a performance optimization.

## Route 2 Multi-Window Diagnostic

The diagnostic used 48 prompts, up to three full verification windows, and
always-verifier labels. The combined multi-window sample contains 580
request-windows and 200 positives.

- Canonical-margin AUC: 0.677.
- Shadow-uncertainty AUC: 0.469.
- Margin-within-uncertainty AUC: 0.648.
- Current dynamic gate recall: 97.5% at an 86.7% trigger budget.
- Best global 100%-recall budget: 94.3%.
- Leave-one-prompt-out calibration does not reach 100% recall.

Separability drops in the second and third windows. The current perturbed
replay signal does not support Route 1 same-forward engineering, and it is not
yet competitive even as a standalone verification predictor.

## Adaptive Window Follow-up

An adaptive window prototype now checks risky prefixes at 16, 32, 48, and 64
tokens. Verification keeps the fixed 64-token shape by padding each request,
and `llm42_verified_tokens` advances only by the actual predicted length.

On 16 prompts and 70 output tokens:

- Fixed 64: 1.4930 s, 750.19 tok/s, 47 verification windows, 184 rollback tokens.
- Adaptive: 1.5530 s, 721.25 tok/s, 67 verification windows, 55 rollback tokens.
- All 16 output digests match.

Early verification reduced rollback damage by 70% but increased verifier work
by 43%. Adaptive length is feasible, but not yet a throughput improvement.

## Dual Fast-Decode Follow-up

Canonical and perturbed fast decoders now run as separate server instances on
GPU0 and GPU1. The perturbed request uses an eight-prompt batch with the target
at the middle position. A deterministic verifier shares GPU1 and starts only
after the SSV4D gate selects the request.

Across three rounds:

- Gate selected 15/24 requests.
- All 6 first-window mismatches were selected; no mismatches were skipped.
- Gated time was 7.664 s per round versus 9.446 s for verify-all audit.
- Canonical, perturbed, and verifier latencies were 0.469, 0.563, and 0.611 s.

The topology demonstrates a 100%-recall selective verifier with 62.5% verifier
traffic. It is 18.9% faster than verify-all, but direct deterministic
verification remains faster for a 64-token-only workload.

## Long Canonical Stream Follow-up

The canonical fast branch now streams 130 tokens instead of stopping at the
64-token gate input. The gate and verifier run while canonical continues:

- Canonical duration: 0.9172 s.
- Perturbed fast duration: 0.6107 s.
- Selected verifier duration: 0.6049 s.
- Verifier overlap with canonical tail: 0.3238 s.
- Additional wait after canonical completion: 0.2814 s.

The gate still selected 15/24 requests and covered all six observed first-window
mismatches. GPU0 does not wait for verification; only final commit has a logical
verifier dependency. The full 130-token output is not yet verified, since this
probe verifies only the first 64 tokens.

## Large-Scale Baseline Matrix

The 32-prompt, three-round matrix makes the tradeoff explicit:

- Canonical-only: 949.70 tok/s, 52.1% full-sequence match, 29/32 repeat-stable.
- Deterministic verifier-only: 461.55 tok/s, 100% sequence match.
- Gated: 424.55 tok/s, 53.1% full-sequence match, 81/96 verifier calls.
- Audit: 502.34 tok/s, 53.1% full-sequence match.

The gate selected 81 requests, with 30 true positives, 51 false positives, and
3 false negatives against audit labels. First-window recall dropped to 90.9%.
The current gated path is stable across repeated runs but is not equal to the
deterministic baseline and is 8% slower than direct deterministic verification.

## Duplicate-Prompt Self-Shadow

The perturbed fast server can be removed by placing eight or sixteen identical
copies of the protected prompt in one fast batch. Position zero is canonical;
other positions are self-shadows.

An eight-prompt pilot achieved 100% first-window recall, but the 32-prompt study
did not generalize:

- Batch 8 recall: 77.1%; fast batch latency 0.782 s.
- Batch 16 recall: 78.8%; fast batch latency 1.069 s.

Self-shadowing is cheaper than a second fast instance, but duplicate prompts
alone are not a reliable deterministic gate.

## Shuffled Mixed Self-Shadow Batch

Eight logical prompts were each duplicated eight times and interleaved into one
shuffled 64-row fast batch.

- Per-prompt batch 8: recall 77.1%, fast latency 0.782 s.
- Per-prompt batch 16: recall 78.8%, fast latency 1.069 s.
- Shuffled mixed batch 64: recall 70.3%, fast latency 0.883 s.

The shuffled layout was both slower and less predictive than per-prompt batch
8. SGLang's scheduler already mixes concurrent requests, so explicit shuffling
adds numerical order noise without a corresponding improvement in rollback
prediction.

## Full Verified Frontier

Strict frontier tracking now records every verification and final commit. On 32
prompts and three rounds:

- Always-on: 96/96 fully verified, 0 frontier violations, 32/32 outputs match.
- Gate threshold 0.0625: 96/96 mechanically fully verified, 0 frontier
  violations, but only 28/32 outputs match always-on.

The frontier correctly prevents partially unverified requests from finishing,
but skipped windows are considered accepted. A mechanically full frontier does
not fix gate false negatives.

The native LLM-42 baseline and SSV4D always-on path matched all 32 outputs and
both used 171 verification windows. SSV4D instrumentation added 3.99% latency,
reducing throughput from 1563.30 to 1505.45 tok/s. This quantifies the current
profiling/logprob/KV-probe overhead before gate savings are considered.

## Lightweight Gate Models

Logistic regression and a `15 -> 32 -> 16 -> 1` MLP were evaluated over 580
audit windows with prompt-grouped folds and three split seeds.

- Margin-only AUC: 0.677.
- Canonical logistic AUC: 0.661.
- Canonical MLP AUC: 0.647.
- Adding shadow features reduced AUC.

The MLP's apparent 89.6% full-recall trigger budget relies on an oracle
out-of-fold threshold. With grouped calibration it reaches only 97.2% recall.
No lightweight model is ready for runtime integration.

## Safe Adaptive Window

The gate question was reframed from "whether to verify" to "when to verify".
A logistic early-rollback model selects 16/32/48 checkpoints, while 64 and
finished tails are always verified.

At threshold 0.1:

- Outputs: 32/32 equal to native LLM-42.
- Rollback tokens: 787 -> 170.
- Verification windows: 171 -> 218.
- Throughput: 1563.30 -> 1466.76 tok/s.

This preserves correctness while reducing rollback damage, but remains slower
because verification is still inline and globally serialized. The next step is
non-blocking verification scheduling with the same frontier.

## Non-Blocking Frontier Prototype

Canonical fast decode now streams on GPU0 while deterministic verification
advances a frontier on GPU1. Matching windows commit speculative tokens;
mismatches commit the common prefix and regenerate the deterministic tail.

Across 32 prompts and three rounds:

- canonical stream: 0.936 s;
- first verifier start: 0.479 s;
- verifier/canonical overlap: 0.457 s;
- commit: 1.972 s;
- full frontier: 96/96 requests.

The mechanism works, but remains service-level. It replays full prompt prefixes
and does not reuse scheduler K/V state, so it is not yet a throughput win.
