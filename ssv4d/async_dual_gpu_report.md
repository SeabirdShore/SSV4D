# SSV4D Dual-GPU Async Draft/Verifier Experiment

This experiment tests whether SSV4D gating can overlap a fast, non-deterministic
draft stream on GPU 0 with deterministic verification on GPU 1.

The fixed `1.0` margin threshold used here is a baseline. The implemented SSV4D
dynamic threshold is documented in `dynamic_gate_report.md`; it is computed
from canonical and perturbed fast drafts at runtime.

## Scope and Architecture

The prototype starts two independent SGLang servers:

- GPU 0: fast draft server, `is_deterministic=false`.
- GPU 1: deterministic verifier server, `is_deterministic=true`.

For each request, the client reads streamed top-k logprobs from GPU 0. Once the
first 64 generated tokens are available, it computes the minimum top-1/top-2
margin over the window. If the margin is at most the gate threshold, it starts
the deterministic generation request on GPU 1 while GPU 0 continues decoding.
The verifier result is joined only after the draft stream completes.

This is a **service-level** overlap experiment. It does not yet modify the
LLM-42 scheduler or share K/V state between the two GPUs. In particular:

1. The verifier covers only the first 64 generated tokens.
2. The per-request verified prefix is compared but not committed into the
   complete draft output.
3. The two servers duplicate model weight loading and allocate independent K/V
   pools.

Timing includes HTTP, batching, and queueing. Model/server startup is excluded.

## Workload

- Model: `NousResearch/Meta-Llama-3.1-8B-Instruct`
- GPUs: 2 x H100 80GB HBM3
- Prompts: the first 8 built-in SSV4D prompts, launched concurrently
- Output length: 130 tokens
- Verification window: 64 tokens
- Margin threshold: 1.0
- Rounds: 3 consecutive warm rounds after server readiness

All three modes use the same prompts and concurrency. Each mode is measured with
its own freshly started server or server pair.

## Results

| Mode | Mean round time | Std. dev. | Output tok/s | Coverage |
|---|---:|---:|---:|---|
| Fast draft only | 1.1596 s | 0.1460 s | 896.86 | 130 tokens |
| Deterministic verifier only | 1.3336 s | 0.1758 s | 779.86 | 130 tokens |
| Async draft + first-window verifier | 1.2227 s | 0.1308 s | 850.59 | draft 130, verify 64 |

Per-request async timing over 24 request-rounds:

| Measurement | Mean |
|---|---:|
| Draft latency | 1.1114 s |
| Verifier latency | 0.6730 s |
| Draft/verifier overlap | 0.5637 s |
| Gate decision latency from request start | 0.5446 s |
| Commit latency | 1.2207 s |

The async path hides about 83.8% of verifier latency. Relative to the
deterministic verifier-only path, mean round time falls by 8.3% and output
throughput rises by 9.1%. However, it is still 5.4% slower than fast draft only,
and the verifier does not yet cover the complete output.

The result is therefore a systems feasibility result, not a complete
speculative-decoding speedup. The current safe baseline is already close to the
draft-only latency, so most of the theoretical overlap benefit is consumed by
the asynchronous verifier tail and duplicate serving overhead.

## Gate Behavior

Every request launched verification in all three rounds. The observed first-
window minimum margins ranged from 0.0 to 0.25, all below the threshold 1.0.
This workload therefore tests the overlap path, not selective verification
savings.

The first 64 draft and verifier tokens matched for 18 of 24 request-rounds.
The six mismatches are the same two prompts in every round:

| Prompt | Mismatch positions | Mismatch count |
|---|---|---:|
| `plan_weekend` | 31-63 | 29 |
| `science_photosynthesis` | 63 | 1 |

The deterministic verifier's 130-token outputs were stable across all eight
prompts and all three rounds. The repeated errors are localized within the
verification window, with one mismatch exactly at its final token. This supports
the sparse/latency-sensitive premise, but also shows why the verified result
must be committed rather than merely measured.

## Interpretation

GPU-asynchronous draft and verification is technically viable at the service
level. The client can issue verification while the draft continues, and the
verifier can return before the joined request commits.

The current experiment does not justify replacing always-on deterministic
verification. To claim a useful SSV4D speedup, the next implementation must:

1. Commit verifier output for the verified prefix and continue from the safe
   state.
2. Support repeated window-level verification rather than one prefix check.
3. Compare against LLM-42 always-on verification on the identical workload.
4. Remove duplicate model/K/V serving or implement scheduler-native overlap.
5. Evaluate workloads with many high-margin windows so the gate can skip enough
   verifier work to pay for the shadow path.

## Artifacts

- `results/async_dual_gpu_3round.json`
- `results/async_draft_only_3round.json`
- `results/async_verifier_only_130_3round.json`
- `results/async_dual_gpu.json`
- `results/async_dual_gpu_forced.json`
- `logs/async_dual_gpu_3round_draft_server.log`
- `logs/async_dual_gpu_3round_verifier_server.log`
- `logs/async_draft_only_3round_draft_server.log`
- `logs/async_verifier_only_130_3round_verifier_server.log`
- `scripts/run_async_dual_gpu.sh`
