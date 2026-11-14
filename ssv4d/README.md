# SSV4D

Unified research code and experiment artifacts for **SSV4D: Shaded Speculative
Verification for LLM Determinism**.

## Layout

- `client.py`: native SGLang `/generate` client returning token IDs, top-k
  logprobs, margins, and timing.
- `perturbation.py`: batch-size, batch-order, and protected-prompt-position
  perturbations.
- `metrics.py`: token-flip sparsity, margin separation, shadow vote/JSD
  disagreement, window aggregation, and AUC.
- `policy.py`: always-on, margin-only, shadow-only, and margin-or-shadow gate
  calibration with leave-one-prompt-out grouped cross-validation.
- `llm42_hook.py`: fail-open LLM-42 verification gate adapter.
- `run_experiment.py`: end-to-end probe and report generator.
- `runtime.py`: live LLM-42 verification instrumentation and bootstrap gate.
- `async_dual_gpu.py`: fast draft on GPU 0, asynchronous deterministic verifier
  on GPU 1, plus draft-only and verifier-only baselines.
- `dynamic_gate.py`: reference-free dynamic margin threshold computed from
  canonical versus perturbed fast-draft margins.
- `dynamic_gate_probe.py`: parallel canonical, perturbed-fast-draft, and
  deterministic-verifier trace collection.
- `prompts_extended.py`: 48-prompt suite for multi-window generalization tests.
- `analyze_route2.py`: AUC, full-recall budget, grouped-CV, and window-level
  predictor analysis.
- `dual_fast_gate.py`: separate canonical/perturbed fast servers with SSV4D
  gate before a deterministic verifier.
- `runtime.py`: also contains the in-process batched window replay shadow used
  by the scheduler-native dynamic gate.
- `launch_runtime_server.py`: instrumented server launcher.
- `runtime_benchmark.py`: deterministic concurrent workload generator.
- `prefill_probe.py`: real ModelRunner prefill perturbation probe.
- `tests/`: unit tests.
- `scripts/run_ssv4d.sh`: default experiment launcher.
- `scripts/run_runtime_probe.sh`: always-on or bootstrap-gated runtime benchmark.
- `scripts/run_prefill_probe.sh`: real ModelRunner prefill perturbation probe.
- `scripts/run_async_dual_gpu.sh`: dual-GPU async and serial baseline matrix.
- `scripts/run_dynamic_gate_probe.sh`: dynamic-threshold trace collection.
- `scripts/run_route2_probe.sh`: multi-batch, multi-window predictor diagnostic.
- `logs/`: execution logs.
- `results/`: raw traces, features, gate metrics, figures, and per-run reports.
- `report.md`: latest generated research report.
- `research_notes.md`: related work, hypotheses, findings, and next steps.
- `runtime_prefill_report.md`: worker-level logprob/KV/timing results.
- `async_dual_gpu_report.md`: dual-GPU overlap experiment and limitations.
- `dynamic_gate_report.md`: dynamic threshold rule, audit, and runtime cost.
- `scheduler_dynamic_shadow_report.md`: in-process replay-shadow
  implementation and end-to-end results.
- `route2_multi_window_report.md`: Route 2 multi-window go/no-go diagnostic.
- `adaptive_window_report.md`: checkpointed variable-length verification
  experiment.
- `dual_fast_gate_report.md`: two-fast-instance gate topology and audit.
- `dual_fast_longstream_report.md`: 130-token canonical stream with verifier
  overlap and commit barrier.
- `dual_fast_matrix_report.md`: 32-prompt baseline matrix for determinism and
  throughput.
- `self_shadow_gate_report.md`: duplicate-prompt self-shadow gate on one fast
  server.
- `shuffled_self_shadow_report.md`: mixed 64-row shuffled self-shadow batch.
- `full_verified_frontier_report.md`: strict frontier and final-commit audit.
- `gate_model_ablation_report.md`: grouped-CV logistic/MLP gate experiment.
- `safe_adaptive_window_report.md`: early-rollback model that chooses when to
  verify while preserving the full frontier.
- `nonblocking_frontier_report.md`: two-GPU non-blocking verified frontier.
- `long_output_1024_report.md`: 1024-token correctness and throughput recheck.
- `multi_shape_verification_report.md`: fixed 16/32/48/64 verification pools.
- `trie_1024_report.md`: prefix-trie and top-k analysis at 1024 tokens.
- `trie_verification_scheme.md`: proposed bounded tree-verification design.
- `large_scale_trie_study.md`: 48-prompt multi-layout trie evaluation.
- `trie_frontier.py`: implemented trie/frontier/rollback state machine.
- `validate_trie_frontier.py`: replay validation for trie semantics.
- `python/sglang/srt/llm42/trie_spec.py`: flattened tree spec and causal mask.
- `gpu_tree_implementation_status.md`: implemented GPU-tree pieces and blockers.

## Environment

```bash
cd /home/xiaosibo/llm-42
source .venv/bin/activate
export LD_LIBRARY_PATH=/home/xiaosibo/.local/llm42-numa/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

Start an LLM-42 server:

```bash
cd sosp-artifact/0-basic
MODEL=NousResearch/Meta-Llama-3.1-8B-Instruct \
GPU_SHORT_NAME=h100_pcie \
ENABLE_TORCH_SYMM_MEM=0 \
DISABLE_CUSTOM_ALL_REDUCE=0 \
bash launch_server.sh
```

Run the experiment:

```bash
cd /home/xiaosibo/llm-42
python -m ssv4d.run_experiment
```

Run tests:

```bash
python -m unittest ssv4d.tests.test_core -v
```

## Latest Finding

The 16-prompt, 128-token experiment observed a 0.61% token flip rate.
Margin-only gating recovered all observed flips with a 73.91% window budget and
a 4.16% token budget. Shadow vote/JSD disagreement had AUC 0.5000/0.3525 versus
0.9855 for negative margin, so calibration rejected the shadow feature and fell
back to margin-only for this workload.

The worker-level follow-up adds a bootstrap verification window and dynamic
margin gate. It preserved all 16 runtime outputs while reducing verification
windows from 77 to 68 and increasing stable-round throughput from 752 to 835
tok/s. A serial in-worker prefill-shadow probe was correct but slower, so the
next implementation step was a parallel path.

The dual-GPU prototype now overlaps a 64-token deterministic verifier with a
130-token fast draft stream. Across three warm rounds, async mean round time was
1.223 s versus 1.334 s for deterministic verifier-only, an 8.3% reduction, but
it was 5.4% slower than fast draft only and verified only the first 64 tokens.
The experiment confirms service-level overlap is feasible; full verified-prefix
commit and scheduler-native K/V sharing remain open.

The dynamic gate now computes a per-token threshold from parallel perturbed
fast drafts:

```text
tau(t) = max_shadow |margin(canonical, t) - margin(shadow, t)|
```

It selected 15/24 first windows, detected every observed canonical/verifier
mismatch in the audit, and skipped only stable requests. However, shadow HTTP
batch overhead outweighed the verifier savings in the current service-level
implementation. Scheduler-native shadow execution is required for a throughput
gain.

A scheduler-native batched window replay is now implemented in `runtime.py`.
It runs one isolated batch=16 shadow extend per 16-request window. The gated
path preserved all output digests and reduced verification windows from 77 to
75, but always-on LLM-42 remained 12.7% faster end-to-end. The first prototype
therefore establishes correctness and a zero-miss audit, not yet a performance
win.

The Route 2 multi-window study was also negative. Across 580 request-windows,
shadow-uncertainty AUC was 0.469 versus 0.677 for canonical margin, and the best
100%-recall budget remained 94.3%. The current predictor signal does not justify
same-forward Route 1 engineering.

Adaptive verification windows are now implemented with checkpoints
`16/32/48/64`. The verifier keeps a fixed 64-token shape and pads each request
to its predicted length. All 16 outputs matched the fixed baseline, and rollback
tokens dropped from 184 to 55, but verifier windows rose from 47 to 67 and
wall-time increased by about 4.0%.

The separate two-fast-instance experiment selected 15/24 first windows,
covered all 6 canonical/verifier mismatches, and skipped only stable requests.
It was 18.9% faster than verifying every request in audit, but still slower
than running the deterministic verifier directly for this 64-token workload.

The canonical branch now streams 130 tokens while the perturbed branch computes
the first 64-token gate input. Selected verifier work overlaps the canonical
tail by about 0.324 s; selected requests commit about 0.281 s after canonical
finishes. All observed first-window mismatches remain covered.

The 32-prompt baseline matrix found canonical-only at 949.70 tok/s with 52.1%
full-sequence determinism, and deterministic verifier-only at 461.55 tok/s with
100% determinism. Gated mode reached 424.55 tok/s and 53.1% full-sequence
determinism. First-window gate recall fell to 90.9%, showing that the current
64-token verification scope is not sufficient for full-output guarantees.

The duplicate-prompt self-shadow topology removes the GPU1 perturbed fast
server. On eight prompts it recovered every first-window mismatch, but on 32
prompts recall fell to 77.1% for batch size 8 and 78.8% for batch size 16.

Explicitly shuffling eight prompts and eight copies of each into one 64-row
batch did not improve the predictor: recall fell to 70.3% and fast latency rose
to 0.883 s. Scheduler-generated mixed batches already provide most of the
available ordering perturbation.

The verified frontier is now tracked and strictly validated for every finished
LLM-42 request. On 32 prompts, all 96 finished request-rounds reached
`verified_tokens == output_tokens` with no monotonicity violations. However,
gate threshold 0.0625 produced four incorrect outputs despite a mechanically
full frontier, proving that frontier integrity and gate correctness must both
hold.

Lightweight logistic and MLP gates were also evaluated on 580 audit windows
with prompt-grouped splits. Canonical margin remained the strongest stable
baseline; shadow features reduced model AUC, and no model reached 100% recall
under deployable grouped calibration.

The new safe adaptive direction predicts verification timing rather than
skipping verification. Across 32 prompts it reduced rollback tokens by 78%
while preserving all outputs, but increased verification windows by 28% and
reduced throughput by 6%. Non-blocking verification scheduling is now the
remaining prerequisite for a throughput win.

A two-GPU non-blocking prototype now streams fast decode on GPU0 while a
deterministic verifier advances a committed frontier on GPU1. Across 32 prompts
and three rounds, all 96 requests reached the full 130-token frontier, with
about 0.46 s verifier/decode overlap. The implementation is service-level and
does not yet reuse scheduler K/V state.

Compared with native LLM-42, the fully instrumented always-on path produced
identical outputs and the same 171 verification windows, but added 3.99% wall
time and reduced throughput by 3.84%.

At 1024 output tokens, all tested configurations remained deterministic but the
Safe Adaptive advantage did not scale. On 32 extended prompts and three rounds,
Safe Adaptive threshold 0.1 reduced rollback tokens by 78.02% but increased
verification windows from 716 to 1786 and reduced warm-round throughput from
2572.81 to 1590.04 tok/s. Checkpoint 16 accounted for 90.43% of selected
verification windows. The detailed long-output report also retains the
instrumentation fix and the service-level non-blocking probe.

Multi-shape fixed verification pools are now implemented for LLM-42 modes 1
and 2. Mode 3 automatically falls back to the 64-token pool because its verifier
is deterministic only within one fixed shape. In a 32-prompt mode-2 run, all
32 outputs matched fixed-64, but throughput fell from 2025.09 to 1229.94 tok/s:
16-token and 64-token verifier calls both cost about 32-35 ms at batch size 8,
so extra calls outweighed the smaller token window.

At 1024 output tokens, observed completion variation remains sparse and highly
prefix-shared. Across 832 canonical, shadow, and verifier paths, all 183 shadow
first branches and 49/50 verifier first branches were within canonical top-2;
top-3 covered 50/50 verifier branches. The largest observed trie had about
three thousand nodes, with binary branching and an active frontier of at most
three nodes.

The follow-up 48-prompt study found a rank-5 shadow branch at batch size 16, so
the candidate was revised to top-8 logical candidates plus a lazy overflow edge.
Across 1536 additional 1024-token paths, tries remained binary, frontier width
stayed at most three, and a margin threshold of 0.375 covered every observed
branch while triggering on only 4.3%-4.6% of positions. Tree-aware verification
would reduce independent-path verifier calls by 47.8%-56.0%, but the
scheduler-native tree verifier is not implemented yet, so no candidate
end-to-end speedup is claimed.
