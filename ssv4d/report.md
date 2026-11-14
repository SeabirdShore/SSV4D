# SSV4D Research Report

- Run: `run_20261112_135628`
- Model: `NousResearch/Meta-Llama-3.1-8B-Instruct`
- Output tokens per request: 128
- Window size: 32
- Shadow plans per prompt: 2 repeat(s) x 9 plan(s) + bs1
- Results: `/home/xiaosibo/llm-42/ssv4d/results/run_20261112_135628`
- Detailed research notes: `ssv4d/research_notes.md`
- Closest prior work: MarginGate, arXiv:2605.30218
- Runtime/KV/prefill follow-up: `ssv4d/runtime_prefill_report.md`

## Research Question

Can pre-prefill batch perturbations expose batch-induced token-instability with enough precision to gate LLM-42 verification and reduce verifier budget?

## Method

For each protected prompt, SSV4D concurrently runs a deterministic LLM-42 reference and a shadow branch. The shadow branch inserts the protected prompt into perturbed batches of size 1/4/8/16, at first/middle/last positions, with rotated filler order. For each generated token it records output IDs, top-k logprobs, top-1/top-2 margin, and cross-shadow vote/JS disagreement. These signals are aggregated over verification windows. Gate thresholds are evaluated with leave-one-prompt-out grouped cross-validation, which keeps calibration and evaluation prompts disjoint.

## Sparse Error Observation

- Shadow traces: 304
- Synchronous decode steps: 24966
- Token flips: 152 (0.61%)
- Window positives: 8/46 (17.39%)
- Synchronous token positions: 1322
- Token positives: 8
- Median flip position: 32.0
- Flip positions: [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
- Median shadow margin at flips: 0.0625
- Median stable-step margin: 4.2500

## Gate Comparison

| Policy | Trigger rate | Window recall | Prompt recall | Prompt determinism |
|---|---:|---:|---:|---:|
| always_on | 100.00% | 100.00% | 100.00% | 100.00% |
| margin_only | 73.91% | 100.00% | 100.00% | 100.00% |
| shadow_only | 97.83% | 87.50% | 87.50% | 93.75% |
| ssv4d | 73.91% | 100.00% | 100.00% | 100.00% |

## Token-level Selective Verification

| Policy | Verify-token rate | Token recall | Prompt determinism |
|---|---:|---:|---:|
| always_on | 100.00% | 100.00% | 100.00% |
| margin_only | 4.16% | 100.00% | 100.00% |
| shadow_only | 100.00% | 100.00% | 100.00% |
| ssv4d | 4.16% | 100.00% | 100.00% |

Feature AUC on token positives (higher is better):

- Negative min-margin: 0.9855
- Shadow vote disagreement: 0.5000
- Shadow top-k JSD: 0.3525


## Selected SSV4D Gate

- Margin threshold: 0.187500
- Shadow enabled: False
- Shadow AUC: 0.5000
- Shadow threshold: inf
- Rule: `verify = min_margin <= tau_margin` (shadow branch rejected by calibration AUC gate)
- Shadow risk: top-1 vote disagreement + min(1, 100 * top-k JSD)
- Token margin threshold: 0.187500
- Token shadow threshold: inf

The gate is a prototype for the LLM-42 verification hook in `ssv4d/llm42_hook.py`. The current experiment measures the selective verification frontier offline; it does not yet run the shadow branch inside the GPU scheduler or perform current-column K/V repair.

## Interpretation

Calibration rejected the shadow disagreement feature because its held-in AUC (0.5000) did not exceed the configured minimum. SSV4D therefore falls back to margin-only verification for this model/workload; the shadow branch remains an online diagnostic.
SSV4D selected 73.91% of windows versus 100% always-on verification, while recovering 100.00% of observed flip windows and 100.00% prompt-level determinism on the leave-one-prompt-out evaluation.
Under grouped cross-validation, margin-only triggered 73.91% with 100.00% recall; shadow-only triggered 97.83% with 87.50% recall.
At token granularity, margin-only verified 4.16% of positions with 100.00% recall; SSV4D verified 4.16% with 100.00% recall. The gate fell back to margin-only behavior because calibration rejected the shadow feature.
The observed flips are sparse and their first positions are listed above. The window-level budget numbers are therefore the primary result; sequence-level claims require more prompts and repeated trials.

## Artifacts

- `token_flips.jsonl`
- `window_features.csv`
- `token_features.csv`
- `gate_metrics.csv`
- `token_gate_metrics.csv`
- `selected_token_gate.json`
- `raw_traces.json`
- `flip_positions.png`
- `margin_separation.png`
- `policy_comparison.png`
- `token_policy_comparison.png`
- `window_sparsity.png`

## Limitations and Next Steps

1. The shadow branch currently runs through the public `/generate` API rather than inside the scheduler; this isolates the numerical signal but includes HTTP and tokenizer overhead.
2. The evaluation uses a small built-in prompt set and one model. It is a research prototype, not a paper-scale reproduction.
3. The LLM-42 hook gates request-level verification. The next step is a window-level scheduler callback with online risk state and current-column K/V repair.
4. Future calibration should use a larger held-out stream and target explicit latency/determinism constraints rather than recall alone.
