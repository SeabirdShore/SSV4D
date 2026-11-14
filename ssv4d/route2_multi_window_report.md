# Route 2: Multi-Window Verification Predictor Diagnostic

This experiment tests whether perturbed fast-draft uncertainty predicts
LLM-42 rollback windows well enough to justify a same-forward performance
prototype.

## Method

- Model: `NousResearch/Meta-Llama-3.1-8B-Instruct`
- Prompt suite: 48 built-in prompts (`ssv4d/prompts_extended.py`)
- Window size: 64 tokens
- Audit mode: every window is verified, so each prediction receives a
  deterministic rollback label.
- Batch stability cases: 8 standard prompts, 16 standard prompts, and 48
  extended prompts.
- Multi-window sample: 48 prompts, three rounds at 130 tokens and two rounds at
  194 tokens.

The worker runs one isolated replay extend per verification window. Each replay
uses fresh Req/K-V state and records canonical/shadow top-1 margins and top-1
agreement at every position.

## Full Multi-Window Sample

The combined 48-prompt audit contains 580 request-windows:

- 200 positive rollback windows, positive rate 34.5%.
- The current dynamic gate triggers 503 windows (86.7%).
- It covers 195/200 positives: recall 97.5%, precision 38.8%.
- Five rollback windows are missed.

AUC, where higher is better:

| Feature | AUC |
|---|---:|
| Negative canonical minimum margin | 0.677 |
| Shadow uncertainty `max abs(margin delta)` | 0.469 |
| Margin-within-uncertainty score | 0.648 |

Shadow uncertainty is close to random ranking and does not improve over the
canonical margin.

## Full-Recall Budget

At 100% recall over the complete sample:

| Score | Trigger budget |
|---|---:|
| Margin-only | 97.2% |
| Shadow uncertainty | 96.6% |
| Margin-within-uncertainty | 94.3% |

The best observed full-recall budget is still 94.3%; it does not provide the
large verification reduction needed for a performance prototype.

Leave-one-prompt-out calibration is harder. In the 580-window sample:

| Score | Grouped-CV recall | Trigger budget |
|---|---:|---:|
| Margin-only | 99.0% | 96.9% |
| Shadow uncertainty | 97.5% | 95.7% |
| Margin-within-uncertainty | 97.5% | 93.4% |

No feature reaches exact 100% recall on unseen prompt groups.

## Window Stability

For the 48-prompt multi-window run:

| Window | Samples | Positives | Margin AUC | Shadow AUC | Dynamic recall | Dynamic budget |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 240 | 90 | 0.700 | 0.529 | 100.0% | 83.3% |
| 1 | 240 | 90 | 0.680 | 0.513 | 94.4% | 89.6% |
| 2 | 96 | 20 | 0.537 | 0.518 | 100.0% | 87.5% |

Separability and recall degrade after the first window. A first-window-only
predictor is not enough for a complete SSV4D gate.

## Concurrency/Batch Stability

Independent audit runs:

| Concurrent prompts | Request-windows | Positive rate | Margin AUC | Shadow AUC | Dynamic recall | Dynamic budget |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 48 | 25.0% | 0.740 | 0.229 | 100.0% | 75.0% |
| 16 | 99 | 24.2% | 0.753 | 0.368 | 100.0% | 81.8% |
| 48 | 580 | 34.5% | 0.677 | 0.469 | 97.5% | 86.7% |

For the same eight first-window prompts, dynamic decisions were identical at
batch sizes 8 and 16. At batch size 48, 15/16 decisions from the 16-prompt run
remained identical. Basic decision stability is acceptable, but predictive
quality does not improve with scale.

## Decision

The Route 2 go/no-go check is negative:

1. Shadow uncertainty does not beat or reliably complement canonical margin.
2. Full-recall verification budgets remain near always-on.
3. Recall and AUC degrade in later windows and on the larger prompt set.
4. Grouped calibration does not transfer at 100% recall.

The expected benefit does not justify proceeding to a same-forward Route 1
implementation. More performance engineering on the current signal is unlikely
to recover the missing selectivity.

SSV4D can still be positioned as a diagnostic for batch-shape sensitivity, but
the present perturbed-replay feature is not a competitive verification
predictor.

## Artifacts

- `prompts_extended.py`
- `analyze_route2.py`
- `scripts/run_route2_probe.sh`
- `results/r2_batch8.json`
- `results/r2_batch16.json`
- `results/r2_multi_audit.json`
- `results/r2_multi_audit_194.json`
- `results/route2_batch_stability_summary.json`
- `results/route2_decision_stability.json`
- `results/route2_multi_window_summary.json`
- `logs/r2_batch8.jsonl`
- `logs/r2_batch16.jsonl`
- `logs/r2_multi_audit.jsonl`
