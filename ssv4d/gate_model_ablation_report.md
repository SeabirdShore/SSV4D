# Lightweight Gate Model Ablation

This experiment tests whether a small classifier can predict verification
windows better than the hand-written SSV4D gate.

## Dataset

- Source: `logs/r2_multi_audit.jsonl`
- 580 request-windows
- 200 rollback-window positives
- 48 prompt groups
- Labels come from always-verify deterministic rollback results

Features are computed before verification:

- canonical margin minimum, q10, mean, standard deviation
- fraction of margins below `0`, `0.0625`, `0.125`
- position of minimum margin
- window index
- shadow margin statistics
- canonical/shadow margin delta mean/max
- top-1 disagreement rate
- uncertainty-minus-margin score

The shadow feature set is optional and tested as an ablation.

## Models

Two lightweight models were evaluated:

1. Logistic regression: 15 learned parameters.
2. MLP: `15 -> 32 -> 16 -> 1`, about 1k parameters.

Training uses:

- five prompt-grouped folds
- three model seeds per fold
- positive-class weighting
- score averaging across seeds
- threshold selected on each calibration fold to maximize recall

Token-level random splits were not used.

## Results Across Three Split Seeds

Values are means over seeds 42, 43, and 44.

| Policy | AUC | Grouped-CV recall | Grouped-CV trigger rate | Oracle full-recall trigger |
|---|---:|---:|---:|---:|
| Margin-only | 0.677 | 99.0% | 95.9% | 97.2% |
| Canonical logistic | 0.661 | 98.2% | 96.4% | 98.0% |
| Canonical MLP | 0.647 | 97.2% | 88.4% | 89.6% |
| Canonical + shadow logistic | 0.659 | 97.8% | 94.4% | 97.4% |
| Canonical + shadow MLP | 0.622 | 97.5% | 94.5% | 96.4% |

## Interpretation

The MLP can reduce the trigger rate to about 88%, but it loses recall:

- grouped-CV recall is only 97.2%
- the 100%-recall trigger rate of 89.6% is obtained only by choosing a
  threshold with access to all out-of-fold labels

That oracle threshold is not deployable for unseen prompts. Under realistic
prompt-grouped calibration, no model reaches 100% recall.

Canonical logistic regression is slightly above margin-only on one split, but
its AUC ranges from 0.635 to 0.687 across seeds and averages below margin-only.
The model has no stable advantage.

Adding shadow features consistently degrades both models:

- canonical logistic: 0.661 mean AUC
- canonical + shadow logistic: 0.659 mean AUC
- canonical MLP: 0.647 mean AUC
- canonical + shadow MLP: 0.622 mean AUC

## Decision

The lightweight-model approach is technically feasible and inference cost would
be negligible. However, the available features do not support a deployable
100%-recall gate.

Do not integrate these models into the runtime yet. The next prerequisite is
additional predictive features or substantially more grouped training data,
not a larger neural network.

## Artifacts

- `train_gate_model.py`
- `scripts/run_gate_model_ablation.sh`
- `results/gate_model_ablation_seed42.json`
- `results/gate_model_ablation_seed43.json`
- `results/gate_model_ablation_seed44.json`
- `results/gate_model_ablation_summary.json`
