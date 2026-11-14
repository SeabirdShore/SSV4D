"""Train lightweight SSV4D verification-window risk models."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn


CANONICAL_FEATURES = (
    "neg_min_margin",
    "neg_q10_margin",
    "neg_mean_margin",
    "margin_std",
    "frac_margin_le_0",
    "frac_margin_le_00625",
    "frac_margin_le_0125",
    "min_margin_position",
    "window_index",
)

SHADOW_FEATURES = (
    "neg_min_shadow_margin",
    "shadow_margin_std",
    "shadow_delta_mean",
    "shadow_delta_max",
    "top1_disagreement_rate",
    "uncertainty_minus_margin",
)


def _load_samples(paths: list[Path]) -> list[dict]:
    samples = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") != "verify":
                continue
            for _, features in (
                event.get("replay_window_features") or {}
            ).items():
                positions = features.get("positions") or []
                if len(positions) < 2:
                    continue
                canonical = np.asarray(
                    [
                        float(position["canonical_margin"])
                        for position in positions
                    ]
                )
                shadow = np.asarray(
                    [
                        float(position["shadow_margin"])
                        for position in positions
                    ]
                )
                delta = np.abs(canonical - shadow)
                top1_disagreement = np.asarray(
                    [
                        not bool(position["top1_agreement"])
                        for position in positions
                    ],
                    dtype=float,
                )
                min_position = float(np.argmin(canonical)) / max(
                    1, len(canonical) - 1
                )
                combined = {
                    "neg_min_margin": -float(np.min(canonical)),
                    "neg_q10_margin": -float(
                        np.quantile(canonical, 0.1)
                    ),
                    "neg_mean_margin": -float(np.mean(canonical)),
                    "margin_std": float(np.std(canonical)),
                    "frac_margin_le_0": float(np.mean(canonical <= 0)),
                    "frac_margin_le_00625": float(
                        np.mean(canonical <= 0.0625)
                    ),
                    "frac_margin_le_0125": float(
                        np.mean(canonical <= 0.125)
                    ),
                    "min_margin_position": min_position,
                    "window_index": float(features["window_index"]),
                    "neg_min_shadow_margin": -float(np.min(shadow)),
                    "shadow_margin_std": float(np.std(shadow)),
                    "shadow_delta_mean": float(np.mean(delta)),
                    "shadow_delta_max": float(np.max(delta)),
                    "top1_disagreement_rate": float(
                        np.mean(top1_disagreement)
                    ),
                    "uncertainty_minus_margin": float(
                        np.max(delta - canonical)
                    ),
                }
                samples.append(
                    {
                        "group": features["request_fingerprint"],
                        "window_key": features["window_key"],
                        "label": int(features["rollback_tokens"] > 0),
                        "features": combined,
                        "dynamic_decision": bool(
                            features["decision"]["should_verify"]
                        ),
                    }
                )
    return samples


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if not len(positives) or not len(negatives):
        return float("nan")
    comparisons = (
        (positives[:, None] > negatives[None, :]).sum()
        + 0.5 * (positives[:, None] == negatives[None, :]).sum()
    )
    return float(comparisons / (len(positives) * len(negatives)))


def _metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict:
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    return {
        "samples": int(len(labels)),
        "positives": int(labels.sum()),
        "triggered": tp + fp,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "trigger_rate": (tp + fp) / len(labels),
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
    }


class LogisticModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, inputs):
        return self.linear(inputs).squeeze(-1)


class MLPModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, inputs):
        return self.network(inputs).squeeze(-1)


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    kind: str,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    xe = torch.tensor(x_eval, dtype=torch.float32)
    model = (
        LogisticModel(xt.shape[1])
        if kind == "logistic"
        else MLPModel(xt.shape[1])
    )
    positives = max(1, int(yt.sum().item()))
    negatives = max(1, len(yt) - positives)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.03 if kind == "logistic" else 0.01,
        weight_decay=1e-3,
    )
    model.train()
    for _ in range(500):
        optimizer.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(xe)).numpy()


def _fold_assignment(
    groups: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> np.ndarray:
    unique = np.asarray(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    mapping = {
        group: index % folds for index, group in enumerate(unique)
    }
    return np.asarray([mapping[group] for group in groups])


def _evaluate_feature_set(
    samples: list[dict],
    feature_names: tuple[str, ...],
    *,
    folds: int,
    seed: int,
) -> dict:
    x = np.asarray(
        [
            [sample["features"][name] for name in feature_names]
            for sample in samples
        ],
        dtype=np.float32,
    )
    y = np.asarray([sample["label"] for sample in samples], dtype=int)
    groups = np.asarray([sample["group"] for sample in samples])
    fold_ids = _fold_assignment(groups, folds=folds, seed=seed)

    predictions = {
        name: np.zeros(len(samples), dtype=float)
        for name in ("logistic", "mlp")
    }
    for fold in range(folds):
        train_mask = fold_ids != fold
        eval_mask = ~train_mask
        if not eval_mask.any():
            continue
        mean = x[train_mask].mean(axis=0)
        scale = x[train_mask].std(axis=0)
        scale[scale < 1e-6] = 1.0
        x_train = (x[train_mask] - mean) / scale
        x_eval = (x[eval_mask] - mean) / scale
        for offset, kind in enumerate(("logistic", "mlp")):
            seed_scores = [
                _fit_predict(
                    x_train,
                    y[train_mask],
                    x_eval,
                    kind=kind,
                    seed=seed + fold * 17 + rep,
                )
                for rep in range(3)
            ]
            predictions[kind][eval_mask] = np.mean(seed_scores, axis=0)

    result = {}
    for kind, scores in predictions.items():
        calibrated_predictions = np.zeros(len(samples), dtype=int)
        thresholds = []
        for fold in range(folds):
            train_mask = fold_ids != fold
            eval_mask = ~train_mask
            positive_scores = scores[train_mask][y[train_mask] == 1]
            threshold = (
                float(positive_scores.min())
                if positive_scores.size
                else float("inf")
            )
            thresholds.append(threshold)
            calibrated_predictions[eval_mask] = (
                scores[eval_mask] >= threshold
            ).astype(int)
        oracle_threshold = (
            float(scores[y == 1].min())
            if np.any(y == 1)
            else float("inf")
        )
        result[kind] = {
            "auc": _auc(scores, y),
            "grouped_cv_thresholds": {
                "mean": float(np.mean(thresholds)),
                "min": float(np.min(thresholds)),
                "max": float(np.max(thresholds)),
            },
            "grouped_full_recall_calibration": _metrics(
                calibrated_predictions, y
            ),
            "oracle_oof_full_recall": {
                "threshold": oracle_threshold,
                **_metrics(
                    (scores >= oracle_threshold).astype(int), y
                ),
            },
        }
    return result


def analyze(paths: list[Path], *, folds: int, seed: int) -> dict:
    samples = _load_samples(paths)
    y = np.asarray([sample["label"] for sample in samples])
    margin_scores = np.asarray(
        [sample["features"]["neg_min_margin"] for sample in samples]
    )
    uncertainty_scores = np.asarray(
        [
            sample["features"]["shadow_delta_max"]
            for sample in samples
        ]
    )
    combined_scores = np.asarray(
        [
            sample["features"]["uncertainty_minus_margin"]
            for sample in samples
        ]
    )
    dynamic = np.asarray(
        [sample["dynamic_decision"] for sample in samples], dtype=int
    )
    groups = np.asarray([sample["group"] for sample in samples])
    fold_ids = _fold_assignment(groups, folds=folds, seed=seed)

    def fixed_metrics(scores: np.ndarray) -> dict:
        global_threshold = float(scores[y == 1].min())
        grouped_predictions = np.zeros(len(samples), dtype=int)
        thresholds = []
        for fold in range(folds):
            train_mask = fold_ids != fold
            eval_mask = ~train_mask
            positive_scores = scores[train_mask][y[train_mask] == 1]
            threshold = (
                float(positive_scores.min())
                if positive_scores.size
                else float("inf")
            )
            thresholds.append(threshold)
            grouped_predictions[eval_mask] = (
                scores[eval_mask] >= threshold
            ).astype(int)
        oracle_threshold = float(scores[y == 1].min())
        return {
            "auc": _auc(scores, y),
            "global_full_recall": {
                "threshold": global_threshold,
                **_metrics(
                    (scores >= global_threshold).astype(int), y
                ),
            },
            "grouped_full_recall_calibration": _metrics(
                grouped_predictions, y
            ),
            "grouped_thresholds": {
                "mean": float(np.mean(thresholds)),
                "min": float(np.min(thresholds)),
                "max": float(np.max(thresholds)),
            },
            "oracle_oof_full_recall": {
                "threshold": oracle_threshold,
                **_metrics(
                    (scores >= oracle_threshold).astype(int), y
                ),
            },
        }

    result = {
        "samples": len(samples),
        "positives": int(y.sum()),
        "groups": len({sample["group"] for sample in samples}),
        "baselines": {
            "margin_only": fixed_metrics(margin_scores),
            "shadow_uncertainty": fixed_metrics(uncertainty_scores),
            "margin_uncertainty": fixed_metrics(combined_scores),
            "dynamic_gate": {
                **_metrics(dynamic, y),
                "auc": _auc(
                    dynamic.astype(float)
                    + 1e-6
                    * np.asarray(
                        [
                            sample["features"][
                                "uncertainty_minus_margin"
                            ]
                            for sample in samples
                        ]
                    ),
                    y,
                ),
            },
        },
        "models": {},
    }
    for name, features in (
        ("canonical", CANONICAL_FEATURES),
        ("canonical_shadow", CANONICAL_FEATURES + SHADOW_FEATURES),
    ):
        result["models"][name] = _evaluate_feature_set(
            samples,
            features,
            folds=folds,
            seed=seed,
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="SSV4D runtime JSONL logs with replay_window_features.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ssv4d/results/gate_model_ablation.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(args.inputs, folds=args.folds, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
