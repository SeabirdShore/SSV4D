"""Train lightweight early-rollback predictors for safe adaptive windows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


CHECKPOINTS = (16, 32, 48)


def _window_features(
    canonical: np.ndarray,
    window_index: int,
) -> list[float]:
    return [
        -float(np.min(canonical)),
        -float(np.quantile(canonical, 0.1)),
        -float(np.mean(canonical)),
        float(np.std(canonical)),
        float(np.mean(canonical <= 0)),
        float(np.mean(canonical <= 0.0625)),
        float(np.mean(canonical <= 0.125)),
        float(np.argmin(canonical)) / max(1, len(canonical) - 1),
        float(window_index),
    ]


def _load_samples(
    paths: list[Path],
) -> dict[int, list[dict]]:
    result = {checkpoint: [] for checkpoint in CHECKPOINTS}
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
                if not positions:
                    continue
                canonical = np.asarray(
                    [
                        float(position["canonical_margin"])
                        for position in positions
                    ]
                )
                rollback_tokens = int(features["rollback_tokens"])
                first_mismatch = (
                    len(canonical) - rollback_tokens
                    if rollback_tokens > 0
                    else None
                )
                for checkpoint in CHECKPOINTS:
                    if len(canonical) < checkpoint:
                        continue
                    result[checkpoint].append(
                        {
                            "group": features["request_fingerprint"],
                            "label": int(
                                first_mismatch is not None
                                and first_mismatch < checkpoint
                            ),
                            "rollback_position": first_mismatch,
                            "window_length": len(canonical),
                            "features": _window_features(
                                canonical[:checkpoint],
                                int(features["window_index"]),
                            ),
                        }
                    )
    return result


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    comparisons = (
        (positive[:, None] > negative[None, :]).sum()
        + 0.5 * (positive[:, None] == negative[None, :]).sum()
    )
    return float(comparisons / (len(positive) * len(negative)))


class Logistic(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, values):
        return self.linear(values).squeeze(-1)


def _fit_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    xn = (x - mean) / scale
    xt = torch.tensor(xn, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    positives = max(1, int(yt.sum().item()))
    negatives = max(1, len(yt) - positives)
    model = Logistic(x.shape[1])
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.03, weight_decay=1e-3
    )
    model.train()
    for _ in range(500):
        optimizer.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        optimizer.step()
    weights = model.linear.weight.detach().numpy()[0].tolist()
    bias = float(model.linear.bias.detach().item())
    return np.asarray(weights), np.asarray(mean), scale, bias


def _predict(
    x: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    bias: float,
) -> np.ndarray:
    logits = ((x - mean) / scale) @ weights + bias
    return 1.0 / (1.0 + np.exp(-logits))


def _fold_ids(groups: np.ndarray, folds: int, seed: int) -> np.ndarray:
    unique = np.asarray(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    mapping = {
        group: index % folds for index, group in enumerate(unique)
    }
    return np.asarray([mapping[group] for group in groups])


def train(
    paths: list[Path],
    *,
    folds: int,
    seed: int,
) -> tuple[dict, dict]:
    samples_by_checkpoint = _load_samples(paths)
    models = {}
    report = {}
    for checkpoint, samples in samples_by_checkpoint.items():
        if not samples:
            continue
        x = np.asarray(
            [sample["features"] for sample in samples],
            dtype=np.float32,
        )
        y = np.asarray(
            [sample["label"] for sample in samples], dtype=int
        )
        groups = np.asarray([sample["group"] for sample in samples])
        fold_ids = _fold_ids(groups, folds, seed)
        oof = np.zeros(len(samples))
        for fold in range(folds):
            train_mask = fold_ids != fold
            eval_mask = ~train_mask
            weights, mean, scale, bias = _fit_model(
                x[train_mask],
                y[train_mask],
                seed=seed + fold,
            )
            oof[eval_mask] = _predict(
                x[eval_mask], weights, mean, scale, bias
            )
        weights, mean, scale, bias = _fit_model(x, y, seed=seed)
        thresholds = {}
        for threshold in (0.05, 0.1, 0.2, 0.3, 0.5):
            predictions = oof >= threshold
            tp = int(np.sum(predictions & (y == 1)))
            fp = int(np.sum(predictions & (y == 0)))
            fn = int(np.sum(~predictions & (y == 1)))
            thresholds[str(threshold)] = {
                "trigger_rate": float(np.mean(predictions)),
                "early_rollback_recall": (
                    tp / (tp + fn) if tp + fn else 1.0
                ),
                "false_positives": fp,
                "false_negatives": fn,
            }
        models[str(checkpoint)] = {
            "weights": weights.tolist(),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "bias": bias,
        }
        report[str(checkpoint)] = {
            "samples": len(samples),
            "positive_early_rollbacks": int(y.sum()),
            "positive_rate": float(y.mean()),
            "auc": _auc(oof, y),
            "thresholds": thresholds,
        }
    return models, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("ssv4d/results/safe_window_model.json"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("ssv4d/results/safe_window_model_report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models, report = train(args.inputs, folds=args.folds, seed=args.seed)
    for path, payload in (
        (args.model_output, models),
        (args.report_output, report),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
