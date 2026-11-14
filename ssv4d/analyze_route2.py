"""Analyze multi-window dynamic-shadow verification prediction quality."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _finite_json(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    return value


def _auc(scores: list[float], labels: list[bool]) -> float:
    positives = np.asarray(
        [score for score, label in zip(scores, labels) if label],
        dtype=float,
    )
    negatives = np.asarray(
        [score for score, label in zip(scores, labels) if not label],
        dtype=float,
    )
    if not len(positives) or not len(negatives):
        return float("nan")
    comparisons = (
        (positives[:, None] > negatives[None, :]).sum()
        + 0.5 * (positives[:, None] == negatives[None, :]).sum()
    )
    return float(comparisons / (len(positives) * len(negatives)))


def _binary_metrics(
    labels: list[bool],
    predictions: list[bool],
) -> dict:
    tp = sum(label and prediction for label, prediction in zip(labels, predictions))
    fp = sum(
        not label and prediction
        for label, prediction in zip(labels, predictions)
    )
    fn = sum(
        label and not prediction
        for label, prediction in zip(labels, predictions)
    )
    tn = sum(
        not label and not prediction
        for label, prediction in zip(labels, predictions)
    )
    return {
        "requests": len(labels),
        "positives": tp + fn,
        "triggered": tp + fp,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "trigger_rate": (tp + fp) / len(labels) if labels else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
    }


def _threshold_for_full_recall(
    scores: list[float],
    labels: list[bool],
) -> float:
    positive_scores = [
        score for score, label in zip(scores, labels) if label
    ]
    if not positive_scores:
        return float("inf")
    threshold = min(positive_scores)
    while math.isfinite(threshold) and any(
        label and score < threshold
        for score, label in zip(scores, labels)
    ):
        threshold = math.nextafter(threshold, -math.inf)
    return threshold


def _full_recall_global_metrics(
    samples: list[dict],
    score_name: str,
) -> dict:
    labels = [sample["label"] for sample in samples]
    scores = [sample[score_name] for sample in samples]
    threshold = _threshold_for_full_recall(scores, labels)
    predictions = [score >= threshold for score in scores]
    return {
        "threshold": threshold,
        **_binary_metrics(labels, predictions),
    }


def _full_recall_grouped_metrics(
    samples: list[dict],
    score_name: str,
) -> dict:
    groups = sorted({sample["prompt"] for sample in samples})
    predictions_by_key = {}
    thresholds = {}
    for held_out in groups:
        calibration = [
            sample for sample in samples if sample["prompt"] != held_out
        ]
        threshold = _threshold_for_full_recall(
            [sample[score_name] for sample in calibration],
            [sample["label"] for sample in calibration],
        )
        thresholds[held_out] = threshold
        for sample in samples:
            if sample["prompt"] == held_out:
                predictions_by_key[sample["key"]] = (
                    sample[score_name] >= threshold
                )
    predictions = [
        predictions_by_key[sample["key"]] for sample in samples
    ]
    return {
        "mean_threshold": float(np.mean(list(thresholds.values()))),
        **_binary_metrics(
            [sample["label"] for sample in samples],
            predictions,
        ),
    }


def analyze(path: Path) -> dict:
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    samples = []
    for event in events:
        if event.get("event") != "verify":
            continue
        for rid, features in (
            event.get("replay_window_features") or {}
        ).items():
            positions = features.get("positions") or []
            if not positions:
                continue
            canonical_margins = [
                float(position["canonical_margin"])
                for position in positions
            ]
            shadow_margins = [
                float(position["shadow_margin"])
                for position in positions
            ]
            delta = [
                abs(canonical - shadow)
                for canonical, shadow in zip(
                    canonical_margins, shadow_margins
                )
            ]
            decision = features.get("decision") or {}
            top1_disagreement = any(
                not position["top1_agreement"] for position in positions
            )
            samples.append(
                {
                    "key": f"{features['request_fingerprint']}:{features['window_start']}",
                    "prompt": features["request_fingerprint"],
                    "window": int(features["window_index"]),
                    "label": features["rollback_tokens"] > 0,
                    "min_margin": min(canonical_margins),
                    "margin_score": -min(canonical_margins),
                    "shadow_uncertainty": max(delta),
                    "margin_uncertainty_score": max(
                        uncertainty - canonical
                        for canonical, uncertainty in zip(
                            canonical_margins, delta
                        )
                    ),
                    "top1_disagreement": top1_disagreement,
                    "dynamic_decision": bool(
                        decision.get("should_verify", False)
                    ),
                    "dynamic_reason": decision.get("trigger_reason"),
                }
            )

    labels = [sample["label"] for sample in samples]
    score_names = (
        "margin_score",
        "shadow_uncertainty",
        "margin_uncertainty_score",
    )
    by_window = {}
    for window in sorted({sample["window"] for sample in samples}):
        subset = [sample for sample in samples if sample["window"] == window]
        by_window[str(window)] = {
            "samples": len(subset),
            "positives": sum(sample["label"] for sample in subset),
            "auc": {
                name: _auc(
                    [sample[name] for sample in subset],
                    [sample["label"] for sample in subset],
                )
                for name in score_names
            },
            "dynamic": _binary_metrics(
                [sample["label"] for sample in subset],
                [sample["dynamic_decision"] for sample in subset],
            ),
            "global_full_recall": {
                name: _full_recall_global_metrics(subset, name)
                for name in score_names
            },
        }

    return {
        "path": str(path),
        "requests": len(samples),
        "positive_windows": sum(labels),
        "positive_rate": (
            sum(labels) / len(labels) if labels else 0.0
        ),
        "auc": {
            name: _auc(
                [sample[name] for sample in samples],
                labels,
            )
            for name in score_names
        },
        "dynamic": _binary_metrics(
            labels,
            [sample["dynamic_decision"] for sample in samples],
        ),
        "dynamic_reasons": dict(
            Counter(
                sample["dynamic_reason"]
                for sample in samples
                if sample["dynamic_decision"]
            )
        ),
        "global_full_recall": {
            name: _full_recall_global_metrics(samples, name)
            for name in score_names
        },
        "grouped_full_recall": {
            name: _full_recall_grouped_metrics(samples, name)
            for name in score_names
        },
        "by_window": by_window,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ssv4d/results/route2_multi_window_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = _finite_json([analyze(path) for path in args.inputs])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
