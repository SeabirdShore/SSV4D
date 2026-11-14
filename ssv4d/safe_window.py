"""Lightweight early-rollback predictor for safe adaptive verification."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class CheckpointModel:
    weights: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    bias: float


class SafeWindowPredictor:
    def __init__(
        self,
        models: dict[str, CheckpointModel],
        thresholds: dict[int, float],
    ):
        self.models = models
        self.thresholds = thresholds

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        threshold: float = 0.1,
    ) -> "SafeWindowPredictor":
        payload = json.loads(path.read_text(encoding="utf-8"))
        models = {
            str(checkpoint): CheckpointModel(
                weights=np.asarray(row["weights"], dtype=float),
                mean=np.asarray(row["mean"], dtype=float),
                scale=np.asarray(row["scale"], dtype=float),
                bias=float(row["bias"]),
            )
            for checkpoint, row in payload.items()
        }
        return cls(
            models=models,
            thresholds={int(key): threshold for key in models},
        )

    @staticmethod
    def features(
        margins: Iterable[float],
        *,
        window_index: int,
    ) -> np.ndarray:
        values = np.asarray(
            [float(value) for value in margins if math.isfinite(float(value))],
            dtype=float,
        )
        if not values.size:
            return np.zeros(9, dtype=float)
        return np.asarray(
            [
                -float(np.min(values)),
                -float(np.quantile(values, 0.1)),
                -float(np.mean(values)),
                float(np.std(values)),
                float(np.mean(values <= 0)),
                float(np.mean(values <= 0.0625)),
                float(np.mean(values <= 0.125)),
                float(np.argmin(values)) / max(1, len(values) - 1),
                float(window_index),
            ],
            dtype=float,
        )

    def probability(
        self,
        checkpoint: int,
        margins: Iterable[float],
        *,
        window_index: int,
    ) -> float:
        model = self.models.get(str(checkpoint))
        if model is None:
            return 1.0
        features = self.features(margins, window_index=window_index)
        normalized = (features - model.mean) / model.scale
        logit = float(normalized @ model.weights + model.bias)
        return 1.0 / (1.0 + math.exp(-logit))

    def should_verify(
        self,
        checkpoint: int,
        margins: Iterable[float],
        *,
        window_index: int,
    ) -> tuple[bool, float]:
        probability = self.probability(
            checkpoint,
            margins,
            window_index=window_index,
        )
        return (
            probability >= self.thresholds.get(checkpoint, 1.0),
            probability,
        )
