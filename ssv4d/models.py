"""Data structures shared by the SSV4D experiment pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class GenerationTrace:
    prompt_id: str
    scenario: str
    batch_size: int
    protected_index: int
    output_ids: list[int]
    text: str
    margins: list[float]
    top_ids: list[list[int]]
    top_logprobs: list[list[float]]
    latency_s: float
    deterministic: bool
    plan_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WindowSample:
    prompt_id: str
    window_index: int
    start: int
    end: int
    label: int
    mismatch_positions: list[int]
    min_margin: float
    q10_margin: float
    mean_margin: float
    margin_spread: float
    shadow_vote_disagreement: float
    shadow_jsd: float
    n_traces: int
    reference_shadow_jsd: float = 0.0
    reference_shadow_vote: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateMetrics:
    policy: str
    threshold: float
    calibration_target_recall: float
    split: str
    n_windows: int
    n_positive_windows: int
    n_selected: int
    true_positives: int
    false_positives: int
    false_negatives: int
    trigger_rate: float
    window_recall: float
    window_precision: float
    prompt_recall: float
    prompt_determinism: float
    weight_margin: float = 0.0
    weight_shadow: float = 0.0
    weight_jsd: float = 0.0
    extra: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
