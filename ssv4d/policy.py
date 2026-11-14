"""Threshold calibration and comparison of verification-gate policies."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from ssv4d.models import GateMetrics, WindowSample


DEFAULT_MARGIN_GUARD = 0.0625
DEFAULT_MIN_SHADOW_AUC = 0.55


def _limited_candidates(values: Iterable[float], max_candidates: int = 48) -> list[float]:
    unique = np.unique(np.asarray(list(values), dtype=float))
    if unique.size <= max_candidates:
        return [float(x) for x in unique]
    indices = np.linspace(0, unique.size - 1, max_candidates).round().astype(int)
    return [float(unique[i]) for i in sorted(set(indices.tolist()))]


@dataclass
class PolicySpec:
    name: str
    threshold: float
    weight_margin: float = 1.0
    weight_shadow: float = 0.0
    weight_jsd: float = 0.0


@dataclass
class CompositeRiskGate:
    threshold: float
    median: dict[str, float] = field(default_factory=dict)
    scale: dict[str, float] = field(default_factory=dict)
    weight_margin: float = 1.0
    weight_shadow: float = 0.5
    weight_jsd: float = 0.25
    policy_name: str = "ssv4d"

    def fit_normalizer(self, samples: Iterable[WindowSample]) -> None:
        sample_list = list(samples)
        raw = {
            "margin": np.asarray([-s.min_margin for s in sample_list], dtype=float),
            "shadow": np.asarray(
                [s.shadow_vote_disagreement for s in sample_list], dtype=float
            ),
            "jsd": np.asarray([s.shadow_jsd for s in sample_list], dtype=float),
        }
        for key, values in raw.items():
            self.median[key] = float(np.median(values)) if values.size else 0.0
            q75, q25 = (
                np.percentile(values, [75, 25]) if values.size else (1.0, 0.0)
            )
            self.scale[key] = float(max(q75 - q25, 1e-6))

    def score(self, sample: WindowSample, spec: PolicySpec) -> float:
        z_margin = (self._z(-sample.min_margin, "margin"))
        z_shadow = self._z(sample.shadow_vote_disagreement, "shadow")
        z_jsd = self._z(sample.shadow_jsd, "jsd")
        return (
            spec.weight_margin * z_margin
            + spec.weight_shadow * z_shadow
            + spec.weight_jsd * z_jsd
        )

    def _z(self, value: float, key: str) -> float:
        return (float(value) - self.median.get(key, 0.0)) / self.scale.get(
            key, 1.0
        )

    def trigger(self, sample: WindowSample) -> bool:
        spec = PolicySpec(
            name=self.policy_name,
            threshold=self.threshold,
            weight_margin=self.weight_margin,
            weight_shadow=self.weight_shadow,
            weight_jsd=self.weight_jsd,
        )
        return self.score(sample, spec) >= self.threshold


@dataclass
class DualThresholdGate:
    """Deployable margin-or-shadow gate."""

    margin_threshold: float
    shadow_threshold: float
    shadow_enabled: bool = True
    shadow_auc: float = float("nan")
    policy_name: str = "ssv4d"

    @staticmethod
    def shadow_risk(sample: WindowSample) -> float:
        # Vote disagreement and top-k distribution shift are complementary.
        # JSD is compressed because stable top-k distributions can differ in
        # float noise while genuine flips produce much larger movement.
        return sample.shadow_vote_disagreement + min(
            1.0, sample.shadow_jsd * 100.0
        )

    def trigger(self, sample: WindowSample) -> bool:
        return (
            sample.min_margin <= self.margin_threshold
            or (
                self.shadow_enabled
                and self.shadow_risk(sample) >= self.shadow_threshold
            )
        )


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_predictions(
    samples: list[WindowSample],
    predictions: list[bool],
    *,
    policy_name: str,
    threshold: float,
    target_recall: float,
    split: str,
    weight_margin: float = 0.0,
    weight_shadow: float = 0.0,
    weight_jsd: float = 0.0,
) -> GateMetrics:
    if len(samples) != len(predictions):
        raise ValueError("samples and predictions differ in length")
    labels = [bool(s.label) for s in samples]
    tp = sum(1 for y, p in zip(labels, predictions) if y and p)
    fp = sum(1 for y, p in zip(labels, predictions) if not y and p)
    fn = sum(1 for y, p in zip(labels, predictions) if y and not p)
    positives = sum(labels)
    selected = sum(predictions)

    prompt_total: dict[str, bool] = defaultdict(bool)
    prompt_hit: dict[str, int] = defaultdict(int)
    prompt_positive: dict[str, int] = defaultdict(int)
    for sample, prediction in zip(samples, predictions):
        prompt_total[sample.prompt_id] = True
        if sample.label:
            prompt_positive[sample.prompt_id] += 1
            if prediction:
                prompt_hit[sample.prompt_id] += 1

    prompts_with_flips = [p for p, n in prompt_positive.items() if n]
    covered_prompts = [
        p for p in prompts_with_flips if prompt_hit[p] == prompt_positive[p]
    ]
    prompt_recall = _safe_div(len(covered_prompts), len(prompts_with_flips))
    prompts_without_flips = [
        p for p in prompt_total if prompt_positive[p] == 0
    ]
    deterministic_prompts = len(covered_prompts) + len(prompts_without_flips)
    prompt_determinism = _safe_div(deterministic_prompts, len(prompt_total))

    return GateMetrics(
        policy=policy_name,
        threshold=float(threshold),
        calibration_target_recall=target_recall,
        split=split,
        n_windows=len(samples),
        n_positive_windows=positives,
        n_selected=selected,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        trigger_rate=_safe_div(selected, len(samples)),
        window_recall=_safe_div(tp, positives) if positives else 1.0,
        window_precision=_safe_div(tp, tp + fp) if tp + fp else 1.0,
        prompt_recall=prompt_recall if prompts_with_flips else 1.0,
        prompt_determinism=prompt_determinism,
        weight_margin=weight_margin,
        weight_shadow=weight_shadow,
        weight_jsd=weight_jsd,
    )


def _calibrate_margin_threshold(
    samples: list[WindowSample],
    target_recall: float,
    margin_guard: float,
) -> tuple[float, GateMetrics]:
    values = sorted({s.min_margin for s in samples})
    candidates = [float("-inf"), *values]
    best: Optional[tuple[float, GateMetrics]] = None
    for threshold in candidates:
        predictions = [s.min_margin <= threshold for s in samples]
        metrics = evaluate_predictions(
            samples,
            predictions,
            policy_name="margin_only",
            threshold=threshold,
            target_recall=target_recall,
            split="calibration",
            weight_margin=1.0,
        )
        if metrics.window_recall + 1e-12 < target_recall:
            continue
        if best is None:
            best = (threshold, metrics)
        elif (
            metrics.trigger_rate,
            -metrics.window_precision,
        ) < (
            best[1].trigger_rate,
            -best[1].window_precision,
        ):
            best = (threshold, metrics)
    if best is None:
        return float("inf"), evaluate_predictions(
            samples,
            [True] * len(samples),
            policy_name="margin_only",
            threshold=float("inf"),
            target_recall=target_recall,
            split="calibration",
            weight_margin=1.0,
        )
    threshold = best[0]
    if math.isfinite(threshold):
        threshold += margin_guard
    metrics = evaluate_predictions(
        samples,
        [s.min_margin <= threshold for s in samples],
        policy_name="margin_only",
        threshold=threshold,
        target_recall=target_recall,
        split="calibration",
        weight_margin=1.0,
    )
    return threshold, metrics


def _calibrate_shadow_threshold(
    samples: list[WindowSample],
    target_recall: float,
) -> tuple[float, GateMetrics]:
    values = sorted({DualThresholdGate.shadow_risk(s) for s in samples})
    candidates = [*values, values[-1] + 1e-9]
    best: Optional[tuple[float, GateMetrics]] = None
    for threshold in candidates:
        predictions = [
            DualThresholdGate.shadow_risk(s) >= threshold for s in samples
        ]
        metrics = evaluate_predictions(
            samples,
            predictions,
            policy_name="shadow_only",
            threshold=threshold,
            target_recall=target_recall,
            split="calibration",
            weight_shadow=1.0,
            weight_jsd=1.0,
        )
        if metrics.window_recall + 1e-12 < target_recall:
            continue
        if best is None:
            best = (threshold, metrics)
        elif (
            metrics.trigger_rate,
            -metrics.window_precision,
        ) < (
            best[1].trigger_rate,
            -best[1].window_precision,
        ):
            best = (threshold, metrics)
    if best is None:
        return float("-inf"), evaluate_predictions(
            samples,
            [True] * len(samples),
            policy_name="shadow_only",
            threshold=float("-inf"),
            target_recall=target_recall,
            split="calibration",
            weight_shadow=1.0,
            weight_jsd=1.0,
        )
    return best


def calibrate_dual_gate(
    samples: list[WindowSample],
    *,
    target_recall: float = 1.0,
    margin_guard: float = DEFAULT_MARGIN_GUARD,
    min_shadow_auc: float = DEFAULT_MIN_SHADOW_AUC,
) -> tuple[DualThresholdGate, GateMetrics]:
    """Find the lowest-budget margin-or-shadow threshold pair."""

    from ssv4d.metrics import feature_auc

    shadow_auc = max(
        feature_auc(samples, "shadow_vote_disagreement"),
        feature_auc(samples, "shadow_jsd"),
    )
    margin_threshold, margin_metrics = _calibrate_margin_threshold(
        samples, target_recall, margin_guard
    )
    if not math.isfinite(shadow_auc) or shadow_auc <= min_shadow_auc:
        return (
            DualThresholdGate(
                margin_threshold=margin_threshold,
                shadow_threshold=float("inf"),
                shadow_enabled=False,
                shadow_auc=shadow_auc,
            ),
            margin_metrics,
        )

    margin_values = _limited_candidates(s.min_margin for s in samples)
    shadow_values = _limited_candidates(
        DualThresholdGate.shadow_risk(s) for s in samples
    )
    margin_candidates = [float("-inf"), *margin_values]
    shadow_candidates = [*shadow_values, shadow_values[-1] + 1e-9]

    best: Optional[tuple[DualThresholdGate, GateMetrics]] = None
    for margin_threshold in margin_candidates:
        for shadow_threshold in shadow_candidates:
            gate = DualThresholdGate(margin_threshold, shadow_threshold)
            predictions = [gate.trigger(s) for s in samples]
            metrics = evaluate_predictions(
                samples,
                predictions,
                policy_name="ssv4d",
                threshold=0.0,
                target_recall=target_recall,
                split="calibration",
                weight_margin=1.0,
                weight_shadow=1.0,
                weight_jsd=1.0,
            )
            if metrics.window_recall + 1e-12 < target_recall:
                continue
            if best is None or (
                metrics.trigger_rate,
                -metrics.window_precision,
            ) < (
                best[1].trigger_rate,
                -best[1].window_precision,
            ):
                best = (gate, metrics)

    if best is None:
        gate = DualThresholdGate(float("inf"), float("-inf"))
        metrics = evaluate_predictions(
            samples,
            [True] * len(samples),
            policy_name="ssv4d",
            threshold=0.0,
            target_recall=target_recall,
            split="calibration",
            weight_margin=1.0,
            weight_shadow=1.0,
            weight_jsd=1.0,
        )
        return gate, metrics
    selected_gate, _ = best
    guarded_gate = DualThresholdGate(
        margin_threshold=selected_gate.margin_threshold + margin_guard,
        shadow_threshold=selected_gate.shadow_threshold,
        shadow_enabled=True,
        shadow_auc=shadow_auc,
    )
    metrics = evaluate_dual_gate(
        guarded_gate,
        samples,
        target_recall=target_recall,
        split="calibration",
    )
    return guarded_gate, metrics


def evaluate_dual_gate(
    gate: DualThresholdGate,
    samples: list[WindowSample],
    *,
    target_recall: float,
    split: str,
) -> GateMetrics:
    return evaluate_predictions(
        samples,
        [gate.trigger(s) for s in samples],
        policy_name="ssv4d",
        threshold=0.0,
        target_recall=target_recall,
        split=split,
        weight_margin=1.0,
        weight_shadow=1.0,
        weight_jsd=1.0,
    )


def grouped_cross_validate(
    samples: list[WindowSample],
    *,
    target_recall: float = 1.0,
    margin_guard: float = DEFAULT_MARGIN_GUARD,
    min_shadow_auc: float = DEFAULT_MIN_SHADOW_AUC,
) -> list[GateMetrics]:
    """Leave-one-prompt-out evaluation for all gate policies."""

    prompt_ids = sorted({s.prompt_id for s in samples})
    if len(prompt_ids) < 2:
        raise ValueError("Grouped CV requires at least two prompt groups")

    policy_predictions: dict[str, list[bool]] = {
        "always_on": [],
        "margin_only": [],
        "shadow_only": [],
        "ssv4d": [],
    }
    ordered_samples: list[WindowSample] = []
    fold_configs: list[dict] = []

    for held_prompt in prompt_ids:
        calibration = [s for s in samples if s.prompt_id != held_prompt]
        held_out = [s for s in samples if s.prompt_id == held_prompt]
        if not calibration or not held_out:
            continue

        margin_threshold, _ = _calibrate_margin_threshold(
            calibration, target_recall, margin_guard
        )
        shadow_threshold, _ = _calibrate_shadow_threshold(
            calibration, target_recall
        )
        gate, _ = calibrate_dual_gate(
            calibration,
            target_recall=target_recall,
            margin_guard=margin_guard,
            min_shadow_auc=min_shadow_auc,
        )
        fold_configs.append(
            {
                "held_prompt": held_prompt,
                "margin_threshold": margin_threshold,
                "shadow_threshold": shadow_threshold,
                "dual_margin_threshold": gate.margin_threshold,
                "dual_shadow_threshold": gate.shadow_threshold,
            }
        )
        ordered_samples.extend(held_out)
        policy_predictions["always_on"].extend([True] * len(held_out))
        policy_predictions["margin_only"].extend(
            s.min_margin <= margin_threshold for s in held_out
        )
        policy_predictions["shadow_only"].extend(
            DualThresholdGate.shadow_risk(s) >= shadow_threshold
            for s in held_out
        )
        policy_predictions["ssv4d"].extend(gate.trigger(s) for s in held_out)

    results = []
    for policy, predictions in policy_predictions.items():
        metrics = evaluate_predictions(
            ordered_samples,
            predictions,
            policy_name=policy,
            threshold=0.0,
            target_recall=target_recall,
            split="grouped_cv",
            weight_margin=1.0 if policy in ("margin_only", "ssv4d") else 0.0,
            weight_shadow=1.0 if policy in ("shadow_only", "ssv4d") else 0.0,
            weight_jsd=1.0 if policy in ("shadow_only", "ssv4d") else 0.0,
        )
        metrics.extra = {"folds": fold_configs}
        results.append(metrics)
    return results


def _calibrate_spec(
    samples: list[WindowSample],
    normalizer: CompositeRiskGate,
    spec: PolicySpec,
    target_recall: float,
) -> tuple[PolicySpec, GateMetrics]:
    scores = np.asarray([normalizer.score(s, spec) for s in samples], dtype=float)
    if scores.size == 0:
        raise ValueError("Calibration set is empty")

    candidates = [float(scores.min() - 1e-9)] + sorted(
        {float(x) for x in scores}
    )
    best: Optional[tuple[PolicySpec, GateMetrics]] = None
    for threshold in candidates:
        predictions = (scores >= threshold).tolist()
        metrics = evaluate_predictions(
            samples,
            predictions,
            policy_name=spec.name,
            threshold=threshold,
            target_recall=target_recall,
            split="calibration",
            weight_margin=spec.weight_margin,
            weight_shadow=spec.weight_shadow,
            weight_jsd=spec.weight_jsd,
        )
        if metrics.window_recall + 1e-12 < target_recall:
            continue
        if best is None:
            best = (
                PolicySpec(
                    spec.name,
                    threshold,
                    spec.weight_margin,
                    spec.weight_shadow,
                    spec.weight_jsd,
                ),
                metrics,
            )
            continue
        old_spec, old_metrics = best
        key_new = (metrics.trigger_rate, -metrics.window_precision)
        key_old = (old_metrics.trigger_rate, -old_metrics.window_precision)
        if key_new < key_old:
            best = (
                PolicySpec(
                    spec.name,
                    threshold,
                    spec.weight_margin,
                    spec.weight_shadow,
                    spec.weight_jsd,
                ),
                metrics,
            )

    if best is None:
        # No threshold reaches the target. Fall back to always-on, which is the
        # safest operating point and makes the uncovered calibration region
        # explicit in the report.
        threshold = float(scores.min() - 1e-9)
        predictions = [True] * len(samples)
        metrics = evaluate_predictions(
            samples,
            predictions,
            policy_name=spec.name,
            threshold=threshold,
            target_recall=target_recall,
            split="calibration",
            weight_margin=spec.weight_margin,
            weight_shadow=spec.weight_shadow,
            weight_jsd=spec.weight_jsd,
        )
        best = (
            PolicySpec(
                spec.name,
                threshold,
                spec.weight_margin,
                spec.weight_shadow,
                spec.weight_jsd,
            ),
            metrics,
        )
    return best


def calibrate_and_evaluate(
    calibration: list[WindowSample],
    evaluation: list[WindowSample],
    *,
    target_recall: float = 1.0,
) -> tuple[CompositeRiskGate, list[GateMetrics]]:
    """Fit SSV4D and baseline thresholds on calibration windows."""

    if not calibration or not evaluation:
        raise ValueError("Calibration and evaluation sets must both be non-empty")

    normalizer = CompositeRiskGate(threshold=0.0)
    normalizer.fit_normalizer(calibration)

    specs = [
        PolicySpec(
            name="always_on",
            threshold=float("-inf"),
            weight_margin=0.0,
            weight_shadow=0.0,
            weight_jsd=0.0,
        ),
        PolicySpec(
            name="margin_only",
            threshold=0.0,
            weight_margin=1.0,
            weight_shadow=0.0,
            weight_jsd=0.0,
        ),
        PolicySpec(
            name="shadow_only",
            threshold=0.0,
            weight_margin=0.0,
            weight_shadow=0.0,
            weight_jsd=1.0,
        ),
        PolicySpec(
            name="ssv4d",
            threshold=0.0,
            weight_margin=1.0,
            weight_shadow=0.5,
            weight_jsd=0.25,
        ),
    ]

    results: list[GateMetrics] = []
    selected_gate: Optional[CompositeRiskGate] = None
    for spec in specs:
        calibrated, calibration_metrics = _calibrate_spec(
            calibration, normalizer, spec, target_recall
        )
        results.append(calibration_metrics)

        if spec.name == "always_on":
            evaluation_predictions = [True] * len(evaluation)
        else:
            evaluation_predictions = [
                normalizer.score(sample, calibrated) >= calibrated.threshold
                for sample in evaluation
            ]
        metrics = evaluate_predictions(
            evaluation,
            evaluation_predictions,
            policy_name=spec.name,
            threshold=calibrated.threshold,
            target_recall=target_recall,
            split="evaluation",
            weight_margin=calibrated.weight_margin,
            weight_shadow=calibrated.weight_shadow,
            weight_jsd=calibrated.weight_jsd,
        )
        results.append(metrics)

        if spec.name == "ssv4d":
            selected_gate = CompositeRiskGate(
                threshold=calibrated.threshold,
                median=dict(normalizer.median),
                scale=dict(normalizer.scale),
                weight_margin=calibrated.weight_margin,
                weight_shadow=calibrated.weight_shadow,
                weight_jsd=calibrated.weight_jsd,
                policy_name=spec.name,
            )

    assert selected_gate is not None
    return selected_gate, results
