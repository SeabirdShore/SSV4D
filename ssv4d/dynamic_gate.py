"""Reference-free dynamic verification threshold from parallel fast drafts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

from ssv4d.features import topk_dict, topk_margin
from ssv4d.models import GenerationTrace


@dataclass(frozen=True)
class DynamicPositionRisk:
    position: int
    canonical_margin: float
    dynamic_threshold: float
    top1_disagreement: bool
    branch_divergence: bool
    should_verify: bool
    reason: str


@dataclass(frozen=True)
class DynamicWindowDecision:
    should_verify: bool
    trigger_position: int | None
    trigger_threshold: float
    trigger_reason: str
    max_threshold: float
    n_positions: int
    positions: tuple[DynamicPositionRisk, ...]

    def to_dict(self) -> dict:
        result = asdict(self)
        for key in ("trigger_threshold", "max_threshold"):
            if not math.isfinite(float(result[key])):
                result[key] = None
        result["positions"] = []
        for position in self.positions:
            row = asdict(position)
            for key in (
                "canonical_margin",
                "dynamic_threshold",
            ):
                if not math.isfinite(float(row[key])):
                    row[key] = None
            result["positions"].append(row)
        return result


def _valid_topk(logprobs, token_ids) -> bool:
    return len(topk_dict(logprobs, token_ids)) >= 2


def _first_token_id(top_ids) -> int | None:
    return int(top_ids[0]) if top_ids else None


def evaluate_dynamic_window(
    canonical: GenerationTrace,
    shadows: Sequence[GenerationTrace],
    *,
    window: int | None = None,
) -> DynamicWindowDecision:
    """Compute a per-position margin threshold from parallel fast drafts.

    At position ``t`` the dynamic threshold is the largest absolute margin
    difference between the canonical fast draft and any perturbed fast draft::

        tau_t = max_shadow |margin(canonical, t) - margin(shadow, t)|

    Verification is requested when the canonical top-1/top-2 margin is not
    larger than that perturbation-induced margin uncertainty. Top-1 identity
    disagreement and sampled-token divergence also trigger verification.

    No absolute margin threshold or calibration constant is used. The threshold
    is recomputed from the four fast-draft branches for every token position.
    """

    if not shadows:
        return DynamicWindowDecision(
            should_verify=True,
            trigger_position=None,
            trigger_threshold=float("inf"),
            trigger_reason="no_shadow",
            max_threshold=float("inf"),
            n_positions=0,
            positions=(),
        )

    end = min(
        len(canonical.output_ids),
        *(len(shadow.output_ids) for shadow in shadows),
    )
    if window is not None:
        end = min(end, window)

    positions: list[DynamicPositionRisk] = []
    trigger = None
    for position in range(end):
        canonical_margin = topk_margin(
            canonical.top_logprobs[position],
            canonical.top_ids[position],
        )
        if not _valid_topk(
            canonical.top_logprobs[position],
            canonical.top_ids[position],
        ):
            risk = DynamicPositionRisk(
                position=position,
                canonical_margin=canonical_margin,
                dynamic_threshold=float("inf"),
                top1_disagreement=False,
                branch_divergence=False,
                should_verify=True,
                reason="missing_canonical_logprob",
            )
        else:
            shadow_margins: list[float] = []
            top1_disagreement = False
            branch_divergence = False
            canonical_top1 = _first_token_id(canonical.top_ids[position])
            for shadow in shadows:
                has_logprob = _valid_topk(
                    shadow.top_logprobs[position],
                    shadow.top_ids[position],
                )
                if not has_logprob:
                    shadow_margins.append(float("nan"))
                else:
                    shadow_margins.append(
                        topk_margin(
                            shadow.top_logprobs[position],
                            shadow.top_ids[position],
                        )
                    )
                shadow_top1 = _first_token_id(shadow.top_ids[position])
                if (
                    canonical_top1 is not None
                    and shadow_top1 is not None
                    and canonical_top1 != shadow_top1
                ):
                    top1_disagreement = True
                if shadow.output_ids[position] != canonical.output_ids[position]:
                    branch_divergence = True

            finite_margins = [
                value for value in shadow_margins if math.isfinite(value)
            ]
            missing_shadow_logprob = len(finite_margins) != len(shadow_margins)
            uncertainty = max(
                (
                    abs(canonical_margin - shadow_margin)
                    for shadow_margin in finite_margins
                ),
                default=float("inf"),
            )
            margin_risk = canonical_margin <= uncertainty
            should_verify = (
                margin_risk
                or top1_disagreement
                or branch_divergence
                or missing_shadow_logprob
            )
            if missing_shadow_logprob:
                reason = "missing_shadow_logprob"
            elif branch_divergence:
                reason = "branch_divergence"
            elif top1_disagreement:
                reason = "top1_disagreement"
            elif margin_risk:
                reason = "margin_within_shadow_uncertainty"
            else:
                reason = "stable"
            risk = DynamicPositionRisk(
                position=position,
                canonical_margin=canonical_margin,
                dynamic_threshold=uncertainty,
                top1_disagreement=top1_disagreement,
                branch_divergence=branch_divergence,
                should_verify=should_verify,
                reason=reason,
            )
        positions.append(risk)
        if trigger is None and risk.should_verify:
            trigger = risk

    if trigger is None:
        return DynamicWindowDecision(
            should_verify=False,
            trigger_position=None,
            trigger_threshold=float("nan"),
            trigger_reason="stable",
            max_threshold=max(
                (position.dynamic_threshold for position in positions),
                default=0.0,
            ),
            n_positions=len(positions),
            positions=tuple(positions),
        )
    return DynamicWindowDecision(
        should_verify=True,
        trigger_position=trigger.position,
        trigger_threshold=trigger.dynamic_threshold,
        trigger_reason=trigger.reason,
        max_threshold=max(
            (position.dynamic_threshold for position in positions),
            default=trigger.dynamic_threshold,
        ),
        n_positions=len(positions),
        positions=tuple(positions),
    )
