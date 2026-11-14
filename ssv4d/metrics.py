"""Sparse-error metrics and window feature extraction."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable, Optional

import numpy as np

from ssv4d.models import GenerationTrace, WindowSample


def first_mismatch(reference: list[int], candidate: list[int]) -> int:
    n = min(len(reference), len(candidate))
    for index in range(n):
        if reference[index] != candidate[index]:
            return index
    return n


def _nanpercentile(values: Iterable[float], q: float, default: float) -> float:
    finite = [x for x in values if math.isfinite(x)]
    return float(np.percentile(finite, q)) if finite else default


def _finite(values: Iterable[float], default: float = 0.0) -> list[float]:
    return [float(x) for x in values if math.isfinite(float(x))]


def top1_vote_disagreement(top_ids: Iterable[list[int]]) -> float:
    votes = [row[0] for row in top_ids if row]
    if len(votes) <= 1:
        return 0.0
    counts = Counter(votes)
    return 1.0 - max(counts.values()) / len(votes)


def topk_js_divergence(top_logprobs: Iterable[list[float]]) -> float:
    """Jensen-Shannon divergence among normalized top-k distributions."""

    distributions: list[np.ndarray] = []
    for row in top_logprobs:
        vals = np.asarray([x for x in row if math.isfinite(x)], dtype=np.float64)
        if vals.size < 2:
            continue
        vals = vals - np.max(vals)
        probs = np.exp(vals)
        distributions.append(probs / probs.sum())
    if len(distributions) <= 1:
        return 0.0

    # Rows are top-k ranked, not vocabulary-aligned. Comparing rank-normalized
    # mass captures shape disagreement while top1-vote disagreement captures
    # identity disagreement.
    width = min(len(row) for row in distributions)
    if width < 2:
        return 0.0
    matrix = np.stack([row[:width] for row in distributions])
    mean = matrix.mean(axis=0)

    def entropy(p: np.ndarray) -> float:
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    mean_entropy = entropy(mean)
    component_entropy = float(np.mean([entropy(row) for row in matrix]))
    return max(0.0, mean_entropy - component_entropy)


def summarize_token_flips(
    references: dict[str, GenerationTrace],
    shadows: dict[str, list[GenerationTrace]],
) -> tuple[list[dict], dict]:
    records: list[dict] = []
    flip_margins: list[float] = []
    stable_margins: list[float] = []
    positions: list[int] = []
    n_sync_steps = 0

    for prompt_id, reference in references.items():
        for trace in shadows.get(prompt_id, []):
            mismatch_pos = first_mismatch(reference.output_ids, trace.output_ids)
            n_sync_steps += mismatch_pos
            for pos in range(min(mismatch_pos + 1, len(trace.margins))):
                margin = trace.margins[pos]
                if math.isfinite(margin):
                    if pos == mismatch_pos and mismatch_pos < len(trace.output_ids):
                        flip_margins.append(margin)
                    elif pos < mismatch_pos:
                        stable_margins.append(margin)
            if mismatch_pos < len(reference.output_ids):
                positions.append(mismatch_pos)
            records.append(
                {
                    "prompt_id": prompt_id,
                    "plan_name": trace.plan_name,
                    "scenario": trace.scenario,
                    "batch_size": trace.batch_size,
                    "protected_index": trace.protected_index,
                    "first_mismatch": mismatch_pos,
                    "is_flip": mismatch_pos < len(reference.output_ids),
                    "prompt_tokens": len(reference.output_ids),
                }
            )

    total_traces = sum(len(v) for v in shadows.values())
    summary = {
        "n_reference_prompts": len(references),
        "n_shadow_traces": total_traces,
        "n_synchronous_steps": n_sync_steps,
        "n_flip_events": len(positions),
        "token_flip_rate": (
            len(positions) / n_sync_steps if n_sync_steps else 0.0
        ),
        "flip_margin_median": _nanpercentile(flip_margins, 50, float("nan")),
        "stable_margin_median": _nanpercentile(stable_margins, 50, float("nan")),
        "flip_positions": positions,
        "flip_position_median": _nanpercentile(positions, 50, float("nan")),
    }
    return records, summary


def build_window_dataset(
    references: dict[str, GenerationTrace],
    shadows: dict[str, list[GenerationTrace]],
    *,
    window_size: int = 32,
) -> list[WindowSample]:
    """Aggregate token-level observations into verification-window features."""

    samples: list[WindowSample] = []
    for prompt_id, reference in references.items():
        traces = shadows.get(prompt_id, [])
        if not traces:
            continue
        n_tokens = len(reference.output_ids)
        for window_index, start in enumerate(range(0, n_tokens, window_size)):
            end = min(n_tokens, start + window_size)
            mismatch_positions: list[int] = []
            trace_slices: list[GenerationTrace] = []
            slice_starts: list[int] = []

            for trace in traces:
                mismatch_pos = first_mismatch(reference.output_ids, trace.output_ids)
                if mismatch_pos < start:
                    # This trace had already branched before the window.
                    continue
                if mismatch_pos < end:
                    mismatch_positions.append(mismatch_pos)
                trace_slices.append(trace)
                slice_starts.append(start)

            if not trace_slices:
                continue

            per_position_margins: list[float] = []
            per_position_vote: list[float] = []
            per_position_jsd: list[float] = []
            reference_shadow_vote: list[float] = []
            reference_shadow_jsd: list[float] = []
            per_trace_margin_means: list[float] = []
            spread: list[float] = []

            for pos in range(start, end):
                margins = [
                    trace.margins[pos]
                    for trace in trace_slices
                    if pos < len(trace.margins)
                    and math.isfinite(trace.margins[pos])
                ]
                if margins:
                    per_position_margins.append(float(min(margins)))
                    spread.append(float(max(margins) - min(margins)))

                top_ids = [
                    trace.top_ids[pos]
                    for trace in trace_slices
                    if pos < len(trace.top_ids) and trace.top_ids[pos]
                ]
                if top_ids:
                    per_position_vote.append(top1_vote_disagreement(top_ids))

                top_lps = [
                    trace.top_logprobs[pos]
                    for trace in trace_slices
                    if pos < len(trace.top_logprobs) and trace.top_logprobs[pos]
                ]
                if top_lps:
                    per_position_jsd.append(topk_js_divergence(top_lps))

                reference_top = (
                    reference.top_ids[pos]
                    if pos < len(reference.top_ids)
                    else []
                )
                if reference_top:
                    comparisons = [
                        trace.top_ids[pos]
                        for trace in trace_slices
                        if pos < len(trace.top_ids) and trace.top_ids[pos]
                    ]
                    if comparisons:
                        reference_vote = sum(
                            row[0] != reference_top[0]
                            for row in comparisons
                            if row
                        ) / len(comparisons)
                        reference_shadow_vote.append(reference_vote)

                reference_lps = (
                    reference.top_logprobs[pos]
                    if pos < len(reference.top_logprobs)
                    else []
                )
                if reference_lps:
                    comparison_lps = [
                        trace.top_logprobs[pos]
                        for trace in trace_slices
                        if pos < len(trace.top_logprobs)
                        and trace.top_logprobs[pos]
                    ]
                    if comparison_lps:
                        reference_shadow_jsd.append(
                            topk_js_divergence([reference_lps, *comparison_lps])
                        )

            if not per_position_margins:
                continue
            for trace in trace_slices:
                row = _finite(trace.margins[start:end])
                if row:
                    per_trace_margin_means.append(float(np.mean(row)))

            samples.append(
                WindowSample(
                    prompt_id=prompt_id,
                    window_index=window_index,
                    start=start,
                    end=end,
                    label=int(bool(mismatch_positions)),
                    mismatch_positions=sorted(set(mismatch_positions)),
                    min_margin=float(min(per_position_margins)),
                    q10_margin=_nanpercentile(per_position_margins, 10, 1e9),
                    mean_margin=float(np.mean(per_position_margins)),
                    margin_spread=float(max(spread, default=0.0)),
                    shadow_vote_disagreement=float(
                        max(per_position_vote, default=0.0)
                    ),
                    shadow_jsd=float(max(per_position_jsd, default=0.0)),
                    n_traces=len(trace_slices),
                    reference_shadow_jsd=float(
                        max(reference_shadow_jsd, default=0.0)
                    ),
                    reference_shadow_vote=float(
                        max(reference_shadow_vote, default=0.0)
                    ),
                )
            )
    return samples


def build_token_dataset(
    references: dict[str, GenerationTrace],
    shadows: dict[str, list[GenerationTrace]],
) -> list[WindowSample]:
    """Build position-level verification samples.

    Each sample represents one synchronous decode position. Traces that have
    already branched are excluded from that position. The label is positive if
    any active shadow trace diverges from the deterministic reference at this
    exact position.
    """

    samples: list[WindowSample] = []
    for prompt_id, reference in references.items():
        traces = shadows.get(prompt_id, [])
        if not traces:
            continue
        mismatch_positions = [
            first_mismatch(reference.output_ids, trace.output_ids)
            for trace in traces
        ]
        for pos in range(len(reference.output_ids)):
            active = [
                (trace, mismatch)
                for trace, mismatch in zip(traces, mismatch_positions)
                if mismatch >= pos
            ]
            if not active:
                continue
            label = int(any(mismatch == pos for _, mismatch in active))
            margins = [
                trace.margins[pos]
                for trace, _ in active
                if pos < len(trace.margins)
                and math.isfinite(trace.margins[pos])
            ]
            if not margins:
                continue
            top_ids = [
                trace.top_ids[pos]
                for trace, _ in active
                if pos < len(trace.top_ids) and trace.top_ids[pos]
            ]
            top_lps = [
                trace.top_logprobs[pos]
                for trace, _ in active
                if pos < len(trace.top_logprobs)
                and trace.top_logprobs[pos]
            ]
            samples.append(
                WindowSample(
                    prompt_id=prompt_id,
                    window_index=pos,
                    start=pos,
                    end=pos + 1,
                    label=label,
                    mismatch_positions=[pos] if label else [],
                    min_margin=float(min(margins)),
                    q10_margin=_nanpercentile(margins, 10, 1e9),
                    mean_margin=float(np.mean(margins)),
                    margin_spread=float(max(margins) - min(margins)),
                    shadow_vote_disagreement=top1_vote_disagreement(top_ids),
                    shadow_jsd=topk_js_divergence(top_lps),
                    n_traces=len(active),
                )
            )
    return samples


def feature_auc(
    samples: list[WindowSample],
    feature: str,
    *,
    higher_is_riskier: bool = True,
) -> float:
    """Mann-Whitney AUC without a scikit-learn dependency."""

    values = np.asarray([float(getattr(s, feature)) for s in samples])
    labels = np.asarray([bool(s.label) for s in samples])
    positives = values[labels]
    negatives = values[~labels]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    comparisons = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    auc = (comparisons + 0.5 * ties) / (positives.size * negatives.size)
    return float(auc if higher_is_riskier else 1.0 - auc)


def split_by_prompt(
    samples: list[WindowSample],
    *,
    seed: int = 2026,
    calibration_fraction: float = 0.5,
) -> tuple[list[WindowSample], list[WindowSample]]:
    prompt_ids = sorted({sample.prompt_id for sample in samples})
    if len(prompt_ids) < 2:
        raise ValueError("At least two prompt groups are required for calibration")
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(prompt_ids))
    n_cal = max(1, min(len(shuffled) - 1, int(len(shuffled) * calibration_fraction)))
    calibration_ids = set(shuffled[:n_cal])
    calibration = [s for s in samples if s.prompt_id in calibration_ids]
    evaluation = [s for s in samples if s.prompt_id not in calibration_ids]
    return calibration, evaluation


def aggregate_token_margin_buckets(
    records: list[dict],
    references: dict[str, GenerationTrace],
    shadows: dict[str, list[GenerationTrace]],
) -> dict[str, list[float]]:
    """Return margin distributions for stable and flipped synchronous steps."""

    stable: list[float] = []
    flipped: list[float] = []
    for record in records:
        prompt_id = record["prompt_id"]
        ref = references[prompt_id]
        trace = next(
            (
                t
                for t in shadows[prompt_id]
                if t.plan_name == record["plan_name"]
            ),
            None,
        )
        if trace is None:
            continue
        mismatch = record["first_mismatch"]
        for pos, margin in enumerate(trace.margins[: mismatch + 1]):
            if not math.isfinite(margin):
                continue
            if record["is_flip"] and pos == mismatch:
                flipped.append(margin)
            else:
                stable.append(margin)
    return {"stable": stable, "flipped": flipped}
