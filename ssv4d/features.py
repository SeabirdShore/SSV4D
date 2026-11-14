"""Numerical features for SSV4D runtime and prefill experiments."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch


def _to_list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def topk_dict(
    logprobs: Sequence[float] | torch.Tensor,
    token_ids: Sequence[int] | torch.Tensor,
) -> dict[int, float]:
    """Return a vocabulary-aligned top-k logprob dictionary."""

    lps = _to_list(logprobs) or []
    ids = _to_list(token_ids) or []
    result = {}
    for token_id, logprob in zip(ids, lps):
        if logprob is None or not math.isfinite(float(logprob)):
            continue
        result[int(token_id)] = float(logprob)
    return result


def topk_margin(
    logprobs: Sequence[float] | torch.Tensor,
    token_ids: Sequence[int] | torch.Tensor,
) -> float:
    """Top-1/top-2 logprob margin, or zero when undefined."""

    values = sorted(topk_dict(logprobs, token_ids).values(), reverse=True)
    return values[0] - values[1] if len(values) >= 2 else 0.0


def vocab_aligned_divergence(
    canonical_logprobs: Sequence[float] | torch.Tensor,
    canonical_ids: Sequence[int] | torch.Tensor,
    shadow_logprobs: Sequence[float] | torch.Tensor,
    shadow_ids: Sequence[int] | torch.Tensor,
    *,
    missing_logprob: float = -20.0,
) -> dict[str, float]:
    """KL/JSD between two top-k distributions after aligning token IDs.

    Tokens absent from one top-k list receive a small floor probability. The
    union is renormalized, so the metric measures disagreement inside the
    observed leading support rather than assuming a complete vocabulary vector.
    """

    canonical = topk_dict(canonical_logprobs, canonical_ids)
    shadow = topk_dict(shadow_logprobs, shadow_ids)
    if not canonical or not shadow:
        return {
            "kl_canonical_shadow": float("nan"),
            "kl_shadow_canonical": float("nan"),
            "jsd": float("nan"),
            "top1_agreement": float("nan"),
            "margin_delta": float("nan"),
        }

    vocabulary = sorted(set(canonical) | set(shadow))
    p_values = [math.exp(canonical.get(token, missing_logprob)) for token in vocabulary]
    q_values = [math.exp(shadow.get(token, missing_logprob)) for token in vocabulary]
    p_sum = sum(p_values)
    q_sum = sum(q_values)
    p = [value / p_sum for value in p_values]
    q = [value / q_sum for value in q_values]
    m = [(a + b) / 2 for a, b in zip(p, q)]

    def kl(x: Iterable[float], y: Iterable[float]) -> float:
        value = 0.0
        for a, b in zip(x, y):
            if a > 0 and b > 0:
                value += a * math.log(a / b)
        return value

    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    canonical_top = max(canonical, key=canonical.get)
    shadow_top = max(shadow, key=shadow.get)
    canonical_margin = topk_margin(canonical_logprobs, canonical_ids)
    shadow_margin = topk_margin(shadow_logprobs, shadow_ids)
    return {
        "kl_canonical_shadow": kl(p, q),
        "kl_shadow_canonical": kl(q, p),
        "jsd": jsd,
        "top1_agreement": float(canonical_top == shadow_top),
        "margin_delta": shadow_margin - canonical_margin,
    }


def kv_column_deviation(
    canonical_k: torch.Tensor,
    canonical_v: torch.Tensor,
    shadow_k: torch.Tensor,
    shadow_v: torch.Tensor,
) -> dict[str, float]:
    """Measure current-column K/V movement between two executions."""

    with torch.no_grad():
        ck = canonical_k.detach().float()
        cv = canonical_v.detach().float()
        sk = shadow_k.detach().float()
        sv = shadow_v.detach().float()
        eps = 1e-12
        k_l2 = torch.linalg.vector_norm(sk - ck).item()
        v_l2 = torch.linalg.vector_norm(sv - cv).item()
        k_base = torch.linalg.vector_norm(ck).item()
        v_base = torch.linalg.vector_norm(cv).item()
        k_cos = torch.nn.functional.cosine_similarity(
            ck.reshape(-1), sk.reshape(-1), dim=0
        ).item()
        v_cos = torch.nn.functional.cosine_similarity(
            cv.reshape(-1), sv.reshape(-1), dim=0
        ).item()
    return {
        "k_l2": k_l2,
        "v_l2": v_l2,
        "k_relative_l2": k_l2 / (k_base + eps),
        "v_relative_l2": v_l2 / (v_base + eps),
        "k_cosine": k_cos,
        "v_cosine": v_cos,
    }


def aggregate_feature_rows(rows: list[dict]) -> dict[str, float]:
    """Aggregate numeric feature rows with mean/min/max."""

    if not rows:
        return {}
    keys = {
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    result = {}
    for key in keys:
        values = [
            float(row[key])
            for row in rows
            if key in row and math.isfinite(float(row[key]))
        ]
        if not values:
            continue
        result[f"{key}_mean"] = sum(values) / len(values)
        result[f"{key}_min"] = min(values)
        result[f"{key}_max"] = max(values)
    return result
