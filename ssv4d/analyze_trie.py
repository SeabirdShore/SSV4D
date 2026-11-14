"""Analyze observed completion variation as prefix tries."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


TOPK_LEVELS = (1, 2, 3, 4, 8)


def first_mismatch(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def token_rank(token: int, top_ids: list[int]) -> int | None:
    try:
        return top_ids.index(token) + 1
    except ValueError:
        return None


def build_trie(sequences: Iterable[list[int]]) -> dict[str, Any]:
    root: dict[int, dict] = {}
    unique_sequences = set()
    total_tokens = 0
    for sequence in sequences:
        token_tuple = tuple(sequence)
        unique_sequences.add(token_tuple)
        total_tokens += len(sequence)
        node = root
        for token in sequence:
            node = node.setdefault(int(token), {})

    node_count = 0
    branch_depths = []
    frontier = Counter()
    max_degree = 0

    stack = [(child, 1) for child in root.values()]
    while stack:
        node, depth = stack.pop()
        node_count += 1
        frontier[depth] += 1
        degree = len(node)
        max_degree = max(max_degree, degree)
        if degree > 1:
            branch_depths.append(depth)
        stack.extend((child, depth + 1) for child in node.values())

    return {
        "node_count": node_count,
        "branch_nodes": len(branch_depths),
        "branch_depths": branch_depths,
        "max_degree": max_degree,
        "max_frontier_width": max(frontier.values(), default=0),
        "frontier_width_sum": sum(frontier.values()),
        "unique_sequences": len(unique_sequences),
        "uncompressed_tokens": total_tokens,
        "sharing_ratio": (
            1.0 - node_count / total_tokens if total_tokens else 0.0
        ),
    }


def _rank_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [row["rank"] for row in rows]
    return {
        "branch_pairs": len(rows),
        "topk_counts": {
            str(k): sum(rank is not None and rank <= k for rank in ranks)
            for k in TOPK_LEVELS
        },
        "topk_coverage": {
            str(k): (
                sum(rank is not None and rank <= k for rank in ranks)
                / len(ranks)
                if ranks
                else 1.0
            )
            for k in TOPK_LEVELS
        },
        "rank_counts": {
            "out": sum(rank is None for rank in ranks),
            **{
                str(rank): count
                for rank, count in sorted(Counter(
                    value for value in ranks if value is not None
                ).items())
            },
        },
        "positions": [row["position"] for row in rows],
    }


def analyze(
    payload: dict[str, Any],
    *,
    max_length: int | None = None,
) -> dict[str, Any]:
    shadow_branches = []
    verifier_branches = []
    trie_rows = []
    margin_rows = []
    verifier_reuse = Counter()

    for result in payload.get("results", []):
        canonical = result["canonical"]
        canonical_ids = canonical["output_ids"]
        if max_length is not None:
            canonical_ids = canonical_ids[:max_length]
        canonical_top = canonical.get("top_ids", [])
        canonical_logprobs = canonical.get("top_logprobs", [])
        sequences = [canonical_ids]
        shadow_branch_tokens = {}
        branch_positions = set()

        for index, shadow in enumerate(result.get("shadows", [])):
            shadow_ids = shadow["output_ids"]
            if max_length is not None:
                shadow_ids = shadow_ids[:max_length]
            sequences.append(shadow_ids)
            position = first_mismatch(canonical_ids, shadow_ids)
            if position is None:
                continue
            top_ids = (
                canonical_top[position]
                if position < len(canonical_top)
                else []
            )
            target = shadow_ids[position]
            shadow_branch_tokens[(position, target)] = True
            branch_positions.add(position)
            shadow_branches.append(
                {
                    "prompt_id": result["prompt_id"],
                    "shadow_index": index,
                    "position": position,
                    "rank": token_rank(target, top_ids),
                    "target": target,
                    "canonical": canonical_ids[position],
                }
            )

        verifier = result.get("verifier")
        if verifier is not None:
            verifier_ids = verifier["output_ids"]
            if max_length is not None:
                verifier_ids = verifier_ids[:max_length]
            sequences.append(verifier_ids)
            position = first_mismatch(canonical_ids, verifier_ids)
            if position is not None:
                top_ids = (
                    canonical_top[position]
                    if position < len(canonical_top)
                    else []
                )
                target = verifier_ids[position]
                branch_positions.add(position)
                verifier_reuse["branches"] += 1
                if (position, target) in shadow_branch_tokens:
                    verifier_reuse["reused_from_shadow"] += 1
                verifier_branches.append(
                    {
                        "prompt_id": result["prompt_id"],
                        "position": position,
                        "rank": token_rank(target, top_ids),
                        "target": target,
                        "canonical": canonical_ids[position],
                    }
                )

        trie = build_trie(sequences)
        for position, logprobs in enumerate(canonical_logprobs):
            if len(logprobs) < 2:
                continue
            margin_rows.append(
                {
                    "margin": float(logprobs[0] - logprobs[1]),
                    "branch": position in branch_positions,
                }
            )
        trie_rows.append(
            {
                "prompt_id": result["prompt_id"],
                **trie,
            }
        )

    shadow_rank_stats = _rank_stats(shadow_branches)
    verifier_rank_stats = _rank_stats(verifier_branches)
    shadow_positions = shadow_rank_stats.pop("positions")
    verifier_positions = verifier_rank_stats.pop("positions")

    def summary(values: list[float]) -> dict[str, float]:
        return {
            "mean": statistics.mean(values) if values else 0.0,
            "median": statistics.median(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }

    margin_thresholds = {}
    for threshold in (0.0625, 0.125, 0.25, 0.375, 0.5):
        predictions = [
            row["margin"] <= threshold for row in margin_rows
        ]
        labels = [row["branch"] for row in margin_rows]
        tp = sum(pred and label for pred, label in zip(predictions, labels))
        fp = sum(pred and not label for pred, label in zip(predictions, labels))
        fn = sum(
            not pred and label for pred, label in zip(predictions, labels)
        )
        margin_thresholds[str(threshold)] = {
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "trigger_rate": (
                sum(predictions) / len(predictions) if predictions else 0.0
            ),
            "false_positives": fp,
        }

    window = int(max_length or payload.get("window") or 0)
    window_calls = math.ceil(window / 64) if window else 0

    return {
        "num_prompts": payload.get("num_prompts", len(trie_rows)),
        "completion_tokens": window,
        "batch_size": payload.get("batch_size"),
        "requests_with_shadow_branch": len(
            {row["prompt_id"] for row in shadow_branches}
        ),
        "shadow_vs_canonical": shadow_rank_stats,
        "verifier_vs_canonical": verifier_rank_stats,
        "verifier_branch_reuse": {
            "branches": verifier_reuse["branches"],
            "reused_from_shadow": verifier_reuse["reused_from_shadow"],
            "reuse_rate": (
                verifier_reuse["reused_from_shadow"]
                / verifier_reuse["branches"]
                if verifier_reuse["branches"]
                else 0.0
            ),
        },
        "margin_policy": {
            "positions": len(margin_rows),
            "branch_positions": sum(row["branch"] for row in margin_rows),
            "thresholds": margin_thresholds,
        },
        "tree_verification": {
            "window_size": 64,
            "flat_window_calls": sum(
                row["unique_sequences"] * window_calls for row in trie_rows
            ),
            "tree_window_calls": len(trie_rows) * window_calls,
            "flat_nodes": sum(
                row["uncompressed_tokens"] for row in trie_rows
            ),
            "tree_nodes": sum(row["node_count"] for row in trie_rows),
        },
        "first_branch_position": summary(shadow_positions),
        "verifier_first_branch_position": summary(verifier_positions),
        "trie": {
            "node_count": summary(
                [float(row["node_count"]) for row in trie_rows]
            ),
            "branch_nodes": summary(
                [float(row["branch_nodes"]) for row in trie_rows]
            ),
            "max_degree": max(
                (row["max_degree"] for row in trie_rows), default=0
            ),
            "max_frontier_width": max(
                (row["max_frontier_width"] for row in trie_rows), default=0
            ),
            "sharing_ratio": summary(
                [float(row["sharing_ratio"]) for row in trie_rows]
            ),
            "per_prompt": trie_rows,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--lengths",
        help="Comma-separated prefix lengths to analyze, e.g. 128,512,1024",
    )
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if args.lengths:
        results = {
            str(length): analyze(payload, max_length=length)
            for length in (
                int(value) for value in args.lengths.split(",")
            )
        }
        result = {
            "input": str(args.input),
            "lengths": results,
        }
    else:
        result = analyze(payload)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
