"""Replay audit traces through the trie verified-frontier state machine."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from ssv4d.analyze_trie import first_mismatch
from ssv4d.trie_frontier import PrefixTrie


def validate(payload: dict) -> dict:
    node_counts = []
    branch_counts = []
    frontier_widths = []
    requested = 0
    valid = 0

    for result in payload.get("results", []):
        canonical = result["canonical"]["output_ids"]
        verifier = result["verifier"]["output_ids"]
        trie = PrefixTrie()
        trie.insert(canonical, source="canonical")
        for shadow in result.get("shadows", []):
            trie.insert(shadow["output_ids"], source="shadow")
        trie.verify_path(verifier)

        mismatch = first_mismatch(canonical, verifier)
        expected_prefix = (
            len(canonical) if mismatch is None else mismatch
        )
        rollback = trie.longest_verified_prefix(canonical)
        stats = trie.stats()
        node_counts.append(stats["nodes"])
        branch_counts.append(stats["branch_nodes"])
        frontier_widths.append(stats["max_frontier_width"])
        requested += 1
        if (
            rollback.depth == expected_prefix
            and trie.node_for_path(verifier) is not None
        ):
            valid += 1

    return {
        "requests": requested,
        "valid_verified_frontiers": valid,
        "invalid_verified_frontiers": requested - valid,
        "nodes": {
            "mean": statistics.mean(node_counts) if node_counts else 0.0,
            "max": max(node_counts, default=0),
        },
        "branch_nodes": {
            "mean": statistics.mean(branch_counts) if branch_counts else 0.0,
            "max": max(branch_counts, default=0),
        },
        "max_frontier_width": max(frontier_widths, default=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = {}
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        results[str(path)] = validate(payload)

    text = json.dumps(results, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
