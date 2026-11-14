"""Audit verified-frontier transitions in SSV4D runtime logs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def analyze(path: Path) -> dict:
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    final = [event for event in events if event.get("event") == "frontier_final"]
    transitions = []
    for event in events:
        if event.get("event") != "verify":
            continue
        for frontier in event.get("frontiers") or []:
            transitions.append(frontier)

    by_rid = defaultdict(list)
    for frontier in transitions:
        by_rid[frontier["rid"]].append(frontier)
    monotonic_violations = []
    for rid, rows in by_rid.items():
        previous = 0
        for row in rows:
            current = int(row["verified_tokens"])
            if current < previous:
                monotonic_violations.append(
                    {
                        "rid": rid,
                        "previous": previous,
                        "current": current,
                    }
                )
            previous = current

    return {
        "path": str(path),
        "event_counts": dict(Counter(event.get("event") for event in events)),
        "final_requests": len(final),
        "fully_verified_final": sum(
            event["verified_tokens"] == event["output_tokens"]
            for event in final
        ),
        "final_unverified_tokens": sum(
            event["unverified_tokens"] for event in final
        ),
        "verify_transitions": len(transitions),
        "rollback_transitions": sum(
            int(row.get("rollback_tokens", 0)) > 0
            for row in transitions
        ),
        "monotonic_violations": monotonic_violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ssv4d/results/frontier_audit_summary.json"),
    )
    args = parser.parse_args()
    result = [analyze(path) for path in args.inputs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
