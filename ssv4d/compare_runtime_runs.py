"""Compare two SSV4D runtime benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path, round_index: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["rounds"][round_index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    baseline = _load(args.baseline, args.round)
    candidate = _load(args.candidate, args.round)
    baseline_digests = {
        row["prompt_id"]: row["digest"] for row in baseline["requests"]
    }
    candidate_digests = {
        row["prompt_id"]: row["digest"] for row in candidate["requests"]
    }
    differing = sorted(
        prompt_id
        for prompt_id in baseline_digests
        if baseline_digests.get(prompt_id) != candidate_digests.get(prompt_id)
    )
    result = {
        "round": args.round,
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "baseline_total_s": baseline["total_s"],
        "candidate_total_s": candidate["total_s"],
        "latency_reduction_pct": (
            (baseline["total_s"] - candidate["total_s"])
            / baseline["total_s"]
            * 100
        ),
        "baseline_output_tokens_per_s": baseline["output_tokens_per_s"],
        "candidate_output_tokens_per_s": candidate["output_tokens_per_s"],
        "throughput_gain_pct": (
            (
                candidate["output_tokens_per_s"]
                / baseline["output_tokens_per_s"]
                - 1
            )
            * 100
        ),
        "baseline_verification_windows": baseline["verification_windows"],
        "candidate_verification_windows": candidate["verification_windows"],
        "verification_window_reduction_pct": (
            (
                1
                - candidate["verification_windows"]
                / baseline["verification_windows"]
            )
            * 100
            if baseline["verification_windows"]
            else 0.0
        ),
        "same_outputs": len(differing) == 0,
        "matching_prompts": len(baseline_digests) - len(differing),
        "total_prompts": len(baseline_digests),
        "differing_prompts": differing,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
