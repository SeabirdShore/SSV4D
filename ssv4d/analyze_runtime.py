"""Summarize SSV4D runtime instrumentation JSONL."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def _mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.mean(values) if values else None


def analyze(path: Path) -> dict:
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    verify = [event for event in events if event.get("event") == "verify"]
    skipped = [event for event in events if event.get("event") == "verify_skip"]
    kv_values = [event.get("kv", {}) for event in verify]
    logprob_values = [event.get("logprob", {}) for event in verify]
    return {
        "events": len(events),
        "verify_windows": len(verify),
        "skipped_windows": len(skipped),
        "verified_requests": sum(event.get("num_reqs", 0) for event in verify),
        "rollback_windows": sum(bool(event.get("rollbacks")) for event in verify),
        "rollback_tokens": sum(
            item.get("tokens", 0)
            for event in verify
            for item in event.get("rollbacks", [])
        ),
        "verify_wall_ms_mean": _mean(
            event.get("verify_wall_ms") for event in verify
        ),
        "verify_gpu_ms_mean": _mean(
            event.get("verify_gpu_ms") for event in verify
        ),
        "draft_ms_mean": _mean(event.get("draft_ms") for event in verify),
        "kv_k_relative_l2_mean": _mean(
            value.get("k_relative_l2_mean") for value in kv_values
        ),
        "kv_v_relative_l2_mean": _mean(
            value.get("v_relative_l2_mean") for value in kv_values
        ),
        "kv_k_relative_l2_max": _mean(
            value.get("k_relative_l2_max") for value in kv_values
        ),
        "kv_v_relative_l2_max": _mean(
            value.get("v_relative_l2_max") for value in kv_values
        ),
        "kl_canonical_shadow_mean": _mean(
            value.get("kl_canonical_shadow_mean") for value in logprob_values
        ),
        "jsd_mean": _mean(value.get("jsd_mean") for value in logprob_values),
        "top1_agreement_mean": _mean(
            value.get("top1_agreement_mean") for value in logprob_values
        ),
        "decision_counts": dict(
            Counter(
                decision.get("should_verify")
                for event in verify + skipped
                for decision in event.get("decisions", {}).values()
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = {str(path): analyze(path) for path in args.inputs}
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
