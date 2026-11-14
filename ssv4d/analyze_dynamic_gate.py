"""Summarize SSV4D dynamic-threshold experiment JSON files."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def _finite(value) -> bool:
    return value is not None and math.isfinite(float(value))


def summarize(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["results"]
    triggered = 0
    verified = 0
    mismatches = 0
    covered = 0
    missed = 0
    false_positives = 0
    gate_latencies = []
    token_rates = []
    trigger_reasons = Counter()
    trigger_thresholds = []
    for row in rows:
        gate = row.get("gate_decision") or {}
        dynamic = row.get("dynamic_gate") or gate.get("dynamic_gate") or {}
        trigger = bool(dynamic.get("should_verify", False))
        has_verifier = bool(
            row.get("verifier_output_ids") or row.get("verifier")
        )
        if row.get("verifier"):
            window = row["verifier"].get("output_ids", [])
            canonical = row.get("canonical", {}).get("output_ids", [])
            n_tokens = min(len(window), len(canonical))
            is_mismatch = any(
                canonical[index] != window[index]
                for index in range(n_tokens)
            )
        else:
            is_mismatch = (
                has_verifier and row.get("first_window_match") is False
            )
        triggered += trigger
        verified += has_verifier
        mismatches += is_mismatch
        covered += trigger and is_mismatch
        missed += (not trigger) and is_mismatch
        false_positives += trigger and has_verifier and not is_mismatch
        if _finite(gate.get("gate_latency_from_start_s")):
            gate_latencies.append(
                float(gate["gate_latency_from_start_s"])
            )
        positions = dynamic.get("positions") or []
        if positions:
            token_rates.append(
                sum(bool(item["should_verify"]) for item in positions)
                / len(positions)
            )
        if trigger:
            trigger_reasons[dynamic.get("trigger_reason", "unknown")] += 1
            threshold = dynamic.get("trigger_threshold")
            if _finite(threshold):
                trigger_thresholds.append(float(threshold))

    return {
        "path": str(path),
        "requests": len(rows),
        "verified_requests": verified,
        "dynamic_triggered_requests": triggered,
        "window_trigger_rate": triggered / len(rows) if rows else 0.0,
        "observed_mismatches": mismatches,
        "covered_mismatches": covered,
        "missed_mismatches": missed,
        "false_positive_windows": false_positives,
        "window_recall": covered / mismatches if mismatches else 1.0,
        "mean_token_trigger_rate": (
            statistics.mean(token_rates) if token_rates else 0.0
        ),
        "mean_gate_latency_s": (
            statistics.mean(gate_latencies) if gate_latencies else None
        ),
        "gate_latency_std_s": (
            statistics.stdev(gate_latencies)
            if len(gate_latencies) > 1
            else 0.0
        ),
        "trigger_reasons": dict(trigger_reasons),
        "trigger_threshold_values": sorted(set(trigger_thresholds)),
        "mean_round_s": (
            statistics.mean(
                round_metric["total_s"]
                for round_metric in payload.get("round_metrics", [])
            )
            if payload.get("round_metrics")
            else None
        ),
        "output_tokens_per_s": payload.get("output_tokens_per_s"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ssv4d/results/dynamic_gate_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = [summarize(path) for path in args.inputs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
