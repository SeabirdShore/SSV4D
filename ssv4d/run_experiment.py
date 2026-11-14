"""Run the first SSV4D sparse-shadow research experiment."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ssv4d.client import SGLangShadowClient
from ssv4d.metrics import (
    aggregate_token_margin_buckets,
    build_token_dataset,
    build_window_dataset,
    feature_auc,
    first_mismatch,
    summarize_token_flips,
)
from ssv4d.perturbation import make_perturbation_plans
from ssv4d.policy import calibrate_dual_gate, grouped_cross_validate
from ssv4d.prompts import FILLER_PROMPTS, PROMPTS


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "ssv4d" / "results"
DEFAULT_LOG_DIR = REPO_ROOT / "ssv4d" / "logs"


def _configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ssv4d")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


async def _wait_for_server(client: SGLangShadowClient, timeout_s: int = 300) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            if await client.health():
                return
        except Exception:
            pass
        await asyncio.sleep(2)
    raise TimeoutError(f"Server did not become healthy within {timeout_s}s")


async def collect_one_prompt(
    client: SGLangShadowClient,
    prompt_id: str,
    prompt_text: str,
    *,
    max_new_tokens: int,
    repeats: int,
) -> tuple[Any, list[Any]]:
    plans = make_perturbation_plans(
        prompt_text,
        FILLER_PROMPTS,
        batch_sizes=(4, 8, 16),
        positions=("first", "middle", "last"),
        repeats=repeats,
    )

    async def reference_task():
        traces = await client.generate_batch(
            [prompt_text],
            deterministic=True,
            max_new_tokens=max_new_tokens,
        )
        trace = traces[0]
        trace.prompt_id = prompt_id
        trace.plan_name = "reference"
        trace.scenario = "reference"
        return trace

    async def shadow_task(plan):
        traces = await client.generate_batch(
            plan.prompts,
            deterministic=False,
            max_new_tokens=max_new_tokens,
        )
        trace = traces[plan.protected_index]
        trace.prompt_id = prompt_id
        trace.plan_name = plan.name
        trace.scenario = plan.family
        trace.batch_size = plan.batch_size
        trace.protected_index = plan.protected_index
        return trace

    result = await asyncio.gather(
        reference_task(),
        *(shadow_task(plan) for plan in plans),
    )
    return result[0], result[1:]


def _plot_results(
    output_dir: Path,
    token_records: list[dict],
    margin_buckets: dict[str, list[float]],
    window_samples: list[Any],
    metrics: list[Any],
    token_metrics: list[Any],
) -> None:
    # Flip location histogram.
    positions = [
        row["first_mismatch"] for row in token_records if row["is_flip"]
    ]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    if positions:
        ax.hist(positions, bins=min(24, max(5, len(positions))), color="#1f77b4")
    else:
        ax.text(0.5, 0.5, "No token flips observed", ha="center", va="center")
    ax.set_xlabel("Decode position")
    ax.set_ylabel("Flip count")
    ax.set_title("Shadow token flips by position")
    fig.tight_layout()
    fig.savefig(output_dir / "flip_positions.png", dpi=180)
    plt.close(fig)

    # Margin separation.
    fig, ax = plt.subplots(figsize=(7, 3.6))
    stable = margin_buckets.get("stable", [])
    flipped = margin_buckets.get("flipped", [])
    bins = np.linspace(
        min(stable + flipped + [0.0]),
        max(stable + flipped + [2.0]),
        36,
    )
    if stable:
        ax.hist(stable, bins=bins, alpha=0.55, label="stable", color="#2ca02c")
    if flipped:
        ax.hist(flipped, bins=bins, alpha=0.7, label="flip", color="#d62728")
    ax.set_xlabel("top-1 / top-2 logprob margin")
    ax.set_ylabel("Token count")
    ax.set_title("Near-tie signal at shadow flips")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "margin_separation.png", dpi=180)
    plt.close(fig)

    # Trigger rate and recall on evaluation split.
    eval_metrics = [m for m in metrics if m.split == "grouped_cv"]
    labels = [m.policy for m in eval_metrics]
    rates = [m.trigger_rate * 100 for m in eval_metrics]
    recalls = [m.window_recall * 100 for m in eval_metrics]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, rates, width, label="trigger rate", color="#1f77b4")
    ax.bar(x + width / 2, recalls, width, label="window recall", color="#ff7f0e")
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Percent")
    ax.set_title("Verification budget vs coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "policy_comparison.png", dpi=180)
    plt.close(fig)

    token_eval = [m for m in token_metrics if m.split == "grouped_cv"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    labels = [m.policy for m in token_eval]
    rates = [m.trigger_rate * 100 for m in token_eval]
    recalls = [m.window_recall * 100 for m in token_eval]
    x = np.arange(len(labels))
    ax.bar(x - width / 2, rates, width, label="verify-token rate", color="#9467bd")
    ax.bar(x + width / 2, recalls, width, label="token recall", color="#ff7f0e")
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Percent")
    ax.set_title("Token-level verification budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "token_policy_comparison.png", dpi=180)
    plt.close(fig)

    # Sparse window labels.
    counts = [s.label for s in window_samples]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(
        ["stable windows", "risk windows"],
        [sum(1 for x in counts if x == 0), sum(counts)],
        color=["#2ca02c", "#d62728"],
    )
    ax.set_ylabel("Window count")
    ax.set_title("Sparsity of batch-induced errors")
    fig.tight_layout()
    fig.savefig(output_dir / "window_sparsity.png", dpi=180)
    plt.close(fig)


def _build_report(
    output_dir: Path,
    run_id: str,
    args: argparse.Namespace,
    token_summary: dict,
    token_records: list[dict],
    window_samples: list[Any],
    metrics: list[Any],
    gate: Any,
    token_samples: list[Any],
    token_metrics: list[Any],
    token_gate: Any,
    feature_aucs: dict[str, float],
) -> str:
    positive_windows = sum(s.label for s in window_samples)
    trigger = {m.policy: m for m in metrics if m.split == "grouped_cv"}
    token_trigger = {
        m.policy: m for m in token_metrics if m.split == "grouped_cv"
    }
    ssv4d = trigger.get("ssv4d")
    always = trigger.get("always_on")
    margin = trigger.get("margin_only")
    shadow = trigger.get("shadow_only")
    flip_positions = token_summary.get("flip_positions", [])

    lines = [
        "# SSV4D Research Report",
        "",
        f"- Run: `{run_id}`",
        f"- Model: `{args.model}`",
        f"- Output tokens per request: {args.max_new_tokens}",
        f"- Window size: {args.window_size}",
        f"- Shadow plans per prompt: {args.repeats} repeat(s) x 9 plan(s) + bs1",
        f"- Results: `{output_dir}`",
        "- Detailed research notes: `ssv4d/research_notes.md`",
        "- Closest prior work: MarginGate, arXiv:2605.30218",
        "",
        "## Research Question",
        "",
        "Can pre-prefill batch perturbations expose batch-induced token-instability "
        "with enough precision to gate LLM-42 verification and reduce verifier budget?",
        "",
        "## Method",
        "",
        "For each protected prompt, SSV4D concurrently runs a deterministic LLM-42 "
        "reference and a shadow branch. The shadow branch inserts the protected prompt "
        "into perturbed batches of size 1/4/8/16, at first/middle/last positions, with "
        "rotated filler order. For each generated token it records output IDs, top-k "
        "logprobs, top-1/top-2 margin, and cross-shadow vote/JS disagreement. These "
        "signals are aggregated over verification windows. Gate thresholds are "
        "evaluated with leave-one-prompt-out grouped cross-validation, which keeps "
        "calibration and evaluation prompts disjoint.",
        "",
        "## Sparse Error Observation",
        "",
        f"- Shadow traces: {token_summary['n_shadow_traces']}",
        f"- Synchronous decode steps: {token_summary['n_synchronous_steps']}",
        f"- Token flips: {token_summary['n_flip_events']} "
        f"({token_summary['token_flip_rate'] * 100:.2f}%)",
        f"- Window positives: {positive_windows}/{len(window_samples)} "
        f"({positive_windows / len(window_samples) * 100 if window_samples else 0:.2f}%)",
        f"- Synchronous token positions: {len(token_samples)}",
        f"- Token positives: {sum(s.label for s in token_samples)}",
        f"- Median flip position: {token_summary['flip_position_median']}",
        f"- Flip positions: {flip_positions}",
        f"- Median shadow margin at flips: "
        f"{token_summary['flip_margin_median']:.4f}",
        f"- Median stable-step margin: "
        f"{token_summary['stable_margin_median']:.4f}",
        "",
        "## Gate Comparison",
        "",
        "| Policy | Trigger rate | Window recall | Prompt recall | Prompt determinism |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in ("always_on", "margin_only", "shadow_only", "ssv4d"):
        metric = trigger.get(policy)
        if metric is None:
            continue
        lines.append(
            f"| {policy} | {metric.trigger_rate * 100:.2f}% | "
            f"{metric.window_recall * 100:.2f}% | "
            f"{metric.prompt_recall * 100:.2f}% | "
            f"{metric.prompt_determinism * 100:.2f}% |"
        )

    lines += [
        "",
        "## Token-level Selective Verification",
        "",
        "| Policy | Verify-token rate | Token recall | Prompt determinism |",
        "|---|---:|---:|---:|",
    ]
    for policy in ("always_on", "margin_only", "shadow_only", "ssv4d"):
        metric = token_trigger.get(policy)
        if metric is None:
            continue
        lines.append(
            f"| {policy} | {metric.trigger_rate * 100:.2f}% | "
            f"{metric.window_recall * 100:.2f}% | "
            f"{metric.prompt_determinism * 100:.2f}% |"
        )
    lines += [
        "",
        "Feature AUC on token positives (higher is better):",
        "",
        f"- Negative min-margin: {feature_aucs.get('margin', float('nan')):.4f}",
        f"- Shadow vote disagreement: "
        f"{feature_aucs.get('shadow_vote', float('nan')):.4f}",
        f"- Shadow top-k JSD: {feature_aucs.get('shadow_jsd', float('nan')):.4f}",
        "",
    ]

    lines += [
        "",
        "## Selected SSV4D Gate",
        "",
        f"- Margin threshold: {gate.margin_threshold:.6f}",
        f"- Shadow enabled: {gate.shadow_enabled}",
        f"- Shadow AUC: {gate.shadow_auc:.4f}",
        f"- Shadow threshold: {gate.shadow_threshold:.6f}",
        (
            "- Rule: `verify = min_margin <= tau_margin or shadow_risk >= tau_shadow`"
            if gate.shadow_enabled
            else "- Rule: `verify = min_margin <= tau_margin` "
            "(shadow branch rejected by calibration AUC gate)"
        ),
        "- Shadow risk: top-1 vote disagreement + min(1, 100 * top-k JSD)",
        f"- Token margin threshold: {token_gate.margin_threshold:.6f}",
        f"- Token shadow threshold: {token_gate.shadow_threshold:.6f}",
        "",
        "The gate is a prototype for the LLM-42 verification hook in "
        "`ssv4d/llm42_hook.py`. The current experiment measures the selective "
        "verification frontier offline; it does not yet run the shadow branch inside "
        "the GPU scheduler or perform current-column K/V repair.",
        "",
        "## Interpretation",
        "",
    ]
    if not gate.shadow_enabled:
        lines.append(
            "Calibration rejected the shadow disagreement feature because its "
            f"held-in AUC ({gate.shadow_auc:.4f}) did not exceed the configured "
            "minimum. SSV4D therefore falls back to margin-only verification for "
            "this model/workload; the shadow branch remains an online diagnostic."
        )
    if ssv4d and always:
        lines.append(
            f"SSV4D selected {ssv4d.trigger_rate * 100:.2f}% of windows versus "
            f"100% always-on verification, while recovering "
            f"{ssv4d.window_recall * 100:.2f}% of observed flip windows and "
            f"{ssv4d.prompt_determinism * 100:.2f}% prompt-level determinism on "
            "the leave-one-prompt-out evaluation."
        )
    if margin and shadow:
        lines.append(
            f"Under grouped cross-validation, margin-only triggered "
            f"{margin.trigger_rate * 100:.2f}% with {margin.window_recall * 100:.2f}% "
            f"recall; shadow-only triggered {shadow.trigger_rate * 100:.2f}% with "
            f"{shadow.window_recall * 100:.2f}% recall."
        )
    token_margin = token_trigger.get("margin_only")
    token_ssv4d = token_trigger.get("ssv4d")
    if token_margin and token_ssv4d:
        comparison = (
            "The gate fell back to margin-only behavior because calibration "
            "rejected the shadow feature."
            if not token_gate.shadow_enabled
            else (
                "SSV4D used a larger budget at equal recall, so shadow "
                "disagreement did not add useful coverage."
                if token_ssv4d.trigger_rate > token_margin.trigger_rate + 1e-9
                else "SSV4D reduced the token budget at equal recall."
            )
        )
        lines.append(
            f"At token granularity, margin-only verified "
            f"{token_margin.trigger_rate * 100:.2f}% of positions with "
            f"{token_margin.window_recall * 100:.2f}% recall; SSV4D verified "
            f"{token_ssv4d.trigger_rate * 100:.2f}% with "
            f"{token_ssv4d.window_recall * 100:.2f}% recall. {comparison}"
        )
    if not flip_positions:
        lines.append(
            "No flips were observed with these prompts and batch shapes. This is a "
            "valid sparse-regime outcome, but it prevents drawing a strong conclusion "
            "about gate recall. Increase prompt count, output length, or batch-shape "
            "diversity before using the selected threshold."
        )
    else:
        lines.append(
            "The observed flips are sparse and their first positions are listed above. "
            "The window-level budget numbers are therefore the primary result; "
            "sequence-level claims require more prompts and repeated trials."
        )

    lines += [
        "",
        "## Artifacts",
        "",
        "- `token_flips.jsonl`",
        "- `window_features.csv`",
        "- `token_features.csv`",
        "- `gate_metrics.csv`",
        "- `token_gate_metrics.csv`",
        "- `selected_token_gate.json`",
        "- `raw_traces.json`",
        "- `flip_positions.png`",
        "- `margin_separation.png`",
        "- `policy_comparison.png`",
        "- `token_policy_comparison.png`",
        "- `window_sparsity.png`",
        "",
        "## Limitations and Next Steps",
        "",
        "1. The shadow branch currently runs through the public `/generate` API rather "
        "than inside the scheduler; this isolates the numerical signal but includes "
        "HTTP and tokenizer overhead.",
        "2. The evaluation uses a small built-in prompt set and one model. It is a "
        "research prototype, not a paper-scale reproduction.",
        "3. The LLM-42 hook gates request-level verification. The next step is a "
        "window-level scheduler callback with online risk state and current-column "
        "K/V repair.",
        "4. Future calibration should use a larger held-out stream and target explicit "
        "latency/determinism constraints rather than recall alone.",
    ]
    return "\n".join(lines) + "\n"


async def run(args: argparse.Namespace) -> Path:
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(DEFAULT_LOG_DIR / "probe.log")
    client = SGLangShadowClient(
        base_url=args.base_url,
        timeout_s=args.timeout_s,
        top_logprobs_num=args.top_logprobs,
    )
    await _wait_for_server(client, args.startup_timeout)
    await client.reset_llm42_stats()

    logger.info("Starting %s against %s", run_id, args.base_url)
    logger.info(
        "model=%s prompts=%d window=%d tokens=%d top_k=%d repeats=%d",
        args.model,
        args.num_prompts,
        args.window_size,
        args.max_new_tokens,
        args.top_logprobs,
        args.repeats,
    )

    references = {}
    shadows = {}
    selected_prompts = PROMPTS[: args.num_prompts]
    for index, prompt in enumerate(selected_prompts, start=1):
        logger.info(
            "[%d/%d] probing %s (%s)",
            index,
            len(selected_prompts),
            prompt.prompt_id,
            prompt.category,
        )
        reference, prompt_shadows = await collect_one_prompt(
            client,
            prompt.prompt_id,
            prompt.text,
            max_new_tokens=args.max_new_tokens,
            repeats=args.repeats,
        )
        references[prompt.prompt_id] = reference
        shadows[prompt.prompt_id] = prompt_shadows
        flips = sum(
            first_mismatch(reference.output_ids, trace.output_ids)
            < len(reference.output_ids)
            for trace in prompt_shadows
        )
        logger.info(
            "  reference=%d tokens, shadow_traces=%d, flips=%d",
            len(reference.output_ids),
            len(prompt_shadows),
            flips,
        )

    token_records, token_summary = summarize_token_flips(references, shadows)
    window_samples = build_window_dataset(
        references, shadows, window_size=args.window_size
    )
    token_samples = build_token_dataset(references, shadows)
    if not window_samples:
        raise RuntimeError("No window samples were produced")
    if not token_samples:
        raise RuntimeError("No token samples were produced")
    metrics = grouped_cross_validate(
        window_samples,
        target_recall=args.target_recall,
        margin_guard=args.margin_guard,
        min_shadow_auc=args.min_shadow_auc,
    )
    gate, deployment_metrics = calibrate_dual_gate(
        window_samples,
        target_recall=args.target_recall,
        margin_guard=args.margin_guard,
        min_shadow_auc=args.min_shadow_auc,
    )
    deployment_metrics.split = "full_fit"
    metrics.append(deployment_metrics)
    token_metrics = grouped_cross_validate(
        token_samples,
        target_recall=args.target_recall,
        margin_guard=args.margin_guard,
        min_shadow_auc=args.min_shadow_auc,
    )
    token_gate, token_deployment_metrics = calibrate_dual_gate(
        token_samples,
        target_recall=args.target_recall,
        margin_guard=args.margin_guard,
        min_shadow_auc=args.min_shadow_auc,
    )
    token_deployment_metrics.split = "full_fit"
    token_metrics.append(token_deployment_metrics)
    feature_aucs = {
        "margin": feature_auc(token_samples, "min_margin", higher_is_riskier=False),
        "shadow_vote": feature_auc(
            token_samples, "shadow_vote_disagreement", higher_is_riskier=True
        ),
        "shadow_jsd": feature_auc(
            token_samples, "shadow_jsd", higher_is_riskier=True
        ),
    }
    margin_buckets = aggregate_token_margin_buckets(
        token_records, references, shadows
    )

    raw_traces = {
        "references": {
            key: value.to_dict() for key, value in references.items()
        },
        "shadows": {
            key: [trace.to_dict() for trace in traces]
            for key, traces in shadows.items()
        },
    }
    _write_json(output_dir / "raw_traces.json", raw_traces)
    _write_json(
        output_dir / "config.json",
        {
            **vars(args),
            "run_id": run_id,
            "prompt_ids": [p.prompt_id for p in selected_prompts],
        },
    )
    _write_jsonl(output_dir / "token_flips.jsonl", token_records)
    _write_csv(
        output_dir / "window_features.csv",
        [sample.to_dict() for sample in window_samples],
    )
    _write_csv(
        output_dir / "token_features.csv",
        [sample.to_dict() for sample in token_samples],
    )
    _write_csv(
        output_dir / "gate_metrics.csv",
        [metric.to_dict() for metric in metrics],
    )
    _write_csv(
        output_dir / "token_gate_metrics.csv",
        [metric.to_dict() for metric in token_metrics],
    )
    _write_json(
        output_dir / "selected_gate.json",
        {
            "margin_threshold": gate.margin_threshold,
            "shadow_threshold": gate.shadow_threshold,
            "shadow_enabled": gate.shadow_enabled,
            "shadow_auc": gate.shadow_auc,
            "rule": (
                "min_margin <= margin_threshold "
                "or shadow_risk >= shadow_threshold"
            ),
            "token_summary": token_summary,
        },
    )
    _write_json(
        output_dir / "selected_token_gate.json",
        {
            "margin_threshold": token_gate.margin_threshold,
            "shadow_threshold": token_gate.shadow_threshold,
            "shadow_enabled": token_gate.shadow_enabled,
            "shadow_auc": token_gate.shadow_auc,
            "rule": (
                "min_margin <= margin_threshold "
                "or shadow_risk >= shadow_threshold"
            ),
            "feature_auc": feature_aucs,
        },
    )
    _plot_results(
        output_dir,
        token_records,
        margin_buckets,
        window_samples,
        metrics,
        token_metrics,
    )
    report = _build_report(
        output_dir,
        run_id,
        args,
        token_summary,
        token_records,
        window_samples,
        metrics,
        gate,
        token_samples,
        token_metrics,
        token_gate,
        feature_aucs,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (REPO_ROOT / "ssv4d" / "report.md").write_text(report, encoding="utf-8")
    logger.info("Completed run. Report: %s", REPO_ROOT / "ssv4d" / "report.md")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--num-prompts", type=int, default=len(PROMPTS))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--top-logprobs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--target-recall", type=float, default=1.0)
    parser.add_argument("--margin-guard", type=float, default=0.0625)
    parser.add_argument("--min-shadow-auc", type=float, default=0.55)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
