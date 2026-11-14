"""Large-scale baseline matrix for dual fast-decode SSV4D gating."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp

from ssv4d.async_dual_gpu import (
    _launch_server,
    _terminate,
    _wait_ready,
)
from ssv4d.client import SGLangShadowClient
from ssv4d.dual_fast_gate import _stream_canonical, run_one
from ssv4d.prompts import PROMPTS
from ssv4d.prompts_extended import EXTENDED_PROMPTS


async def canonical_only(
    canonical_url: str,
    prompt,
    *,
    canonical_tokens: int,
    window: int,
    timeout_s: float,
) -> dict:
    ready_event = asyncio.Event()
    ready_state = {}
    result = await _stream_canonical(
        canonical_url,
        prompt.text,
        max_tokens=canonical_tokens,
        window=window,
        timeout_s=timeout_s,
        ready_event=ready_event,
        ready_state=ready_state,
    )
    return {
        "prompt_id": prompt.prompt_id,
        "output_ids": result["output_ids"],
        "completion_tokens": len(result["output_ids"]),
        "canonical_latency_s": result["latency_s"],
        "commit_latency_s": result["latency_s"],
        "verifier_launched": False,
    }


async def verifier_only(
    verifier_client: SGLangShadowClient,
    prompt,
    *,
    canonical_tokens: int,
) -> dict:
    started = time.perf_counter()
    rows = await verifier_client.generate_batch(
        [prompt.text],
        deterministic=True,
        max_new_tokens=canonical_tokens,
    )
    latency = time.perf_counter() - started
    trace = rows[0]
    return {
        "prompt_id": prompt.prompt_id,
        "output_ids": trace.output_ids,
        "completion_tokens": len(trace.output_ids),
        "verifier_latency_s": latency,
        "commit_latency_s": latency,
        "verifier_launched": True,
    }


def compact_gated(result: dict) -> dict:
    return {
        "prompt_id": result["prompt_id"],
        "output_ids": result["canonical"]["output_ids"],
        "verifier_output_ids": (
            result["verifier"]["output_ids"]
            if result["verifier"] is not None
            else None
        ),
        "completion_tokens": len(result["canonical"]["output_ids"]),
        "gate_should_verify": result["gate_should_verify"],
        "gate_reason": result["gate"]["trigger_reason"],
        "gate_position": result["gate"]["trigger_position"],
        "verifier_launched": result["verifier_launched"],
        "first_window_match": result["first_window_match"],
        "canonical_latency_s": result["canonical_latency_s"],
        "shadow_latency_s": result["shadow_latency_s"],
        "verifier_latency_s": result["verifier_latency_s"],
        "gate_ready_latency_s": result["gate_ready_latency_s"],
        "verifier_started_s": result["verifier_started_s"],
        "canonical_finished_s": result["canonical_finished_s"],
        "verifier_finished_s": result["verifier_finished_s"],
        "overlap_s": result["overlap_s"],
        "post_canonical_wait_s": result["post_canonical_wait_s"],
        "commit_latency_s": result["commit_latency_s"],
    }


def _token_agreement(reference: list[int], candidate: list[int]) -> float:
    length = max(len(reference), len(candidate))
    if length == 0:
        return 1.0
    matches = sum(
        left == right
        for left, right in zip(reference, candidate)
    )
    return matches / length


def summarize_modes(payload: dict) -> dict:
    modes = payload["modes"]
    reference_rows = {
        (row["round"], row["prompt_id"]): row
        for row in modes["verifier-only"]["results"]
    }
    summary = {}
    for mode, mode_payload in modes.items():
        rows = mode_payload["results"]
        round_metrics = mode_payload["round_metrics"]
        exact = 0
        first_window_exact = 0
        token_agreements = []
        total_tokens = 0
        total_s = 0.0
        by_prompt = {}
        for row in rows:
            reference = reference_rows[(row["round"], row["prompt_id"])]
            reference_ids = reference["output_ids"]
            candidate_ids = row["output_ids"]
            exact += candidate_ids == reference_ids
            first_window_exact += (
                candidate_ids[: payload["window"]]
                == reference_ids[: payload["window"]]
            )
            token_agreements.append(
                _token_agreement(reference_ids, candidate_ids)
            )
            by_prompt.setdefault(row["prompt_id"], []).append(candidate_ids)
            total_tokens += len(candidate_ids)
        total_s = sum(metric["total_s"] for metric in round_metrics)
        summary[mode] = {
            "requests": len(rows),
            "full_sequence_matches": exact,
            "full_sequence_match_rate": exact / len(rows) if rows else 0.0,
            "first_window_matches": first_window_exact,
            "first_window_match_rate": (
                first_window_exact / len(rows) if rows else 0.0
            ),
            "mean_token_agreement": (
                statistics.mean(token_agreements)
                if token_agreements
                else 0.0
            ),
            "cross_round_deterministic_prompts": sum(
                len({tuple(output_ids) for output_ids in prompt_outputs}) == 1
                for prompt_outputs in by_prompt.values()
            ),
            "total_unique_prompts": len(by_prompt),
            "total_s": total_s,
            "output_tokens_per_s": total_tokens / total_s if total_s else 0.0,
            "verifier_launched": sum(
                row.get("verifier_launched", False) for row in rows
            ),
            "mean_commit_latency_s": statistics.mean(
                row["commit_latency_s"] for row in rows
            ),
        }
    return summary


async def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    stem = output.stem
    log_dir = Path("ssv4d/logs")
    processes = [
        _launch_server(
            0,
            args.canonical_port,
            args.model,
            deterministic=False,
            log_path=log_dir / f"{stem}_canonical_server.log",
            mem_fraction_static=args.mem_fraction_static,
        ),
        _launch_server(
            1,
            args.shadow_port,
            args.model,
            deterministic=False,
            log_path=log_dir / f"{stem}_shadow_server.log",
            mem_fraction_static=args.mem_fraction_static,
        ),
    ]
    try:
        canonical_url = f"http://127.0.0.1:{args.canonical_port}"
        shadow_url = f"http://127.0.0.1:{args.shadow_port}"
        verifier_url = f"http://127.0.0.1:{args.verifier_port}"
        timeout = aiohttp.ClientTimeout(total=args.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await asyncio.gather(
                _wait_ready(session, canonical_url, args.startup_timeout),
                _wait_ready(session, shadow_url, args.startup_timeout),
            )
            processes.append(
                _launch_server(
                    1,
                    args.verifier_port,
                    args.model,
                    deterministic=True,
                    log_path=log_dir / f"{stem}_verifier_server.log",
                    mem_fraction_static=args.mem_fraction_static,
                )
            )
            await _wait_ready(
                session, verifier_url, args.startup_timeout
            )

        shadow_client = SGLangShadowClient(
            shadow_url, timeout_s=args.timeout_s
        )
        verifier_client = SGLangShadowClient(
            verifier_url, timeout_s=args.timeout_s
        )
        prompt_set = (
            EXTENDED_PROMPTS
            if args.prompt_set == "extended"
            else PROMPTS
        )
        prompts = list(prompt_set[: args.num_prompts])
        semaphore = asyncio.Semaphore(args.prompt_concurrency)

        async def limited(coro):
            async with semaphore:
                return await coro

        modes = {}
        for mode in (
            "canonical-only",
            "verifier-only",
            "gated",
            "audit",
        ):
            rounds = []
            for round_index in range(args.rounds):
                started = time.perf_counter()
                if mode == "canonical-only":
                    rows = await asyncio.gather(
                        *(
                            limited(
                                canonical_only(
                                    canonical_url,
                                    prompt,
                                    canonical_tokens=args.canonical_tokens,
                                    window=args.window,
                                    timeout_s=args.timeout_s,
                                )
                            )
                            for prompt in prompts
                        )
                    )
                elif mode == "verifier-only":
                    rows = await asyncio.gather(
                        *(
                            limited(
                                verifier_only(
                                    verifier_client,
                                    prompt,
                                    canonical_tokens=args.canonical_tokens,
                                )
                            )
                            for prompt in prompts
                        )
                    )
                else:
                    rows = await asyncio.gather(
                        *(
                            limited(
                                run_one(
                                    canonical_url,
                                    shadow_client,
                                    verifier_client,
                                    prompt,
                                    shadow_plan_prefix=args.shadow_plan,
                                    window=args.window,
                                    canonical_tokens=args.canonical_tokens,
                                    timeout_s=args.timeout_s,
                                    force_verify=mode == "audit",
                                )
                            )
                            for prompt in prompts
                        )
                    )
                    rows = [compact_gated(row) for row in rows]
                total_s = time.perf_counter() - started
                rounds.append(
                    {
                        "round": round_index,
                        "total_s": total_s,
                        "output_tokens_per_s": (
                            sum(row["completion_tokens"] for row in rows)
                            / total_s
                        ),
                        "results": [
                            {
                                "round": round_index,
                                **row,
                            }
                            for row in rows
                        ],
                    }
                )
            modes[mode] = {
                "round_metrics": [
                    {
                        "round": row["round"],
                        "total_s": row["total_s"],
                        "output_tokens_per_s": row[
                            "output_tokens_per_s"
                        ],
                    }
                    for row in rounds
                ],
                "results": [
                    result
                    for row in rounds
                    for result in row["results"]
                ],
            }
    finally:
        _terminate(processes)

    payload = {
        "model": args.model,
        "prompt_set": args.prompt_set,
        "num_prompts": len(prompts),
        "rounds": args.rounds,
        "prompt_concurrency": args.prompt_concurrency,
        "canonical_tokens": args.canonical_tokens,
        "window": args.window,
        "shadow_plan": args.shadow_plan,
        "modes": modes,
    }
    payload["summary"] = summarize_modes(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        "--prompt-set",
        choices=("standard", "extended"),
        default="extended",
    )
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--prompt-concurrency", type=int, default=8)
    parser.add_argument("--canonical-tokens", type=int, default=130)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--shadow-plan", default="bs8_middle")
    parser.add_argument("--canonical-port", type=int, default=30010)
    parser.add_argument("--shadow-port", type=int, default=30011)
    parser.add_argument("--verifier-port", type=int, default=30012)
    parser.add_argument("--mem-fraction-static", type=float, default=0.28)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--timeout-s", type=float, default=900)
    parser.add_argument(
        "--output",
        default="ssv4d/results/dual_fast_matrix.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
