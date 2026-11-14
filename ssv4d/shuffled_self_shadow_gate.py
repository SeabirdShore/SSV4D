"""Shuffled shared-batch self-shadow gate."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import aiohttp

from ssv4d.async_dual_gpu import _launch_server, _terminate, _wait_ready
from ssv4d.client import SGLangShadowClient
from ssv4d.dynamic_gate import evaluate_dynamic_window
from ssv4d.prompts import PROMPTS
from ssv4d.prompts_extended import EXTENDED_PROMPTS


async def _timed(awaitable):
    started = time.perf_counter()
    result = await awaitable
    return result, time.perf_counter() - started


async def run_wave(
    fast_client: SGLangShadowClient,
    verifier_client: SGLangShadowClient,
    prompts,
    *,
    copies_per_prompt: int,
    window: int,
    force_verify: bool,
    shuffle_seed: int,
) -> tuple[list[dict], dict]:
    entries = [
        (prompt_index, copy_index)
        for prompt_index in range(len(prompts))
        for copy_index in range(copies_per_prompt)
    ]
    random.Random(shuffle_seed).shuffle(entries)
    texts = [prompts[prompt_index].text for prompt_index, _ in entries]
    started = time.perf_counter()
    traces, fast_latency = await _timed(
        fast_client.generate_batch(
            texts,
            deterministic=False,
            max_new_tokens=window,
        )
    )
    positions = {index: [] for index in range(len(prompts))}
    for position, (prompt_index, _) in enumerate(entries):
        positions[prompt_index].append(position)

    decisions = {}
    for prompt_index, prompt in enumerate(prompts):
        ordered_positions = sorted(positions[prompt_index])
        canonical_position = ordered_positions[0]
        shadow_positions = ordered_positions[1:]
        canonical = traces[canonical_position]
        canonical.prompt_id = prompt.prompt_id
        canonical.plan_name = f"canonical_pos_{canonical_position}"
        shadows = [traces[position] for position in shadow_positions]
        decision = evaluate_dynamic_window(
            canonical,
            shadows,
            window=window,
        )
        decisions[prompt_index] = (
            canonical,
            shadows,
            decision,
            canonical_position,
            shadow_positions,
        )

    async def verify_one(prompt_index):
        prompt = prompts[prompt_index]
        canonical, _, _, _, _ = decisions[prompt_index]
        should_verify = (
            force_verify
            or decisions[prompt_index][2].should_verify
        )
        verifier = None
        verifier_latency = 0.0
        if should_verify:
            rows, verifier_latency = await _timed(
                verifier_client.generate_batch(
                    [prompt.text],
                    deterministic=True,
                    max_new_tokens=window,
                )
            )
            verifier = rows[0]
        first_window_match = (
            canonical.output_ids[:window]
            == verifier.output_ids[:window]
            if verifier is not None
            else None
        )
        return {
            "prompt_id": prompt.prompt_id,
            "verifier": verifier.to_dict() if verifier is not None else None,
            "verifier_launched": verifier is not None,
            "verifier_latency_s": verifier_latency,
            "first_window_match": first_window_match,
        }

    verified_rows = await asyncio.gather(
        *(verify_one(index) for index in range(len(prompts)))
    )
    finished = time.perf_counter()
    rows = []
    for prompt_index, prompt in enumerate(prompts):
        (
            canonical,
            shadows,
            decision,
            canonical_position,
            shadow_positions,
        ) = decisions[prompt_index]
        verified = verified_rows[prompt_index]
        rows.append(
            {
                "prompt_id": prompt.prompt_id,
                "batch_size": len(entries),
                "copies_per_prompt": copies_per_prompt,
                "canonical_position": canonical_position,
                "shadow_positions": shadow_positions,
                "canonical": canonical.to_dict(),
                "shadows": [shadow.to_dict() for shadow in shadows],
                "gate": decision.to_dict(),
                "gate_should_verify": decision.should_verify,
                "verifier": verified["verifier"],
                "verifier_launched": verified["verifier_launched"],
                "verifier_latency_s": verified["verifier_latency_s"],
                "first_window_match": verified["first_window_match"],
                "fast_latency_s": fast_latency,
                "wave_latency_s": finished - started,
                "commit_latency_s": finished - started,
            }
        )
    layout = {
        "batch_size": len(entries),
        "copies_per_prompt": copies_per_prompt,
        "prompt_positions": {
            prompt.prompt_id: [
                position
                for position, (prompt_index, _) in enumerate(entries)
                if prompt_index == target_index
            ]
            for target_index, prompt in enumerate(prompts)
        },
    }
    return rows, layout


async def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    stem = output.stem
    log_dir = Path("ssv4d/logs")
    processes = [
        _launch_server(
            0,
            args.fast_port,
            args.model,
            deterministic=False,
            log_path=log_dir / f"{stem}_fast_server.log",
        ),
        _launch_server(
            1,
            args.verifier_port,
            args.model,
            deterministic=True,
            log_path=log_dir / f"{stem}_verifier_server.log",
        ),
    ]
    try:
        fast_url = f"http://127.0.0.1:{args.fast_port}"
        verifier_url = f"http://127.0.0.1:{args.verifier_port}"
        timeout = aiohttp.ClientTimeout(total=args.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await asyncio.gather(
                _wait_ready(session, fast_url, args.startup_timeout),
                _wait_ready(
                    session, verifier_url, args.startup_timeout
                ),
            )
        fast_client = SGLangShadowClient(
            fast_url, timeout_s=args.timeout_s
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
        rounds = []
        for round_index in range(args.rounds):
            rounds_started = time.perf_counter()
            wave_rows = []
            layouts = []
            for wave_index, start in enumerate(
                range(0, len(prompts), args.prompt_concurrency)
            ):
                wave_prompts = prompts[
                    start : start + args.prompt_concurrency
                ]
                rows, layout = await run_wave(
                    fast_client,
                    verifier_client,
                    wave_prompts,
                    copies_per_prompt=args.copies_per_prompt,
                    window=args.window,
                    force_verify=args.force_verify,
                    shuffle_seed=(
                        args.shuffle_seed
                        + round_index * 1009
                        + wave_index
                    ),
                )
                wave_rows.extend(rows)
                layouts.append(layout)
            rounds.append(
                {
                    "round": round_index,
                    "total_s": time.perf_counter() - rounds_started,
                    "layouts": layouts,
                    "results": wave_rows,
                }
            )
    finally:
        _terminate(processes)

    results = [
        {"round": row["round"], **result}
        for row in rounds
        for result in row["results"]
    ]
    verified = [row for row in results if row["verifier_launched"]]
    payload = {
        "model": args.model,
        "prompt_set": args.prompt_set,
        "num_prompts": len(prompts),
        "rounds": args.rounds,
        "prompt_concurrency": args.prompt_concurrency,
        "copies_per_prompt": args.copies_per_prompt,
        "window": args.window,
        "force_verify": args.force_verify,
        "round_metrics": [
            {
                "round": row["round"],
                "total_s": row["total_s"],
            }
            for row in rounds
        ],
        "layout_rounds": [
            {
                "round": row["round"],
                "layouts": row["layouts"],
            }
            for row in rounds
        ],
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "mean_round_s": statistics.mean(
                    row["total_s"] for row in rounds
                ),
                "requests": len(results),
                "gate_triggered": sum(
                    row["gate_should_verify"] for row in results
                ),
                "verifier_launched": len(verified),
                "first_window_matches": sum(
                    row["first_window_match"] is True
                    for row in verified
                ),
                "mean_fast_latency_s": statistics.mean(
                    row["fast_latency_s"] for row in results
                ),
                "mean_wave_latency_s": statistics.mean(
                    row["wave_latency_s"] for row in results
                ),
            },
            indent=2,
        )
    )


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
    parser.add_argument("--copies-per-prompt", type=int, default=8)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--force-verify", action="store_true")
    parser.add_argument("--fast-port", type=int, default=30010)
    parser.add_argument("--verifier-port", type=int, default=30011)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument(
        "--output",
        default="ssv4d/results/shuffled_self_shadow.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
