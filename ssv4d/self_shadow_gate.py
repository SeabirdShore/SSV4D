"""SSV4D gate using duplicate prompts inside the canonical fast batch."""

from __future__ import annotations

import argparse
import asyncio
import json
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


async def run_one(
    fast_client: SGLangShadowClient,
    verifier_client: SGLangShadowClient,
    prompt,
    *,
    batch_size: int,
    window: int,
    force_verify: bool,
) -> dict:
    started = time.perf_counter()
    traces, fast_latency = await _timed(
        fast_client.generate_batch(
            [prompt.text] * batch_size,
            deterministic=False,
            max_new_tokens=window,
        )
    )
    canonical = traces[0]
    shadows = traces[1:]
    canonical.prompt_id = prompt.prompt_id
    canonical.plan_name = "self_canonical_0"
    for index, trace in enumerate(shadows, start=1):
        trace.prompt_id = prompt.prompt_id
        trace.plan_name = f"self_shadow_{index}"
        trace.batch_size = batch_size
        trace.protected_index = index

    gate_started = time.perf_counter()
    decision = evaluate_dynamic_window(
        canonical,
        shadows,
        window=window,
    )
    gate_cpu_s = time.perf_counter() - gate_started
    should_verify = force_verify or decision.should_verify

    verifier = None
    verifier_latency = 0.0
    if should_verify:
        verifier_rows, verifier_latency = await _timed(
            verifier_client.generate_batch(
                [prompt.text],
                deterministic=True,
                max_new_tokens=window,
            )
        )
        verifier = verifier_rows[0]

    first_window_match = (
        canonical.output_ids[:window] == verifier.output_ids[:window]
        if verifier is not None
        else None
    )
    return {
        "prompt_id": prompt.prompt_id,
        "batch_size": batch_size,
        "canonical": canonical.to_dict(),
        "shadows": [trace.to_dict() for trace in shadows],
        "verifier": verifier.to_dict() if verifier is not None else None,
        "gate": decision.to_dict(),
        "gate_should_verify": decision.should_verify,
        "verifier_launched": verifier is not None,
        "fast_latency_s": fast_latency,
        "gate_cpu_s": gate_cpu_s,
        "verifier_latency_s": verifier_latency,
        "commit_latency_s": (
            time.perf_counter() - started
        ),
        "first_window_match": first_window_match,
    }


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
        semaphore = asyncio.Semaphore(args.prompt_concurrency)

        async def run_limited(prompt):
            async with semaphore:
                return await run_one(
                    fast_client,
                    verifier_client,
                    prompt,
                    batch_size=args.batch_size,
                    window=args.window,
                    force_verify=args.force_verify,
                )

        rounds = []
        for round_index in range(args.rounds):
            started = time.perf_counter()
            rows = await asyncio.gather(
                *(run_limited(prompt) for prompt in prompts)
            )
            rounds.append(
                {
                    "round": round_index,
                    "total_s": time.perf_counter() - started,
                    "results": rows,
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
        "batch_size": args.batch_size,
        "window": args.window,
        "force_verify": args.force_verify,
        "round_metrics": [
            {
                "round": row["round"],
                "total_s": row["total_s"],
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
                "mean_verifier_latency_s": statistics.mean(
                    row["verifier_latency_s"] for row in verified
                )
                if verified
                else 0.0,
                "mean_commit_latency_s": statistics.mean(
                    row["commit_latency_s"] for row in results
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
        default="standard",
    )
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--prompt-concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--force-verify", action="store_true")
    parser.add_argument("--fast-port", type=int, default=30010)
    parser.add_argument("--verifier-port", type=int, default=30011)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument(
        "--output",
        default="ssv4d/results/self_shadow_gate.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
