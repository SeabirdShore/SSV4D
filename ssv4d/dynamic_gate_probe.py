"""Collect canonical, perturbed-fast-draft, and verifier traces in parallel."""

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
from ssv4d.perturbation import make_perturbation_plans
from ssv4d.prompts import FILLER_PROMPTS, PROMPTS


async def _timed(awaitable):
    started = time.perf_counter()
    result = await awaitable
    return result, time.perf_counter() - started


async def collect_one(
    fast_client: SGLangShadowClient,
    verifier_client: SGLangShadowClient,
    prompt,
    *,
    window: int,
) -> dict:
    plans = make_perturbation_plans(
        prompt.text,
        FILLER_PROMPTS,
        batch_sizes=(4, 8),
        positions=("first", "middle", "last"),
        repeats=1,
    )
    wanted = ("bs4_first", "bs8_middle", "bs8_last")
    plans = [
        plan
        for plan in plans
        if any(plan.name.startswith(name) for name in wanted)
    ]
    if len(plans) != len(wanted):
        raise RuntimeError(
            f"Expected {len(wanted)} perturbation plans, found {len(plans)}"
        )

    async def run_reference():
        traces = await fast_client.generate_batch(
            [prompt.text],
            deterministic=False,
            max_new_tokens=window,
        )
        return traces[0]

    async def run_shadow(plan):
        traces = await fast_client.generate_batch(
            plan.prompts,
            deterministic=False,
            max_new_tokens=window,
        )
        trace = traces[plan.protected_index]
        trace.plan_name = plan.name
        trace.batch_size = plan.batch_size
        trace.protected_index = plan.protected_index
        return trace

    async def run_verifier():
        traces = await verifier_client.generate_batch(
            [prompt.text],
            deterministic=True,
            max_new_tokens=window,
        )
        return traces[0]

    tasks = [
        _timed(run_reference()),
        _timed(run_verifier()),
        *(_timed(run_shadow(plan)) for plan in plans),
    ]
    results = await asyncio.gather(*tasks)
    reference, reference_wall = results[0]
    verifier, verifier_wall = results[1]
    shadows = []
    for plan, (trace, wall) in zip(plans, results[2:]):
        trace.metadata["gate_probe_wall_s"] = wall
        trace.prompt_id = prompt.prompt_id
        shadows.append(trace)
    reference.prompt_id = prompt.prompt_id
    reference.plan_name = "canonical"
    verifier.prompt_id = prompt.prompt_id
    verifier.plan_name = "verifier"
    decision = evaluate_dynamic_window(
        reference,
        shadows,
        window=window,
    )
    return {
        "prompt_id": prompt.prompt_id,
        "canonical": reference.to_dict(),
        "shadows": [trace.to_dict() for trace in shadows],
        "verifier": verifier.to_dict(),
        "dynamic_gate": decision.to_dict(),
        "canonical_wall_s": reference_wall,
        "verifier_wall_s": verifier_wall,
    }


async def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    stem = output.stem
    log_dir = Path("ssv4d/logs")
    processes = []
    if not args.reuse_servers:
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
                _wait_ready(session, verifier_url, args.startup_timeout),
            )

        fast_client = SGLangShadowClient(
            fast_url,
            model=args.model,
            timeout_s=args.timeout_s,
        )
        verifier_client = SGLangShadowClient(
            verifier_url,
            model=args.model,
            timeout_s=args.timeout_s,
        )
        prompts = list(PROMPTS[: args.num_prompts])
        rounds = []
        for round_index in range(args.rounds):
            started = time.perf_counter()
            rows = []
            for prompt in prompts:
                rows.append(
                    await collect_one(
                        fast_client,
                        verifier_client,
                        prompt,
                        window=args.window,
                    )
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

    all_rows = [
        {"round": row["round"], **result}
        for row in rounds
        for result in row["results"]
    ]
    payload = {
        "model": args.model,
        "num_prompts": len(prompts),
        "rounds": args.rounds,
        "window": args.window,
        "round_metrics": [
            {
                "round": row["round"],
                "total_s": row["total_s"],
            }
            for row in rounds
        ],
        "results": all_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    matches = sum(
        row["canonical"]["output_ids"][: args.window]
        == row["verifier"]["output_ids"][: args.window]
        for row in all_rows
    )
    print(
        json.dumps(
            {
                "total_s": sum(row["total_s"] for row in rounds),
                "mean_round_s": statistics.mean(
                    row["total_s"] for row in rounds
                ),
                "requests": len(all_rows),
                "canonical_verifier_matches": matches,
                "fast_shadow_requests_per_request": 3,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--fast-port", type=int, default=30010)
    parser.add_argument("--verifier-port", type=int, default=30011)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--reuse-servers", action="store_true")
    parser.add_argument(
        "--output",
        default="ssv4d/results/dynamic_gate_probe.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
