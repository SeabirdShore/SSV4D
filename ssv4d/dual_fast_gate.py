"""Two fast-decode instances with SSV4D gate before verification."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp

from ssv4d.async_dual_gpu import (
    _extend_stream_state,
    _launch_server,
    _stream_trace,
    _terminate,
    _wait_ready,
)
from ssv4d.client import SGLangShadowClient
from ssv4d.dynamic_gate import evaluate_dynamic_window
from ssv4d.perturbation import make_perturbation_plans
from ssv4d.prompts import FILLER_PROMPTS, PROMPTS


async def _timed(awaitable):
    started = time.perf_counter()
    result = await awaitable
    return result, time.perf_counter() - started


async def _stream_canonical(
    url: str,
    prompt: str,
    *,
    max_tokens: int,
    window: int,
    timeout_s: float,
    ready_event: asyncio.Event,
    ready_state: dict,
) -> dict:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_tokens,
            "ignore_eos": True,
            "is_deterministic": False,
            "sampling_seed": 42,
        },
        "return_logprob": True,
        "logprob_start_len": -1,
        "top_logprobs_num": 8,
        "stream": True,
    }
    ids: list[int] = []
    top_logprobs: list[list[float]] = []
    top_ids: list[list[int]] = []
    started = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{url}/generate", json=payload) as response:
            if response.status != 200:
                raise RuntimeError(await response.text())
            async for raw in response.content:
                line = raw.strip()
                if not line.startswith(b"data: "):
                    continue
                chunk = line[6:]
                if chunk == b"[DONE]":
                    break
                data = json.loads(chunk)
                _extend_stream_state(
                    ids, top_logprobs, top_ids, data
                )
                if not ready_event.is_set() and len(ids) >= window:
                    ready_state["prefix"] = _stream_trace(
                        ids,
                        top_logprobs,
                        top_ids,
                        window=window,
                    )
                    ready_state["prefix_ready_at"] = time.perf_counter()
                    ready_event.set()
    finished = time.perf_counter()
    if not ready_event.is_set():
        ready_state["prefix"] = _stream_trace(
            ids,
            top_logprobs,
            top_ids,
            window=window,
        )
        ready_state["prefix_ready_at"] = finished
        ready_event.set()
    return {
        "output_ids": ids,
        "top_logprobs": top_logprobs,
        "top_ids": top_ids,
        "started": started,
        "finished": finished,
        "latency_s": finished - started,
    }


async def run_one(
    canonical_url: str,
    shadow_client: SGLangShadowClient,
    verifier_client: SGLangShadowClient,
    prompt,
    *,
    shadow_plan_prefix: str,
    window: int,
    canonical_tokens: int,
    timeout_s: float,
    force_verify: bool,
) -> dict:
    plans = make_perturbation_plans(
        prompt.text,
        FILLER_PROMPTS,
        batch_sizes=(8,),
        positions=("middle",),
        repeats=1,
    )
    plan = next(
        plan
        for plan in plans
        if plan.name.startswith(shadow_plan_prefix)
    )

    async def run_shadow():
        traces = await shadow_client.generate_batch(
            plan.prompts,
            deterministic=False,
            max_new_tokens=window,
        )
        trace = traces[plan.protected_index]
        trace.prompt_id = prompt.prompt_id
        trace.plan_name = plan.name
        trace.batch_size = plan.batch_size
        trace.protected_index = plan.protected_index
        return trace

    started = time.perf_counter()
    ready_event = asyncio.Event()
    ready_state = {}
    canonical_task = asyncio.create_task(
        _stream_canonical(
            canonical_url,
            prompt.text,
            max_tokens=canonical_tokens,
            window=window,
            timeout_s=timeout_s,
            ready_event=ready_event,
            ready_state=ready_state,
        )
    )
    shadow, shadow_latency = await _timed(run_shadow())
    await ready_event.wait()
    canonical = ready_state["prefix"]
    canonical.prompt_id = prompt.prompt_id
    canonical.plan_name = "canonical"
    gate_ready_at = time.perf_counter()
    gate_started = time.perf_counter()
    decision = evaluate_dynamic_window(
        canonical,
        [shadow],
        window=window,
    )
    gate_cpu_s = time.perf_counter() - gate_started
    should_verify = force_verify or decision.should_verify

    verifier = None
    verifier_latency = 0.0
    verifier_task = None
    verifier_started = None
    if should_verify:
        async def run_verifier():
            return await verifier_client.generate_batch(
                [prompt.text],
                deterministic=True,
                max_new_tokens=window,
            )

        verifier_started = time.perf_counter()
        verifier_task = asyncio.create_task(
            _timed(run_verifier())
        )

    canonical_full = await canonical_task
    canonical_finished = time.perf_counter()
    verifier_finished = canonical_finished
    if verifier_task is not None:
        verifier_rows, verifier_latency = await verifier_task
        verifier = verifier_rows[0]
        verifier_finished = time.perf_counter()

    canonical_output_ids = canonical_full["output_ids"]

    first_window_match = (
        canonical_output_ids[:window]
        == verifier.output_ids[:window]
        if verifier is not None
        else None
    )
    overlap_s = 0.0
    post_canonical_wait_s = 0.0
    if verifier_started is not None:
        overlap_s = max(
            0.0,
            min(canonical_finished, verifier_finished)
            - verifier_started,
        )
        post_canonical_wait_s = max(
            0.0, verifier_finished - canonical_finished
        )
    commit_finished = max(canonical_finished, verifier_finished)
    return {
        "prompt_id": prompt.prompt_id,
        "plan": plan.name,
        "canonical": {
            **canonical.to_dict(),
            "output_ids": canonical_output_ids,
            "full_top_logprobs": canonical_full["top_logprobs"],
            "full_top_ids": canonical_full["top_ids"],
        },
        "shadow": shadow.to_dict(),
        "verifier": verifier.to_dict() if verifier is not None else None,
        "gate": decision.to_dict(),
        "gate_should_verify": decision.should_verify,
        "force_verify": force_verify,
        "verifier_launched": verifier is not None,
        "canonical_latency_s": canonical_full["latency_s"],
        "shadow_latency_s": shadow_latency,
        "verifier_latency_s": verifier_latency,
        "gate_cpu_s": gate_cpu_s,
        "gate_ready_latency_s": gate_ready_at - started,
        "verifier_started_s": (
            verifier_started - started
            if verifier_started is not None
            else None
        ),
        "canonical_finished_s": canonical_finished - started,
        "verifier_finished_s": (
            verifier_finished - started
            if verifier_started is not None
            else None
        ),
        "overlap_s": overlap_s,
        "post_canonical_wait_s": post_canonical_wait_s,
        "commit_latency_s": commit_finished - started,
        "first_window_match": first_window_match,
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
                _wait_ready(
                    session, canonical_url, args.startup_timeout
                ),
                _wait_ready(
                    session, shadow_url, args.startup_timeout
                ),
            )
            if not args.reuse_servers:
                processes.append(
                    _launch_server(
                        1,
                        args.verifier_port,
                        args.model,
                        deterministic=True,
                        log_path=log_dir
                        / f"{stem}_verifier_server.log",
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

        prompts = list(PROMPTS[: args.num_prompts])
        rounds = []
        for round_index in range(args.rounds):
            started = time.perf_counter()
            rows = []
            for prompt in prompts:
                rows.append(
                    await run_one(
                        canonical_url,
                        shadow_client,
                        verifier_client,
                        prompt,
                        shadow_plan_prefix=args.shadow_plan,
                        window=args.window,
                        canonical_tokens=args.canonical_tokens,
                        timeout_s=args.timeout_s,
                        force_verify=args.force_verify,
                    )
                )
            rounds.append(
                {
                    "round": round_index,
                    "total_s": time.perf_counter() - started,
                    "results": rows,
                }
            )
            if args.round_delay_s:
                await asyncio.sleep(args.round_delay_s)
    finally:
        _terminate(processes)

    all_results = [
        {"round": row["round"], **result}
        for row in rounds
        for result in row["results"]
    ]
    verified = [result for result in all_results if result["verifier_launched"]]
    payload = {
        "model": args.model,
        "num_prompts": len(prompts),
        "rounds": args.rounds,
        "window": args.window,
        "canonical_tokens": args.canonical_tokens,
        "shadow_plan": args.shadow_plan,
        "force_verify": args.force_verify,
        "mem_fraction_static": args.mem_fraction_static,
        "round_metrics": [
            {
                "round": row["round"],
                "total_s": row["total_s"],
            }
            for row in rounds
        ],
        "results": all_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "total_s": sum(row["total_s"] for row in rounds),
                "mean_round_s": statistics.mean(
                    row["total_s"] for row in rounds
                ),
                "requests": len(all_results),
                "gate_triggered": sum(
                    result["gate_should_verify"]
                    for result in all_results
                ),
                "verifier_launched": len(verified),
                "first_window_matches": sum(
                    result["first_window_match"] is True
                    for result in verified
                ),
                "mean_canonical_latency_s": statistics.mean(
                    result["canonical_latency_s"]
                    for result in all_results
                ),
                "mean_shadow_latency_s": statistics.mean(
                    result["shadow_latency_s"]
                    for result in all_results
                ),
                "mean_verifier_latency_s": statistics.mean(
                    result["verifier_latency_s"]
                    for result in verified
                )
                if verified
                else 0.0,
                "mean_overlap_s": statistics.mean(
                    result["overlap_s"] for result in verified
                )
                if verified
                else 0.0,
                "mean_commit_latency_s": statistics.mean(
                    result["commit_latency_s"]
                    for result in all_results
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
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--canonical-tokens", type=int, default=130)
    parser.add_argument(
        "--shadow-plan", default="bs8_middle"
    )
    parser.add_argument("--force-verify", action="store_true")
    parser.add_argument("--canonical-port", type=int, default=30010)
    parser.add_argument("--shadow-port", type=int, default=30011)
    parser.add_argument("--verifier-port", type=int, default=30012)
    parser.add_argument("--mem-fraction-static", type=float, default=0.28)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--round-delay-s", type=float, default=0.0)
    parser.add_argument("--reuse-servers", action="store_true")
    parser.add_argument(
        "--output",
        default="ssv4d/results/dual_fast_gate.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
