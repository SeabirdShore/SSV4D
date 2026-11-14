"""Dual-GPU asynchronous draft/verifier prototype."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import aiohttp

from ssv4d.client import SGLangShadowClient
from ssv4d.dynamic_gate import evaluate_dynamic_window
from ssv4d.features import topk_margin
from ssv4d.models import GenerationTrace
from ssv4d.perturbation import make_perturbation_plans
from ssv4d.prompts import FILLER_PROMPTS, PROMPTS


REPO_ROOT = Path(__file__).resolve().parents[1]


def _server_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PATH"] = f"{REPO_ROOT}/.venv/bin:/usr/local/cuda/bin:" + env.get("PATH", "")
    numa = "/home/xiaosibo/.local/llm42-numa/usr/lib/x86_64-linux-gnu"
    env["LD_LIBRARY_PATH"] = numa + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    return env


def _launch_server(
    gpu: int,
    port: int,
    model: str,
    *,
    deterministic: bool,
    log_path: Path,
    mem_fraction_static: float | None = None,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tp",
        "1",
        "--attention-backend",
        "fa3",
        "--disable-radix-cache",
        "--disable-chunked-prefix-cache",
        "--disable-overlap-schedule",
        "--chunked-prefill-size",
        "-1",
        "--random-seed",
        "42",
    ]
    if mem_fraction_static is not None:
        cmd.extend(
            ["--mem-fraction-static", str(mem_fraction_static)]
        )
    if deterministic:
        cmd.extend(["--enable-deterministic-inference", "2"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        env=_server_env(gpu),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    process._ssv4d_log_handle = handle
    return process


async def _wait_ready(session: aiohttp.ClientSession, url: str, timeout_s: float) -> None:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout_s:
        try:
            async with session.get(f"{url}/v1/models") as response:
                if response.status == 200:
                    return
        except aiohttp.ClientError:
            pass
        await asyncio.sleep(2)
    raise TimeoutError(f"Server {url} not ready after {timeout_s}s")


def _extend_stream_state(
    ids: list[int],
    top_logprobs: list[list[float]],
    top_ids: list[list[int]],
    data: dict,
) -> None:
    meta = data.get("meta_info") or {}
    new_ids = data.get("output_ids") or []
    completion_tokens = meta.get("completion_tokens")
    if completion_tokens is not None and len(new_ids) == completion_tokens:
        ids[:] = list(new_ids)
    else:
        ids.extend(new_ids)

    values = meta.get("output_top_logprobs") or []
    indices = meta.get("output_top_logprobs_idx") or []
    if values and values[0] and isinstance(values[0][0], (list, tuple)):
        indices = [
            [int(entry[1]) for entry in row]
            for row in values
        ]
        values = [
            [float(entry[0]) for entry in row]
            for row in values
        ]
    if not values and top_logprobs:
        return
    if completion_tokens is not None and len(values) == completion_tokens:
        top_logprobs[:] = [list(row or []) for row in values]
        top_ids[:] = [list(row or []) for row in indices]
    else:
        top_logprobs.extend([list(row or []) for row in values])
        top_ids.extend([list(row or []) for row in indices])


def _stream_trace(
    ids: list[int],
    top_logprobs: list[list[float]],
    top_ids: list[list[int]],
    *,
    window: int,
) -> GenerationTrace:
    end = min(window, len(ids), len(top_logprobs), len(top_ids))
    return GenerationTrace(
        prompt_id="canonical",
        scenario="shadow",
        batch_size=1,
        protected_index=0,
        output_ids=list(ids[:end]),
        text="",
        margins=[
            topk_margin(top_logprobs[index], top_ids[index])
            for index in range(end)
        ],
        top_ids=[list(row) for row in top_ids[:end]],
        top_logprobs=[list(row) for row in top_logprobs[:end]],
        latency_s=0.0,
        deterministic=False,
        plan_name="canonical",
    )


async def _collect_shadow_traces(
    client: SGLangShadowClient,
    prompt: str,
    *,
    window: int,
    plan_prefixes: tuple[str, ...],
) -> list[GenerationTrace]:
    plans = make_perturbation_plans(
        prompt,
        FILLER_PROMPTS,
        batch_sizes=(4, 8),
        positions=("first", "middle", "last"),
        repeats=1,
    )
    plans = [
        plan
        for plan in plans
        if any(plan.name.startswith(name) for name in plan_prefixes)
    ]
    batches = await asyncio.gather(
        *(
            client.generate_batch(
                plan.prompts,
                deterministic=False,
                max_new_tokens=window,
            )
            for plan in plans
        )
    )
    traces = []
    for plan, batch in zip(plans, batches):
        trace = batch[plan.protected_index]
        trace.plan_name = plan.name
        trace.batch_size = plan.batch_size
        trace.protected_index = plan.protected_index
        traces.append(trace)
    return traces


async def _draft_stream(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    *,
    max_new_tokens: int,
    window: int,
    gate_threshold: float,
    force_verify: bool,
    verifier_url: str,
    model: str,
    async_verify: bool = True,
    dynamic_gate: bool = False,
    fast_client: SGLangShadowClient | None = None,
    shadow_plans: tuple[str, ...] = (
        "bs4_first",
        "bs8_middle",
        "bs8_last",
    ),
) -> dict:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
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
    prompt_started = time.perf_counter()
    gate_time = None
    verifier_task = None
    shadow_task = None
    if dynamic_gate:
        if fast_client is None:
            raise ValueError("Dynamic gate requires a fast-draft client")
        shadow_task = asyncio.create_task(
            _collect_shadow_traces(
                fast_client,
                prompt,
                window=window,
                plan_prefixes=shadow_plans,
            )
        )

    async def run_verifier():
        verifier_start = time.perf_counter()
        result = await _generate_once(
            session,
            verifier_url,
            prompt,
            max_new_tokens=window,
            deterministic=True,
        )
        result["started"] = verifier_start
        result["finished"] = time.perf_counter()
        return result

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
            _extend_stream_state(ids, top_logprobs, top_ids, data)
            if gate_time is None and len(ids) >= window and len(top_logprobs) >= window:
                gate_time = time.perf_counter()
                canonical_trace = _stream_trace(
                    ids,
                    top_logprobs,
                    top_ids,
                    window=window,
                )
                margins = canonical_trace.margins
                min_margin = min(margins)
                dynamic_decision = None
                if dynamic_gate:
                    try:
                        shadows = await shadow_task
                        dynamic_decision = evaluate_dynamic_window(
                            canonical_trace,
                            shadows,
                            window=window,
                        ).to_dict()
                        should_verify = dynamic_decision["should_verify"]
                        gate_reason = dynamic_decision["trigger_reason"]
                        dynamic_threshold = dynamic_decision[
                            "trigger_threshold"
                        ]
                    except Exception as exc:
                        should_verify = True
                        gate_reason = "shadow_error"
                        dynamic_threshold = None
                        dynamic_decision = {
                            "should_verify": True,
                            "trigger_reason": gate_reason,
                            "error": repr(exc),
                        }
                else:
                    should_verify = min_margin <= gate_threshold
                    gate_reason = "margin_threshold"
                    dynamic_threshold = None
                requested_verify = force_verify or should_verify
                if requested_verify and async_verify:
                    verifier_task = asyncio.create_task(run_verifier())
                gate_decision = {
                    "min_margin": min_margin,
                    "should_verify": requested_verify,
                    "gate_reason": (
                        "forced" if force_verify else gate_reason
                    ),
                    "dynamic_threshold": dynamic_threshold,
                    "dynamic_gate": dynamic_decision,
                    "verifier_launched": verifier_task is not None,
                    "gate_latency_from_start_s": gate_time - prompt_started,
                }

    if shadow_task is not None and not shadow_task.done():
        shadow_task.cancel()
    draft_end = time.perf_counter()
    verifier = await verifier_task if verifier_task is not None else None
    if verifier is None:
        verifier = {
            "output_ids": [],
            "started": None,
            "finished": None,
            "latency_s": 0.0,
        }
    first_window_match = (
        verifier["output_ids"][:window] == ids[:window]
        if verifier["output_ids"]
        else None
    )
    overlap_s = 0.0
    if verifier["started"] is not None:
        overlap_s = max(
            0.0,
            min(draft_end, verifier["finished"]) - verifier["started"],
        )
    return {
        "draft_output_ids": ids,
        "draft_latency_s": draft_end - prompt_started,
        "gate_decision": gate_decision if gate_time is not None else None,
        "verifier_output_ids": verifier["output_ids"],
        "verifier_latency_s": verifier["latency_s"],
        "verifier_started": verifier["started"],
        "verifier_finished": verifier["finished"],
        "draft_finished": draft_end,
        "overlap_s": overlap_s,
        "commit_latency_s": max(draft_end, verifier["finished"] or 0) - prompt_started,
        "first_window_match": first_window_match,
    }


async def _verify_only(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    *,
    max_new_tokens: int,
) -> dict:
    started = time.perf_counter()
    result = await _generate_once(
        session,
        url,
        prompt,
        max_new_tokens=max_new_tokens,
        deterministic=True,
    )
    finished = time.perf_counter()
    return {
        "draft_output_ids": [],
        "draft_latency_s": 0.0,
        "gate_decision": None,
        "verifier_output_ids": result["output_ids"],
        "verifier_latency_s": result["latency_s"],
        "verifier_started": started,
        "verifier_finished": finished,
        "draft_finished": None,
        "overlap_s": 0.0,
        "commit_latency_s": finished - started,
        "first_window_match": None,
    }


async def _generate_once(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    *,
    max_new_tokens: int,
    deterministic: bool,
) -> dict:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
            "is_deterministic": deterministic,
            "sampling_seed": 42,
        },
        "return_logprob": False,
        "stream": False,
    }
    started = time.perf_counter()
    async with session.post(f"{url}/generate", json=payload) as response:
        data = await response.json(content_type=None)
        if response.status != 200:
            raise RuntimeError(data)
    return {
        "output_ids": data.get("output_ids") or [],
        "latency_s": time.perf_counter() - started,
    }


def _terminate(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        handle = getattr(process, "_ssv4d_log_handle", None)
        if handle is not None:
            handle.close()


async def run(args: argparse.Namespace) -> None:
    processes = []
    if not args.reuse_servers:
        log_dir = Path("ssv4d/logs")
        stem = Path(args.output).stem
        if args.mode != "verifier-only":
            processes.append(
                _launch_server(
                    0,
                    args.draft_port,
                    args.model,
                    deterministic=False,
                    log_path=log_dir / f"{stem}_draft_server.log",
                )
            )
        if args.mode != "draft-only":
            processes.append(
                _launch_server(
                    1,
                    args.verifier_port,
                    args.model,
                    deterministic=True,
                    log_path=log_dir / f"{stem}_verifier_server.log",
                )
            )
    try:
        timeout = aiohttp.ClientTimeout(total=args.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            draft_url = f"http://127.0.0.1:{args.draft_port}"
            verifier_url = f"http://127.0.0.1:{args.verifier_port}"
            ready_urls = []
            if args.mode != "verifier-only":
                ready_urls.append(draft_url)
            if args.mode != "draft-only":
                ready_urls.append(verifier_url)
            await asyncio.gather(
                *(
                    _wait_ready(session, url, args.startup_timeout)
                    for url in ready_urls
                )
            )
            prompts = list(PROMPTS[: args.num_prompts])
            fast_client = (
                SGLangShadowClient(
                    draft_url,
                    model=args.model,
                    timeout_s=args.timeout_s,
                )
                if args.mode == "dynamic-async"
                else None
            )
            round_payloads = []
            for round_index in range(args.rounds):
                started = time.perf_counter()
                if args.mode == "verifier-only":
                    round_results = await asyncio.gather(
                        *(
                            _verify_only(
                                session,
                                verifier_url,
                                prompt.text,
                                max_new_tokens=args.max_new_tokens,
                            )
                            for prompt in prompts
                        )
                    )
                else:
                    round_results = await asyncio.gather(
                        *(
                            _draft_stream(
                                session,
                                draft_url,
                                prompt.text,
                                max_new_tokens=args.max_new_tokens,
                                window=args.window,
                                gate_threshold=args.gate_threshold,
                                force_verify=args.force_verify,
                                verifier_url=verifier_url,
                                model=args.model,
                                async_verify=args.mode
                                in {"async", "dynamic-async"},
                                dynamic_gate=args.mode == "dynamic-async",
                                fast_client=fast_client,
                                shadow_plans=tuple(
                                    part
                                    for part in args.shadow_plans.split(",")
                                    if part
                                ),
                            )
                            for prompt in prompts
                        )
                    )
                total_s = time.perf_counter() - started
                round_payloads.append(
                    {
                        "round": round_index,
                        "total_s": total_s,
                        "output_tokens_per_s": (
                            len(prompts) * args.max_new_tokens / total_s
                        ),
                        "results": [
                            {"prompt_id": prompt.prompt_id, **result}
                            for prompt, result in zip(prompts, round_results)
                        ],
                    }
                )
    finally:
        _terminate(processes)

    results = [
        {"round": row["round"], **result}
        for row in round_payloads
        for result in row["results"]
    ]
    total_s = sum(row["total_s"] for row in round_payloads)
    payload = {
        "num_prompts": len(prompts),
        "rounds": args.rounds,
        "mode": args.mode,
        "window": args.window,
        "max_new_tokens": args.max_new_tokens,
        "force_verify": args.force_verify,
        "gate_threshold": (
            None if args.mode == "dynamic-async" else args.gate_threshold
        ),
        "total_s": total_s,
        "output_tokens_per_s": (
            len(prompts) * args.rounds * args.max_new_tokens / total_s
        ),
        "round_metrics": [
            {
                "round": row["round"],
                "total_s": row["total_s"],
                "output_tokens_per_s": row["output_tokens_per_s"],
            }
            for row in round_payloads
        ],
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    verified = [row for row in payload["results"] if row["verifier_output_ids"]]
    print(
        json.dumps(
            {
                "mode": args.mode,
                "rounds": args.rounds,
                "total_s": total_s,
                "mean_round_s": statistics.mean(
                    row["total_s"] for row in payload["round_metrics"]
                ),
                "output_tokens_per_s": (
                    len(prompts)
                    * args.rounds
                    * args.max_new_tokens
                    / total_s
                ),
                "verified_requests": len(verified),
                "first_window_matches": sum(
                    row["first_window_match"] is True for row in verified
                ),
                "mean_draft_latency_s": statistics.mean(
                    row["draft_latency_s"] for row in payload["results"]
                ),
                "mean_verifier_latency_s": statistics.mean(
                    row["verifier_latency_s"] for row in verified
                )
                if verified
                else 0.0,
                "mean_overlap_s": statistics.mean(
                    row["overlap_s"] for row in verified
                )
                if verified
                else 0.0,
                "mean_commit_latency_s": statistics.mean(
                    row["commit_latency_s"] for row in payload["results"]
                ),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "async",
            "dynamic-async",
            "draft-only",
            "verifier-only",
        ),
        default="async",
        help=(
            "Async first-window verification, the dynamic-threshold variant, "
            "or a single-path baseline."
        ),
    )
    parser.add_argument(
        "--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=130)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--gate-threshold", type=float, default=1.0)
    parser.add_argument("--force-verify", action="store_true")
    parser.add_argument(
        "--shadow-plans",
        default="bs4_first,bs8_middle,bs8_last",
        help="Comma-separated perturbation-plan prefixes for dynamic gating.",
    )
    parser.add_argument("--draft-port", type=int, default=30010)
    parser.add_argument("--verifier-port", type=int, default=30011)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--reuse-servers", action="store_true")
    parser.add_argument(
        "--output", default="ssv4d/results/async_dual_gpu.json"
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
