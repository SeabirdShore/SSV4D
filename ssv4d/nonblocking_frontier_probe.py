"""Non-blocking verified-frontier service prototype."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

from ssv4d.async_dual_gpu import _launch_server, _terminate, _wait_ready
from ssv4d.prompts import PROMPTS
from ssv4d.prompts_extended import EXTENDED_PROMPTS


async def _stream_ids(
    session: aiohttp.ClientSession,
    url: str,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
) -> dict:
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
            "is_deterministic": False,
            "sampling_seed": 42,
        },
        "return_logprob": False,
        "stream": True,
    }
    ids: list[int] = []
    started = time.perf_counter()
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
            new_ids = data.get("output_ids") or []
            completion = (data.get("meta_info") or {}).get(
                "completion_tokens"
            )
            if completion is not None and len(new_ids) == completion:
                ids[:] = [int(value) for value in new_ids]
            else:
                ids.extend(int(value) for value in new_ids)
    return {
        "output_ids": ids,
        "latency_s": time.perf_counter() - started,
        "finished": time.perf_counter(),
    }


async def _generate_ids(
    session: aiohttp.ClientSession,
    url: str,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
) -> dict:
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
            "is_deterministic": True,
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
        "output_ids": [int(value) for value in data.get("output_ids") or []],
        "latency_s": time.perf_counter() - started,
    }


async def _wait_for_ids(
    ids: list[int],
    target: int,
    task: asyncio.Task,
    timeout_s: float,
) -> None:
    started = time.perf_counter()
    while len(ids) < target:
        if task.done():
            await task
            if len(ids) < target:
                raise RuntimeError(
                    f"Canonical stream ended at {len(ids)} tokens, "
                    f"expected {target}"
                )
        if time.perf_counter() - started > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for canonical prefix {target}"
            )
        await asyncio.sleep(0.003)


async def run_prompt(
    session: aiohttp.ClientSession,
    fast_url: str,
    verifier_url: str,
    prompt,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    window: int,
    tail_merge_tokens: int,
) -> dict:
    started = time.perf_counter()
    canonical_ids: list[int] = []
    # Share the mutable list with the streaming parser.
    stream_state_task = asyncio.create_task(
        _stream_ids_into(
            session,
            fast_url,
            prompt_ids,
            canonical_ids,
            max_new_tokens=max_new_tokens,
        )
    )

    committed: list[int] = []
    frontier = 0
    verify_slices = []
    first_verify_started = None
    while frontier < max_new_tokens:
        target = frontier + window
        if max_new_tokens - target <= tail_merge_tokens:
            target = max_new_tokens
        target = min(target, max_new_tokens)
        await _wait_for_ids(
            canonical_ids,
            target,
            stream_state_task,
            timeout_s=600.0,
        )
        speculative = canonical_ids[frontier:target]
        verify_prefix = prompt_ids + committed
        verify_started = time.perf_counter()
        if first_verify_started is None:
            first_verify_started = verify_started
        verified = await _generate_ids(
            session,
            verifier_url,
            verify_prefix,
            max_new_tokens=len(speculative),
        )
        verify_ended = time.perf_counter()
        verify_ids = verified["output_ids"]
        n = min(len(speculative), len(verify_ids))
        mismatch = next(
            (
                index
                for index in range(n)
                if speculative[index] != verify_ids[index]
            ),
            n,
        )
        verify_slices.append(
            {
                "frontier_before": frontier,
                "length": len(speculative),
                "mismatch": mismatch if mismatch < n else None,
                "verifier_latency_s": verified["latency_s"],
                "started_s": verify_started - started,
                "finished_s": verify_ended - started,
            }
        )
        if mismatch == n:
            committed.extend(speculative[:n])
            frontier += n
            continue

        committed.extend(speculative[:mismatch])
        committed.append(verify_ids[mismatch])
        frontier += mismatch + 1
        remaining = max_new_tokens - frontier
        if remaining > 0:
            tail = await _generate_ids(
                session,
                verifier_url,
                prompt_ids + committed,
                max_new_tokens=remaining,
            )
            committed.extend(tail["output_ids"][:remaining])
            frontier += min(remaining, len(tail["output_ids"]))
        break

    canonical_finished_abs = time.perf_counter()
    if stream_state_task.done():
        stream_result = await stream_state_task
        canonical_finished_abs = stream_result["finished"]
    else:
        stream_state_task.cancel()
        try:
            await stream_state_task
        except asyncio.CancelledError:
            pass
    commit_finished_abs = time.perf_counter()
    canonical_finished_s = canonical_finished_abs - started
    commit_finished_s = commit_finished_abs - started
    overlap_s = 0.0
    if first_verify_started is not None:
        overlap_s = max(
            0.0,
            min(canonical_finished_abs, commit_finished_abs)
            - first_verify_started,
        )
    return {
        "prompt_id": prompt.prompt_id,
        "prompt_ids": prompt_ids,
        "canonical_output_ids": canonical_ids,
        "committed_output_ids": committed,
        "frontier": frontier,
        "max_new_tokens": max_new_tokens,
        "canonical_finished_s": canonical_finished_s,
        "commit_finished_s": commit_finished_s,
        "overlap_s": overlap_s,
        "first_verify_started_s": first_verify_started - started
        if first_verify_started is not None
        else None,
        "verify_slices": verify_slices,
    }


async def _stream_ids_into(
    session: aiohttp.ClientSession,
    url: str,
    prompt_ids: list[int],
    output_ids: list[int],
    *,
    max_new_tokens: int,
) -> dict:
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
            "is_deterministic": False,
            "sampling_seed": 42,
        },
        "return_logprob": False,
        "stream": True,
    }
    started = time.perf_counter()
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
            new_ids = data.get("output_ids") or []
            completion = (data.get("meta_info") or {}).get(
                "completion_tokens"
            )
            if completion is not None and len(new_ids) == completion:
                output_ids[:] = [int(value) for value in new_ids]
            else:
                output_ids.extend(int(value) for value in new_ids)
    return {
        "output_ids": output_ids,
        "latency_s": time.perf_counter() - started,
        "finished": time.perf_counter(),
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
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            prompt_set = (
                EXTENDED_PROMPTS
                if args.prompt_set == "extended"
                else PROMPTS
            )
            prompts = list(prompt_set[: args.num_prompts])
            semaphore = asyncio.Semaphore(args.prompt_concurrency)

            async def one(prompt):
                async with semaphore:
                    prompt_ids = tokenizer.encode(prompt.text)
                    return await run_prompt(
                        session,
                        fast_url,
                        verifier_url,
                        prompt,
                        prompt_ids,
                        max_new_tokens=args.max_new_tokens,
                        window=args.window,
                        tail_merge_tokens=args.tail_merge_tokens,
                    )

            rounds = []
            for round_index in range(args.rounds):
                started = time.perf_counter()
                rows = await asyncio.gather(
                    *(one(prompt) for prompt in prompts)
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
    payload = {
        "model": args.model,
        "prompt_set": args.prompt_set,
        "num_prompts": len(prompts),
        "rounds": args.rounds,
        "prompt_concurrency": args.prompt_concurrency,
        "window": args.window,
        "max_new_tokens": args.max_new_tokens,
        "round_metrics": [
            {"round": row["round"], "total_s": row["total_s"]}
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
                "fully_verified": sum(
                    row["frontier"] == row["max_new_tokens"]
                    for row in results
                ),
                "mean_commit_s": statistics.mean(
                    row["commit_finished_s"] for row in results
                ),
                "mean_canonical_s": statistics.mean(
                    row["canonical_finished_s"] for row in results
                ),
                "mean_first_verify_start_s": statistics.mean(
                    row["first_verify_started_s"]
                    for row in results
                    if row["first_verify_started_s"] is not None
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
    parser.add_argument("--prompt-concurrency", type=int, default=4)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--tail-merge-tokens", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=130)
    parser.add_argument("--fast-port", type=int, default=30010)
    parser.add_argument("--verifier-port", type=int, default=30011)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument(
        "--output",
        default="ssv4d/results/nonblocking_frontier.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
