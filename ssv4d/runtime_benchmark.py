"""Concurrent LLM-42 workload for SSV4D runtime wall-time comparisons."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path

import aiohttp

from ssv4d.prompts import PROMPTS
from ssv4d.prompts_extended import EXTENDED_PROMPTS


async def _request(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt_id: str,
    prompt: str,
    max_new_tokens: int,
) -> dict:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
            "is_deterministic": True,
            "sampling_seed": 42,
        },
        "return_logprob": True,
        "logprob_start_len": -1,
        "top_logprobs_num": 8,
        "stream": False,
    }
    started = time.perf_counter()
    async with session.post(f"{base_url.rstrip('/')}/generate", json=payload) as response:
        raw = await response.text()
        if response.status != 200:
            raise RuntimeError(f"{prompt_id}: HTTP {response.status}: {raw[:500]}")
        data = json.loads(raw)
    wall_s = time.perf_counter() - started
    output_ids = data.get("output_ids") or []
    meta = data.get("meta_info") or {}
    return {
        "prompt_id": prompt_id,
        "wall_s": wall_s,
        "e2e_latency_s": meta.get("e2e_latency"),
        "output_ids": output_ids,
        "digest": hashlib.sha256(
            json.dumps(output_ids).encode("utf-8")
        ).hexdigest(),
        "rollbacks": meta.get("llm42_num_rollbacks", 0),
        "tokens_rolled_back": meta.get("llm42_tokens_rolled_back", 0),
        "verification_windows": meta.get("llm42_num_verification_windows", 0),
        "completion_tokens": meta.get("completion_tokens", len(output_ids)),
    }


async def run(args: argparse.Namespace) -> None:
    prompt_set = (
        EXTENDED_PROMPTS if args.prompt_set == "extended" else PROMPTS
    )
    prompts = list(prompt_set[: args.num_prompts])
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    rounds = []
    prompt_concurrency = args.prompt_concurrency or len(prompts)
    semaphore = asyncio.Semaphore(prompt_concurrency)

    async def one(session: aiohttp.ClientSession, prompt):
        async with semaphore:
            return await _request(
                session,
                args.base_url,
                prompt.prompt_id,
                prompt.text,
                args.max_new_tokens,
            )

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for round_index in range(args.rounds):
            started = time.perf_counter()
            results = await asyncio.gather(
                *(
                    one(session, prompt)
                    for prompt in prompts
                )
            )
            total_s = time.perf_counter() - started
            latencies = [row["wall_s"] for row in results]
            total_output = sum(row["completion_tokens"] for row in results)
            rounds.append(
                {
                    "round": round_index,
                    "total_s": total_s,
                    "output_tokens_per_s": total_output / total_s if total_s else 0,
                    "mean_latency_s": statistics.mean(latencies),
                    "median_latency_s": statistics.median(latencies),
                    "p99_latency_s": sorted(latencies)[
                        max(0, int(len(latencies) * 0.99) - 1)
                    ],
                    "rollbacks": sum(row["rollbacks"] for row in results),
                    "tokens_rolled_back": sum(
                        row["tokens_rolled_back"] for row in results
                    ),
                    "verification_windows": sum(
                        row["verification_windows"] for row in results
                    ),
                    "requests": [
                        {key: value for key, value in row.items() if key != "output_ids"}
                        for row in results
                    ],
                }
            )

    deterministic = True
    if len(rounds) > 1:
        reference = {
            row["prompt_id"]: row["digest"] for row in rounds[0]["requests"]
        }
        for round_data in rounds[1:]:
            current = {
                row["prompt_id"]: row["digest"] for row in round_data["requests"]
            }
            deterministic &= current == reference

    payload = {
        "base_url": args.base_url,
        "num_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "rounds": rounds,
        "prompt_concurrency": prompt_concurrency,
        "cross_round_deterministic": deterministic,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument(
        "--prompt-set",
        choices=("standard", "extended"),
        default="standard",
    )
    parser.add_argument("--max-new-tokens", type=int, default=130)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--prompt-concurrency",
        type=int,
        default=0,
        help="Maximum concurrently active prompts; 0 means all prompts.",
    )
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--output",
        default="ssv4d/results/runtime_benchmark.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
