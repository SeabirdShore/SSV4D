"""Generate candidate trie suffixes on a separate fast-model server."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable

import aiohttp

from ssv4d.trie_runtime import TrieBranchManager


@dataclass(frozen=True)
class BranchRollout:
    candidate_token: int
    suffix_tokens: tuple[int, ...]
    path_tokens: tuple[int, ...]


class GPU1BranchProducer:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:30011",
        *,
        timeout_s: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def produce(
        self,
        prompt_ids: Iterable[int],
        prefix_tokens: Iterable[int],
        candidate_tokens: Iterable[int],
        *,
        suffix_length: int,
    ) -> list[BranchRollout]:
        prompt_ids = [int(token) for token in prompt_ids]
        prefix_tokens = [int(token) for token in prefix_tokens]
        candidates = [int(token) for token in candidate_tokens]
        if not candidates or suffix_length <= 0:
            return []

        payload = {
            "input_ids": [
                prompt_ids + prefix_tokens + [candidate]
                for candidate in candidates
            ],
            "sampling_params": [
                {
                    "temperature": 0.0,
                    "max_new_tokens": suffix_length,
                    "ignore_eos": True,
                    "is_deterministic": False,
                    "sampling_seed": 42,
                }
                for _ in candidates
            ],
            "return_logprob": False,
            "stream": False,
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.base_url}/generate",
                json=payload,
            ) as response:
                text = await response.text()
                if response.status != 200:
                    raise RuntimeError(
                        f"GPU1 branch rollout failed: {response.status}: {text[:500]}"
                    )
                data = await response.json(content_type=None)

        rows = data if isinstance(data, list) else [data]
        if len(rows) != len(candidates):
            raise RuntimeError(
                f"Expected {len(candidates)} branch rows, got {len(rows)}"
            )
        rollouts = []
        for candidate, row in zip(candidates, rows):
            suffix = tuple(int(token) for token in row.get("output_ids", []))
            if len(suffix) != suffix_length:
                raise RuntimeError(
                    f"Expected suffix length {suffix_length}, got {len(suffix)}"
                )
            rollouts.append(
                BranchRollout(
                    candidate_token=candidate,
                    suffix_tokens=suffix,
                    path_tokens=tuple(prefix_tokens) + (candidate,) + suffix,
                )
            )
        return rollouts

    def produce_sync(
        self,
        prompt_ids: Iterable[int],
        prefix_tokens: Iterable[int],
        candidate_tokens: Iterable[int],
        *,
        suffix_length: int,
    ) -> list[BranchRollout]:
        """Synchronous worker-safe wrapper around the GPU1 rollout."""

        producers = self.produce(
            prompt_ids,
            prefix_tokens,
            candidate_tokens,
            suffix_length=suffix_length,
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(producers)
        raise RuntimeError(
            "produce_sync cannot run inside an active asyncio loop"
        )

    @staticmethod
    def write_to_manager(
        manager: TrieBranchManager,
        rollouts: Iterable[BranchRollout],
    ) -> None:
        for rollout in rollouts:
            manager.trie.insert(
                rollout.path_tokens,
                source="shadow",
            )


async def _smoke() -> None:
    producer = GPU1BranchProducer()
    rollouts = await producer.produce(
        [128000, 9906],
        [],
        [9906, 40],
        suffix_length=8,
    )
    manager = TrieBranchManager()
    producer.write_to_manager(manager, rollouts)
    print(
        {
            "rollouts": len(rollouts),
            "trie": manager.trie.stats(),
            "lengths": [len(row.suffix_tokens) for row in rollouts],
        }
    )


if __name__ == "__main__":
    asyncio.run(_smoke())
