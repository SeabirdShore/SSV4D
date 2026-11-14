"""Async client for collecting canonical and shadow generation traces."""

from __future__ import annotations

import asyncio
import math
from typing import Any, Iterable, Optional

import aiohttp

from ssv4d.models import GenerationTrace


class SGLangShadowClient:
    """Small client around SGLang's native ``/generate`` endpoint.

    A single batch request is used for each perturbation plan. This matters
    because batch shape, rather than independent HTTP concurrency, is the
    numerical perturbation we want to observe.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:30000",
        model: Optional[str] = None,
        timeout_s: float = 600.0,
        top_logprobs_num: int = 8,
        retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.top_logprobs_num = top_logprobs_num
        self.retries = retries

    async def health(self) -> bool:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{self.base_url}/v1/models") as response:
                return response.status == 200

    async def reset_llm42_stats(self) -> None:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.base_url}/reset_llm42_stats") as response:
                await response.read()

    async def generate_batch(
        self,
        prompts: Iterable[str],
        *,
        deterministic: bool,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        sampling_seed: int = 42,
        extra_sampling_params: Optional[dict[str, Any]] = None,
    ) -> list[GenerationTrace]:
        prompts = list(prompts)
        if not prompts:
            raise ValueError("prompts must not be empty")

        sampling_params: dict[str, Any] = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
            "is_deterministic": deterministic,
            "sampling_seed": sampling_seed,
        }
        if extra_sampling_params:
            sampling_params.update(extra_sampling_params)

        payload = {
            "text": prompts,
            "sampling_params": [dict(sampling_params) for _ in prompts],
            "return_logprob": True,
            "logprob_start_len": -1,
            "top_logprobs_num": self.top_logprobs_num,
            "stream": False,
        }

        last_error: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            try:
                raw = await self._post_generate(payload)
                items = raw if isinstance(raw, list) else [raw]
                if len(items) != len(prompts):
                    raise RuntimeError(
                        f"Expected {len(prompts)} results, received {len(items)}"
                    )
                return [
                    self._parse_trace(
                        item,
                        deterministic=deterministic,
                        batch_size=len(prompts),
                    )
                    for item in items
                ]
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as exc:
                last_error = exc
                if attempt == self.retries:
                    break
                await asyncio.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"/generate failed after retries: {last_error}")

    async def _post_generate(self, payload: dict[str, Any]) -> Any:
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.base_url}/generate", json=payload
            ) as response:
                text = await response.text()
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {text[:1000]}")
                return await response.json(content_type=None)

    def _parse_trace(
        self,
        item: dict[str, Any],
        *,
        deterministic: bool,
        batch_size: int,
    ) -> GenerationTrace:
        meta = item.get("meta_info") or {}
        output_ids = [int(x) for x in (item.get("output_ids") or [])]
        top_logprobs_raw = meta.get("output_top_logprobs") or []

        margins: list[float] = []
        top_ids: list[list[int]] = []
        top_logprobs: list[list[float]] = []
        for token_topk in top_logprobs_raw:
            if not token_topk:
                margins.append(float("nan"))
                top_ids.append([])
                top_logprobs.append([])
                continue

            ids_row: list[int] = []
            lps_row: list[float] = []
            for entry in token_topk:
                if entry is None or len(entry) < 2:
                    continue
                lps_row.append(float(entry[0]))
                ids_row.append(int(entry[1]))
            top_ids.append(ids_row)
            top_logprobs.append(lps_row)
            if len(lps_row) >= 2:
                margins.append(lps_row[0] - lps_row[1])
            else:
                margins.append(float("nan"))

        # Keep feature vectors aligned to output_ids. Missing top-k entries are
        # represented by NaN and filtered by downstream aggregations.
        n_tokens = len(output_ids)
        margins = margins[:n_tokens]
        top_ids = top_ids[:n_tokens]
        top_logprobs = top_logprobs[:n_tokens]
        while len(margins) < n_tokens:
            margins.append(float("nan"))
            top_ids.append([])
            top_logprobs.append([])

        latency = float(meta.get("e2e_latency") or 0.0)
        return GenerationTrace(
            prompt_id="",
            scenario="reference" if deterministic else "shadow",
            batch_size=batch_size,
            protected_index=0,
            output_ids=output_ids,
            text=item.get("text") or "",
            margins=margins,
            top_ids=top_ids,
            top_logprobs=top_logprobs,
            latency_s=latency if math.isfinite(latency) else 0.0,
            deterministic=deterministic,
            metadata={
                "finish_reason": meta.get("finish_reason"),
                "completion_tokens": meta.get("completion_tokens"),
                "llm42_num_rollbacks": meta.get("llm42_num_rollbacks", 0),
                "llm42_tokens_rolled_back": meta.get(
                    "llm42_tokens_rolled_back", 0
                ),
            },
        )
