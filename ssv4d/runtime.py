"""In-process instrumentation and gating for LLM-42 verification workers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import torch

from ssv4d.dynamic_gate import evaluate_dynamic_window
from ssv4d.features import (
    aggregate_feature_rows,
    kv_column_deviation,
    topk_margin,
    vocab_aligned_divergence,
)
from ssv4d.frontier import frontier_snapshot, validate_frontier
from ssv4d.models import GenerationTrace
from ssv4d.trie_frontier import plan_candidate_edges
from ssv4d.trie_runtime import TrieBranchManager


_INSTALLED = False
_LOG_LOCK = threading.Lock()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int_list(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    values = []
    for part in raw.split(","):
        try:
            value = int(part.strip())
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return tuple(sorted(set(values))) or default


def _runtime_config() -> dict[str, Any]:
    return {
        "gate_enabled": _env_bool("SSV4D_GATE_ENABLE", False),
        "dynamic_gate_enabled": _env_bool(
            "SSV4D_DYNAMIC_GATE_ENABLE", False
        ),
        "dynamic_audit": _env_bool("SSV4D_DYNAMIC_AUDIT", False),
        "frontier_strict": _env_bool(
            "SSV4D_FRONTIER_STRICT", False
        ),
        "adaptive_window_enabled": _env_bool(
            "SSV4D_ADAPTIVE_WINDOW_ENABLE", False
        ),
        "adaptive_checkpoints": _env_int_list(
            "SSV4D_ADAPTIVE_CHECKPOINTS", (16, 32, 48, 64)
        ),
        "safe_window_enabled": _env_bool(
            "SSV4D_SAFE_WINDOW_ENABLE", False
        ),
        "safe_window_model": os.environ.get(
            "SSV4D_SAFE_WINDOW_MODEL",
            str(
                Path(__file__).resolve().parent
                / "results"
                / "safe_window_model.json"
            ),
        ),
        "safe_window_threshold": _env_float(
            "SSV4D_SAFE_WINDOW_THRESHOLD", 0.1
        ),
        "margin_threshold": _env_float("SSV4D_MARGIN_THRESHOLD", 0.1875),
        "verify_top_k": _env_int("SSV4D_VERIFY_TOP_K", 8),
        "sync_timing": _env_bool("SSV4D_SYNC_TIMING", False),
        "prefill_shadow_enabled": _env_bool(
            "SSV4D_PREFILL_SHADOW_ENABLE", False
        ),
        "prefill_jsd_threshold": _env_float(
            "SSV4D_PREFILL_JSD_THRESHOLD", 1e-3
        ),
        "prefill_kv_threshold": _env_float(
            "SSV4D_PREFILL_KV_THRESHOLD", 0.01
        ),
        "replay_shadow_enabled": _env_bool(
            "SSV4D_REPLAY_SHADOW_ENABLE", False
        ),
        "trie_manager_enabled": _env_bool(
            "SSV4D_TRIE_MANAGER_ENABLE", False
        ),
        "trie_top_k": _env_int("SSV4D_TRIE_TOP_K", 3),
        "gpu1_branch_url": os.environ.get("SSV4D_GPU1_BRANCH_URL"),
        "replay_shadow_batch_size": _env_int(
            "SSV4D_REPLAY_SHADOW_BATCH_SIZE", 8
        ),
        "replay_shadow_position": _env_int(
            "SSV4D_REPLAY_SHADOW_POSITION", 7
        ),
        "log_path": os.environ.get(
            "SSV4D_RUNTIME_LOG",
            str(Path(__file__).resolve().parent / "logs" / "runtime_metrics.jsonl"),
        ),
    }


def _append_event(event: dict[str, Any]) -> None:
    path = Path(event.pop("_log_path"))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(event, sort_keys=True) + "\n"
    with _LOG_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(payload)
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _as_rows(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    if isinstance(values, (list, tuple)):
        return list(values)
    return []


def _slice_rows(values: Any, start: int, count: int) -> list[Any]:
    if values is None or count <= 0:
        return []
    if isinstance(values, torch.Tensor):
        return values[start : start + count].detach().cpu().tolist()
    if isinstance(values, (list, tuple)):
        return list(values[start : start + count])
    return []


def _tail_rows(values: Any, count: int) -> list[Any]:
    if values is None or count <= 0:
        return []
    if isinstance(values, torch.Tensor):
        return values[-count:].detach().cpu().tolist()
    if isinstance(values, (list, tuple)):
        return list(values[-count:])
    return []


def _draft_topk(req: Any, start: int, count: int) -> list[tuple[Any, Any]]:
    values = getattr(req, "output_top_logprobs_val", None)
    indices = getattr(req, "output_top_logprobs_idx", None)
    return list(
        zip(
            _slice_rows(values, start, count),
            _slice_rows(indices, start, count),
        )
    )


def _window_margins(req: Any, window_size: int) -> list[float]:
    values = _tail_rows(
        getattr(req, "output_top_logprobs_val", None), window_size
    )
    indices = _tail_rows(
        getattr(req, "output_top_logprobs_idx", None), window_size
    )
    margins = []
    for logprobs, token_ids in zip(values, indices):
        if logprobs is not None and token_ids is not None:
            margins.append(topk_margin(logprobs, token_ids))
    return margins


def _actual_window_len(req: Any, worker: Any) -> int:
    unverified = max(0, len(req.output_ids) - req.llm42_verified_tokens)
    requested = getattr(req, "ssv4d_verify_tokens", worker._window_size)
    return min(unverified, worker._window_size, max(1, int(requested)))


def _replay_window_key(req: Any, limit: int) -> str:
    return f"{req.rid}:{req.llm42_verified_tokens}:{limit}"


def _selected_layers(model_runner: Any) -> list[int]:
    raw = os.environ.get("SSV4D_KV_LAYERS", "-1")
    num_layers = int(
        getattr(model_runner.model_config, "num_hidden_layers", 0)
    )
    layers = []
    for part in raw.split(","):
        try:
            layer = int(part.strip())
        except ValueError:
            continue
        if layer < 0 and num_layers:
            layer += num_layers
        if 0 <= layer < num_layers:
            layers.append(layer)
    return sorted(set(layers))


def _snapshot_current_kv(worker: Any, reqs: list[Any]) -> dict:
    model_runner = worker.target_worker.model_runner
    kv_pool = getattr(model_runner, "token_to_kv_pool", None)
    req_pool = getattr(model_runner, "req_to_token_pool", None)
    if kv_pool is None or req_pool is None:
        return {}
    layers = _selected_layers(model_runner)
    if not layers:
        return {}

    snapshots = {}
    for req in reqs:
        if getattr(req, "req_pool_idx", None) is None:
            continue
        position = len(req.origin_input_ids) + len(req.output_ids) - 2
        if position < 0:
            continue
        slot = req_pool.req_to_token[req.req_pool_idx, position]
        if slot.item() == 0:
            continue
        for layer in layers:
            k_buffer, v_buffer = kv_pool.get_kv_buffer(layer)
            snapshots[(req.rid, layer)] = (
                k_buffer[slot].detach().clone(),
                v_buffer[slot].detach().clone(),
            )
    return snapshots


def _after_kv_features(worker: Any, snapshots: dict) -> dict[str, float]:
    if not snapshots:
        return {}
    model_runner = worker.target_worker.model_runner
    kv_pool = model_runner.token_to_kv_pool
    req_pool = model_runner.req_to_token_pool
    values = []
    for (rid, layer), (before_k, before_v) in snapshots.items():
        req = worker._ssv4d_req_by_rid.get(rid)
        if req is None or getattr(req, "req_pool_idx", None) is None:
            continue
        position = len(req.origin_input_ids) + len(req.output_ids) - 2
        if position < 0:
            continue
        slot = req_pool.req_to_token[req.req_pool_idx, position]
        if slot.item() == 0:
            continue
        k_buffer, v_buffer = kv_pool.get_kv_buffer(layer)
        row = kv_column_deviation(
            before_k, before_v, k_buffer[slot], v_buffer[slot]
        )
        row["layer"] = layer
        values.append(row)
    return aggregate_feature_rows(values)


def _verify_output_features(
    reqs: list[Any],
    worker: Any,
    verify_output: Any,
    start_indices: dict[str, int],
    window_lengths: dict[str, int],
) -> dict[str, float]:
    if verify_output is None or verify_output.logits_output is None:
        return {}
    logits = verify_output.logits_output
    verify_values = _as_rows(getattr(logits, "next_token_top_logprobs_val", None))
    verify_indices = _as_rows(getattr(logits, "next_token_top_logprobs_idx", None))
    rows = []
    offset = 0
    for req in reqs:
        actual_len = window_lengths.get(req.rid, 0)
        start = start_indices.get(req.rid, req.llm42_verified_tokens)
        draft_rows = _draft_topk(req, start, actual_len)
        for index in range(actual_len):
            position = offset + index
            if position >= len(verify_values) or index >= len(draft_rows):
                break
            draft_lps, draft_ids = draft_rows[index]
            verify_lps = verify_values[position]
            verify_ids = verify_indices[position]
            if (
                draft_lps is None
                or draft_ids is None
                or verify_lps is None
                or verify_ids is None
            ):
                continue
            row = vocab_aligned_divergence(
                draft_lps, draft_ids, verify_lps, verify_ids
            )
            row["draft_margin"] = topk_margin(draft_lps, draft_ids)
            row["verify_margin"] = topk_margin(verify_lps, verify_ids)
            rows.append(row)
        offset += actual_len
    return aggregate_feature_rows(rows)


class _ProbeTreeCache:
    def __init__(self, page_size, device, allocator):
        self.page_size = page_size
        self.device = device
        self.token_to_kv_pool_allocator = allocator

    def supports_swa(self):
        return False

    def supports_mamba(self):
        return False

    def is_chunk_cache(self):
        return False

    def is_tree_cache(self):
        return True


def _prefill_once(
    model_runner: Any,
    input_ids: list[int],
    batch_size: int,
    protected_index: int,
    layer: int,
) -> dict[str, Any]:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    sampling = SamplingParams(temperature=0, max_new_tokens=1)
    reqs = []
    for index in range(batch_size):
        req = Req(
            rid=f"ssv4d-probe-{id(model_runner)}-{id(input_ids)}-{index}",
            origin_input_text="",
            origin_input_ids=list(input_ids),
            sampling_params=sampling,
        )
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(
            len(req.fill_ids) - len(req.prefix_indices)
        )
        reqs.append(req)

    tree_cache = _ProbeTreeCache(
        page_size=model_runner.server_args.page_size,
        device=model_runner.device,
        allocator=model_runner.token_to_kv_pool_allocator,
    )
    batch = None
    try:
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(worker_batch, model_runner)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            output = model_runner.forward(forward_batch)
        end.record()
        end.synchronize()
        gpu_ms = start.elapsed_time(end)

        logits = output.logits_output.next_token_logits[
            protected_index
        ].detach().float()
        logprobs = torch.log_softmax(logits, dim=-1)
        top_values, top_ids = torch.topk(logprobs, 8)
        target = reqs[protected_index]
        slot = model_runner.req_to_token_pool.req_to_token[
            target.req_pool_idx, len(target.origin_input_ids) - 1
        ]
        k_buffer, v_buffer = model_runner.token_to_kv_pool.get_kv_buffer(layer)
        return {
            "top_logprobs": top_values.cpu().tolist(),
            "top_ids": top_ids.cpu().tolist(),
            "k": k_buffer[slot].detach().clone(),
            "v": v_buffer[slot].detach().clone(),
            "gpu_ms": gpu_ms,
        }
    finally:
        if batch is not None and batch.out_cache_loc is not None:
            model_runner.token_to_kv_pool_allocator.free(batch.out_cache_loc)
        for req in reqs:
            if req.req_pool_idx is not None:
                model_runner.req_to_token_pool.free(req)


def _run_replay_shadow(
    worker: Any,
    req: Any,
    *,
    window_size: int,
    batch_size: int,
    protected_index: int,
) -> dict[str, Any]:
    """Replay the first decode window in an isolated perturbed prefill batch.

    The shadow batch uses a fresh request slot and fresh KV cache slots. It never
    overwrites the canonical request's KV state. Target logprobs are computed for
    all tokens in the replayed window in one extend forward.
    """

    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    model_runner = worker.target_worker.model_runner
    unverified = req.output_ids[req.llm42_verified_tokens :]
    actual_len = min(window_size, len(unverified))
    if actual_len <= 1:
        raise ValueError("Replay shadow needs at least two unverified tokens")

    # Input tokens end one position before the last target token. The extend
    # logits then predict every token in the requested verification window.
    replayed_ids = list(req.origin_input_ids) + list(
        req.output_ids[: req.llm42_verified_tokens + actual_len - 1]
    )
    sampling = SamplingParams(temperature=0, max_new_tokens=1)
    batch_size = max(1, batch_size)
    protected_index = min(max(protected_index, 0), batch_size - 1)
    reqs = []
    for index in range(batch_size):
        shadow_req = Req(
            rid=f"ssv4d-replay-{req.rid}-{index}",
            origin_input_text="",
            origin_input_ids=list(replayed_ids),
            sampling_params=sampling,
            return_logprob=index == protected_index,
            top_logprobs_num=8 if index == protected_index else 0,
        )
        shadow_req.fill_ids = shadow_req.origin_input_ids
        shadow_req.logprob_start_len = (
            0 if index == protected_index else -1
        )
        shadow_req.set_extend_input_len(len(shadow_req.fill_ids))
        reqs.append(shadow_req)

    tree_cache = _ProbeTreeCache(
        page_size=model_runner.server_args.page_size,
        device=model_runner.device,
        allocator=model_runner.token_to_kv_pool_allocator,
    )
    batch = None
    started = time.perf_counter()
    try:
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.return_logprob = True
        batch.prepare_for_extend()
        model_worker_batch = batch.get_model_worker_batch()
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_start.record()
        output = worker.target_worker.forward_batch_generation(
            model_worker_batch,
            is_verify=True,
        )
        cuda_end.record()
        cuda_end.synchronize()
        gpu_ms = cuda_start.elapsed_time(cuda_end)
        logits = output.logits_output
        all_top_values = _as_rows(logits.input_top_logprobs_val)
        all_top_ids = _as_rows(logits.input_top_logprobs_idx)
        top_values = (
            all_top_values[protected_index]
            if protected_index < len(all_top_values)
            else []
        )
        top_ids = (
            all_top_ids[protected_index]
            if protected_index < len(all_top_ids)
            else []
        )
        verified_tokens = req.llm42_verified_tokens
        origin_len = len(req.origin_input_ids)
        shadow_start = origin_len + verified_tokens - 1
        shadow_values = top_values[shadow_start : shadow_start + actual_len]
        shadow_ids = top_ids[shadow_start : shadow_start + actual_len]
        canonical_values = _as_rows(req.output_top_logprobs_val)[
            verified_tokens : verified_tokens + len(shadow_values)
        ]
        canonical_ids = _as_rows(req.output_top_logprobs_idx)[
            verified_tokens : verified_tokens + len(shadow_ids)
        ]
        aligned_len = min(
            len(canonical_values),
            len(canonical_ids),
            len(shadow_values),
            len(shadow_ids),
        )
        canonical_values = canonical_values[:aligned_len]
        canonical_ids = canonical_ids[:aligned_len]
        shadow_values = shadow_values[:aligned_len]
        shadow_ids = shadow_ids[:aligned_len]
        output_slice = list(
            req.output_ids[verified_tokens : verified_tokens + aligned_len]
        )
        canonical_trace = GenerationTrace(
            prompt_id=req.rid,
            scenario="shadow",
            batch_size=1,
            protected_index=0,
            output_ids=output_slice,
            text="",
            margins=[
                topk_margin(values, ids)
                for values, ids in zip(canonical_values, canonical_ids)
            ],
            top_ids=canonical_ids,
            top_logprobs=canonical_values,
            latency_s=0.0,
            deterministic=False,
            plan_name="canonical",
        )
        shadow_trace = GenerationTrace(
            prompt_id=req.rid,
            scenario="shadow",
            batch_size=batch_size,
            protected_index=protected_index,
            output_ids=output_slice,
            text="",
            margins=[
                topk_margin(values, ids)
                for values, ids in zip(shadow_values, shadow_ids)
            ],
            top_ids=shadow_ids,
            top_logprobs=shadow_values,
            latency_s=0.0,
            deterministic=False,
            plan_name="replay_shadow",
        )
        decision = evaluate_dynamic_window(
            canonical_trace,
            [shadow_trace],
            window=aligned_len,
        )
        top1_agreement = [
            bool(cids and sids and cids[0] == sids[0])
            for cids, sids in zip(canonical_ids, shadow_ids)
        ]
        return {
            "actual_len": actual_len,
            "replayed_tokens": len(replayed_ids),
            "origin_len": origin_len,
            "verified_tokens": verified_tokens,
            "shadow_start": shadow_start,
            "aligned_len": aligned_len,
            "batch_size": batch_size,
            "protected_index": protected_index,
            "top_values": top_values,
            "top_ids": top_ids,
            "canonical_values": canonical_values,
            "canonical_ids": canonical_ids,
            "shadow_values": shadow_values,
            "shadow_ids": shadow_ids,
            "top1_agreement_mean": (
                sum(top1_agreement) / len(top1_agreement)
                if top1_agreement
                else 0.0
            ),
            "decision": decision.to_dict(),
            "gpu_ms": gpu_ms,
            "wall_ms": (time.perf_counter() - started) * 1000.0,
        }
    finally:
        if batch is not None and batch.out_cache_loc is not None:
            model_runner.token_to_kv_pool_allocator.free(batch.out_cache_loc)
        for shadow_req in reqs:
            if shadow_req.req_pool_idx is not None:
                model_runner.req_to_token_pool.free(shadow_req)


def _run_replay_shadows(
    worker: Any,
    source_reqs: list[Any],
    *,
    window_size: int,
    min_batch_size: int,
    window_limits: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run all first-window replay shadows in one isolated extend batch."""

    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    model_runner = worker.target_worker.model_runner
    replay_specs = []
    for req in source_reqs:
        unverified = req.output_ids[req.llm42_verified_tokens :]
        requested_limit = (
            window_limits.get(req.rid, window_size)
            if window_limits is not None
            else window_size
        )
        actual_len = min(requested_limit, window_size, len(unverified))
        if actual_len <= 1:
            continue
        replayed_ids = list(req.origin_input_ids) + list(
            req.output_ids[: req.llm42_verified_tokens + actual_len - 1]
        )
        replay_specs.append((req, replayed_ids, actual_len))
    if not replay_specs:
        return {}

    target_count = len(replay_specs)
    batch_size = max(min_batch_size, target_count)
    sampling = SamplingParams(temperature=0, max_new_tokens=1)
    shadow_reqs = []
    for index in range(batch_size):
        source_index = index if index < target_count else 0
        _, replayed_ids, _ = replay_specs[source_index]
        shadow_req = Req(
            rid=f"ssv4d-batch-replay-{id(source_reqs)}-{index}",
            origin_input_text="",
            origin_input_ids=list(replayed_ids),
            sampling_params=sampling,
            return_logprob=index < target_count,
            top_logprobs_num=8 if index < target_count else 0,
        )
        shadow_req.fill_ids = shadow_req.origin_input_ids
        shadow_req.logprob_start_len = 0 if index < target_count else -1
        shadow_req.set_extend_input_len(len(shadow_req.fill_ids))
        shadow_reqs.append(shadow_req)

    tree_cache = _ProbeTreeCache(
        page_size=model_runner.server_args.page_size,
        device=model_runner.device,
        allocator=model_runner.token_to_kv_pool_allocator,
    )
    batch = None
    started = time.perf_counter()
    try:
        batch = ScheduleBatch.init_new(
            reqs=shadow_reqs,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.return_logprob = True
        batch.prepare_for_extend()
        model_worker_batch = batch.get_model_worker_batch()
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_start.record()
        output = worker.target_worker.forward_batch_generation(
            model_worker_batch,
            is_verify=True,
        )
        cuda_end.record()
        cuda_end.synchronize()
        gpu_ms = cuda_start.elapsed_time(cuda_end)
        all_top_values = _as_rows(output.logits_output.input_top_logprobs_val)
        all_top_ids = _as_rows(output.logits_output.input_top_logprobs_idx)
        wall_ms = (time.perf_counter() - started) * 1000.0

        probes = {}
        for index, (req, replayed_ids, actual_len) in enumerate(replay_specs):
            top_values = (
                all_top_values[index] if index < len(all_top_values) else []
            )
            top_ids = all_top_ids[index] if index < len(all_top_ids) else []
            verified_tokens = req.llm42_verified_tokens
            origin_len = len(req.origin_input_ids)
            shadow_start = origin_len + verified_tokens - 1
            shadow_values = top_values[shadow_start : shadow_start + actual_len]
            shadow_ids = top_ids[shadow_start : shadow_start + actual_len]
            canonical_values = _as_rows(req.output_top_logprobs_val)[
                verified_tokens : verified_tokens + len(shadow_values)
            ]
            canonical_ids = _as_rows(req.output_top_logprobs_idx)[
                verified_tokens : verified_tokens + len(shadow_ids)
            ]
            aligned_len = min(
                len(canonical_values),
                len(canonical_ids),
                len(shadow_values),
                len(shadow_ids),
            )
            canonical_values = canonical_values[:aligned_len]
            canonical_ids = canonical_ids[:aligned_len]
            shadow_values = shadow_values[:aligned_len]
            shadow_ids = shadow_ids[:aligned_len]
            output_slice = list(
                req.output_ids[
                    verified_tokens : verified_tokens + aligned_len
                ]
            )
            canonical_trace = GenerationTrace(
                prompt_id=req.rid,
                scenario="shadow",
                batch_size=1,
                protected_index=0,
                output_ids=output_slice,
                text="",
                margins=[
                    topk_margin(values, ids)
                    for values, ids in zip(canonical_values, canonical_ids)
                ],
                top_ids=canonical_ids,
                top_logprobs=canonical_values,
                latency_s=0.0,
                deterministic=False,
                plan_name="canonical",
            )
            shadow_trace = GenerationTrace(
                prompt_id=req.rid,
                scenario="shadow",
                batch_size=batch_size,
                protected_index=index,
                output_ids=output_slice,
                text="",
                margins=[
                    topk_margin(values, ids)
                    for values, ids in zip(shadow_values, shadow_ids)
                ],
                top_ids=shadow_ids,
                top_logprobs=shadow_values,
                latency_s=0.0,
                deterministic=False,
                plan_name="batch_replay_shadow",
            )
            decision = evaluate_dynamic_window(
                canonical_trace,
                [shadow_trace],
                window=aligned_len,
            )
            top1_agreement = [
                bool(cids and sids and cids[0] == sids[0])
                for cids, sids in zip(canonical_ids, shadow_ids)
            ]
            window_key = _replay_window_key(req, requested_limit)
            probes[window_key] = {
                "rid": req.rid,
                "window_key": window_key,
                "window_start": verified_tokens,
                "window_limit": requested_limit,
                "actual_len": actual_len,
                "replayed_tokens": len(replayed_ids),
                "origin_len": origin_len,
                "verified_tokens": verified_tokens,
                "shadow_start": shadow_start,
                "aligned_len": aligned_len,
                "batch_size": batch_size,
                "protected_index": index,
                "top_values": top_values,
                "top_ids": top_ids,
                "canonical_values": canonical_values,
                "canonical_ids": canonical_ids,
                "shadow_values": shadow_values,
                "shadow_ids": shadow_ids,
                "top1_agreement_mean": (
                    sum(top1_agreement) / len(top1_agreement)
                    if top1_agreement
                    else 0.0
                ),
                "decision": decision.to_dict(),
                "gpu_ms": gpu_ms,
                "wall_ms": wall_ms,
            }
        return probes
    finally:
        if batch is not None and batch.out_cache_loc is not None:
            model_runner.token_to_kv_pool_allocator.free(batch.out_cache_loc)
        for shadow_req in shadow_reqs:
            if shadow_req.req_pool_idx is not None:
                model_runner.req_to_token_pool.free(shadow_req)


def _run_prefill_shadow(worker: Any, req: Any) -> dict[str, float]:
    model_runner = worker.target_worker.model_runner
    num_layers = int(model_runner.model_config.num_hidden_layers)
    layer = num_layers - 1
    started = time.perf_counter()
    canonical = _prefill_once(
        model_runner, req.origin_input_ids, 1, 0, layer
    )
    shadow = _prefill_once(
        model_runner, req.origin_input_ids, 8, 7, layer
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    divergence = vocab_aligned_divergence(
        canonical["top_logprobs"],
        canonical["top_ids"],
        shadow["top_logprobs"],
        shadow["top_ids"],
    )
    kv = kv_column_deviation(
        canonical["k"], canonical["v"], shadow["k"], shadow["v"]
    )
    return {
        **divergence,
        **kv,
        "probe_wall_ms": wall_ms,
        "canonical_prefill_gpu_ms": canonical["gpu_ms"],
        "shadow_prefill_gpu_ms": shadow["gpu_ms"],
    }


def install_runtime() -> bool:
    """Install SSV4D instrumentation on the current LLM42Worker class."""

    global _INSTALLED
    if _INSTALLED:
        return True

    from sglang.srt.llm42.llm42_info import LLM42Info
    from sglang.srt.llm42.llm42_worker import LLM42Worker

    if getattr(LLM42Worker, "_ssv4d_runtime_installed", False):
        _INSTALLED = True
        return True

    config = _runtime_config()
    safe_window_predictor = None
    if config["safe_window_enabled"]:
        from ssv4d.safe_window import SafeWindowPredictor

        safe_window_predictor = SafeWindowPredictor.from_file(
            Path(config["safe_window_model"]),
            threshold=config["safe_window_threshold"],
        )
    original_prepare = LLM42Info.prepare_verify_batch
    original_check = LLM42Worker.check_and_verify_deterministic_requests
    original_draft = LLM42Worker.forward_batch_generation
    original_verify = LLM42Worker._run_verification

    def prepare_verify_batch(self, *args, **kwargs):
        verify_batch = original_prepare(self, *args, **kwargs)
        top_k = config["verify_top_k"]
        if top_k > 0 and getattr(verify_batch, "top_logprobs_nums", None) is not None:
            verify_batch.top_logprobs_nums = [
                top_k for _ in verify_batch.top_logprobs_nums
            ]
        return verify_batch

    def forward_batch_generation(self, model_worker_batch=None, **kwargs):
        started = time.perf_counter()
        result = original_draft(self, model_worker_batch, **kwargs)
        self._ssv4d_last_draft_ms = (time.perf_counter() - started) * 1000.0
        self.stats["ssv4d_draft_calls"] = (
            self.stats.get("ssv4d_draft_calls", 0) + 1
        )
        return result

    def check_and_verify_deterministic_requests(self, batch):
        decisions = {}
        denied = []
        self._ssv4d_risky_rids = getattr(self, "_ssv4d_risky_rids", set())
        self._ssv4d_trie_managers = getattr(
            self, "_ssv4d_trie_managers", {}
        )
        self._ssv4d_gpu1_producer = getattr(
            self, "_ssv4d_gpu1_producer", None
        )
        self._ssv4d_prefill_metrics = getattr(
            self, "_ssv4d_prefill_metrics", {}
        )
        self._ssv4d_prefill_risky = getattr(
            self, "_ssv4d_prefill_risky", set()
        )
        self._ssv4d_producer_ms = None
        self._ssv4d_replay_shadow = getattr(
            self, "_ssv4d_replay_shadow", {}
        )
        self._ssv4d_adaptive_processed = getattr(
            self, "_ssv4d_adaptive_processed", set()
        )
        self._ssv4d_finished_logged = getattr(
            self, "_ssv4d_finished_logged", set()
        )
        self.stats.setdefault("ssv4d_windows_seen", 0)
        self.stats.setdefault("ssv4d_windows_skipped", 0)
        self.stats.setdefault("ssv4d_windows_verified", 0)
        self.stats.setdefault("ssv4d_tokens_skipped", 0)

        adaptive_limits = {}
        replay_reqs = []
        if config["gate_enabled"] and batch.reqs:
            for req in batch.reqs:
                if (
                    not getattr(req, "is_deterministic", False)
                    or req.req_pool_idx is None
                    or getattr(req, "is_retracted", False)
                ):
                    continue
                unverified = (
                    len(req.output_ids) - req.llm42_verified_tokens
                )
                if config["adaptive_window_enabled"]:
                    for checkpoint in config["adaptive_checkpoints"]:
                        checkpoint = min(checkpoint, self._window_size)
                        if unverified < checkpoint:
                            continue
                        key = _replay_window_key(req, checkpoint)
                        if key in self._ssv4d_adaptive_processed:
                            continue
                        adaptive_limits[req.rid] = checkpoint
                        if config["replay_shadow_enabled"]:
                            replay_reqs.append(req)
                        break
                elif (
                    config["replay_shadow_enabled"]
                    and unverified >= self._window_size
                ):
                    key = _replay_window_key(req, self._window_size)
                    if key not in self._ssv4d_replay_shadow:
                        adaptive_limits[req.rid] = self._window_size
                        replay_reqs.append(req)
        if config["replay_shadow_enabled"] and batch.reqs:
            if replay_reqs:
                try:
                    probes = _run_replay_shadows(
                        self,
                        replay_reqs,
                        window_size=self._window_size,
                        min_batch_size=config["replay_shadow_batch_size"],
                        window_limits=adaptive_limits,
                    )
                    self._ssv4d_replay_shadow.update(probes)
                    for window_key, probe in probes.items():
                        decision = probe["decision"]
                        _append_event(
                            {
                                "_log_path": config["log_path"],
                                "event": "replay_shadow",
                                "timestamp": time.time(),
                                "rid": probe["rid"],
                                "window_key": window_key,
                                "batch_replay": True,
                                **{
                                    key: value
                                    for key, value in probe.items()
                                    if key
                                    not in {
                                        "top_values",
                                        "top_ids",
                                        "canonical_values",
                                        "canonical_ids",
                                        "shadow_values",
                                        "shadow_ids",
                                        "decision",
                                    }
                                },
                                "top_values_len": len(probe["top_values"]),
                                "top_ids_len": len(probe["top_ids"]),
                                "shadow_window_len": len(
                                    probe["shadow_values"]
                                ),
                                "canonical_window_len": len(
                                    probe["canonical_values"]
                                ),
                                "dynamic_should_verify": decision[
                                    "should_verify"
                                ],
                                "dynamic_trigger_position": decision[
                                    "trigger_position"
                                ],
                                "dynamic_threshold": decision[
                                    "trigger_threshold"
                                ],
                                "dynamic_reason": decision[
                                    "trigger_reason"
                                ],
                            }
                        )
                except Exception as exc:
                    _append_event(
                        {
                            "_log_path": config["log_path"],
                            "event": "replay_shadow_error",
                            "timestamp": time.time(),
                            "num_reqs": len(replay_reqs),
                            "error": repr(exc),
                        }
                    )

        deferred = []
        if config["gate_enabled"] and getattr(batch, "reqs", None):
            for req in batch.reqs:
                if not getattr(req, "is_deterministic", False):
                    continue
                if req.req_pool_idx is None or getattr(req, "is_retracted", False):
                    continue
                unverified = len(req.output_ids) - req.llm42_verified_tokens
                if unverified <= 0:
                    continue
                if (
                    config["prefill_shadow_enabled"]
                    and req.rid not in self._ssv4d_prefill_metrics
                ):
                    try:
                        probe = _run_prefill_shadow(self, req)
                        self._ssv4d_prefill_metrics[req.rid] = probe
                        is_prefill_risky = (
                            probe.get("jsd", 0.0)
                            >= config["prefill_jsd_threshold"]
                            or probe.get("k_relative_l2", 0.0)
                            >= config["prefill_kv_threshold"]
                            or probe.get("v_relative_l2", 0.0)
                            >= config["prefill_kv_threshold"]
                        )
                        if is_prefill_risky:
                            self._ssv4d_prefill_risky.add(req.rid)
                        _append_event(
                            {
                                "_log_path": config["log_path"],
                                "event": "prefill_shadow",
                                "timestamp": time.time(),
                                "rid": req.rid,
                                "risky": is_prefill_risky,
                                **probe,
                            }
                        )
                    except Exception as exc:
                        _append_event(
                            {
                                "_log_path": config["log_path"],
                                "event": "prefill_shadow_error",
                                "timestamp": time.time(),
                                "rid": req.rid,
                                "error": repr(exc),
                            }
                        )
                margins = _window_margins(req, self._window_size)
                min_margin = min(margins) if margins else float("inf")
                if config["trie_manager_enabled"]:
                    manager = self._ssv4d_trie_managers.setdefault(
                        req.rid, TrieBranchManager()
                    )
                    manager.trie.insert(req.output_ids, source="canonical")
                    if margins:
                        start = max(
                            0, len(req.output_ids) - len(margins)
                        )
                        risk_index = min(
                            range(len(margins)),
                            key=lambda index: margins[index],
                        )
                        topk = _draft_topk(
                            req, start + risk_index, 1
                        )
                        if topk:
                            logprobs, token_ids = topk[0]
                            edges = plan_candidate_edges(
                                logprobs,
                                token_ids,
                                max_candidates=config["trie_top_k"],
                            )
                            if edges.expanded:
                                manager.record_candidate_labels(
                                    req.output_ids[: start + risk_index],
                                    edges.candidate_tokens,
                                )
                                if (
                                    config["gpu1_branch_url"]
                                    and self._ssv4d_gpu1_producer is None
                                ):
                                    from ssv4d.gpu1_branch_producer import (
                                        GPU1BranchProducer,
                                    )

                                    self._ssv4d_gpu1_producer = (
                                        GPU1BranchProducer(
                                            config["gpu1_branch_url"]
                                        )
                                    )
                                if self._ssv4d_gpu1_producer is not None:
                                    suffix_length = max(
                                        1,
                                        min(
                                            self._window_size - risk_index,
                                            len(req.output_ids)
                                            - start
                                            - risk_index
                                            - 1,
                                        ),
                                    )
                                    if suffix_length > 0:
                                        started = time.perf_counter()
                                        rollouts = (
                                            self._ssv4d_gpu1_producer.produce_sync(
                                                req.origin_input_ids,
                                                req.output_ids[
                                                    : start + risk_index
                                                ],
                                                edges.candidate_tokens[
                                                    : config["trie_top_k"]
                                                ],
                                                suffix_length=suffix_length,
                                            )
                                        )
                                        self._ssv4d_gpu1_producer.write_to_manager(
                                            manager, rollouts
                                        )
                                        self._ssv4d_producer_ms = (
                                            time.perf_counter() - started
                                        ) * 1000.0
                bootstrap = req.llm42_num_verification_windows == 0
                marked_risky = req.rid in self._ssv4d_risky_rids
                prefill_risky = req.rid in self._ssv4d_prefill_risky
                checkpoint = adaptive_limits.get(req.rid)
                replay_limit = (
                    checkpoint
                    if checkpoint is not None
                    else self._window_size
                )
                window_key = _replay_window_key(req, replay_limit)
                replay = self._ssv4d_replay_shadow.get(window_key)
                replay_decision = (
                    replay.get("decision") if replay is not None else None
                )
                dynamic_reason = None
                dynamic_threshold = None
                predicted_length = None
                is_finished = req.finished_reason is not None
                if config["dynamic_gate_enabled"]:
                    if config["adaptive_window_enabled"]:
                        if is_finished:
                            should_verify = True
                            predicted_length = max(
                                1,
                                min(unverified, self._window_size),
                            )
                            req.ssv4d_verify_tokens = predicted_length
                            dynamic_reason = "finished_partial_window"
                        elif config["safe_window_enabled"]:
                            if checkpoint is None:
                                deferred.append(req)
                                should_verify = False
                                dynamic_reason = "before_next_checkpoint"
                            else:
                                checkpoint_margins = _window_margins(
                                    req, checkpoint
                                )
                                (
                                    model_trigger,
                                    probability,
                                ) = safe_window_predictor.should_verify(
                                    checkpoint,
                                    checkpoint_margins,
                                    window_index=req.llm42_num_verification_windows,
                                )
                                dynamic_threshold = probability
                                if (
                                    checkpoint == self._window_size
                                    or model_trigger
                                    or marked_risky
                                    or prefill_risky
                                ):
                                    should_verify = True
                                    predicted_length = checkpoint
                                    req.ssv4d_verify_tokens = predicted_length
                                    dynamic_reason = (
                                        "safe_model_max_checkpoint"
                                        if checkpoint == self._window_size
                                        else "safe_model_early_verify"
                                    )
                                else:
                                    deferred.append(req)
                                    should_verify = False
                                    dynamic_reason = "safe_model_defer"
                                self._ssv4d_adaptive_processed.add(
                                    window_key
                                )
                        elif checkpoint is None:
                            deferred.append(req)
                            should_verify = False
                            dynamic_reason = "before_next_checkpoint"
                        elif replay_decision is None:
                            should_verify = True
                            req.ssv4d_verify_tokens = checkpoint
                            predicted_length = checkpoint
                            dynamic_reason = "replay_shadow_missing"
                        else:
                            trigger_position = replay_decision[
                                "trigger_position"
                            ]
                            risk = bool(
                                replay_decision["should_verify"]
                                or marked_risky
                                or prefill_risky
                                or not margins
                            )
                            if risk:
                                should_verify = True
                                # Verify at the first risky checkpoint rather
                                # than the raw trigger position. This avoids
                                # one-token verification thrash when position 0
                                # remains numerically unstable.
                                predicted_length = checkpoint
                                req.ssv4d_verify_tokens = predicted_length
                                dynamic_reason = replay_decision[
                                    "trigger_reason"
                                ]
                                dynamic_threshold = replay_decision[
                                    "trigger_threshold"
                                ]
                            elif checkpoint < self._window_size:
                                deferred.append(req)
                                should_verify = False
                                dynamic_reason = "no_risk_before_checkpoint"
                            else:
                                req.ssv4d_verify_tokens = self._window_size
                                predicted_length = self._window_size
                                should_verify = bool(config["dynamic_audit"])
                                dynamic_reason = "no_risk_at_max_checkpoint"
                        if checkpoint is not None:
                            self._ssv4d_adaptive_processed.add(window_key)
                    elif replay_decision is not None:
                        dynamic_reason = replay_decision["trigger_reason"]
                        dynamic_threshold = replay_decision[
                            "trigger_threshold"
                        ]
                        should_verify = (
                            replay_decision["should_verify"]
                            or marked_risky
                            or prefill_risky
                            or not margins
                        )
                    else:
                        # A missing replay is fail-closed.
                        should_verify = True
                        dynamic_reason = "replay_shadow_missing"
                else:
                    should_verify = (
                        not margins
                        or bootstrap
                        or marked_risky
                        or prefill_risky
                        or min_margin <= config["margin_threshold"]
                    )
                if config["dynamic_gate_enabled"] and config["dynamic_audit"]:
                    should_verify = True
                decisions[req.rid] = {
                    "min_margin": min_margin,
                    "should_verify": should_verify,
                    "bootstrap": bootstrap,
                    "marked_risky": marked_risky,
                    "prefill_risky": prefill_risky,
                    "dynamic_reason": dynamic_reason,
                    "dynamic_threshold": dynamic_threshold,
                    "replay_window_key": window_key,
                    "replay_window_start": (
                        replay["window_start"] if replay is not None else None
                    ),
                    "dynamic_audit": config["dynamic_audit"],
                    "adaptive_checkpoint": checkpoint,
                    "predicted_length": predicted_length,
                    "frontier_before": frontier_snapshot(
                        req
                    ).verified_tokens,
                }
                self.stats["ssv4d_windows_seen"] += 1
                if not should_verify and req not in deferred:
                    frontier_before = frontier_snapshot(req)
                    denied.append(req)
                    # Accept this draft window without verification. Advancing
                    # the verified frontier keeps the request in LLM-42's
                    # deterministic set while preventing the same window from
                    # being reconsidered on every subsequent decode step.
                    req.llm42_verified_tokens = len(req.output_ids)
                    frontier_after = frontier_snapshot(req)
                    decisions[req.rid][
                        "frontier_after"
                    ] = frontier_after.verified_tokens
                    decisions[req.rid][
                        "output_tokens"
                    ] = frontier_after.output_tokens
                    self.stats["ssv4d_windows_skipped"] += 1
                    self.stats["ssv4d_tokens_skipped"] += unverified

        self._ssv4d_decisions = decisions
        for req in deferred:
            req._ssv4d_deferred_deterministic = req.is_deterministic
            req.is_deterministic = False
        try:
            result = original_check(self, batch)
        finally:
            for req in deferred:
                req.is_deterministic = getattr(
                    req, "_ssv4d_deferred_deterministic", True
                )
        for req in getattr(batch, "reqs", None) or []:
            if (
                not getattr(req, "is_deterministic", False)
                or req.finished_reason is None
                or req.rid in self._ssv4d_finished_logged
            ):
                continue
            snapshot = frontier_snapshot(req)
            try:
                validate_frontier(snapshot)
            except ValueError:
                if config["frontier_strict"]:
                    raise
            self._ssv4d_finished_logged.add(req.rid)
            trie_final = None
            if config["trie_manager_enabled"]:
                manager = self._ssv4d_trie_managers.pop(req.rid, None)
                trie_final = manager.trie.stats() if manager else None
            _append_event(
                {
                    "_log_path": config["log_path"],
                    "event": "frontier_final",
                    "timestamp": time.time(),
                    "trie": trie_final,
                    **snapshot.to_dict(),
                }
            )
        if denied:
            _append_event(
                {
                    "_log_path": config["log_path"],
                    "event": "verify_skip",
                    "timestamp": time.time(),
                    "requests": {
                        req.rid: decisions.get(req.rid, {}) for req in denied
                    },
                    "draft_ms": getattr(self, "_ssv4d_last_draft_ms", None),
                }
            )
        return result

    def run_verification(self, original_batch, reqs, *args, **kwargs):
        request_list = list(reqs)
        self._ssv4d_req_by_rid = {req.rid: req for req in request_list}
        decisions = getattr(self, "_ssv4d_decisions", {})
        if not request_list:
            return []

        start_indices = {
            req.rid: req.llm42_verified_tokens for req in request_list
        }
        window_indices = {
            req.rid: req.llm42_num_verification_windows
            for req in request_list
        }
        window_lengths = {
            req.rid: _actual_window_len(req, self) for req in request_list
        }
        verification_shapes = sorted(
            {
                int(getattr(req, "ssv4d_verify_shape", self._window_size))
                for req in request_list
            }
        )
        kv_before = _snapshot_current_kv(self, request_list)
        captured = {}
        target_worker = self.target_worker
        target_forward = target_worker.forward_batch_generation

        def capture_forward(batch, **forward_kwargs):
            output = target_forward(batch, **forward_kwargs)
            captured["output"] = output
            return output

        target_worker.forward_batch_generation = capture_forward
        cuda_start = cuda_end = None
        if torch.cuda.is_available():
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
        started = time.perf_counter()
        try:
            result = original_verify(
                self, original_batch, request_list, *args, **kwargs
            )
        finally:
            target_worker.forward_batch_generation = target_forward
        wall_ms = (time.perf_counter() - started) * 1000.0
        gpu_ms = None
        if cuda_start is not None:
            cuda_end.record()
            if config["sync_timing"]:
                cuda_end.synchronize()
            try:
                gpu_ms = cuda_start.elapsed_time(cuda_end)
            except RuntimeError:
                pass

        self.stats["ssv4d_windows_verified"] += len(request_list)
        trie_stats = {}
        if config["trie_manager_enabled"]:
            for req in request_list:
                manager = self._ssv4d_trie_managers.get(req.rid)
                if manager is None:
                    continue
                manager.verify_path(
                    req.output_ids[: req.llm42_verified_tokens]
                )
                trie_stats[req.rid] = manager.trie.stats()
        for req, tokens in result:
            if tokens > 0:
                self._ssv4d_risky_rids.add(req.rid)
        logprob_features = _verify_output_features(
            request_list,
            self,
            captured.get("output"),
            start_indices,
            window_lengths,
        )
        kv_features = _after_kv_features(self, kv_before)
        captured_logits = (
            captured["output"].logits_output
            if captured.get("output") is not None
            else None
        )
        verify_top_rows = _as_rows(
            getattr(captured_logits, "next_token_top_logprobs_val", None)
        )
        draft_top_counts = {
            req.rid: len(
                _as_rows(getattr(req, "output_top_logprobs_val", None))
            )
            for req in request_list
        }
        replay_window_features = {}
        rollback_by_rid = {
            req.rid: tokens for req, tokens in result
        }
        frontiers = []
        for req in request_list:
            snapshot = frontier_snapshot(req)
            try:
                validate_frontier(snapshot)
            except ValueError:
                if config["frontier_strict"]:
                    raise
            frontiers.append(
                {
                    **snapshot.to_dict(),
                    "frontier_before": start_indices[req.rid],
                    "rollback_tokens": rollback_by_rid.get(
                        req.rid, 0
                    ),
                }
            )
        for req in request_list:
            window_start = start_indices[req.rid]
            window_key = decisions.get(req.rid, {}).get(
                "replay_window_key"
            )
            if window_key is None:
                continue
            replay = self._ssv4d_replay_shadow.get(window_key)
            if replay is None:
                continue
            positions = []
            for index, (canonical_values, canonical_ids, shadow_values, shadow_ids) in enumerate(
                zip(
                    replay["canonical_values"],
                    replay["canonical_ids"],
                    replay["shadow_values"],
                    replay["shadow_ids"],
                )
            ):
                positions.append(
                    {
                        "position": index,
                        "canonical_margin": topk_margin(
                            canonical_values, canonical_ids
                        ),
                        "shadow_margin": topk_margin(
                            shadow_values, shadow_ids
                        ),
                        "top1_agreement": bool(
                            canonical_ids
                            and shadow_ids
                            and canonical_ids[0] == shadow_ids[0]
                        ),
                    }
                )
            replay_window_features[req.rid] = {
                "window_start": window_start,
                "window_key": window_key,
                "window_index": window_indices[req.rid],
                "request_fingerprint": hashlib.sha1(
                    json.dumps(req.origin_input_ids).encode("utf-8")
                ).hexdigest()[:16],
                "rollback_tokens": rollback_by_rid.get(req.rid, 0),
                "decision": replay["decision"],
                "positions": positions,
            }
        _append_event(
            {
                "_log_path": config["log_path"],
                "event": "verify",
                "timestamp": time.time(),
                "num_reqs": len(request_list),
                "window_size": self._window_size,
                "verification_shapes": verification_shapes,
                "draft_ms": getattr(self, "_ssv4d_last_draft_ms", None),
                "verify_wall_ms": wall_ms,
                "verify_gpu_ms": gpu_ms,
                "producer_ms": getattr(
                    self, "_ssv4d_producer_ms", None
                ),
                "decisions": decisions,
                "logprob": logprob_features,
                "logprob_debug": {
                    "captured_output": captured.get("output") is not None,
                    "has_logits_output": bool(
                        captured.get("output") is not None
                        and captured["output"].logits_output is not None
                    ),
                    "top_values": (
                        len(
                            _as_rows(
                                captured["output"].logits_output.next_token_top_logprobs_val
                            )
                        )
                        if captured.get("output") is not None
                        and captured["output"].logits_output is not None
                        else 0
                    ),
                    "top_indices": (
                        len(
                            _as_rows(
                                captured["output"].logits_output.next_token_top_logprobs_idx
                            )
                        )
                        if captured.get("output") is not None
                        and captured["output"].logits_output is not None
                        else 0
                    ),
                    "verify_first_type": (
                        type(verify_top_rows[0]).__name__
                        if verify_top_rows
                        else None
                    ),
                    "draft_top_counts": draft_top_counts,
                },
                "kv": kv_features,
                "trie": trie_stats,
                "replay_window_features": replay_window_features,
                "frontiers": frontiers,
                "rollbacks": [
                    {"rid": req.rid, "tokens": tokens}
                    for req, tokens in result
                ],
            }
        )
        for req in request_list:
            for attribute in ("ssv4d_verify_tokens", "ssv4d_verify_shape"):
                if hasattr(req, attribute):
                    delattr(req, attribute)
        return result

    LLM42Info.prepare_verify_batch = prepare_verify_batch
    LLM42Worker.check_and_verify_deterministic_requests = (
        check_and_verify_deterministic_requests
    )
    LLM42Worker.forward_batch_generation = forward_batch_generation
    LLM42Worker._run_verification = run_verification
    LLM42Worker._ssv4d_runtime_installed = True
    _INSTALLED = True
    return True
