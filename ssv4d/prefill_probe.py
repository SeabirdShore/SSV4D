"""Measure real model prefill perturbations at the ModelRunner batch level."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import destroy_distributed_environment
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import suppress_other_loggers
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

from ssv4d.features import topk_margin, vocab_aligned_divergence
from ssv4d.perturbation import make_perturbation_plans
from ssv4d.prompts import FILLER_PROMPTS, PROMPTS


class DummyTreeCache(SimpleNamespace):
    def supports_swa(self) -> bool:
        return False

    def supports_mamba(self) -> bool:
        return False

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return True


def _make_reqs(tokenizer, prompts: list[str]) -> list[Req]:
    sampling = SamplingParams(temperature=0, max_new_tokens=1)
    reqs = []
    for index, prompt in enumerate(prompts):
        token_ids = tokenizer.encode(prompt)
        req = Req(
            rid=index,
            origin_input_text=prompt,
            origin_input_ids=token_ids,
            sampling_params=sampling,
        )
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(
            len(req.fill_ids) - len(req.prefix_indices)
        )
        reqs.append(req)
    return reqs


def _reset(model_runner: ModelRunner) -> None:
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()


@torch.no_grad()
def run_prefill(
    model_runner: ModelRunner,
    tokenizer,
    prompts: list[str],
    protected_index: int,
    *,
    top_k: int = 8,
    kv_layer: int = -1,
) -> dict:
    _reset(model_runner)
    reqs = _make_reqs(tokenizer, prompts)
    tree_cache = DummyTreeCache(
        page_size=model_runner.server_args.page_size,
        device=model_runner.device,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
    )
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
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    start_event.record()
    output = model_runner.forward(forward_batch)
    end_event.record()
    end_event.synchronize()
    gpu_ms = start_event.elapsed_time(end_event)
    wall_ms = (time.perf_counter() - started) * 1000

    logits = output.logits_output.next_token_logits[
        protected_index
    ].detach().float()
    logprobs = torch.log_softmax(logits, dim=-1)
    top_values, top_ids = torch.topk(logprobs, top_k)

    target_req = reqs[protected_index]
    slot = model_runner.req_to_token_pool.req_to_token[
        target_req.req_pool_idx, len(target_req.origin_input_ids) - 1
    ]
    num_layers = model_runner.model_config.num_hidden_layers
    layer = kv_layer if kv_layer >= 0 else num_layers + kv_layer
    k_buffer, v_buffer = model_runner.token_to_kv_pool.get_kv_buffer(layer)
    return {
        "top_logprobs": top_values.cpu().tolist(),
        "top_ids": top_ids.cpu().tolist(),
        "logits": logits.cpu(),
        "k": k_buffer[slot].detach().clone(),
        "v": v_buffer[slot].detach().clone(),
        "prefill_wall_ms": wall_ms,
        "prefill_gpu_ms": gpu_ms,
        "layer": layer,
        "batch_size": len(prompts),
    }


def _select_plans(protected: str, names: set[str]):
    plans = make_perturbation_plans(
        protected,
        FILLER_PROMPTS,
        batch_sizes=(4, 8, 16),
        positions=("first", "middle", "last"),
        repeats=1,
    )
    if not names:
        return plans
    return [plan for plan in plans if plan.name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", default="NousResearch/Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--attention-backend", default="fa3")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--kv-layer", type=int, default=-1)
    parser.add_argument(
        "--plans",
        default="bs1,bs4_first_r0,bs8_first_r0,bs8_middle_r0,bs8_last_r0,bs16_last_r0",
        help="Comma-separated perturbation plan names",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("ssv4d/results/prefill_probe")
    )
    args = parser.parse_args()

    server_args = ServerArgs(
        model_path=args.model_path,
        tp_size=args.tp,
        attention_backend=args.attention_backend,
        disable_cuda_graph=True,
        chunked_prefill_size=-1,
    )
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)
    suppress_other_loggers()
    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=0,
        tp_rank=0,
        tp_size=server_args.tp_size,
        moe_ep_rank=0,
        moe_ep_size=server_args.ep_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if server_args.tp_size > 1:
        dist.barrier()

    # Warm CUDA kernels, allocator, and lazy imports before recording timings.
    _reset(model_runner)
    run_prefill(
        model_runner,
        tokenizer,
        [PROMPTS[-1].text],
        0,
        top_k=args.top_k,
        kv_layer=args.kv_layer,
    )

    selected_names = {
        name.strip() for name in args.plans.split(",") if name.strip()
    }
    rows = []
    for prompt in PROMPTS[: args.num_prompts]:
        plans = _select_plans(prompt.text, selected_names)
        canonical = None
        for plan in plans:
            result = run_prefill(
                model_runner,
                tokenizer,
                list(plan.prompts),
                plan.protected_index,
                top_k=args.top_k,
                kv_layer=args.kv_layer,
            )
            if canonical is None:
                canonical = result
                divergence = {
                    "kl_canonical_shadow": 0.0,
                    "kl_shadow_canonical": 0.0,
                    "jsd": 0.0,
                    "top1_agreement": 1.0,
                    "margin_delta": 0.0,
                    "k_relative_l2": 0.0,
                    "v_relative_l2": 0.0,
                    "k_cosine": 1.0,
                    "v_cosine": 1.0,
                }
            else:
                logprob_features = vocab_aligned_divergence(
                    canonical["top_logprobs"],
                    canonical["top_ids"],
                    result["top_logprobs"],
                    result["top_ids"],
                )
                k_delta = torch.linalg.vector_norm(
                    result["k"].float() - canonical["k"].float()
                ).item()
                v_delta = torch.linalg.vector_norm(
                    result["v"].float() - canonical["v"].float()
                ).item()
                k_base = torch.linalg.vector_norm(canonical["k"].float()).item()
                v_base = torch.linalg.vector_norm(canonical["v"].float()).item()
                divergence = {
                    **logprob_features,
                    "k_relative_l2": k_delta / (k_base + 1e-12),
                    "v_relative_l2": v_delta / (v_base + 1e-12),
                    "k_cosine": torch.nn.functional.cosine_similarity(
                        canonical["k"].float().reshape(-1),
                        result["k"].float().reshape(-1),
                        dim=0,
                    ).item(),
                    "v_cosine": torch.nn.functional.cosine_similarity(
                        canonical["v"].float().reshape(-1),
                        result["v"].float().reshape(-1),
                        dim=0,
                    ).item(),
                }
            rows.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "plan": plan.name,
                    "batch_size": result["batch_size"],
                    "protected_index": plan.protected_index,
                    "prefill_wall_ms": result["prefill_wall_ms"],
                    "prefill_gpu_ms": result["prefill_gpu_ms"],
                    "top1_margin": topk_margin(
                        result["top_logprobs"], result["top_ids"]
                    ),
                    **divergence,
                }
            )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "prefill_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for plan in sorted({row["plan"] for row in rows}):
        plan_rows = [row for row in rows if row["plan"] == plan]
        summary[plan] = {
            key: float(np.mean([row[key] for row in plan_rows]))
            for key in (
                "prefill_wall_ms",
                "prefill_gpu_ms",
                "top1_margin",
                "kl_canonical_shadow",
                "jsd",
                "top1_agreement",
                "margin_delta",
                "k_relative_l2",
                "v_relative_l2",
                "k_cosine",
                "v_cosine",
            )
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))

    if server_args.tp_size > 1:
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
