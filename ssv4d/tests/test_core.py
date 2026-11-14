from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.llm42.llm42_worker import (
    LLM42Worker,
    _configured_verification_windows,
)
from sglang.srt.llm42.trie_spec import build_trie_spec, build_tree_kv_layout
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from ssv4d.llm42_hook import install_verification_gate
from ssv4d.metrics import first_mismatch
from ssv4d.models import GenerationTrace, WindowSample
from ssv4d.perturbation import make_perturbation_plans
from ssv4d.prompts_extended import EXTENDED_PROMPTS
from ssv4d.features import (
    kv_column_deviation,
    topk_margin,
    vocab_aligned_divergence,
)
from ssv4d.frontier import FrontierSnapshot, validate_frontier
from ssv4d.dynamic_gate import evaluate_dynamic_window
from ssv4d.policy import (
    calibrate_and_evaluate,
    evaluate_predictions,
    grouped_cross_validate,
)
from ssv4d.runtime import _slice_rows, _tail_rows
from ssv4d.analyze_trie import analyze, build_trie
from ssv4d.trie_frontier import PrefixTrie, plan_candidate_edges
from ssv4d.trie_runtime import PagePool, TreeVerificationPool, TrieBranchManager
from ssv4d.gpu1_branch_producer import BranchRollout, GPU1BranchProducer
from sglang.srt.llm42.llm42_info import LLM42Info


class MetricTests(unittest.TestCase):
    def test_verified_frontier_validation(self):
        validate_frontier(
            FrontierSnapshot(
                rid="ok",
                verified_tokens=8,
                output_tokens=8,
                unverified_tokens=0,
                finished=True,
            )
        )
        with self.assertRaises(ValueError):
            validate_frontier(
                FrontierSnapshot(
                    rid="bad",
                    verified_tokens=7,
                    output_tokens=8,
                    unverified_tokens=1,
                    finished=True,
                )
            )

    def test_per_request_verification_limit(self):
        req = SimpleNamespace(
            rid="r",
            output_ids=list(range(10)),
            llm42_verified_tokens=2,
            finished_reason=None,
        )
        info = LLM42Info.from_requests(
            [req],
            always_align=False,
            output_limits={"r": 3},
        )
        self.assertEqual(info.output_lens, [3])
        self.assertEqual(info.original_outputs.tolist(), [2, 3, 4])

    def test_first_mismatch(self):
        self.assertEqual(first_mismatch([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(first_mismatch([1, 2], [1, 2]), 2)
        self.assertEqual(first_mismatch([1], [1, 2]), 1)

    def test_vocab_aligned_features(self):
        canonical = [-0.1, -1.1, -4.0]
        canonical_ids = [10, 11, 12]
        shadow = [-0.2, -0.9, -4.1]
        shadow_ids = [10, 11, 12]
        self.assertAlmostEqual(
            topk_margin(canonical, canonical_ids), 1.0, places=6
        )
        result = vocab_aligned_divergence(
            canonical, canonical_ids, shadow, shadow_ids
        )
        self.assertEqual(result["top1_agreement"], 1.0)
        self.assertGreater(result["jsd"], 0.0)
        self.assertAlmostEqual(result["margin_delta"], -0.3, places=6)

    def test_kv_column_deviation(self):
        canonical = torch.tensor([1.0, 2.0, 3.0])
        shadow = torch.tensor([1.0, 2.0, 4.0])
        result = kv_column_deviation(
            canonical, canonical, shadow, shadow
        )
        self.assertGreater(result["k_l2"], 0.0)
        self.assertAlmostEqual(result["k_relative_l2"], result["v_relative_l2"])

    def test_window_row_slicing_matches_full_conversion(self):
        values = torch.arange(24).reshape(8, 3)
        flattened = torch.arange(24)
        self.assertEqual(_tail_rows(values, 3), values[-3:].tolist())
        self.assertEqual(_slice_rows(flattened, 5, 4), [5, 6, 7, 8])

    def test_multi_shape_verification_pool_selection(self):
        with patch.dict(
            os.environ,
            {
                "SSV4D_VERIFY_WINDOW_SIZES": "16,32,48,64",
                "SGLANG_ENABLE_LLM42": "2",
            },
        ):
            self.assertEqual(
                _configured_verification_windows(64),
                (16, 32, 48, 64),
            )
        with patch.dict(
            os.environ,
            {
                "SSV4D_VERIFY_WINDOW_SIZES": "16,32,48,64",
                "SGLANG_ENABLE_LLM42": "3",
            },
        ):
            self.assertEqual(_configured_verification_windows(64), (64,))

        worker = object.__new__(LLM42Worker)
        worker._window_size = 64
        worker.fixed_pools = {
            window: SimpleNamespace(window_size=window)
            for window in (16, 32, 48, 64)
        }
        worker.fixed_pool = worker.fixed_pools[64]
        for requested, expected in (
            (1, 16),
            (16, 16),
            (17, 32),
            (48, 48),
            (64, 64),
        ):
            req = SimpleNamespace(ssv4d_verify_tokens=requested)
            self.assertEqual(
                worker._fixed_pool_for_request(req).window_size,
                expected,
            )

    def test_trie_analysis_merges_common_prefixes(self):
        trie = build_trie([[1, 2, 3], [1, 4, 3]])
        self.assertEqual(trie["node_count"], 5)
        self.assertEqual(trie["branch_nodes"], 1)
        self.assertEqual(trie["max_frontier_width"], 2)

        payload = {
            "num_prompts": 1,
            "window": 3,
            "batch_size": 2,
            "results": [
                {
                    "prompt_id": "p",
                    "canonical": {
                        "output_ids": [1, 2, 3],
                        "top_ids": [[1, 9], [2, 4], [3, 8]],
                    },
                    "shadows": [{"output_ids": [1, 4, 3]}],
                    "verifier": {"output_ids": [1, 2, 3]},
                }
            ],
        }
        result = analyze(payload)
        self.assertEqual(
            result["shadow_vs_canonical"]["topk_counts"]["2"], 1
        )
        self.assertEqual(result["verifier_vs_canonical"]["branch_pairs"], 0)
        short = analyze(payload, max_length=2)
        self.assertEqual(short["completion_tokens"], 2)
        self.assertEqual(short["trie"]["node_count"]["max"], 3)

    def test_prefix_trie_verified_frontier_and_rollback(self):
        trie = PrefixTrie()
        trie.insert([1, 2, 3], source="canonical")
        trie.insert([1, 4, 3], source="shadow")
        trie.verify_path([1, 2, 9])

        stats = trie.stats()
        self.assertEqual(stats["nodes"], 6)
        self.assertEqual(stats["branch_nodes"], 2)
        self.assertEqual(stats["max_branch_degree"], 2)
        self.assertEqual(stats["max_frontier_width"], 3)

        target = trie.longest_verified_prefix([1, 2, 4])
        self.assertEqual(target.depth, 2)
        self.assertEqual(target.node.token_id, 2)
        self.assertIsNotNone(trie.node_for_path([1, 4, 3]))

    def test_lazy_topk_candidate_edges(self):
        stable = plan_candidate_edges(
            [-0.1, -1.1, -2.0, -3.0],
            [10, 11, 12, 13],
        )
        self.assertFalse(stable.expanded)
        self.assertEqual(stable.candidate_tokens, (10,))

        risky = plan_candidate_edges(
            [-0.1, -0.2, -2.0, -3.0],
            [10, 11, 12, 13],
            max_candidates=3,
        )
        self.assertTrue(risky.expanded)
        self.assertEqual(risky.candidate_tokens, (10, 11, 12))

    def test_trie_spec_causal_mask(self):
        spec = build_trie_spec(
            [[1, 2], [1, 3]],
            prefix_len=5,
            device="cpu",
        )
        self.assertEqual(spec.draft_token.tolist(), [1, 2, 3])
        self.assertEqual(spec.parent_indices.tolist(), [-1, 0, 0])
        self.assertEqual(spec.positions.tolist(), [5, 6, 6])

        mask = spec.custom_mask.view(3, 8)
        self.assertTrue(mask[0, :5].all())
        self.assertFalse(mask[0, 5:].any())
        self.assertTrue(mask[1, 5])
        self.assertFalse(mask[1, 6:].any())
        self.assertTrue(mask[2, 5])
        self.assertFalse(mask[2, 6])

        page_table, cache_seqlens = build_tree_kv_layout(
            spec,
            torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32),
            torch.tensor([20, 21, 22], dtype=torch.int32),
        )
        self.assertEqual(cache_seqlens.tolist(), [6, 7, 7])
        self.assertEqual(
            page_table[1].tolist(),
            [10, 11, 12, 13, 14, 20, 21, 0],
        )
        self.assertEqual(
            page_table[2].tolist(),
            [10, 11, 12, 13, 14, 20, 22, 0],
        )

    def test_trie_branch_manager_releases_siblings(self):
        pool = PagePool(capacity=8)
        prefix = pool.allocate(2)
        manager = TrieBranchManager(pool)
        first = manager.add_branch(prefix, [1, 2])
        second = manager.add_branch(prefix, [1, 3])
        self.assertEqual(pool.live_pages(), 6)

        selected = manager.verify_path([1, 2])
        self.assertEqual(selected.token_id, 2)
        self.assertFalse(second.active)
        self.assertIn(first.node.node_id, manager.branches)
        self.assertEqual(pool.live_pages(), 4)

        manager.trie.insert([5, 6], source="canonical")
        labeled = manager.record_candidate_labels(
            [5], [6, 7, 8, 9]
        )
        self.assertEqual(labeled.candidate_labels, (6, 7, 8, 9))
        self.assertEqual(manager.trie.stats()["max_branch_degree"], 2)

    def test_tree_verification_pool_lifecycle(self):
        class Allocator:
            page_size = 1

            def __init__(self):
                self.next_slot = 100
                self.freed = None

            def alloc(self, count):
                result = torch.arange(
                    self.next_slot,
                    self.next_slot + count,
                    dtype=torch.int64,
                )
                self.next_slot += count
                return result

            def free(self, values):
                self.freed = values

        allocator = Allocator()
        request_pool = SimpleNamespace(free_slots=list(range(10, 18)))
        pool = TreeVerificationPool(
            fixed_size=2,
            node_capacity=4,
            req_to_token_pool=request_pool,
            token_to_kv_pool_allocator=allocator,
            device="cpu",
        )
        self.assertEqual(pool.acquire(0), 10)
        self.assertEqual(
            pool.node_cache_locs(0, 3).tolist(),
            [100, 101, 102],
        )
        pool.release(0)
        pool.free(allocator)
        self.assertEqual(
            request_pool.free_slots,
            [12, 13, 14, 15, 16, 17, 10, 11],
        )
        self.assertIsNotNone(allocator.freed)

    def test_fa3_consumes_trie_spec_metadata(self):
        spec = build_trie_spec(
            [[1, 2], [1, 3]],
            prefix_len=3,
            device="cpu",
        )
        req_to_token = torch.tensor(
            [[7, 8, 9, 0, 0, 0, 0, 0]],
            dtype=torch.int32,
        )
        page_table, cache_seqlens = build_tree_kv_layout(
            spec,
            req_to_token[0, :3],
            torch.tensor([20, 21, 22], dtype=torch.int32),
        )
        spec.set_kv_layout(page_table, cache_seqlens)

        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_LLM42_VERIFY,
            seq_lens=torch.tensor([3]),
            seq_lens_cpu=torch.tensor([3]),
            batch_size=1,
            device="cpu",
            spec_info=spec,
            encoder_lens=None,
            req_pool_indices=torch.tensor([0]),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        )
        backend = object.__new__(FlashAttentionBackend)
        backend.use_sliding_window_kv_pool = False
        backend.has_swa = False
        backend.page_size = 1
        backend.init_forward_metadata(forward_batch)
        metadata = backend.forward_metadata
        expand = backend.forward_metadata_spec_decode_expand
        self.assertEqual(metadata.cache_seqlens_int32.tolist(), [3])
        self.assertEqual(metadata.max_seq_len_q, 3)
        self.assertEqual(expand.cache_seqlens_int32.tolist(), [4, 5, 5])
        self.assertEqual(
            expand.cu_seqlens_k.tolist(),
            [0, 4, 9, 14],
        )
        self.assertEqual(expand.page_table.tolist(), page_table.tolist())

    def test_gpu1_rollouts_merge_into_trie(self):
        manager = TrieBranchManager()
        rollouts = [
            BranchRollout(1, (7, 8), (1, 7, 8)),
            BranchRollout(2, (7, 9), (2, 7, 9)),
        ]
        GPU1BranchProducer.write_to_manager(manager, rollouts)
        self.assertEqual(manager.trie.stats()["nodes"], 6)
        self.assertEqual(manager.trie.stats()["max_branch_degree"], 2)


class PerturbationTests(unittest.TestCase):
    def test_plan_placement_and_sizes(self):
        plans = make_perturbation_plans(
            "target", ["a", "b", "c", "d"], batch_sizes=(4,), repeats=1
        )
        self.assertEqual(plans[0].name, "bs1")
        for plan in plans[1:]:
            self.assertEqual(len(plan.prompts), 4)
            self.assertEqual(plan.prompts[plan.protected_index], "target")

    def test_extended_prompt_suite(self):
        self.assertGreaterEqual(len(EXTENDED_PROMPTS), 48)
        self.assertEqual(
            len({prompt.prompt_id for prompt in EXTENDED_PROMPTS}),
            len(EXTENDED_PROMPTS),
        )


class DynamicGateTests(unittest.TestCase):
    @staticmethod
    def _trace(margin, top_id, output_id):
        logprobs = [0.0, -margin, -4.0]
        ids = [top_id, top_id + 1, top_id + 2]
        return GenerationTrace(
            prompt_id="p",
            scenario="shadow",
            batch_size=1,
            protected_index=0,
            output_ids=[output_id],
            text="",
            margins=[margin],
            top_ids=[ids],
            top_logprobs=[logprobs],
            latency_s=0.0,
            deterministic=False,
        )

    def test_threshold_is_derived_from_shadow_margins(self):
        canonical = self._trace(0.04, 10, 10)
        tight_shadow = self._trace(0.08, 10, 10)
        wide_shadow = self._trace(0.0, 10, 10)
        decision = evaluate_dynamic_window(
            canonical, [tight_shadow, wide_shadow]
        )
        self.assertTrue(decision.should_verify)
        self.assertAlmostEqual(decision.trigger_threshold, 0.04)
        self.assertEqual(
            decision.trigger_reason, "margin_within_shadow_uncertainty"
        )

    def test_high_margin_with_small_shadow_delta_is_skipped(self):
        canonical = self._trace(2.0, 10, 10)
        shadow = self._trace(1.9, 10, 10)
        decision = evaluate_dynamic_window(canonical, [shadow])
        self.assertFalse(decision.should_verify)
        self.assertAlmostEqual(decision.max_threshold, 0.1)

    def test_top1_disagreement_triggers_without_fixed_margin(self):
        canonical = self._trace(0.5, 10, 10)
        shadow = self._trace(0.5, 20, 10)
        decision = evaluate_dynamic_window(canonical, [shadow])
        self.assertTrue(decision.should_verify)
        self.assertEqual(decision.trigger_reason, "top1_disagreement")


class HookTests(unittest.TestCase):
    def test_gate_temporarily_skips_denied_requests(self):
        seen = []

        class Worker:
            def check_and_verify_deterministic_requests(self, batch):
                seen.extend(req.is_deterministic for req in batch.reqs)
                return []

        worker = Worker()
        reqs = [
            SimpleNamespace(is_deterministic=True, rid="a"),
            SimpleNamespace(is_deterministic=True, rid="b"),
        ]
        install_verification_gate(
            worker, lambda req: req.rid == "a"
        )
        worker.check_and_verify_deterministic_requests(
            SimpleNamespace(reqs=reqs)
        )
        self.assertEqual(seen, [True, False])
        self.assertTrue(all(req.is_deterministic for req in reqs))


class PolicyTests(unittest.TestCase):
    def _samples(self):
        rows = []
        for i in range(20):
            label = int(i in (2, 7, 14))
            rows.append(
                WindowSample(
                    prompt_id=f"p{i % 8}",
                    window_index=i,
                    start=i * 4,
                    end=i * 4 + 4,
                    label=label,
                    mismatch_positions=[i * 4 + 1] if label else [],
                    min_margin=0.05 if label else 1.5,
                    q10_margin=0.05 if label else 1.2,
                    mean_margin=0.1 if label else 1.8,
                    margin_spread=0.2 if label else 0.01,
                    shadow_vote_disagreement=0.5 if label else 0.0,
                    shadow_jsd=0.2 if label else 0.0,
                    n_traces=4,
                )
            )
        return rows

    def test_metrics_and_calibration(self):
        samples = self._samples()
        calib = samples[:10]
        test = samples[10:]
        gate, metrics = calibrate_and_evaluate(
            calib, test, target_recall=1.0
        )
        self.assertIsNotNone(gate)
        self.assertTrue(any(m.policy == "ssv4d" for m in metrics))
        cv_metrics = grouped_cross_validate(
            samples, target_recall=1.0, margin_guard=0.0625
        )
        self.assertTrue(any(m.policy == "ssv4d" for m in cv_metrics))
        metric = evaluate_predictions(
            samples,
            [bool(s.label) for s in samples],
            policy_name="oracle",
            threshold=0.0,
            target_recall=1.0,
            split="test",
        )
        self.assertEqual(metric.window_recall, 1.0)


if __name__ == "__main__":
    unittest.main()
