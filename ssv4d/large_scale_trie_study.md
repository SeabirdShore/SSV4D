# Large-Scale Trie Study

## Goal

Evaluate whether a bounded prefix trie can represent completion variation from
batch-shape and batch-order perturbations, and decide whether a scheduler-native
tree verifier is worth implementing.

The full candidate design is in `trie_verification_scheme.md`.

## Experiment Coverage

| Workload | Prompts | Batch/layout | Lengths | Rounds |
|---|---:|---|---:|---:|
| Native LLM-42 | 48 | all concurrent | 1024 | 3 |
| Native LLM-42 | 48 | prompt concurrency 4 | 128, 512, 1024 | 3 each |
| Self-shadow | 48 | 8 positions | 1024 | 1 |
| Self-shadow | 48 | 16 positions | 1024 | 1 |
| Shuffled shared batch | 32 | 8 copies/prompt | 1024 | 1 |

The 128- and 512-token trie results are prefixes of the 1024-token traces, so
they measure structural scaling without rerunning generation.

## Native LLM-42 Baseline

Warm round for prompt concurrency 4:

| Output length | Output tok/s | Verification windows | Rollback tokens | p50 latency |
|---:|---:|---:|---:|---:|
| 128 | 387.91 | 226 | 846 | 1.188 s |
| 512 | 434.21 | 521 | 2880 | 4.523 s |
| 1024 | 442.80 | 935 | 4474 | 8.969 s |

All three-round runs are cross-round deterministic.

With all 48 prompts submitted concurrently, the warm 1024-token round reaches
2630.67 output tok/s, 1109 verification windows, and 5013 rollback tokens.
Both scheduling shapes are retained because they are not interchangeable.

Full concurrency warm rounds:

| Output length | Output tok/s | Verification windows | Rollback tokens |
|---:|---:|---:|---:|
| 128 | 1649.40 | 284 | 1223 |
| 512 | 2572.50 | 700 | 3507 |
| 1024 | 2630.67 | 1109 | 5013 |

## Top-K Coverage at 1024 Tokens

| Workload | Shadow branches | Top-2 | Top-3 | Top-8 | Verifier branches | Verifier top-k |
|---|---:|---:|---:|---:|---:|---:|
| 48p batch 8 | 34 | 100% | 100% | 100% | 40 | top-3 100% |
| 48p batch 16 | 276 | 92.4% | 94.9% | 100% | 38 | top-2 100% |
| 32p shuffled | 55 | 100% | 100% | 100% | 26 | top-2 100% |

The 48-prompt batch-16 run invalidates a universal top-3 assumption. One prompt
in `statistics_variance` produced fourteen rank-5 shadow branches at position
544. Top-8 covers all currently observed first branches.

Decision:

- retain top-8 as logical candidates;
- materialize only observed or verifier-demanded children;
- keep a lazy overflow edge;
- never claim a fixed top-k bound without an overflow path.

## Branch-Triggered Expansion

Using canonical top-1/top-2 margin:

| Workload | Threshold | Branch recall | Trigger rate |
|---|---:|---:|---:|
| 48p batch 8 | 0.375 | 100% | 4.35% |
| 48p batch 16 | 0.375 | 100% | 4.58% |
| 32p shuffled | 0.375 | 100% | 4.31% |

The trigger rate remains stable across lengths because margins for a token do
not change when the trace is truncated later.

Decision: use `tau_branch=0.375` for the conservative candidate, with `0.25`
as an aggressive ablation.

## Trie Shape

At 1024 tokens:

| Workload | Mean nodes | Max nodes | Mean branch nodes | Max degree | Max frontier | Sharing |
|---|---:|---:|---:|---:|---:|---:|
| 48p batch 8 | 1804.5 | 2796 | 0.90 | 2 | 3 | 80.4% |
| 48p batch 16 | 2121.1 | 3072 | 1.21 | 2 | 3 | 87.8% |
| 32p shuffled | 1860.4 | 2862 | 0.97 | 2 | 3 | 79.8% |

Across 128/512/1024 lengths, mean nodes grow approximately linearly:

| Workload | 128 | 512 | 1024 |
|---|---:|---:|---:|
| Batch 8 | 176.6 | 840.1 | 1804.5 |
| Batch 16 | 192.0 | 975.0 | 2121.1 |
| Shuffled | 176.0 | 856.6 | 1860.4 |

The observed maximum is 3072 nodes, about 3x a canonical 1024-token path.
Set the hard node cap to 4096 and the active frontier cap to 4.

## Tree-Verification Budget

If every unique path is verified independently in 64-token windows, versus one
tree-aware call per active-window group:

| Workload | Flat calls | Tree calls | Reduction |
|---|---:|---:|---:|
| 48p batch 8 | 1472 | 768 | 47.8% |
| 48p batch 16 | 1744 | 768 | 56.0% |
| 32p shuffled | 1024 | 512 | 50.0% |

This is a call-count upper bound. It does not include tree-mask construction,
KV gather, or extra branch compute, so it is not yet a wall-time claim.

## Shadow Reuse

Only 5%-21.7% of verifier first branches have the same source position and
token as an observed shadow branch. Reusing stored shadow suffixes cannot
replace verifier suffix computation. The trie is useful for prefix sharing,
candidate discovery, and node-level commit, not as a full verifier cache.

## Decisions

Proceed with a runtime prototype having:

- top-8 logical candidates plus overflow;
- lazy branch materialization;
- `tau_branch=0.375`;
- one canonical path plus at most 3 active alternatives;
- node cap 4096/request;
- full observed KV below 3.25x canonical;
- node-level verified frontier;
- one tree-attention verification call per active 64-token window.

Reject:

- top-3-only candidate bounds;
- relying on shadow suffix reuse;
- multi-shape verifier windows as the primary optimization.

## Implemented State Machine

`trie_frontier.py` now implements:

- prefix insertion with canonical/shadow/verifier source counts;
- lazy canonical or top-8 candidate edges;
- verified-node marking;
- node-level rollback to the longest verified prefix;
- active-frontier and trie-size accounting.

Replay validation over 128 request trees at 1024 tokens produced:

| Workload | Requests | Valid frontiers | Max nodes | Max frontier |
|---|---:|---:|---:|---:|
| 48p batch 8 | 48 | 48/48 | 2796 | 3 |
| 48p batch 16 | 48 | 48/48 | 3072 | 3 |
| 32p shuffled | 32 | 32/32 | 2862 | 3 |

This validates trie state and rollback semantics. It does not execute tree
attention on GPU.

`python/sglang/srt/llm42/trie_spec.py` now implements the physical tree layout:

- flattened unique tree nodes;
- parent indices and per-node depths;
- token positions;
- per-path retrieve indices;
- a causal mask where each query attends the common prefix and its ancestors
  but not sibling branches.

The remaining integration gap is not the mask representation. FA3 currently
lacks a `TARGET_LLM42_VERIFY` metadata branch that consumes this mask, and the
runtime still needs a producer for complete alternative suffixes at low-margin
nodes.

## Runtime Work Remaining

The current study does not implement scheduler-native tree attention. Therefore
it cannot yet claim a candidate throughput result against LLM-42. The next
implementation must:

1. Add a trie-node KV representation with reference-counted prefix pages.
2. Build `TARGET_LLM42_VERIFY` batches with a causal tree mask.
3. Add rollback to node-level verified frontiers.
4. Run the full 48-prompt, three-round, three-length, two-concurrency matrix.

The candidate only advances with 48/48 output equality, a complete frontier,
and positive warm-round throughput over native LLM-42.

## Artifacts

- `results/large_scale_native_llm42_1024_b48.json`
- `results/large_scale_native_llm42_1024_b48_c4.json`
- `results/large_scale_native_llm42_128_b48_c4.json`
- `results/large_scale_native_llm42_512_b48_c4.json`
- `results/large_scale_self_shadow_1024_b8.json`
- `results/large_scale_self_shadow_1024_b16.json`
- `results/large_scale_shuffled_self_shadow_1024.json`
- `results/large_scale_trie_b8_lengths.json`
- `results/large_scale_trie_b16_lengths.json`
- `results/large_scale_trie_shuffled_lengths.json`
