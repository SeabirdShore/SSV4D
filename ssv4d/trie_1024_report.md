# 1024-Token Prefix-Trie Extrapolation

> Later 48-prompt batch-16 data found a rank-5 shadow branch. Top-8, not top-3,
> is required for the observed support. See `large_scale_trie_study.md`.

This experiment tests whether non-deterministic completions can be represented
as a small prefix tree instead of storing every complete sequence.

## Setup

- 32 extended prompts.
- One round per batch shape.
- `max_new_tokens=1024`, temperature 0, `ignore_eos=true`.
- One canonical fast completion, seven same-prompt shadow copies, and one
  deterministic verifier completion per prompt.
- Batch 8: 8 fast requests, so each prompt has `1 + 7 + 1 = 9` observed paths.
- Batch 16: 16 fast requests, so each prompt has `1 + 15 + 1 = 17` paths.
- All 832 generated paths reached exactly 1024 tokens.

The top-k rank analysis is for the **first divergence from canonical**, where
both paths share the same source prefix. Tokens after a divergence have their
own conditional distributions and cannot be compared to the canonical top-k.

## Top-K Coverage

Batch 8:

| Comparison | Branches | Top-1 | Top-2 | Top-3 | Outside top-8 |
|---|---:|---:|---:|---:|---:|
| Shadow vs canonical | 25 | 28.0% | 100% | 100% | 0 |
| Verifier vs canonical | 26 | 38.5% | 96.2% | 100% | 0 |

Batch 16:

| Comparison | Branches | Top-1 | Top-2 | Top-3 | Outside top-8 |
|---|---:|---:|---:|---:|---:|
| Shadow vs canonical | 158 | 34.8% | 100% | 100% | 0 |
| Verifier vs canonical | 24 | 37.5% | 100% | 100% | 0 |

Across both runs:

- 183/183 shadow branches are in canonical top-2.
- 49/50 verifier branches are in canonical top-2.
- 50/50 verifier branches are in canonical top-3.
- No observed first branch is outside canonical top-8.

The one rank-3 verifier branch occurred for `fact_honey` at position 208:

```text
canonical = 67038
verifier  = 1579
top-3     = [67038, 14675, 1579]
```

This is why top-2 is not a safe hard bound. A practical prototype should keep at
least top-3 or top-4 plus an overflow edge.

## Trie Size

| Batch | Branching prompts | Mean nodes | Median nodes | Max nodes | Mean branch nodes | Max branch degree | Max active frontier | Sharing |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 5/32 | 1830.8 | 1948 | 3028 | 0.97 | 2 | 3 | 80.1% |
| 16 | 13/32 | 1987.3 | 1986.5 | 3047 | 1.09 | 2 | 3 | 88.6% |

First-branch positions:

| Batch | Shadow mean / median / max | Verifier mean / median / max |
|---:|---:|---:|
| 8 | 22.4 / 7 / 298 | 194.3 / 100 / 993 |
| 16 | 62.3 / 24 / 443 | 210.0 / 122 / 806 |

For batch 16, 17 complete paths require `17 × 1024 = 17408` token entries
without sharing. The observed trie needs a mean of 1987 nodes, an 8.8x
compression. The largest observed trie has only 3047 nodes.

All observed branches are binary, even though the candidate set may contain
top-3 or top-4 options. The maximum number of simultaneously active trie
frontier nodes is three.

## Route Exploration

### Tree-Aware Verification

If each unique path is verified independently with 64-token windows:

| Batch | Flat window calls | Tree window calls | Reduction |
|---:|---:|---:|---:|
| 8 | 1008 | 512 | 49.2% |
| 16 | 1088 | 512 | 52.9% |

These counts assume all nodes in a window can be processed by one
tree-attention verification call. They exclude the additional tree-mask cost,
but show that verification call count can be roughly halved.

### Branch-Triggered Expansion

Canonical top-1/top-2 margin thresholds over all 32768 positions:

| Threshold | Batch-8 recall / trigger | Batch-16 recall / trigger |
|---:|---:|---:|
| 0.0625 | 53.6% / 0.63% | 57.6% / 0.70% |
| 0.125 | 96.4% / 1.81% | 97.0% / 1.94% |
| 0.25 | 96.4% / 2.98% | 100% / 3.17% |
| 0.375 | 100% / 4.00% | 100% / 4.22% |

A threshold of 0.25-0.375 is a plausible branch-expansion trigger. It preserves
every observed branch while opening candidate edges at only 3%-4% of positions.

### Shadow-to-Verifier Reuse

Only 7/50 verifier first branches have the same position and token as an
observed shadow branch. Storing shadow paths helps tree shape and KV sharing,
but cannot generally replace the verifier suffix. The tree should keep the
top-3 candidate edge, but still execute the verifier path when its suffix is not
already materialized.

## Interpretation

The 1024-token data supports the finite-state tree direction:

- Flips remain sparse over eight times the earlier completion length.
- Most prompts have identical canonical, shadow, and verifier paths.
- Observed trees are small and deeply shared.
- `top-3` covers every observed first branch; top-2 misses one verifier branch.

This is not yet proof that all perturbation schedules remain in top-3. The
experiment uses one batch-order realization per batch size and a finite set of
self-shadow permutations. A safe implementation should still include an
overflow branch and a node/KV budget.

## Reproduction

```bash
python -m ssv4d.self_shadow_gate \
  --prompt-set extended \
  --num-prompts 32 \
  --rounds 1 \
  --prompt-concurrency 4 \
  --batch-size 8 \
  --window 1024 \
  --force-verify \
  --output ssv4d/results/self_shadow_1024_b8_audit.json

python -m ssv4d.analyze_trie \
  ssv4d/results/self_shadow_1024_b8_audit.json \
  --output ssv4d/results/self_shadow_1024_b8_trie.json
```

Repeat with `--batch-size 16` and the corresponding output names.

## Artifacts

- `analyze_trie.py`
- `results/self_shadow_1024_b8_audit.json`
- `results/self_shadow_1024_b8_trie.json`
- `results/self_shadow_1024_b16_audit.json`
- `results/self_shadow_1024_b16_trie.json`
