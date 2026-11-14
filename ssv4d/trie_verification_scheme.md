# SSV4D-Trie V1 Scheme and Evaluation Protocol

## Objective

Test whether a bounded prefix trie can represent batch-induced
non-deterministic completions and reduce LLM-42 verification cost without
changing the deterministic output.

The candidate is compared against native LLM-42 under identical prompts,
output lengths, batch sizes, seeds, and rounds.

## Candidate Design

### State

Each request owns one trie:

```text
TrieNode
  node_id
  token_id
  parent
  children: token_id -> node_id
  depth
  kv_pages
  topk_candidates
  canonical_count
  shadow_count
  verifier_count
  verified
```

The deterministic verifier path is authoritative. A node is committable only
after verifier confirmation or when it is an ancestor of a verified node.

### Expansion Policy

- Follow one canonical child when the top-1/top-2 margin is above
  `tau_stable`.
- At `margin <= tau_branch`, retain top-8 logical candidates from logits.
- Materialize top-1 by default and instantiate another child only when shadow
  disagreement or verifier demand makes it active.
- Keep a lazy overflow edge for observations outside top-8.
- Default evaluation thresholds:
  - `tau_branch=0.25`
  - conservative `tau_branch=0.375`
- Maximum observed branch degree is 2. The top-8 set is a logical candidate
  label set, not eight materialized KV branches.
- Maximum active frontier starts at 4. This is above the observed maximum of 3.

### KV Management

- KV pages for the shared prefix use reference counting.
- A branch allocates new pages only for its divergent suffix.
- Branch siblings may release pages independently.
- Node and KV budgets:
  - hard node cap: 4096 nodes/request;
  - hard active frontier: 4;
  - maximum extra KV: configurable; observed full tries stay below 3.1x
    canonical output length.

The largest observed 1024-token trie was 3072 nodes, so the node cap has
headroom while remaining bounded.

### Verification

For each 64-token depth window:

1. Collect all active trie nodes.
2. Gather their branch suffixes.
3. Run one tree-attention `TARGET_LLM42_VERIFY` forward.
4. Mark verifier-selected children as verified.
5. Commit only the verified frontier.
6. Release siblings that cannot beat the verifier path under the pruning
   policy.

The first implementation should only verify observed branches. Candidate edges
are lazy and acquire KV/suffix state only when the verifier needs them.

## Experimental Matrix

### Workloads

- Prompt set: all 48 extended prompts.
- Output lengths: 128, 512, 1024 tokens.
- Fast batch layouts: 8 and 16 duplicate positions.
- Prompt concurrency: 8 and 32.
- Rounds: 3, with round 0 designated as warm-up.
- Temperature 0, fixed seed 42, `ignore_eos=true`.

### Baselines

1. Native LLM-42 fixed 64-token verification.
2. LLM-42 mode-2 fixed 64-token verification for cross-shape comparisons.
3. Flat independent-path replay, used as the non-sharing upper bound.

### Candidate

1. Observed-trie replay with exact observed paths. This is an oracle for
   storage and call-count upper bounds.
2. Lazy top-8 trie with `tau_branch=0.25`.
3. Lazy top-8 trie with `tau_branch=0.375`.
4. Node cap ablations: 1024, 2048, 4096.

## Metrics

### Correctness

- Exact output digest match against native LLM-42.
- Cross-round deterministic rate.
- `verified_tokens == output_tokens` for every finished request.
- Frontier monotonicity violations.
- First-branch top-k coverage. The 48-prompt batch-16 run contains a rank-5
  shadow branch, so the candidate set must extend through top-8.

### Performance

- End-to-end output tokens/s.
- Mean, median, and p99 request latency.
- Decode and verify wall-time.
- Verifier calls and verifier tokens.
- Rollback events and rollback tokens.

### Tree and Memory

- Trie nodes/request.
- Branch nodes and maximum branch degree.
- Maximum active frontier.
- Prefix-sharing ratio.
- KV pages and peak allocated KV tokens.
- Branch prune and overflow rates.

## Acceptance Gates

The candidate advances only if all correctness gates pass:

- 48/48 output matches for every configuration.
- No frontier violations.
- All finished requests fully verified.

Performance gates:

- At least 30% fewer verifier calls than flat independent-path verification.
- At least 10% higher throughput than native LLM-42 at 1024 tokens.
- Full observed trie no more than 3.25x canonical KV; the active working set
  should remain below 1.5x after branch suffix pruning.
- No correctness regression when the active frontier cap is 4.

If the candidate cannot beat native LLM-42 after tree-attention verification is
implemented, it should remain an explainability/diagnostic structure rather
than a serving optimization.

## Current Status

Completed:

- 1024-token top-k and trie characterization.
- Candidate branch-trigger threshold analysis.
- Multi-shape verifier prototype and negative throughput result.
- Native LLM-42 1024-token baseline.

Next:

- Add scheduler-native tree attention and KV copy-on-write.
- Run the full matrix above.
- Report confidence intervals and retain all negative runs.
