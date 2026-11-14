# GPU Tree Verification Implementation Status

## Implemented

### Tree specification

`python/sglang/srt/llm42/trie_spec.py` now provides:

- `SSV4DTrieSpecInput`;
- flattened unique tree nodes;
- token positions and parent indices;
- per-path retrieve indices;
- causal tree masks;
- per-node page tables and cache lengths from shared prefix and trie KV
  slots.

### Runtime bookkeeping

`ssv4d/trie_runtime.py` provides:

- reference-counted page allocation;
- branch page retention and sibling release;
- prefix-preserving branch creation;
- verifier-path selection;
- rollback depth lookup.
- `TreeVerificationPool` for fixed batch scratch KV and request-pool slots;
- worker opt-in initialization via `SSV4D_TRIE_VERIFY_ENABLE=1`;
- scheduler reservation accounting for scratch KV/request slots.

The pool was smoke-tested against the real 8B server with
`SSV4D_TRIE_NODE_CAPACITY=128`; the server reserved the pool and completed a
small request without changing the default LLM-42 path.

### Node frontier

`ssv4d/trie_frontier.py` and `validate_trie_frontier.py` validate node-level
verified frontiers across all large traces.

## Not Integrated

FA3 still needs a `TARGET_LLM42_VERIFY` path carrying
`SSV4DTrieSpecInput.custom_mask` and `build_tree_kv_layout` into a real tree
attention forward. `LLM42Worker` also still needs an online producer for
alternative suffixes at low-margin positions.

The FA3 metadata branch is now implemented: when
`TARGET_LLM42_VERIFY` carries an `SSV4DTrieSpecInput`, the backend builds the
prefix metadata and tree-expand metadata from `tree_page_table` and
`tree_cache_seqlens`, then enables cascade attention using the same mechanism
as EAGLE target verification. It currently enforces batch size 1 until the
batch metadata path is validated.

The metadata path is covered by a direct unit test using
`ForwardMode.TARGET_LLM42_VERIFY + SSV4DTrieSpecInput`; it verifies prefix
cache lengths, tree-expand cache lengths, cumulative key lengths and the
tree page table.

`TrieBranchManager` is now installed in the LLM-42 verification window loop
behind `SSV4D_TRIE_MANAGER_ENABLE=1`:

- canonical paths are inserted at every verification window;
- low-margin nodes record top-8 logical candidate labels without materializing
  them as tree branches;
- the verifier-selected path is marked verified;
- final trie statistics are emitted with the frontier event.

A 2-prompt GPU smoke preserved full 64-token frontiers. One request remained a
single path; the other produced one real branch with degree 2 and two leaves.

The default materialized candidate count is now three (`SSV4D_TRIE_TOP_K=3`).
Measured GPU1 rollout cost for three candidates with 64-token suffixes is
0.477 s/call, versus 0.505 s/call for eight candidates. The latency is dominated
by the model forward rather than suffix count. Since the 48-prompt batch-16
study contains a rank-5 branch, top-3 must retain a lazy overflow edge.

The worker window loop now calls GPU1 automatically when
`SSV4D_GPU1_BRANCH_URL` is set. A 2-prompt, 64-token smoke completed with
three rollback windows and full frontiers, but took 22.97 s (5.57 tok/s)
versus 1.11 s (115.14 tok/s) for native LLM-42 because each verification
window synchronously waits for a 0.44-0.48 s rollout. The producer is correct
but not yet a serving-speed path.

The two gaps are coupled:

1. FA3 cannot build tree metadata without complete candidate paths.
2. Candidate paths cannot be verified efficiently without FA3 tree attention.

The next implementation unit is `TrieBranchManager` inside `LLM42Worker`:

- canonical fast decode of the current window;
- low-margin branch creation;
- bounded branch suffix rollout;
- one tree verifier forward;
- verified-path commit and sibling release.

## Correctness Reference

Native mode 3 is deterministic only under one fixed verifier shape. A
tree-shape verifier must use mode 2 batch-invariant kernels and compare against
the mode-2 fixed-shape baseline. It cannot be claimed to produce identical mode
3 output without retaining an authoritative fixed-64 verification pass.

## Implemented Tests

The following are covered by `ssv4d.tests.test_core`:

- tree mask prefix/ancestor/sibling behavior;
- page tables and cache lengths per node;
- lazy top-8 edge planning;
- verified-prefix rollback;
- COW page reference counts;
- sibling release after path selection.
