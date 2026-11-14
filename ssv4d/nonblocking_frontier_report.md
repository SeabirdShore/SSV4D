# Non-Blocking Verified-Frontier Prototype

This experiment implements the second half of the new direction: fast decode
continues while deterministic verification runs on another GPU.

## Topology

```text
GPU0 fast server
  prompt_ids -> stream 130 fast tokens

GPU1 deterministic verifier
  prompt_ids + committed frontier
      -> generate the next 64-token window

frontier advances only after verifier comparison
    match: commit speculative tokens
    mismatch: commit common prefix, append verifier token, regenerate tail
```

The canonical stream is not awaited before the first verifier starts. The
frontier is advanced from verifier-backed windows. If the canonical stream and
verifier diverge, the speculative tail is discarded and rebuilt from the
verified prefix.

## Tail Handling

For 130 output tokens, fixed windows initially produced:

```text
64 + 64 + 2
```

The final two-token verification had poor latency and dominated the commit
tail. Windows smaller than eight residual tokens are now merged:

```text
64 + 66
```

## Results

Eight standard prompts, concurrency 4:

- First verifier starts at 0.535 s.
- Canonical stream completes at 0.990 s.
- Verifier overlaps canonical decode by 0.455 s.
- Mean commit completes at 1.994 s.
- Full frontier: 8/8.

Thirty-two extended prompts, three rounds, concurrency 4:

| Measurement | Mean |
|---|---:|
| Canonical stream latency | 0.936 s |
| First verifier start | 0.479 s |
| Verifier/canonical overlap | 0.457 s |
| Commit latency | 1.972 s |
| Round time | 16.814 s |
| Output throughput | 247.4 tok/s |
| Fully verified requests | 96/96 |

All requests reached `frontier == 130`.

## 1024-Token Probe

Eight standard prompts, one round, prompt concurrency 4:

| Measurement | Mean |
|---|---:|
| Canonical stream latency | 7.169 s |
| First verifier start | 0.518 s |
| Verifier/canonical overlap | 6.651 s |
| Commit latency | 35.574 s |
| Round time | 71.527 s |
| Output throughput | 114.53 tok/s |
| Fully verified requests | 8/8 |

Every request reached the full 1024-token frontier. Six of eight canonical
outputs differed from the deterministic committed output, and those tails were
repaired from the last common prefix.

At this output length the service-level duplicate-prefix cost dominates:
verification replays increasingly long prompts and the verifier is not sharing
canonical K/V state. This is a correctness probe, not a comparable throughput
result. For reference, native LLM-42 on the same eight prompts completes eight
concurrent requests in 10.393 s in its warm round (788.23 tok/s), but that run
does not use the probe's concurrency-4 waves.

## Findings

The non-blocking mechanism works:

- Verifier starts before canonical decode finishes.
- About 0.46 s of verifier work overlaps canonical generation.
- Commit waits only for the verified frontier.
- Mismatch recovery rebuilds the deterministic tail from the last common
  prefix.

The current service-level implementation remains slower than native LLM-42:

- prompt groups are processed in serial waves;
- verifier requests replay the full prompt prefix;
- the verifier runs at request granularity rather than scheduler verification
  batches;
- no canonical K/V state is shared or repaired across GPUs.

This is a correctness and overlap prototype, not yet the scheduler-native
implementation needed for a throughput win.

## Combined Status

Safe adaptive window:

- Preserves output correctness.
- Reduces rollback tokens by 78%.
- Increases verification calls and is still slower.

Non-blocking frontier:

- Preserves full verification.
- Provides real verifier/fast-decode overlap.
- Is service-level and still has duplicated prefix work.

The scheduler-native next step is:

1. Schedule verification as a separate priority class/stream inside LLM-42.
2. Reuse canonical prompt/KV state on GPU0.
3. Limit commit waits to deterministic requests.
4. Adapt verification batching and window length together.

## Artifacts

- `nonblocking_frontier_probe.py`
- `scripts/run_nonblocking_frontier_probe.sh`
- `results/nonblocking_frontier.json`
- `results/nonblocking_frontier_tailmerge.json`
- `results/nonblocking_frontier_32.json`
- `results/nonblocking_frontier_1024_small.json`
- `logs/nonblocking_frontier_32_fast_server.log`
- `logs/nonblocking_frontier_32_verifier_server.log`
- `logs/nonblocking_frontier_1024_small_fast_server.log`
- `logs/nonblocking_frontier_1024_small_verifier_server.log`
