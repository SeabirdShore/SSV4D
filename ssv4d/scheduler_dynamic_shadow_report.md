# Scheduler-Native Dynamic Shadow

This experiment implements the first scheduler-native SSV4D shadow stage inside
`LLM42Worker`.

> Historical checkpoint: the runtime now replays every verification window.
> The performance numbers below were collected during the earlier
> first-window-only stage. See `route2_multi_window_report.md` for the latest
> multi-window predictor evaluation.

## What Changed

At the first 64-token verification window, the worker now:

1. Creates an isolated shadow `Req` for every ready deterministic request.
2. Copies the canonical prompt plus the first 63 generated tokens.
3. Runs all shadow requests in one in-process extend batch with isolated
   request slots and K/V locations.
4. Computes canonical and shadow top-k distributions without HTTP or a second
   model process.
5. Applies the dynamic per-token threshold to decide whether the canonical
   LLM-42 verification pass is needed.

The shadow batch uses the actual number of protected requests. With 16
concurrent prompts, the replay is one batch=16 extend forward. No shadow state
is written into canonical K/V locations.

Implementation is in `ssv4d/runtime.py`; the threshold policy remains in
`ssv4d/dynamic_gate.py`.

## Dynamic Decision

For each position `t`:

```text
tau(t) = max_shadow |margin(canonical, t) - margin(shadow, t)|
verify if margin(canonical, t) <= tau(t)
       or top-1 tokens disagree
```

The first window has no bootstrap override in dynamic mode. Later windows are
fail-closed and still use the existing deterministic verifier because the first
prototype only covers the initial window.

## Correctness Audit

Three fresh rounds, 16 prompts, 64-token first window:

- 48 first-window requests.
- 15 canonical/verifier rollback requests.
- Dynamic gate selected 42/48 windows, an 87.5% first-window budget.
- It covered 15/15 rollback requests: 100% recall, zero misses.
- 27 selected requests were false positives in this small sample.

The conservative recall is partly caused by exact zero-tie events where the
current rule selects verification even when the perturbed margin is unchanged.
An offline rule requiring nonzero shadow uncertainty selected 36/48 with the
same 15/15 recall, but it was not promoted because it can ignore exact ties.

## Runtime Results

Four warm rounds, 16 prompts, 130 output tokens:

| Configuration | Mean time | Output tok/s | Verification windows |
|---|---:|---:|---:|
| LLM-42 always-on | 2.2865 s | 909.67 | 77 |
| Dynamic shadow, audit all verifier | 2.5640 s | 811.25 | 77 |
| Dynamic shadow, gated, final | 2.5769 s | 807.17 | 75 |

The gated path reduces request-window verifications by 2.6% and preserves all 16
output digests. Relative to the replay audit, gating is effectively flat in wall
time. Relative to always-on LLM-42, the in-process shadow is 12.7% slower.

The batched replay itself takes about 36 ms GPU / 37 ms wall per 16-request
window. Its scheduler and allocation disruption costs more than the raw kernel
time: only two request-window verifications are removed, while the shadow extend
pass and its K/V allocation are added to the critical path.

## Conclusion

Scheduler-native execution removes HTTP and duplicate model serving, and all
outputs remain deterministic. However, this first-window shadow still does not
beat always-on LLM-42 end-to-end.

The current implementation should therefore be treated as a verified systems
prototype, not yet as a performance win. The next decision is whether to:

1. improve shadow selectivity so substantially more verifier windows are
   skipped at 100% recall, or
2. reposition SSV4D as a higher-accuracy verification predictor without
   requiring an end-to-end speedup.

## Artifacts

- `logs/runtime_dynamic_position_features.jsonl`
- `logs/runtime_dynamic_batch_audit.jsonl`
- `logs/runtime_dynamic_batch_gated.jsonl`
- `logs/runtime_dynamic_batch_gated_final.jsonl`
- `logs/runtime_always_current.jsonl`
- `results/runtime_dynamic_position_features.json`
- `results/runtime_always_current_5round.json`
- `results/runtime_dynamic_batch_audit_5round.json`
- `results/runtime_dynamic_batch_gated_5round.json`
- `results/runtime_dynamic_batch_gated_final_5round.json`
- `results/runtime_dynamic_batch_summary.json`
- `scripts/run_runtime_probe.sh dynamic`
- `scripts/run_runtime_probe.sh dynamic-audit`

`runtime_dynamic_batch_gated_5round.json` predates the one-time
`replay_shadow_done` fix and must not be used for final measurements. That bug
could reuse a skipped first-window decision on later windows. The `_final_`
artifact is the corrected result.
