# Full Verified Frontier

This experiment adds strict verified-frontier tracking and final-commit
validation to the LLM-42 worker.

## Frontier Invariant

For every deterministic request:

```text
0 <= verified_tokens <= output_tokens
```

For a finished request:

```text
verified_tokens == output_tokens
```

The existing scheduler output path emits only tokens below
`llm42_verified_tokens`. The new instrumentation records:

- frontier before and after every verification.
- frontier after a gate-skipped window.
- rollback token count.
- final frontier when a request finishes.
- strict validation failures.

`SSV4D_FRONTIER_STRICT=1` turns a finished request with unverified speculative
tokens into a server error.

## Workload

- 32 extended prompts
- 3 rounds
- 130 output tokens
- fixed 64-token LLM-42 verification window

## Results

| Configuration | Warm round | Output tok/s | Verification windows | Deterministic output |
|---|---:|---:|---:|---:|
| Native LLM-42 baseline | 2.6610 s | 1563.30 | 171 | 32/32 |
| Always-on | 2.7633 s | 1505.45 | 171 | 32/32 |
| Gate threshold 1.0 | 3.2202 s | 1291.85 | 171 | 32/32 |
| Gate threshold 0.0625 | 3.1663 s | 1313.82 | 148 | 28/32 |

Native LLM-42 and SSV4D always-on produce identical output digests for all
32 prompts and execute the same 171 verification windows. SSV4D instrumentation
adds 3.99% wall time and reduces throughput by 3.84%.

Gate threshold 0.0625 reduced verification windows by 13.45%, but throughput
was 12.7% lower than always-on due to gate instrumentation and changed request
scheduling.

Four prompts differed from the always-on baseline:

- `fact_capitals`
- `fact_honey`
- `math_sum`
- `probability_cards`

## Frontier Audit

Always-on:

- 96/96 finished requests fully verified.
- Final unverified tokens: 0.
- 513 verification transitions.
- 72 rollback transitions.
- Frontier monotonicity violations: 0.

Gate threshold 0.0625:

- 96/96 finished requests mechanically reached `verified == output`.
- Final unverified tokens: 0.
- 444 verification transitions.
- 66 rollback transitions.
- Frontier monotonicity violations: 0.

The frontier invariant is therefore implemented correctly.

## Critical Interpretation

A mechanically full frontier does not guarantee deterministic correctness when
the gate can incorrectly mark a skipped window as verified.

Threshold 0.0625 produced:

```text
frontier fully advanced: 96/96
output equal to always-on: 28/32
```

This is expected: the frontier tracks what the policy has accepted. If a skipped
window contains a hidden rollback, advancing the frontier commits an
incorrectly accepted token.

Full deterministic correctness therefore requires both:

1. a correct verified frontier,
2. a gate with 100% rollback recall on unseen prompts.

The current gate does not satisfy the second condition.

## Artifacts

- `frontier.py`
- `analyze_frontier.py`
- `scripts/run_frontier_probe.sh`
- `results/frontier_always.json`
- `results/native_llm42_baseline.json`
- `results/native_vs_always_compare.json`
- `results/frontier_gated.json`
- `results/frontier_gated_0p0625.json`
- `results/frontier_always_audit.json`
- `results/frontier_gated_0p0625_audit.json`
- `logs/frontier_always.jsonl`
- `logs/frontier_gated.jsonl`
- `logs/frontier_gated_0p0625.jsonl`
