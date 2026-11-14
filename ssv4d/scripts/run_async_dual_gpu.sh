#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export PATH="$repo_root/.venv/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/home/xiaosibo/.local/llm42-numa/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

rounds="${ROUNDS:-3}"
prompts="${PROMPTS:-8}"
tokens="${MAX_NEW_TOKENS:-130}"
window="${WINDOW:-64}"

python -m ssv4d.async_dual_gpu \
  --mode draft-only \
  --num-prompts "$prompts" \
  --rounds "$rounds" \
  --max-new-tokens "$tokens" \
  --output ssv4d/results/async_draft_only_3round.json

python -m ssv4d.async_dual_gpu \
  --mode verifier-only \
  --num-prompts "$prompts" \
  --rounds "$rounds" \
  --max-new-tokens "$tokens" \
  --output ssv4d/results/async_verifier_only_130_3round.json

python -m ssv4d.async_dual_gpu \
  --mode async \
  --num-prompts "$prompts" \
  --rounds "$rounds" \
  --max-new-tokens "$tokens" \
  --window "$window" \
  --gate-threshold 1.0 \
  --output ssv4d/results/async_dual_gpu_3round.json

python -m ssv4d.async_dual_gpu \
  --mode dynamic-async \
  --num-prompts "$prompts" \
  --rounds "$rounds" \
  --max-new-tokens "$tokens" \
  --window "$window" \
  --shadow-plans bs8_middle \
  --output ssv4d/results/dynamic_async_gate_single_3round.json

python -m ssv4d.async_dual_gpu \
  --mode dynamic-async \
  --num-prompts "$prompts" \
  --rounds "$rounds" \
  --max-new-tokens "$tokens" \
  --window "$window" \
  --shadow-plans bs8_middle \
  --force-verify \
  --output ssv4d/results/dynamic_async_gate_single_audit_3round.json
