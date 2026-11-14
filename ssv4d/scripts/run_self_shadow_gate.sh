#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export PATH="$repo_root/.venv/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/home/xiaosibo/.local/llm42-numa/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

python -m ssv4d.self_shadow_gate \
  --prompt-set extended \
  --num-prompts 32 \
  --rounds 3 \
  --prompt-concurrency 8 \
  --batch-size 8 \
  --window 64 \
  --output ssv4d/results/self_shadow_32.json

python -m ssv4d.self_shadow_gate \
  --prompt-set extended \
  --num-prompts 32 \
  --rounds 3 \
  --prompt-concurrency 8 \
  --batch-size 8 \
  --window 64 \
  --force-verify \
  --output ssv4d/results/self_shadow_32_audit.json

python -m ssv4d.self_shadow_gate \
  --prompt-set extended \
  --num-prompts 32 \
  --rounds 3 \
  --prompt-concurrency 8 \
  --batch-size 16 \
  --window 64 \
  --force-verify \
  --output ssv4d/results/self_shadow_32_b16_audit.json
