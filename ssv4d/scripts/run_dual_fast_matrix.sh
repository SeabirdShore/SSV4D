#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export PATH="$repo_root/.venv/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/home/xiaosibo/.local/llm42-numa/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

python -m ssv4d.dual_fast_matrix \
  --prompt-set extended \
  --num-prompts 32 \
  --rounds 3 \
  --prompt-concurrency 8 \
  --canonical-tokens 130 \
  --window 64 \
  --shadow-plan bs8_middle \
  --mem-fraction-static 0.28 \
  --output ssv4d/results/dual_fast_matrix.json
