#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export PATH="$repo_root/.venv/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/home/xiaosibo/.local/llm42-numa/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

max_new_tokens="${1:-130}"
result_suffix="${2:-}"
token_label="${max_new_tokens}${result_suffix:+_${result_suffix}}"

python -m ssv4d.nonblocking_frontier_probe \
  --prompt-set extended \
  --num-prompts 32 \
  --rounds 3 \
  --prompt-concurrency 4 \
  --window 64 \
  --tail-merge-tokens 8 \
  --max-new-tokens "$max_new_tokens" \
  --output "ssv4d/results/nonblocking_frontier_32_${token_label}.json"
