#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

export PATH="$REPO_ROOT/.venv/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/home/xiaosibo/.local/llm42-numa/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -m ssv4d.prefill_probe \
  --num-prompts 8 \
  --top-k 8 \
  --kv-layer -1 \
  --output-dir ssv4d/results/prefill_probe "$@"
