#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

export PATH="$REPO_ROOT/.venv/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/home/xiaosibo/.local/llm42-numa/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -m ssv4d.run_experiment \
  --num-prompts 16 \
  --max-new-tokens 128 \
  --repeats 2 \
  --window-size 32 \
  --target-recall 1.0 \
  --margin-guard 0.0625 \
  --min-shadow-auc 0.55 \
  "$@"
