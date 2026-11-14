#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export PATH="$repo_root/.venv/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/home/xiaosibo/.local/llm42-numa/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

for seed in 42 43 44; do
  python -m ssv4d.train_gate_model \
    ssv4d/logs/r2_multi_audit.jsonl \
    --folds 5 \
    --seed "$seed" \
    --output "ssv4d/results/gate_model_ablation_seed${seed}.json"
done
