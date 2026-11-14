#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export PATH="$repo_root/.venv/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/home/xiaosibo/.local/llm42-numa/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

python -m ssv4d.dynamic_gate_probe \
  --num-prompts "${PROMPTS:-8}" \
  --rounds "${ROUNDS:-3}" \
  --window "${WINDOW:-64}" \
  --output ssv4d/results/dynamic_gate_probe_3round.json

python -m ssv4d.analyze_dynamic_gate \
  ssv4d/results/dynamic_gate_probe_3round.json \
  --output ssv4d/results/dynamic_gate_probe_summary.json
