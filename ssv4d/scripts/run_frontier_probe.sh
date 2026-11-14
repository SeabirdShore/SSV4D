#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export PATH="$repo_root/.venv/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/home/xiaosibo/.local/llm42-numa/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export SSV4D_RUNTIME_ENABLE=1
export SSV4D_FRONTIER_STRICT=1

run_case() {
  local mode="$1"
  local gate="$2"
  local threshold="$3"
  export SSV4D_RUNTIME_LOG="$repo_root/ssv4d/logs/frontier_${mode}.jsonl"
  export SSV4D_GATE_ENABLE="$gate"
  export SSV4D_MARGIN_THRESHOLD="$threshold"
  : >"$SSV4D_RUNTIME_LOG"
  python -m ssv4d.launch_runtime_server \
    --model-path NousResearch/Meta-Llama-3.1-8B-Instruct \
    --host 127.0.0.1 \
    --port 30000 \
    --tp 1 \
    --enable-llm42 3 \
    --llm42-window-size 64 \
    --llm42-verify-batch-size 8 \
    --attention-backend fa3 \
    --disable-radix-cache \
    --disable-chunked-prefix-cache \
    --disable-overlap-schedule \
    --chunked-prefill-size -1 \
    --random-seed 42 \
    >"$repo_root/ssv4d/logs/frontier_${mode}_server.log" 2>&1 &
  local server_pid=$!
  for _ in $(seq 1 150); do
    if curl -fsS http://127.0.0.1:30000/v1/models >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
  python -m ssv4d.runtime_benchmark \
      --prompt-set extended \
      --num-prompts 32 \
      --max-new-tokens 130 \
      --rounds 3 \
      --output "$repo_root/ssv4d/results/frontier_${mode}.json"
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}

run_case always 0 0.0625
run_case gated_0p0625 1 0.0625

python -m ssv4d.analyze_frontier \
  "$repo_root/ssv4d/logs/frontier_always.jsonl" \
  "$repo_root/ssv4d/logs/frontier_gated_0p0625.jsonl" \
  --output "$repo_root/ssv4d/results/frontier_audit_summary.json"
