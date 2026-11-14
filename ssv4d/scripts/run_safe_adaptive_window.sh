#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export PATH="$repo_root/.venv/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/home/xiaosibo/.local/llm42-numa/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export SSV4D_RUNTIME_ENABLE=1
export SSV4D_GATE_ENABLE=1
export SSV4D_DYNAMIC_GATE_ENABLE=1
export SSV4D_ADAPTIVE_WINDOW_ENABLE=1
export SSV4D_SAFE_WINDOW_ENABLE=1
export SSV4D_SAFE_WINDOW_MODEL="$repo_root/ssv4d/results/safe_window_model.json"
export SSV4D_REPLAY_SHADOW_ENABLE=0
export SSV4D_FRONTIER_STRICT=1

max_new_tokens="${1:-130}"
result_suffix="${2:-}"

run_threshold() {
  local threshold="$1"
  local label="$2"
  local token_label="${max_new_tokens}${result_suffix:+_${result_suffix}}"
  export SSV4D_SAFE_WINDOW_THRESHOLD="$threshold"
  export SSV4D_RUNTIME_LOG="$repo_root/ssv4d/logs/safe_adaptive_${label}_${token_label}.jsonl"
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
    >"$repo_root/ssv4d/logs/safe_adaptive_${label}_${token_label}_server.log" 2>&1 &
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
    --max-new-tokens "$max_new_tokens" \
    --rounds 3 \
    --output "$repo_root/ssv4d/results/safe_adaptive_${label}_${token_label}.json"
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}

run_threshold 0.1 window
run_threshold 0.2 window_0p2
run_threshold 0.3 window_0p3
