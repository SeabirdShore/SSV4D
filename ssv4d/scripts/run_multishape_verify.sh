#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export PATH="$repo_root/.venv/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/home/xiaosibo/.local/llm42-numa/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export SSV4D_RUNTIME_ENABLE=1
export SSV4D_FRONTIER_STRICT=1

max_new_tokens="${1:-1024}"
num_prompts="${2:-32}"
rounds="${3:-1}"

run_case() {
  local label="$1"
  local gate="$2"
  local windows="$3"

  export SSV4D_GATE_ENABLE="$gate"
  export SSV4D_DYNAMIC_GATE_ENABLE="$gate"
  export SSV4D_ADAPTIVE_WINDOW_ENABLE="$gate"
  export SSV4D_SAFE_WINDOW_ENABLE="$gate"
  export SSV4D_REPLAY_SHADOW_ENABLE=0
  export SSV4D_SAFE_WINDOW_MODEL="$repo_root/ssv4d/results/safe_window_model.json"
  export SSV4D_SAFE_WINDOW_THRESHOLD=0.1
  export SSV4D_VERIFY_WINDOW_SIZES="$windows"
  export SSV4D_RUNTIME_LOG="$repo_root/ssv4d/logs/${label}.jsonl"
  : >"$SSV4D_RUNTIME_LOG"

  python -m ssv4d.launch_runtime_server \
    --model-path NousResearch/Meta-Llama-3.1-8B-Instruct \
    --host 127.0.0.1 \
    --port 30000 \
    --tp 1 \
    --enable-llm42 2 \
    --llm42-window-size 64 \
    --llm42-verify-batch-size 8 \
    --attention-backend fa3 \
    --disable-radix-cache \
    --disable-chunked-prefix-cache \
    --disable-overlap-schedule \
    --chunked-prefill-size -1 \
    --random-seed 42 \
    >"$repo_root/ssv4d/logs/${label}_server.log" 2>&1 &
  local server_pid=$!
  for _ in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:30000/v1/models >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
  python -m ssv4d.runtime_benchmark \
    --prompt-set extended \
    --num-prompts "$num_prompts" \
    --max-new-tokens "$max_new_tokens" \
    --rounds "$rounds" \
    --output "$repo_root/ssv4d/results/${label}.json"
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}

run_case "mode2_fixed64_${max_new_tokens}_${num_prompts}" 0 64
run_case "mode2_multishape_${max_new_tokens}_${num_prompts}" 1 16,32,48,64

compare_round=$((rounds > 1 ? rounds - 1 : 0))
python -m ssv4d.compare_runtime_runs \
  "$repo_root/ssv4d/results/mode2_fixed64_${max_new_tokens}_${num_prompts}.json" \
  "$repo_root/ssv4d/results/mode2_multishape_${max_new_tokens}_${num_prompts}.json" \
  --round "$compare_round" \
  --output "$repo_root/ssv4d/results/mode2_multishape_${max_new_tokens}_${num_prompts}_compare.json"
