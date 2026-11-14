#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export PATH="$repo_root/.venv/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/home/xiaosibo/.local/llm42-numa/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export SSV4D_RUNTIME_ENABLE=1
export SSV4D_GATE_ENABLE=1
export SSV4D_DYNAMIC_GATE_ENABLE=1
export SSV4D_DYNAMIC_AUDIT=1
export SSV4D_REPLAY_SHADOW_ENABLE=1

run_case() {
  local name="$1"
  local prompts="$2"
  local tokens="$3"
  local rounds="$4"
  local prompt_set="$5"
  local extra_tokens="${6:-}"
  export SSV4D_RUNTIME_LOG="$repo_root/ssv4d/logs/${name}.jsonl"
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
    >"$repo_root/ssv4d/logs/${name}_server.log" 2>&1 &
  local server_pid=$!

  for _ in $(seq 1 150); do
    if curl -fsS http://127.0.0.1:30000/v1/models >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done

  python -m ssv4d.runtime_benchmark \
    --prompt-set "$prompt_set" \
    --num-prompts "$prompts" \
    --max-new-tokens "$tokens" \
    --rounds "$rounds" \
    --output "$repo_root/ssv4d/results/${name}.json"

  if [[ -n "$extra_tokens" ]]; then
    python -m ssv4d.runtime_benchmark \
      --prompt-set "$prompt_set" \
      --num-prompts "$prompts" \
      --max-new-tokens "$extra_tokens" \
      --rounds 2 \
      --output "$repo_root/ssv4d/results/${name}_${extra_tokens}.json"
  fi

  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}

run_case r2_batch8 8 130 "${ROUNDS:-3}" standard
run_case r2_batch16 16 130 "${ROUNDS:-3}" standard
run_case r2_multi_audit 48 130 "${ROUNDS:-3}" extended 194

python -m ssv4d.analyze_route2 \
  "$repo_root/ssv4d/logs/r2_batch8.jsonl" \
  "$repo_root/ssv4d/logs/r2_batch16.jsonl" \
  "$repo_root/ssv4d/logs/r2_multi_audit.jsonl" \
  --output "$repo_root/ssv4d/results/route2_batch_stability_summary.json"
