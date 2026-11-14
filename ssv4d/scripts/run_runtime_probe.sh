#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-always}"
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

export PATH="$REPO_ROOT/.venv/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/home/xiaosibo/.local/llm42-numa/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SSV4D_RUNTIME_ENABLE=1
export SSV4D_VERIFY_TOP_K=8
export SSV4D_KV_LAYERS=-1

case "$MODE" in
  always)
    export SSV4D_GATE_ENABLE=0
    export SSV4D_DYNAMIC_GATE_ENABLE=0
    export SSV4D_REPLAY_SHADOW_ENABLE=0
    export SSV4D_ADAPTIVE_WINDOW_ENABLE=0
    ;;
  gated)
    export SSV4D_GATE_ENABLE=1
    export SSV4D_DYNAMIC_GATE_ENABLE=0
    export SSV4D_REPLAY_SHADOW_ENABLE=0
    export SSV4D_ADAPTIVE_WINDOW_ENABLE=0
    export SSV4D_MARGIN_THRESHOLD="${SSV4D_MARGIN_THRESHOLD:-1.0}"
    ;;
  dynamic)
    export SSV4D_GATE_ENABLE=1
    export SSV4D_DYNAMIC_GATE_ENABLE=1
    export SSV4D_DYNAMIC_AUDIT=0
    export SSV4D_REPLAY_SHADOW_ENABLE=1
    export SSV4D_ADAPTIVE_WINDOW_ENABLE=0
    ;;
  dynamic-audit)
    export SSV4D_GATE_ENABLE=1
    export SSV4D_DYNAMIC_GATE_ENABLE=1
    export SSV4D_DYNAMIC_AUDIT=1
    export SSV4D_REPLAY_SHADOW_ENABLE=1
    export SSV4D_ADAPTIVE_WINDOW_ENABLE=0
    ;;
  adaptive)
    export SSV4D_GATE_ENABLE=1
    export SSV4D_DYNAMIC_GATE_ENABLE=1
    export SSV4D_DYNAMIC_AUDIT=0
    export SSV4D_REPLAY_SHADOW_ENABLE=1
    export SSV4D_ADAPTIVE_WINDOW_ENABLE=1
    export SSV4D_ADAPTIVE_CHECKPOINTS="${SSV4D_ADAPTIVE_CHECKPOINTS:-16,32,48,64}"
    ;;
  adaptive-audit)
    export SSV4D_GATE_ENABLE=1
    export SSV4D_DYNAMIC_GATE_ENABLE=1
    export SSV4D_DYNAMIC_AUDIT=1
    export SSV4D_REPLAY_SHADOW_ENABLE=1
    export SSV4D_ADAPTIVE_WINDOW_ENABLE=1
    export SSV4D_ADAPTIVE_CHECKPOINTS="${SSV4D_ADAPTIVE_CHECKPOINTS:-16,32,48,64}"
    ;;
  *)
    echo "Usage: $0 [always|gated|dynamic|dynamic-audit|adaptive|adaptive-audit]" >&2
    exit 2
    ;;
esac

export SSV4D_RUNTIME_LOG="$REPO_ROOT/ssv4d/logs/runtime_${MODE}_rerun.jsonl"
SERVER_LOG="$REPO_ROOT/ssv4d/logs/server_${MODE}_rerun.log"

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
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 150); do
  if curl -fsS http://127.0.0.1:30000/v1/models >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

python -m ssv4d.runtime_benchmark \
  --num-prompts 16 \
  --max-new-tokens 130 \
  --rounds 2 \
  --output "ssv4d/results/runtime_${MODE}_rerun.json"
