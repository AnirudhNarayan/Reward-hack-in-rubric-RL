#!/usr/bin/env bash
# start_vllm <hf-model-id> <tensor-parallel> <max-model-len> [extra vllm args...]
#   Starts an OpenAI-compatible vLLM server on this (compute) node in the background, waits for
#   /health, and exports JUDGE_BASE_URL. stop_vllm kills it. Used by grade.slurm, smoke_test.slurm
#   and serve_judge.slurm.
VLLM_PID=""

start_vllm() {
  local model="$1" tp="$2" maxlen="$3"; shift 3
  local port="${VLLM_PORT:-$((8000 + ${SLURM_JOB_ID:-0} % 1000))}"
  local log="${VLLM_LOG:-vllm-${SLURM_JOB_ID:-local}.log}"
  echo "[vllm] serving $model tp=$tp max_len=$maxlen port=$port (log: $log)"
  vllm serve "$model" --tensor-parallel-size "$tp" --port "$port" --host 0.0.0.0 \
       --max-model-len "$maxlen" --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.90}" \
       --enable-prefix-caching "$@" > "$log" 2>&1 &
  VLLM_PID=$!
  local url="http://$(hostname):$port"
  for _ in $(seq 1 180); do                       # up to 30 min (70B load from Lustre)
    if curl -sf "$url/health" >/dev/null 2>&1; then
      export JUDGE_BASE_URL="$url/v1"
      echo "[vllm] ready at $JUDGE_BASE_URL"
      return 0
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "[vllm] server died; tail of log:"; tail -50 "$log"; return 1
    fi
    sleep 10
  done
  echo "[vllm] timed out waiting for /health"; tail -50 "$log"; return 1
}

stop_vllm() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID"; wait "$VLLM_PID" 2>/dev/null || true
  fi
  VLLM_PID=""
}
trap stop_vllm EXIT
