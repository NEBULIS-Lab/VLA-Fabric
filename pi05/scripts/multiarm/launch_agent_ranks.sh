#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || "$4" != "--" ]]; then
  echo "usage: $0 GPU_IDS PORT LOG_DIR -- COMMAND [ARGS...]" >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "$1"
PORT="$2"
LOG_DIR="$3"
shift 4
AGENT_COUNT="${#GPUS[@]}"
if [[ "$AGENT_COUNT" -ne 3 && "$AGENT_COUNT" -ne 4 ]]; then
  echo "GPU_IDS must contain exactly three or four physical GPU IDs" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$LOG_DIR"
PIDS=()

cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

for ((RANK=0; RANK<AGENT_COUNT; RANK++)); do
  (
    cd "$ROOT"
    export CUDA_VISIBLE_DEVICES="${GPUS[$RANK]}"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export PYTHONPATH="$ROOT/src:${OPENPI_ROOT:?set OPENPI_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
    "$@" \
      --distributed-process-count "$AGENT_COUNT" \
      --distributed-process-id "$RANK" \
      --distributed-coordinator-address "127.0.0.1:${PORT}"
  ) >"$LOG_DIR/rank${RANK}.log" 2>&1 &
  PIDS+=("$!")
done

STATUS=0
REMAINING="${#PIDS[@]}"
while (( REMAINING > 0 )); do
  if wait -n; then
    REMAINING=$((REMAINING - 1))
    continue
  fi
  STATUS=$?
  cleanup
  wait || true
  break
done
PIDS=()
trap - EXIT INT TERM
exit "$STATUS"
