#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <checkpoint> <task> <development|disjoint> <output-dir>" >&2
  exit 2
fi

CHECKPOINT="$1"
TASK="$2"
SPLIT="$3"
OUTPUT_DIR="$4"

case "${SPLIT}" in
  development) OFFSET=0 ;;
  disjoint) OFFSET=200 ;;
  *) echo "split must be development or disjoint" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

torchrun --standalone --nproc_per_node=2 \
  -m commvla.evaluation.tabletop_two_process \
  --checkpoint "${CHECKPOINT}" \
  --task-name "${TASK}" \
  --unnorm-key "${TASK}" \
  --benchmark \
  --trials 200 \
  --rollout-offset "${OFFSET}" \
  --action-len 20 \
  --cfg 1.1 \
  --num-denoising-steps 10 \
  --output-dir "${OUTPUT_DIR}"

