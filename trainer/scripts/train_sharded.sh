#!/usr/bin/env bash
# 64-way split training (train_sharded.py).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${TRAINER_DIR}/.." && pwd)"

# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"
ensure_pixi "trainer/scripts/train_sharded.sh" "$@"
load_dotenv

run_sharded() {
  CONFIG_PATH="${CONFIG_PATH:-48}"
  DEVICE_ID="${DEVICE_ID:-0}"
  SAVED_CKPT_PATH="${SAVED_CKPT_PATH:-}"
  CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

  TRAIN_SCRIPT="train_sharded.py"
  RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:29501}"
  NPROC_PER_NODE=1
  run_train_job
}

run_sharded "$@"
