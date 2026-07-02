#!/usr/bin/env bash
# Single-GPU smoke test (smoke.json, same hyperparams as formal training).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${TRAINER_DIR}/.." && pwd)"

# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"
ensure_pixi "trainer/scripts/train_smoke.sh" "$@"
load_dotenv

run_smoke() {
  CONFIG_PATH="${CONFIG_PATH:-smoke}"
  DEVICE_ID="${DEVICE_ID:-0}"
  CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  CKPT_DIRNAME="${CKPT_DIRNAME:-}"
  LOG_DIRNAME="${LOG_DIRNAME:-}"
  OUT_DIRNAME="${OUT_DIRNAME:-}"

  TRAIN_SCRIPT="train.py"
  RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:29500}"
  NPROC_PER_NODE=1
  run_train_job
}

run_smoke "$@"
