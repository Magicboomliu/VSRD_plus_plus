#!/usr/bin/env bash
# Mask-erode ablation (train_ablation.py).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${TRAINER_DIR}/.." && pwd)"

# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"
ensure_pixi "trainer/scripts/train_ablation.sh" "$@"
load_dotenv

run_ablation() {
  CONFIG_PATH="${CONFIG_PATH:-ablation_selective}"
  DEVICE_ID="${DEVICE_ID:-0}"
  ERODE_RATIO="${ERODE_RATIO:-0.03}"
  CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  CKPT_DIRNAME="${CKPT_DIRNAME:-}"
  LOG_DIRNAME="${LOG_DIRNAME:-}"
  OUT_DIRNAME="${OUT_DIRNAME:-}"

  TRAIN_SCRIPT="train_ablation.py"
  RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:22500}"
  NPROC_PER_NODE=1
  run_train_job
}

run_ablation "$@"
