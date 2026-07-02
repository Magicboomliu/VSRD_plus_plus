#!/usr/bin/env bash
# Stage-1 training launcher (pixi + torchrun).
#
# Direct run (auto re-exec under pixi):
#   bash trainer/scripts/train.sh
#
# Presets:
#   MODE=init      → train.py with attribute initialization (default)
#   MODE=no_init   → train_no_init.py
#
# Override any variable:
#   CONFIG_PATH=sequence_07 CUDA_VISIBLE_DEVICES=0 USE_WANDB=1 bash trainer/scripts/train.sh
#
# Or source and call a preset:
#   source trainer/scripts/train.sh
#   run_with_init

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${TRAINER_DIR}/.." && pwd)"

# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"
ensure_pixi "trainer/scripts/train.sh" "$@"
load_dotenv

# ── presets ─────────────────────────────────────────────────────────────────

run_with_init() {
  # ====== 基本参数（都可以在这里直接改默认值）======
  CONFIG_PATH="${CONFIG_PATH:-ablation_selective}"
  DEVICE_ID="${DEVICE_ID:-0}"
  CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

  # ====== 输出目录（留空=使用 train.py 内置默认逻辑）======
  # train.py 默认：
  #   ckpt: TRAIN.CONFIG.replace("configs", "ckpts/${MODEL_TYPE}")
  #   log : TRAIN.CONFIG.replace("configs", "logs")
  #   out : TRAIN.CONFIG.replace("configs", "outs")
  CKPT_DIRNAME="${CKPT_DIRNAME:-}"
  LOG_DIRNAME="${LOG_DIRNAME:-}"
  OUT_DIRNAME="${OUT_DIRNAME:-}"

  # ====== wandb（环境变量开关，默认关闭）======
  # USE_WANDB=1 开启；WANDB_LOG_IMAGES=1 开启图像日志
  USE_WANDB="${USE_WANDB:-0}"
  WANDB_LOG_IMAGES="${WANDB_LOG_IMAGES:-0}"

  # ====== torchrun（单卡）======
  TRAIN_SCRIPT="train.py"
  RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:22500}"
  NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
  run_train_job
}

run_no_init() {
  # ====== 基本参数（都可以在这里直接改默认值）======
  CONFIG_PATH="${CONFIG_PATH:-test}"
  DEVICE_ID="${DEVICE_ID:-0}"
  CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

  # ====== 输出目录（留空=使用 train_no_init.py 内置默认逻辑）======
  CKPT_DIRNAME="${CKPT_DIRNAME:-}"
  LOG_DIRNAME="${LOG_DIRNAME:-}"
  OUT_DIRNAME="${OUT_DIRNAME:-}"

  # ====== wandb（环境变量开关，默认关闭）======
  USE_WANDB="${USE_WANDB:-0}"
  WANDB_LOG_IMAGES="${WANDB_LOG_IMAGES:-0}"

  # ====== torchrun（双卡）======
  TRAIN_SCRIPT="train_no_init.py"
  RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:29501}"
  NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
  run_train_job
}

# ── entry ───────────────────────────────────────────────────────────────────

main() {
  # Optional pass-through flags to python entrypoint (e.g. --wandb_name ...).
  # Example:
  #   pixi run bash trainer/scripts/train.sh --wandb_name "my_run"
  EXTRA_ARGS=("$@")
  case "${MODE:-init}" in
    no_init) run_no_init ;;
    init|*)  run_with_init ;;
  esac
}

# Only auto-run when executed, not when sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
