#!/usr/bin/env bash
# Stage1 VSRD++ full 训练（Casual / VSRD24 两套帧列表，模型配置相同）
#
# 用法：
#   bash trainer/scripts/train.sh                    # 默认 vsrdpp_casual
#   bash trainer/scripts/train.sh vsrdpp_vsrd24
#   bash trainer/scripts/train.sh vsrdpp_casual -- --ckpt_dirname /path/ckpts

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -z "${VSRD_PIXI_WRAPPED:-}" ] && [ "${VSRD_SKIP_PIXI:-0}" != "1" ]; then
  export VSRD_PIXI_WRAPPED=1
  cd "${PROJECT_ROOT}"
  exec pixi run bash trainer/scripts/train.sh "$@"
fi

if [ -f "${PROJECT_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi

# ── 跑哪个 study ──────────────────────────────────────────────────────────
# vsrdpp_casual   → cascual_splits（4143 帧）
# vsrdpp_vsrd24   → vsrd24_splits（6901 帧）
STUDY=vsrdpp_vsrd24 

DEVICE_ID=0
CUDA_DEVICES=0
NPROC_PER_NODE=1
RDZV_ENDPOINT=localhost:22500

SAVED_ROOT_PATH=/media/zliu/data12/IJCV/stage1/vsrdpp
USE_WANDB=1
WANDB_LOG_IMAGES=1

_append_train_extra() {
  local -n _extra=$1
  shift

  if [ "${USE_WANDB}" = "1" ]; then
    _extra+=(--wandb)
    [ -n "${WANDB_PROJECT:-}" ] && _extra+=(--wandb_project "${WANDB_PROJECT}")
    [ -n "${WANDB_ENTITY:-}" ] && _extra+=(--wandb_entity "${WANDB_ENTITY}")
    [ -n "${WANDB_NAME:-}" ] && _extra+=(--wandb_name "${WANDB_NAME}")
    if [ "${WANDB_LOG_IMAGES:-0}" = "1" ]; then
      _extra+=(--wandb_log_images)
    fi
  fi

  if [ "${1:-}" = "--" ]; then
    shift
  fi
  _extra+=("$@")
}

# VSRD++ Full + Casual split
vsrdpp_casual() {
  CKPT_ROOT="${SAVED_ROOT_PATH}/vsrdpp_casual/ckpts"
  LOG_ROOT="${SAVED_ROOT_PATH}/vsrdpp_casual/logs"
  OUT_ROOT="${SAVED_ROOT_PATH}/vsrdpp_casual/outs"

  TRAIN_EXTRA=(
    --ckpt_dirname "${CKPT_ROOT}"
    --log_dirname "${LOG_ROOT}"
    --out_dirname "${OUT_ROOT}"
  )
  _append_train_extra TRAIN_EXTRA "$@"

  python "${SCRIPT_DIR}/launch_train.py" \
    --config_path stage1_trainfiles/vsrdpp_full_casual \
    --device_id "${DEVICE_ID}" \
    --cuda_devices "${CUDA_DEVICES}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --rdzv_endpoint "${RDZV_ENDPOINT}" \
    -- "${TRAIN_EXTRA[@]}"
}

# VSRD++ Full + VSRD24 split
vsrdpp_vsrd24() {
  CKPT_ROOT="${SAVED_ROOT_PATH}/vsrdpp_vsrd24/ckpts"
  LOG_ROOT="${SAVED_ROOT_PATH}/vsrdpp_vsrd24/logs"
  OUT_ROOT="${SAVED_ROOT_PATH}/vsrdpp_vsrd24/outs"

  TRAIN_EXTRA=(
    --ckpt_dirname "${CKPT_ROOT}"
    --log_dirname "${LOG_ROOT}"
    --out_dirname "${OUT_ROOT}"
  )
  _append_train_extra TRAIN_EXTRA "$@"

  python "${SCRIPT_DIR}/launch_train.py" \
    --config_path stage1_trainfiles/vsrdpp_full_vsrd24 \
    --device_id "${DEVICE_ID}" \
    --cuda_devices "${CUDA_DEVICES}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --rdzv_endpoint "${RDZV_ENDPOINT}" \
    -- "${TRAIN_EXTRA[@]}"
}

if [ $# -gt 0 ] && [ "$1" != "--" ]; then
  STUDY="$1"
  shift
fi

if [ "${WANDB_USE_ENV_NAME:-0}" != "1" ]; then
  WANDB_NAME="${STUDY##*/}"
  export WANDB_NAME
fi

case "${STUDY}" in
  vsrdpp_casual|stage1_trainfiles/vsrdpp_full_casual) vsrdpp_casual "$@" ;;
  vsrdpp_vsrd24|stage1_trainfiles/vsrdpp_full_vsrd24) vsrdpp_vsrd24 "$@" ;;
  *)
    echo "STUDY=${STUDY} 无效，请设为：" >&2
    echo "  vsrdpp_casual" >&2
    echo "  vsrdpp_vsrd24" >&2
    exit 1
    ;;
esac
