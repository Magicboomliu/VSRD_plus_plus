#!/usr/bin/env bash
# Stage1 ckpt infer：从训练 ckpt 推理导出 PD + GT JSON（legacy Step1a + Step1b）
#
# 用法：
#   bash validator/scripts/ckpt_infer.sh
#   bash validator/scripts/ckpt_infer.sh ckpt_infer_vsrd24
#   bash validator/scripts/ckpt_infer.sh ckpt_infer_casual -- --ckpt_dirname /other/ckpts

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -z "${VSRD_PIXI_WRAPPED:-}" ] && [ "${VSRD_SKIP_PIXI:-0}" != "1" ]; then
  export VSRD_PIXI_WRAPPED=1
  cd "${PROJECT_ROOT}"
  exec pixi run bash validator/scripts/ckpt_infer.sh "$@"
fi

# ckpt_infer_casual  → cascual_splits（4143 帧）
# ckpt_infer_vsrd24  → vsrd24_splits（6901 帧）
STUDY=ckpt_infer_casual

DATASET_ROOT=/media/zliu/data12/dataset/KITTI/KITTI360_For_Upload
CKPT_DIRNAME=/media/zliu/data12/IJCV/ablations/saved_models/vsrdpp_full/ckpts
CKPT_FILENAME=step_2499.pt
OUTPUT_ROOT=/media/zliu/data12/IJCV/outputs/stage1
NUM_WORKERS=4

_append_ckpt_infer_extra() {
  local -n _extra=$1
  shift
  if [ "${1:-}" = "--" ]; then
    shift
  fi
  _extra+=("$@")
}

_ckpt_infer() {
  local split=$1
  shift

  JSON_PD_OUT_DIR="${OUTPUT_ROOT}/${STUDY}/predictions/json"
  JSON_GT_OUT_DIR="${OUTPUT_ROOT}/${STUDY}/predictions/gt"

  CKPT_INFER_EXTRA=(
    --ckpt_dirname "${CKPT_DIRNAME}"
    --ckpt_filename "${CKPT_FILENAME}"
    --json_pd_out_dirname "${JSON_PD_OUT_DIR}"
    --json_gt_out_dirname "${JSON_GT_OUT_DIR}"
  )
  _append_ckpt_infer_extra CKPT_INFER_EXTRA "$@"

  echo "[ckpt_infer] study=${STUDY} split=${split}"
  echo "[ckpt_infer] ckpt=${CKPT_DIRNAME}/${CKPT_FILENAME}"
  echo "[ckpt_infer] pd_json=${JSON_PD_OUT_DIR}/data_2d_raw/..."
  echo "[ckpt_infer] gt_json=${JSON_GT_OUT_DIR}/data_2d_raw/..."

  python "${SCRIPT_DIR}/run_ckpt_infer.py" "${split}" \
    --num_workers "${NUM_WORKERS}" \
    "${CKPT_INFER_EXTRA[@]}"
}

ckpt_infer_casual() { _ckpt_infer casual "$@"; }
ckpt_infer_vsrd24() { _ckpt_infer vsrd24 "$@"; }

if [ $# -gt 0 ] && [ "$1" != "--" ]; then
  STUDY="$1"
  shift
fi

case "${STUDY}" in
  ckpt_infer_casual) ckpt_infer_casual "$@" ;;
  ckpt_infer_vsrd24) ckpt_infer_vsrd24 "$@" ;;
  *)
    echo "STUDY=${STUDY} 无效，请设为：" >&2
    echo "  ckpt_infer_casual" >&2
    echo "  ckpt_infer_vsrd24" >&2
    exit 1
    ;;
esac
