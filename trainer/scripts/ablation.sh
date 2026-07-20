SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -z "${VSRD_PIXI_WRAPPED:-}" ] && [ "${VSRD_SKIP_PIXI:-0}" != "1" ]; then
  export VSRD_PIXI_WRAPPED=1
  cd "${PROJECT_ROOT}"
  exec pixi run bash trainer/scripts/ablation.sh "$@"
fi

if [ -f "${PROJECT_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi

# ── 跑哪个 study ──────────────────────────────────────────────────────────
# ablations/vsrd_projection_only
# ablations/vsrd_projection_silhouette
# ablations/vsrd_projection_silhouette_rdf
# ablations/vsrdpp_velocity_no_init
# ablations/vsrdpp_full
STUDY=ablations/vsrdpp_full


DEVICE_ID=0
CUDA_DEVICES=0
NPROC_PER_NODE=1
RDZV_ENDPOINT=localhost:22500

SAVED_ROOT_PATH=/media/zliu/data12/IJCV/ablations
# wandb：0=关，1=开（project/entity/name 从 .env 读取）
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


# Project Loss Only
vsrd_projection_only() {
  CKPT_ROOT="${SAVED_ROOT_PATH}/projection_only/ckpts"
  LOG_ROOT="${SAVED_ROOT_PATH}/projection_only/logs"
  OUT_ROOT="${SAVED_ROOT_PATH}/projection_only/outs"

  TRAIN_EXTRA=(
    --ckpt_dirname "${CKPT_ROOT}"
    --log_dirname "${LOG_ROOT}"
    --out_dirname "${OUT_ROOT}"
  )
  _append_train_extra TRAIN_EXTRA "$@"

  python "${SCRIPT_DIR}/launch_train.py" \
    --config_path ablations/vsrd_projection_only \
    --device_id "${DEVICE_ID}" \
    --cuda_devices "${CUDA_DEVICES}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --rdzv_endpoint "${RDZV_ENDPOINT}" \
    -- "${TRAIN_EXTRA[@]}"
}


# Project Loss + Silhouette
vsrd_projection_silhouette() {
  CKPT_ROOT="${SAVED_ROOT_PATH}/projection_silhouette/ckpts"
  LOG_ROOT="${SAVED_ROOT_PATH}/projection_silhouette/logs"
  OUT_ROOT="${SAVED_ROOT_PATH}/projection_silhouette/outs"

  TRAIN_EXTRA=(
    --ckpt_dirname "${CKPT_ROOT}"
    --log_dirname "${LOG_ROOT}"
    --out_dirname "${OUT_ROOT}"
  )
  _append_train_extra TRAIN_EXTRA "$@"

  python "${SCRIPT_DIR}/launch_train.py" \
    --config_path ablations/vsrd_projection_silhouette \
    --device_id "${DEVICE_ID}" \
    --cuda_devices "${CUDA_DEVICES}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --rdzv_endpoint "${RDZV_ENDPOINT}" \
    -- "${TRAIN_EXTRA[@]}"
}


# Project Loss + Silhouette + RDF
vsrd_projection_silhouette_rdf() {
  CKPT_ROOT="${SAVED_ROOT_PATH}/projection_silhouette_rdf/ckpts"
  LOG_ROOT="${SAVED_ROOT_PATH}/projection_silhouette_rdf/logs"
  OUT_ROOT="${SAVED_ROOT_PATH}/projection_silhouette_rdf/outs"

  TRAIN_EXTRA=(
    --ckpt_dirname "${CKPT_ROOT}"
    --log_dirname "${LOG_ROOT}"
    --out_dirname "${OUT_ROOT}"
  )
  _append_train_extra TRAIN_EXTRA "$@"

  python "${SCRIPT_DIR}/launch_train.py" \
    --config_path ablations/vsrd_projection_silhouette_rdf \
    --device_id "${DEVICE_ID}" \
    --cuda_devices "${CUDA_DEVICES}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --rdzv_endpoint "${RDZV_ENDPOINT}" \
    -- "${TRAIN_EXTRA[@]}"
}

# Project Loss + Silhouette + RDF + No Initial Velocity
vsrdpp_velocity_no_init() {
  CKPT_ROOT="${SAVED_ROOT_PATH}/vsrdpp_velocity_no_init/ckpts"
  LOG_ROOT="${SAVED_ROOT_PATH}/vsrdpp_velocity_no_init/logs"
  OUT_ROOT="${SAVED_ROOT_PATH}/vsrdpp_velocity_no_init/outs"

  TRAIN_EXTRA=(
    --ckpt_dirname "${CKPT_ROOT}"
    --log_dirname "${LOG_ROOT}"
    --out_dirname "${OUT_ROOT}"
  )
  _append_train_extra TRAIN_EXTRA "$@"

  python "${SCRIPT_DIR}/launch_train.py" \
    --config_path ablations/vsrdpp_velocity_no_init \
    --device_id "${DEVICE_ID}" \
    --cuda_devices "${CUDA_DEVICES}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --rdzv_endpoint "${RDZV_ENDPOINT}" \
    -- "${TRAIN_EXTRA[@]}"
}


# Project Loss + Silhouette + RDF + No Initial Velocity + Full VSRD++
vsrdpp_full() {

  CKPT_ROOT="${SAVED_ROOT_PATH}/vsrdpp_full/ckpts"
  LOG_ROOT="${SAVED_ROOT_PATH}/vsrdpp_full/logs"
  OUT_ROOT="${SAVED_ROOT_PATH}/vsrdpp_full/outs"

  TRAIN_EXTRA=(
    --ckpt_dirname "${CKPT_ROOT}"
    --log_dirname "${LOG_ROOT}"
    --out_dirname "${OUT_ROOT}"
  )
  _append_train_extra TRAIN_EXTRA "$@"

  python "${SCRIPT_DIR}/launch_train.py" \
    --config_path ablations/vsrdpp_full \
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

# wandb run name 默认跟随 STUDY（避免 .env 里固定 vsrdpp_full 导致名实不符）
# 若要坚持用 .env 的 WANDB_NAME，启动前设 WANDB_USE_ENV_NAME=1
if [ "${WANDB_USE_ENV_NAME:-0}" != "1" ]; then
  WANDB_NAME="${STUDY##*/}"
  export WANDB_NAME
fi

case "${STUDY}" in
  ablations/vsrd_projection_only|vsrd_projection_only) vsrd_projection_only "$@" ;;
  ablations/vsrd_projection_silhouette|vsrd_projection_silhouette) vsrd_projection_silhouette "$@" ;;
  ablations/vsrd_projection_silhouette_rdf|vsrd_projection_silhouette_rdf) vsrd_projection_silhouette_rdf "$@" ;;
  ablations/vsrdpp_velocity_no_init|vsrdpp_velocity_no_init) vsrdpp_velocity_no_init "$@" ;;
  ablations/vsrdpp_full|vsrdpp_full) vsrdpp_full "$@" ;;
  *)
    echo "STUDY=${STUDY} 无效，请设为：" >&2
    echo "  ablations/vsrd_projection_only" >&2
    echo "  ablations/vsrd_projection_silhouette" >&2
    echo "  ablations/vsrd_projection_silhouette_rdf" >&2
    echo "  ablations/vsrdpp_velocity_no_init" >&2
    echo "  ablations/vsrdpp_full" >&2
    exit 1
    ;;
esac
