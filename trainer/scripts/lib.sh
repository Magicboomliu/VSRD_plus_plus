#!/usr/bin/env bash
# Shared helpers for trainer/scripts/*.sh

# ── paths (set by caller before sourcing, or via init_script_paths) ────────────

init_script_paths() {
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
  TRAINER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
  PROJECT_ROOT="$(cd "${TRAINER_DIR}/.." && pwd)"
}

# ── pixi: re-exec once under pixi env ────────────────────────────────────────

ensure_pixi() {
  local script_relpath="$1"
  shift
  if [ -n "${VSRD_PIXI_WRAPPED:-}" ] || [ "${VSRD_SKIP_PIXI:-0}" = "1" ]; then
    return 0
  fi
  if ! command -v pixi >/dev/null 2>&1; then
    echo "[lib.sh] pixi not found; run from repo root after: pixi install" >&2
    exit 1
  fi
  export VSRD_PIXI_WRAPPED=1
  cd "${PROJECT_ROOT}"
  exec pixi run bash "${script_relpath}" "$@"
}

# ── .env (WANDB_API_KEY, WANDB_PROJECT, …) ───────────────────────────────────

load_dotenv() {
  local env_file="${PROJECT_ROOT}/.env"
  if [ -f "${env_file}" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
  fi
}

# ── wandb CLI flags ──────────────────────────────────────────────────────────

build_wandb_args() {
  WANDB_ARGS=()
  if [ "${USE_WANDB:-${WANDB:-0}}" = "1" ]; then
    WANDB_ARGS+=(--wandb)
    [ -n "${WANDB_PROJECT:-}" ] && WANDB_ARGS+=(--wandb_project "$WANDB_PROJECT")
    [ -n "${WANDB_ENTITY:-}" ] && WANDB_ARGS+=(--wandb_entity "$WANDB_ENTITY")
    [ -n "${WANDB_NAME:-}" ] && WANDB_ARGS+=(--wandb_name "$WANDB_NAME")
    [ -n "${WANDB_TAGS:-}" ] && WANDB_ARGS+=(--wandb_tags "$WANDB_TAGS")
    if [ "${WANDB_LOG_IMAGES:-0}" = "1" ]; then
      WANDB_ARGS+=(--wandb_log_images)
    fi
  fi
}

# ── torchrun launcher ─────────────────────────────────────────────────────────
# Required variables (set by preset functions before calling):
#   TRAIN_SCRIPT, CONFIG_PATH, DEVICE_ID, CUDA_DEVICES,
#   RDZV_ENDPOINT, NPROC_PER_NODE
# Optional: CKPT_DIRNAME, LOG_DIRNAME, OUT_DIRNAME, EXTRA_ARGS (array)

run_train_job() {
  build_wandb_args

  local train_entry="${TRAIN_SCRIPT}"
  local -a launcher=(torchrun)
  if [ "${PIXI_WRAP:-0}" = "1" ]; then
    cd "${PROJECT_ROOT}"
    train_entry="trainer/${TRAIN_SCRIPT}"
    launcher=(pixi run torchrun)
  else
    cd "${TRAINER_DIR}"
  fi

  local -a extra=()
  if [ -n "${CKPT_DIRNAME:-}" ]; then extra+=(--ckpt_dirname "$CKPT_DIRNAME"); fi
  if [ -n "${LOG_DIRNAME:-}" ]; then extra+=(--log_dirname "$LOG_DIRNAME"); fi
  if [ -n "${OUT_DIRNAME:-}" ]; then extra+=(--out_dirname "$OUT_DIRNAME"); fi
  if [ -n "${ERODE_RATIO:-}" ]; then extra+=(--erode_ratio "$ERODE_RATIO"); fi
  if [ -n "${SAVED_CKPT_PATH:-}" ]; then extra+=(--saved_ckpt_path "$SAVED_CKPT_PATH"); fi

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${launcher[@]}" \
    --rdzv_backend c10d \
    --rdzv_endpoint "${RDZV_ENDPOINT}" \
    --nnodes 1 \
    --nproc_per_node "${NPROC_PER_NODE}" \
    "${train_entry}" \
    --config_path "${CONFIG_PATH}" \
    --device_id "${DEVICE_ID}" \
    "${extra[@]}" \
    "${WANDB_ARGS[@]}" \
    ${EXTRA_ARGS:+${EXTRA_ARGS[@]}}
}
