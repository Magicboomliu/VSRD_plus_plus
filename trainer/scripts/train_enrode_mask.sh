#!/usr/bin/env bash
# Mask erode ablation → unified train.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -z "${VSRD_PIXI_WRAPPED:-}" ] && [ "${VSRD_SKIP_PIXI:-0}" != "1" ]; then
  export VSRD_PIXI_WRAPPED=1
  cd "${PROJECT_ROOT}"
  exec pixi run bash trainer/scripts/train_enrode_mask.sh "$@"
fi

ERODE_RATIO="${ERODE_RATIO:-}"
EXTRA=()
[ -n "${ERODE_RATIO}" ] && EXTRA+=(--erode_ratio "${ERODE_RATIO}")

python "${SCRIPT_DIR}/launch_train.py" \
  "${CONFIG_PATH:-ablations/erode_seg_mask_degradation_vsrdpp}" \
  "${EXTRA[@]}" \
  "$@"
