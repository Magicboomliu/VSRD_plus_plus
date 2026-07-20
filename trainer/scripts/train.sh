#!/usr/bin/env bash
# 用法：pixi run bash trainer/scripts/train.sh vsrdpp_sequentials/vsrd_plus_full_seq_10
# 改参数：编辑 trainer/configs/experiment/<name>.yaml（每个文件是完整配置）

set -euo pipefail

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

train() {
  python "${SCRIPT_DIR}/launch_train.py" "$@"
}

train "${1:-vsrdpp_sequentials/vsrd_plus_full_seq_10}" "${@:2}"
