#!/bin/sh
# Generate dynamic_mask.txt from GT bbox velocity (no images / depth).
#
# Output (default):
#   {DATASET_ROOT}/dynamic_attributes_est_gt/<sequence>/dynamic_mask.txt
#
# Usage (from project root):
#   sh preprocessing/scripts/generate_dynamic_labels.sh              # all sequences
#   sh preprocessing/scripts/generate_dynamic_labels.sh sequence_07    # one config
#
# Override threshold (m/frame):
#   THRESHOLD=0.20 sh preprocessing/scripts/generate_dynamic_labels.sh sequence_00

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

THRESHOLD="${THRESHOLD:-0.20}"
CONFIG="${1:-}"

ALL_CONFIGS="sequence_00 sequence_02 sequence_03 sequence_04 sequence_05 sequence_06 sequence_07 sequence_09 sequence_10"

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

run_one() {
    cfg="$1"
    echo "=== ${cfg} (threshold=${THRESHOLD} m/frame) ==="
    PYTHONPATH="${PROJECT_ROOT}" python -m preprocessing.Dynamic_Labels.pipeline \
        --config "${cfg}" \
        --threshold "${THRESHOLD}"
}

if [ -n "${CONFIG}" ]; then
    run_one "${CONFIG}"
else
    for cfg in ${ALL_CONFIGS}; do
        run_one "${cfg}"
    done
fi
