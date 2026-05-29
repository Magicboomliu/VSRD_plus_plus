#!/bin/sh
# Generate pseudo depth maps using WAFT-Stereo.
#
# Must run inside the WAFT pixi environment (PyTorch >= 2.0).
#
# Output location:
#   {DATASET_ROOT}/{OUTPUT_NAME}/{sequence}/image_00/data_rect/{frame}.png
#
# Defaults:
#   DATASET_ROOT=/media/zliu/data12/dataset/KITTI/KITTI360_For_Upload
#   MODEL_NAME=WAFT-Stereo
#   OUTPUT_NAME=pseudo_depth_ssl_waft_stereo   (derived from MODEL_NAME)
#
# Usage (from project root):
#   sh preprocessing/scripts/generate_pseudo_depth_waft.sh              # all sequences
#   sh preprocessing/scripts/generate_pseudo_depth_waft.sh 0006         # one sequence
#
# Override output directory name:
#   OUTPUT_NAME=pseudo_depth_ssl_waft sh preprocessing/scripts/generate_pseudo_depth_waft.sh 0006
#
# Override model (output name follows unless OUTPUT_NAME is set):
#   MODEL_NAME=WAFT sh preprocessing/scripts/generate_pseudo_depth_waft.sh 0006
#
# Custom dataset root:
#   DATASET_ROOT=/path/to/dataset sh preprocessing/scripts/generate_pseudo_depth_waft.sh

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WAFT_DIR="${PROJECT_ROOT}/preprocessing/disparity_estimation/WAFT-Stereo"

DATASET_ROOT="${DATASET_ROOT:-/media/zliu/data12/dataset/KITTI/KITTI360_For_Upload}"
MODEL_NAME="${MODEL_NAME:-WAFT-Stereo}"
DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
SEQ="${1:-}"

if [ -z "${OUTPUT_NAME:-}" ]; then
    OUTPUT_SLUG="$(echo "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
    OUTPUT_NAME="pseudo_depth_ssl_${OUTPUT_SLUG}"
fi

cd "${WAFT_DIR}"

echo "Model:  ${MODEL_NAME}"
echo "Input:  ${DATASET_ROOT}/data_2d_raw/<sequence>/image_00/data_rect/"
echo "Output: ${DATASET_ROOT}/${OUTPUT_NAME}/<sequence>/image_00/data_rect/"
if [ -z "${SEQ}" ]; then
    echo "NOTE: no sequence given — will process ALL sequences (very slow). Example: sh $0 0006"
fi

export PYTHONUNBUFFERED=1

if [ -n "${SEQ}" ]; then
    PYTHONPATH="${PROJECT_ROOT}" pixi run python -m preprocessing.apis.depth_estimator \
        --dataset-root "${DATASET_ROOT}" \
        --model-name "${MODEL_NAME}" \
        --output-name "${OUTPUT_NAME}" \
        --device "cuda:${DEVICE}" \
        --seq "${SEQ}"
else
    PYTHONPATH="${PROJECT_ROOT}" pixi run python -m preprocessing.apis.depth_estimator \
        --dataset-root "${DATASET_ROOT}" \
        --model-name "${MODEL_NAME}" \
        --output-name "${OUTPUT_NAME}" \
        --device "cuda:${DEVICE}"
fi
