#!/usr/bin/env bash
# Tsubame cluster job script for train.py
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=10:00:00
#$ -p -5
#$ -N VSRD_TEST
#$ -m ae
#$ -M liuzihua1004@gmail.com

# Cluster job script (TSUBAME). Uses `pixi run torchrun` from repo root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${TRAINER_DIR}/.." && pwd)"

# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"
load_dotenv

run_tsubame() {
  CONFIG_PATH="${CONFIG_PATH:-sequence_00}"
  DEVICE_ID="${DEVICE_ID:-0}"
  CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

  TRAIN_SCRIPT="train.py"
  RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:29500}"
  NPROC_PER_NODE=1
  PIXI_WRAP=1

  module load cuda/12.0.0 cudnn/9.0.0 ffmpeg/6.1.1
  module load nccl/2.20.5

  nvidia-smi
  run_train_job
}

run_tsubame "$@"
