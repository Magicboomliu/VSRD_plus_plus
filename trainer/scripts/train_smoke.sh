#!/usr/bin/env bash
# Single-GPU smoke test for train.py (uses smoke.json → sequence_00, 50 steps)

CONFIG_PATH="${CONFIG_PATH:-smoke}"
DEVICE_ID="${DEVICE_ID:-0}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" torchrun \
  --rdzv_backend c10d \
  --rdzv_endpoint localhost:29500 \
  --nnodes 1 \
  --nproc_per_node 1 \
  train.py --config_path "$CONFIG_PATH" \
  --device_id "$DEVICE_ID"
