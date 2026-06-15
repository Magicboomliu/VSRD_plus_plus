#!/usr/bin/env bash
# 64-way split training (train_sharded.py)

CONFIG_PATH="${CONFIG_PATH:-48}"
DEVICE_ID="${DEVICE_ID:-0}"
SAVED_CKPT_PATH="${SAVED_CKPT_PATH:-}"

cd ..
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" torchrun \
  --rdzv_backend c10d \
  --rdzv_endpoint localhost:29501 \
  --nnodes 1 \
  --nproc_per_node 1 \
  train_sharded.py \
  --config_path "$CONFIG_PATH" \
  --device_id "$DEVICE_ID" \
  ${SAVED_CKPT_PATH:+--saved_ckpt_path "$SAVED_CKPT_PATH"}
