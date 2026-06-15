#!/usr/bin/env bash
# Mask-erode ablation launcher (train_ablation.py)

CONFIG_PATH="${CONFIG_PATH:-ablation_selective}"
DEVICE_ID="${DEVICE_ID:-0}"
ERODE_RATIO="${ERODE_RATIO:-0.03}"
CKPT_DIRNAME="${CKPT_DIRNAME:-}"
LOG_DIRNAME="${LOG_DIRNAME:-}"
OUT_DIRNAME="${OUT_DIRNAME:-}"

cd ..
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" torchrun \
  --rdzv_backend c10d \
  --rdzv_endpoint localhost:22500 \
  --nnodes 1 \
  --nproc_per_node 1 \
  train_ablation.py \
  --config_path "$CONFIG_PATH" \
  --device_id "$DEVICE_ID" \
  --erode_ratio "$ERODE_RATIO" \
  ${CKPT_DIRNAME:+--ckpt_dirname "$CKPT_DIRNAME"} \
  ${LOG_DIRNAME:+--log_dirname "$LOG_DIRNAME"} \
  ${OUT_DIRNAME:+--out_dirname "$OUT_DIRNAME"}
