#!/usr/bin/env bash
# Launch Stage-1 training with torchrun.
# Usage (from this directory):
#   source train.sh && run_with_init
#   source train.sh && run_no_init

run_with_init() {
  cd ..
  CUDA_VISIBLE_DEVICES=0 torchrun \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:22500 \
    --nnodes 1 \
    --nproc_per_node 1 \
    train.py --config_path ablation_selective \
    --device_id 0
}

run_no_init() {
  cd ..
  CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:29501 \
    --nnodes 1 \
    --nproc_per_node 2 \
    train_no_init.py --config_path test \
    --device_id 0
}

# Default entry when executed directly (edit to switch mode)
run_with_init
