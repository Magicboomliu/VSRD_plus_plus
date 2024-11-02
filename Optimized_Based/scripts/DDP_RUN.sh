# using the initialization
TRAIN_DDP_VSRDPP(){
cd ..
CUDA_VISIBLE_DEVICES=0 torchrun \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:22500 \
    --nnodes 1 \
    --nproc_per_node 1 \
    train_sequence_ddp.py --config_path "ablation_ours" \
    --device_id 0

}

# without using the initialization
TRAIN_DDP_VSRD_SIMPLE(){
cd ..
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:29501 \
    --nnodes 1 \
    --nproc_per_node 2 \
    train_without_initial_sequence_ddp.py --config_path "test" \
    --device_id 0


}

# TRAIN_DDP_VSRD_SIMPLE
TRAIN_DDP_VSRDPP