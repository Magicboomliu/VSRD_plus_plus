TRAIN_DDP_VSRDPP(){
cd ..
CUDA_VISIBLE_DEVICES=0 torchrun \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:29500 \
    --nnodes 1 \
    --nproc_per_node 1 \
    train_dynamic_test.py --config_path "test" \
    --device_id 0

}



TRAIN_DDP_VSRDPP