STATIC_DYNAMIC_EVALUATION(){
cd ..

CUDA_VISIBLE_DEVICES=0 torchrun \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:22500 \
    --nnodes 1 \
    --nproc_per_node 1 \
    static_dynamic_classification.py --config_path "missed" \
    --device_id 0


}


STATIC_DYNAMIC_EVALUATION