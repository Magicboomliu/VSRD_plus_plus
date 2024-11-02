
TRAIN_DDP_VSRDPP_SPLIT(){
cd ..
CUDA_VISIBLE_DEVICES=0 torchrun \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:29501 \
    --nnodes 1 \
    --nproc_per_node 1 \
    train_split64.py --config_path "48" \
    --device_id 0 \
    --saved_ckpt_path "/media/zliu/data12/dataset/VSRD_PP_Sync/output_models"

}

TRAIN_DDP_VSRDPP_SPLIT