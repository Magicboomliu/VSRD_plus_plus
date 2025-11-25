# using the initialization
TRAIN_DDP_VSRDPP(){
cd ..
# Configuration
CONFIG_PATH="ablation_selective"
DEVICE_ID=0
# Mask erode ratio (0.0 = disabled, 0.05 = 5%, 0.10 = 10%)
ERODE_RATIO=0.03
# Custom directories (leave empty to use default from config)
CKPT_DIRNAME="/home/Journals2025/VSRD_plus_plus/TEMP/ckpts/mask_enrode_$ERODE_RATIO"
LOG_DIRNAME="/home/Journals2025/VSRD_plus_plus/TEMP/logs/mask_enrode_$ERODE_RATIO"
OUT_DIRNAME="/home/Journals2025/VSRD_plus_plus/TEMP/results/mask_enrode_$ERODE_RATIO"

CUDA_VISIBLE_DEVICES=0 torchrun \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:22500 \
    --nnodes 1 \
    --nproc_per_node 1 \
    train_sequence_ddp_ablations.py \
    --config_path "$CONFIG_PATH" \
    --device_id $DEVICE_ID \
    --erode_ratio $ERODE_RATIO \
    ${CKPT_DIRNAME:+--ckpt_dirname "$CKPT_DIRNAME"} \
    ${LOG_DIRNAME:+--log_dirname "$LOG_DIRNAME"} \
    ${OUT_DIRNAME:+--out_dirname "$OUT_DIRNAME"}

}

TRAIN_DDP_VSRDPP