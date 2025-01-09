cd ..
torchrun \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:24510 \
    --nnodes 1 \
    --nproc_per_node 1 \
    train_split64_project_loss_only.py --config_path "00" \
    --device_id 0 \
    --saved_ckpt_path "/media/zliu/data12/debug/"
