#!/bin/sh
#$ -cwd
#$ -l node_o=1
#$ -l h_rt=24:00:00
#$ -p -5
#$ -N VSRD_25
#$ -m ae
#$ -M liuzihua1004@gmail.com

# 加载环境
module load cuda/12.0.0 cudnn/9.0.0 ffmpeg/6.1.1
module load nccl/2.20.5
module load intel-mpi/2021.11 openmpi/5.0.2-gcc 
module load forge/23.1.2 intel-vtune/2024.0
module load miniconda/24.1.2 & eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"

conda activate vsrd

nvidia-smi

cd ..
torchrun \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:24525 \
    --nnodes 1 \
    --nproc_per_node 1 \
    train_split64.py --config_path "25" \
    --device_id 0 \
    --saved_ckpt_path "/gs/bs/tga-lab_otm/zliu/VSRD_PP_Sync/output_models"
