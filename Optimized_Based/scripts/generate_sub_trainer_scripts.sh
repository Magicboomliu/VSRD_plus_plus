#!/bin/bash

# 初始端口和文件数量
start_port=24500
num_files=64  # 需要生成的脚本数量
output_dir="../generated_scripts"

# 创建输出目录
mkdir -p "$output_dir"

# 循环生成每个脚本
for i in $(seq 0 $((num_files-1)))
do
    # 格式化 i 为两位数字
    formatted_i=$(printf "%02d" $i)
    
    # 计算当前脚本的端口号
    current_port=$((start_port + i))

    # 动态生成脚本内容
    script_content="#!/bin/sh
#$ -cwd
#$ -l node_o=1
#$ -l h_rt=24:00:00
#$ -p -5
#$ -N VSRD_${formatted_i}
#$ -m ae
#$ -M liuzihua1004@gmail.com

# 加载环境
module load cuda/12.0.0 cudnn/9.0.0 ffmpeg/6.1.1
module load nccl/2.20.5
module load intel-mpi/2021.11 openmpi/5.0.2-gcc 
module load forge/23.1.2 intel-vtune/2024.0
module load miniconda/24.1.2 & eval \"\$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)\"

conda activate vsrd

nvidia-smi

cd ..
torchrun \\
    --rdzv_backend c10d \\
    --rdzv_endpoint localhost:${current_port} \\
    --nnodes 1 \\
    --nproc_per_node 1 \\
    train_split64.py --config_path \"${formatted_i}\" \\
    --device_id 0 \\
    --saved_ckpt_path \"/gs/bs/tga-lab_otm/zliu/VSRD_PP_Sync/output_models_ablations\""

    # 保存脚本文件到指定目录
    echo "$script_content" > "$output_dir/run_script_${formatted_i}.sh"
    chmod +x "$output_dir/run_script_${formatted_i}.sh"
    echo "生成的脚本: $output_dir/run_script_${formatted_i}.sh"
done

