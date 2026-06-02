#!/bin/sh
# Assign dynamic flags to GT KITTI labels from dynamic_mask.txt (Step 3).

Dynamic_Atrribute(){
cd ..
cd tools/Dyanmic_Attribute

ROOT_DIRNAME="${ROOT_DIRNAME:-/media/zliu/data12/dataset/KITTI/KITTI360_For_Upload}"
CKPT_DIRNAME="${CKPT_DIRNAME:-/home/zliu/IJCV/VSRD_plus_plus/trainer/ckpts}"
JSON_FOLDERNAME="${JSON_FOLDERNAME:-predictions}"
OUTPUT_LABELNAME="${OUTPUT_LABELNAME:-GT_with_dynamic}"
DYNAMIC_DIRNAME="${DYNAMIC_DIRNAME:-${ROOT_DIRNAME}/dynamic_attributes_est_gt}"
NUM_WORKERS="${NUM_WORKERS:-4}"

python get_gt_with_dynamic_label.py \
    --root_dirname "$ROOT_DIRNAME" \
    --ckpt_dirname "$CKPT_DIRNAME" \
    --num_workers "$NUM_WORKERS" \
    --json_foldername "$JSON_FOLDERNAME" \
    --output_labelname "$OUTPUT_LABELNAME" \
    --dyanmic_root_filename "$DYNAMIC_DIRNAME"
}

Dynamic_Atrribute
