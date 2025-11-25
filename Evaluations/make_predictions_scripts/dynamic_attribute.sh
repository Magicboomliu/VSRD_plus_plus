Dynamic_Atrribute(){
cd ..
cd tools/Dyanmic_Attribute
ROOT_DIRNAME="/data/dataset/KITTI/KITTI360_For_Docker"
CKPT_DIRNAME="/home/Journals2025/VSRD_plus_plus/Optimized_Based/ckpts/Prefect_GT_Masks"
JSON_FOLDERNAME="predictions"
OUTPUT_LABELNAME='GT_with_dynamic'
NUM_WORKERS=4
DYNAMIC_THRESHOLD=0.01

python get_gt_with_dynamic_label.py \
    --root_dirname $ROOT_DIRNAME \
    --ckpt_dirname $CKPT_DIRNAME \
    --num_workers $NUM_WORKERS \
    --json_foldername $JSON_FOLDERNAME \
    --output_labelname $OUTPUT_LABELNAME \
    --dynamic_threshold $DYNAMIC_THRESHOLD


}

Dynamic_Atrribute