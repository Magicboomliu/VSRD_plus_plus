Convert_To_KITTI_Format(){
cd ..
cd tools/Predictions
ROOT_DIRNAME="/media/zliu/data12/dataset/KITTI/KITTI360_For_Upload"
CKPT_DIRNAME="/home/Journals2025/VSRD_plus_plus/trainer/ckpts/Prefect_GT_Masks"
JSON_FOLDERNAME="predictions"
OUTPUT_LABELNAME='perfect_prediction'
NUM_WORKERS=4


python convert_prediction.py \
    --root_dirname $ROOT_DIRNAME \
    --ckpt_dirname $CKPT_DIRNAME \
    --num_workers $NUM_WORKERS \
    --json_foldername $JSON_FOLDERNAME \
    --output_labelname $OUTPUT_LABELNAME

}

Convert_To_KITTI_Format