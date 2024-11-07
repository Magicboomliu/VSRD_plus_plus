Convert_To_KITTI_Format(){
cd ..
cd tools/Predictions
ROOT_DIRNAME="/media/zliu/data12/dataset/VSRD_PP_Sync/"
CKPT_DIRNAME="/media/zliu/data12/dataset/Tsubame_Results/V1/ckpts/"
JSON_FOLDERNAME="predictions"
OUTPUT_LABELNAME='label_vsrd_pp_full_completed'
NUM_WORKERS=4


python convert_prediction.py \
    --root_dirname $ROOT_DIRNAME \
    --ckpt_dirname $CKPT_DIRNAME \
    --num_workers $NUM_WORKERS \
    --json_foldername $JSON_FOLDERNAME \
    --output_labelname $OUTPUT_LABELNAME

}

Convert_To_KITTI_Format