Make_Json_Prediction(){
cd ..
cd tools/Predictions
ROOT_DIRNAME="/media/zliu/data12/dataset/VSRD_PP_Sync/"
CKPT_DIRNAME="/media/zliu/data12/dataset/Tsubame_Results/V1/ckpts/"
DYNAMIC_DIRNAME="/media/zliu/data12/dataset/VSRD_PP_Sync/est_dynamic_list/"
INPUT_MODEL_TYPE="velocity_with_init"

NUM_WORKERS=4

python make_gt_predictions.py \
    --root_dirname $ROOT_DIRNAME \
    --ckpt_dirname $CKPT_DIRNAME \
    --num_workers $NUM_WORKERS \
    --dyanmic_root_filename $DYNAMIC_DIRNAME \
    --input_model_type $INPUT_MODEL_TYPE


}

Make_Json_Prediction