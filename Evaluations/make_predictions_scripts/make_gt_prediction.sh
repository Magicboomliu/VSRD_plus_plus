Make_Json_Prediction(){
cd ..
cd tools/Predictions
ROOT_DIRNAME="/media/zliu/data12/dataset/VSRD_PP_Sync/"
CKPT_DIRNAME="/home/zliu/CVPR25_Ablations/Ablations/VSRD_Velocity_Only/ckpts/VSRDPP_Abaltions_Velocity_ONLY_V2/"
DYNAMIC_DIRNAME="/media/zliu/data12/dataset/VSRD_PP_Sync/est_dynamic_list/"
INPUT_MODEL_TYPE="velocity_only"

NUM_WORKERS=4

python make_gt_predictions.py \
    --root_dirname $ROOT_DIRNAME \
    --ckpt_dirname $CKPT_DIRNAME \
    --num_workers $NUM_WORKERS \
    --dyanmic_root_filename $DYNAMIC_DIRNAME \
    --input_model_type $INPUT_MODEL_TYPE


}

Make_Json_Prediction