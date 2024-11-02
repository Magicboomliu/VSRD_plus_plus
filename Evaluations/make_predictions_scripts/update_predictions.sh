Update_Predictions(){
cd ..
cd tools/Predictions
ROOT_DIRNAME="/media/zliu/data12/dataset/VSRD_PP_Sync/"
CKPT_DIRNAME="/home/zliu/CVPR25_Ablations/Ablations/VSRD_PP_FULL/ckpts/VSRDPP_FULL/"
OUT_DIRNAME="updated_pred"


python update_predictions.py \
    --root_dirname $ROOT_DIRNAME \
    --ckpt_dirname $CKPT_DIRNAME \
    --out_dirname $OUT_DIRNAME 
}

Update_Predictions