Update_Predictions(){
cd ..
cd tools/Predictions
ROOT_DIRNAME="/media/zliu/data12/dataset/VSRD_PP_Sync/"
CKPT_DIRNAME="/media/zliu/data12/dataset/Tsubame_Results/V1/ckpts/"
OUT_DIRNAME="updated_pred"


python update_predictions.py \
    --root_dirname $ROOT_DIRNAME \
    --ckpt_dirname $CKPT_DIRNAME \
    --out_dirname $OUT_DIRNAME 
}

Update_Predictions
