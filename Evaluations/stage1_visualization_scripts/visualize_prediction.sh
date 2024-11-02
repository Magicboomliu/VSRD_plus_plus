Visualized_Prediction_By_Json(){
cd ..
cd tools/Visualizations
ROOT_DIRNAME="/media/zliu/data12/dataset/VSRD_PP_Sync/"
CKPT_DIRNAME="/home/zliu/CVPR25_Ablations/Ablations/VSRD_Vanilla/ckpts/VSRD_Vanallia_Ablation/"
JSON_FOLDERNAME="predictions"
OUT_DIRNAME="vis_mlp_only"
NUM_WORKERS=4
VIS_OPTION="gt_pd_bev" # "pd_only_3d","gt_only_3d","gt_pd_3d","gt_pd_bev"

python visualize_predictions.py \
    --root_dirname $ROOT_DIRNAME \
    --ckpt_dirname $CKPT_DIRNAME \
    --out_dirname $OUT_DIRNAME \
    --num_workers $NUM_WORKERS \
    --vis_option $VIS_OPTION \
    --json_folder_name $JSON_FOLDERNAME


}

Visualized_Prediction_By_Json