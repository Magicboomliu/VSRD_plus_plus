GET_STAGE_1_IoU(){
cd ..
cd tools/IoU_AP_Analysis/Stage1_Eval/

#/media/zliu/data12/dataset/VSRD_PP_Sync/label_vsrd_pp_complelte_V2/updated_pred
PREDICTION_FOLDER="/media/zliu/data12/dataset/VSRD_PP_Sync/Ablations/VSRD_MLP_Only/label_vsrd_pp_mlp_only/predictions"
GT_FOLDER="/media/zliu/data12/dataset/VSRD_PP_Sync/Ablations/VSRD_MLP_Only/label_vsrd_pp_mlp_only/my_gts_with_dynamic/"
OUTPUT_NAME="/media/zliu/data12/dataset/VSRD_PP_Sync/Ablations/VSRD_MLP_Only/label_vsrd_pp_mlp_only/IoU_Results2"
OPTIONS='hard' #all, easy, hard

python get_IoU.py --prediction_folder $PREDICTION_FOLDER \
                  --gt_folder $GT_FOLDER \
                  --output_name $OUTPUT_NAME \
                  --options $OPTIONS

}


GET_STAGE_1_IoU