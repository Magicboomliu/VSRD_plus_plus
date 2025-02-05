STAGE_1_VISUALIZATION_BOX3D(){

cd /home/zliu/TPAMI25/KITTI360_Benchmarks/KITTI360_Benchmark/Visualizations/Stage1/
ROOT_FOLDER="/media/zliu/data12/dataset/TPAMI_Stage2/NEW_VSRDPP25_LABEL/VSRD_PP_SPLIT/"
STAGE2_LABEL_PATH="/media/zliu/data12/TPAMI_Results/TPAMI_Ablations/Velocity_Projection_Loss_Only/Visaliations"
PD_FOLDER="/media/zliu/data12/TPAMI_Results/TPAMI_Ablations/Synced_For_Evaluations/VSRD_Proj_Only/training/label_2/"
ANNOTATION_FOLDER="/media/zliu/data12/dataset/VSRD_PP_Sync/annotations"
OPTIONS="pd_only" # selected from "pd_only" and "pd_gt"
filelist="/media/zliu/data12/dataset/TPAMI_Stage2/NEW_VSRDPP25_LABEL/Replaced_Version/VSRD_PP_SPLIT_V2/ImageSets/train.txt"

python visualization_project_3dbox.py --root_folder $ROOT_FOLDER \
                                      --stage2_vis_output_folder $STAGE2_LABEL_PATH \
                                      --pd_folder $PD_FOLDER \
                                      --annotation_folder $ANNOTATION_FOLDER \
                                      --options $OPTIONS \
                                      --filelist $filelist
}



STAGE_1_VISUALIZATION_BEV(){

cd /home/zliu/TPAMI25/KITTI360_Benchmarks/KITTI360_Benchmark/Visualizations/Stage1/

ROOT_FOLDER="/media/zliu/data12/dataset/TPAMI_Stage2/NEW_VSRDPP25_LABEL/VSRD_PP_SPLIT/"
STAGE2_LABEL_PATH="/media/zliu/data12/TPAMI_Results/TPAMI_Ablations/Velocity_Projection_Loss_Only/Visaliations/BEV"
PD_FOLDER="/media/zliu/data12/TPAMI_Results/TPAMI_Ablations/Synced_For_Evaluations/VSRD_Proj_Only/training/label_2/"
filelist="/media/zliu/data12/dataset/TPAMI_Stage2/NEW_VSRDPP25_LABEL/Replaced_Version/VSRD_PP_SPLIT_V2/ImageSets/train.txt"

python visualization_bev.py --root_folder $ROOT_FOLDER \
                                      --stage2_vis_output_folder $STAGE2_LABEL_PATH \
                                      --pd_folder $PD_FOLDER  \
                                      --filelist $filelist
}

# STAGE_1_VISUALIZATION_BOX3D
STAGE_1_VISUALIZATION_BEV
# STAGE_2_VISUALIZATION_BEV




# STAGE_1_VISUALIZATION_BOX3D
