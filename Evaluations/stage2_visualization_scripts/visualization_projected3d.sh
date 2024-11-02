STAGE_2_VISUALIZATION_BOX3D(){
cd ..
cd tools/Stage2Visualizations/
ROOT_FOLDER="/media/zliu/data12/dataset/KITTI3D_Format_DEBUG/"
STAGE2_LABEL_PATH=/media/zliu/data12/dataset/DEBUG_VIS/STAGE2_VIS
PD_FOLDER="/media/zliu/data12/dataset/KITTI3D_Format_DEBUG/training/label_est/"
ANNOTATION_FOLDER="/media/zliu/data12/dataset/VSRD_PP_Sync/annotations/"
OPTIONS="pd_only" # selected from "pd_only" and "pd_gt"


python visualization_project_3dbox.py --root_folder $ROOT_FOLDER \
                                      --stage2_vis_output_folder $STAGE2_LABEL_PATH \
                                      --pd_folder $PD_FOLDER \
                                      --annotation_folder $ANNOTATION_FOLDER \
                                      --options $OPTIONS

}

STAGE_2_VISUALIZATION_BOX3D