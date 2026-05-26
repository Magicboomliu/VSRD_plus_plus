STAGE_2_VISUALIZATION_BOX3D(){
cd ..
cd tools/Stage2Visualizations/
ROOT_FOLDER="/home/Journals2025/VSRD_plus_plus/Round1/Seg_Ablations/prefect_seg/"
PD_FOLDER="/home/Journals2025/VSRD_plus_plus/Round1/Seg_Ablations/prefect_seg/training/label_est/"
ANNOTATION_FOLDER="/data/dataset/KITTI/KITTI360_For_Docker/annotations/"
OPTIONS="pd_only" # selected from "pd_only" and "pd_gt"
# Mask erode ratio (0.0 = disabled, 0.05 = 5%, 0.10 = 10%)
ERODE_RATIO=0.05
STAGE2_LABEL_PATH="/home/Journals2025/VSRD_plus_plus/Round1/Seg_Ablations/prefect_seg/visualization/projected3d_enrode_$ERODE_RATIO/"


python visualization_project_3dbox.py --root_folder $ROOT_FOLDER \
                                      --stage2_vis_output_folder $STAGE2_LABEL_PATH \
                                      --pd_folder $PD_FOLDER \
                                      --annotation_folder $ANNOTATION_FOLDER \
                                      --options $OPTIONS \
                                      --erode_ratio $ERODE_RATIO

}

STAGE_2_VISUALIZATION_BOX3D