STAGE_2_VISUALIZATION_BEV(){

cd ..
cd tools/Stage2Visualizations/
ROOT_FOLDER="/home/Journals2025/VSRD_plus_plus/Round1/Seg_Ablations/prefect_seg/"
STAGE2_LABEL_PATH="/home/Journals2025/VSRD_plus_plus/Round1/Seg_Ablations/prefect_seg/visualization/bev"
PD_FOLDER="/home/Journals2025/VSRD_plus_plus/Round1/Seg_Ablations/prefect_seg/training/label_est/"

python visualization_bev.py --root_folder $ROOT_FOLDER \
                                      --stage2_vis_output_folder $STAGE2_LABEL_PATH \
                                      --pd_folder $PD_FOLDER 
}

STAGE_2_VISUALIZATION_BEV