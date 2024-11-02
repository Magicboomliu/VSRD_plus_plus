STAGE_2_VISUALIZATION_BEV(){

cd ..
cd tools/Stage2Visualizations/
ROOT_FOLDER="/media/zliu/data12/dataset/KITTI3D_Format_DEBUG/"
STAGE2_LABEL_PATH=/media/zliu/data12/dataset/DEBUG_VIS/STAGE2_VIS/BEV
PD_FOLDER="/media/zliu/data12/dataset/KITTI3D_Format_DEBUG/training/label_est/"

python visualization_bev.py --root_folder $ROOT_FOLDER \
                                      --stage2_vis_output_folder $STAGE2_LABEL_PATH \
                                      --pd_folder $PD_FOLDER 
}

STAGE_2_VISUALIZATION_BEV