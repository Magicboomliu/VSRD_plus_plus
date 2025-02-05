STAGE_2_VISUALIZATION_BOX3D(){

cd /home/zliu/TPAMI25/KITTI360_Benchmarks/KITTI360_Benchmark/Visualizations/Stage2/

ROOT_FOLDER="/media/zliu/data12/dataset/TPAMI_Stage2/NEW_VSRDPP25_LABEL/VSRD_PP_SPLIT/"
STAGE2_LABEL_PATH="/media/zliu/data12/dataset/TPAMI_Ablations/VSRD_Vanilla/Visualizations"
PD_FOLDER="/media/zliu/data12/dataset/TPAMI_Ablations/Synced_For_Evaluations/VSRD_Vanilla/training/label_2"
ANNOTATION_FOLDER="/media/zliu/data12/dataset/VSRD_PP_Sync/annotations"
OPTIONS="pd_gt" # selected from "pd_only" and "pd_gt"


python visualization_project_3dbox.py --root_folder $ROOT_FOLDER \
                                      --stage2_vis_output_folder $STAGE2_LABEL_PATH \
                                      --pd_folder $PD_FOLDER \
                                      --annotation_folder $ANNOTATION_FOLDER \
                                      --options $OPTIONS

}


STAGE_2_VISUALIZATION_BEV(){

cd /home/zliu/TPAMI25/KITTI360_Benchmarks/KITTI360_Benchmark/Visualizations/Stage2/

ROOT_FOLDER="/media/zliu/data12/dataset/TPAMI_Stage2/NEW_VSRDPP25_LABEL/VSRD_PP_SPLIT/"
STAGE2_LABEL_PATH="/home/zliu/TPAMI25/Experimental_Results/Stage2/Static_Dynamic_Splits/WeakM3D/WeakM3D_Vanilla/Visualizations/bev"
PD_FOLDER="/home/zliu/TPAMI25/Experimental_Results/Stage2/Static_Dynamic_Splits/WeakM3D/WeakM3D_Vanilla/pd_labels/"

python visualization_bev.py --root_folder $ROOT_FOLDER \
                                      --stage2_vis_output_folder $STAGE2_LABEL_PATH \
                                      --pd_folder $PD_FOLDER 
}


STAGE_2_VISUALIZATION_BOX3D
