CONVERSION_KITTI3D_STRUCTURE(){

cd ..
cd tools/organize_to_kitti3d_dataset_structure/
ROOT_DIRNAME="/media/zliu/data12/dataset/VSRD_PP_Sync/"
PREDICTION_LABEL_PATH="/media/zliu/data12/dataset/VSRD_PP_Sync/VSRDPP_Tsubame_Results/label_vsrd_pp_full_completed/predictions/"
GT_LABEL_PATH="/media/zliu/data12/dataset/VSRD_PP_Sync/VSRDPP_Tsubame_Results/label_vsrd_pp_full_completed/my_gts_with_dynamic/"
OUTPUT_FOLDER="/media/zliu/data12/dataset/VSRDPP_Stage2_Dataset/KITTI360_VSRDPP_V1"
TRAINING_SPLIT="00,02,03,04,05,06,07,09" 
TESTING_SPLIT="10"

python conversion_kitt3d_structure.py --root_dirname $ROOT_DIRNAME \
                                        --prediction_label_path $PREDICTION_LABEL_PATH \
                                        --gt_label_path $GT_LABEL_PATH \
                                        --training_split $TRAINING_SPLIT \
                                        --testing_split $TESTING_SPLIT \
                                        --output_folder $OUTPUT_FOLDER
}

CONVERSION_KITTI3D_STRUCTURE