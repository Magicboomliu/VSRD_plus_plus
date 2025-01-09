CONVERSION_KITTI3D_STRUCTURE(){

cd ..
cd tools/organize_to_kitti3d_dataset_structure/
ROOT_DIRNAME="/media/zliu/data12/dataset/VSRD_PP_Sync/"
PREDICTION_LABEL_PATH="/media/zliu/data12/TPAMI_Results/Stage1_Expermental_Results/Autolabels_Re_Train/pseudo_labels_txt/predictions/"
GT_LABEL_PATH="/media/zliu/data12/TPAMI_Results/Stage1_Expermental_Results/Autolabels_Re_Train/pseudo_labels_txt/my_gts_with_dynamic/"
OUTPUT_FOLDER="/media/zliu/data12/TPAMI_Results/Stage1_Expermental_Results/Synced_For_Evaluations/Splits_2023/Autolabels_Re_Train"
TRAINING_SPLIT="00,02,04,05,06,09" 
TESTING_SPLIT="10"

python conversion_kitt3d_structure.py --root_dirname $ROOT_DIRNAME \
                                        --prediction_label_path $PREDICTION_LABEL_PATH \
                                        --gt_label_path $GT_LABEL_PATH \
                                        --training_split $TRAINING_SPLIT \
                                        --testing_split $TESTING_SPLIT \
                                        --output_folder $OUTPUT_FOLDER
}

CONVERSION_KITTI3D_STRUCTURE