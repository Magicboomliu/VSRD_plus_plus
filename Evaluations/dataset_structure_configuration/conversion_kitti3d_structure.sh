CONVERSION_KITTI3D_STRUCTURE(){

cd ..
cd tools/organize_to_kitti3d_dataset_structure/

ROOT_DIRNAME="/media/zliu/data12/dataset/VSRD_PP_Sync/"
PREDICTION_LABEL_PATH="/media/zliu/data12/dataset/VSRD_PP_Sync/Ablations/VSRD_Velcoity_Only/label_vsrd_pp_velocity_only/predictions/"
GT_LABEL_PATH="/media/zliu/data12/dataset/VSRD_PP_Sync/Ablations/VSRD_Velcoity_Only/label_vsrd_pp_velocity_only/my_gts_with_dynamic/"
OUTPUT_FOLDER="/media/zliu/data12/dataset/KITTI3D_Velocity_Only"
TRAINING_SPLIT="03,07" 
TESTING_SPLIT="03"




python conversion_kitt3d_structure.py --root_dirname $ROOT_DIRNAME \
                                        --prediction_label_path $PREDICTION_LABEL_PATH \
                                        --gt_label_path $GT_LABEL_PATH \
                                        --training_split $TRAINING_SPLIT \
                                        --testing_split $TESTING_SPLIT \
                                        --output_folder $OUTPUT_FOLDER





}

CONVERSION_KITTI3D_STRUCTURE