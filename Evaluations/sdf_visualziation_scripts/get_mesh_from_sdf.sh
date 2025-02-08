Make_Mesh_From_VSRD_Prediction(){
cd ..
cd tools/SDF2Mesh
ROOT_DIRNAME="/media/zliu/data12/dataset/VSRD_PP_Sync/"
CKPT_DIRNAME="/media/zliu/data12/dataset/TPAMI_Saved_Ckpts/VSRDPP_saved_ckpts/ckpts/"
DYNAMIC_DIRNAME="/media/zliu/data12/dataset/VSRD_PP_Sync/est_dynamic_list/"
INPUT_MODEL_TYPE="velocity_with_init"

OUTPUT_FOLDER="/media/zliu/data12/TPAMI_Results/SDF_Visualizations/"

NUM_WORKERS=4
python sdf2mesh.py \
    --root_dirname $ROOT_DIRNAME \
    --ckpt_dirname $CKPT_DIRNAME \
    --num_workers $NUM_WORKERS \
    --dyanmic_root_filename $DYNAMIC_DIRNAME \
    --input_model_type $INPUT_MODEL_TYPE \
    --output_folder $OUTPUT_FOLDER


}

Make_Mesh_From_VSRD_Prediction