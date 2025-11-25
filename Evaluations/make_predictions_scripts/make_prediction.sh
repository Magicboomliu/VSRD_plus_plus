Make_Json_Prediction(){
cd ..
cd tools/Predictions
ROOT_DIRNAME="/data/dataset/KITTI/KITTI360_For_Docker"
CKPT_DIRNAME="/home/Journals2025/VSRD_plus_plus/Optimized_Based/ckpts/with_pseudo_depth_ssl_igevstereo"
DYNAMIC_DIRNAME="/data/dataset/KITTI/KITTI360_For_Docker/est_dynamic_list/"
INPUT_MODEL_TYPE="velocity_with_init"
SAVED_PSEUDO_LABEL_FOLDER="predictions"


NUM_WORKERS=4
python make_predictions.py \
    --root_dirname $ROOT_DIRNAME \
    --ckpt_dirname $CKPT_DIRNAME \
    --num_workers $NUM_WORKERS \
    --dyanmic_root_filename $DYNAMIC_DIRNAME \
    --input_model_type $INPUT_MODEL_TYPE \
    --saved_pseudo_folder_path $SAVED_PSEUDO_LABEL_FOLDER


}

Make_Json_Prediction