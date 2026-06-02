#!/bin/sh
# Unified evaluation pipeline script for VSRD++: Step 1-4

Run_Evaluation_Pipeline(){
    cd ..
    cd tools

    ROOT_DIRNAME="${ROOT_DIRNAME:-/media/zliu/data12/dataset/KITTI/KITTI360_For_Upload}"
    CKPT_DIRNAME="${CKPT_DIRNAME:-/home/zliu/IJCV/VSRD_plus_plus/trainer/ckpts}"
    CKPT_FILENAME="${CKPT_FILENAME:-step_2999.pt}"
    DYNAMIC_DIRNAME="${DYNAMIC_DIRNAME:-${ROOT_DIRNAME}/dynamic_attributes_est_gt}"
    INPUT_MODEL_TYPE="${INPUT_MODEL_TYPE:-velocity_with_init}"
    SAVED_PSEUDO_FOLDER_PATH="${SAVED_PSEUDO_FOLDER_PATH:-predictions}"
    SPLIT_DIRNAME="${SPLIT_DIRNAME:-R50-N16-M128-B16}"
    JSON_FOLDERNAME="${JSON_FOLDERNAME:-predictions}"
    OUTPUT_LABELNAME="${OUTPUT_LABELNAME:-perfect_prediction}"
    TRAINING_SPLIT="${TRAINING_SPLIT:-03,07}"
    TESTING_SPLIT="${TESTING_SPLIT:-03,07}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
    CLASS_NAMES="${CLASS_NAMES:-car}"
    OUTPUT_FOLDER="${OUTPUT_FOLDER:-}"

    python evaluation_pipeline.py \
        --root_dirname "$ROOT_DIRNAME" \
        --ckpt_dirname "$CKPT_DIRNAME" \
        --ckpt_filename "$CKPT_FILENAME" \
        --num_workers "$NUM_WORKERS" \
        --class_names "$CLASS_NAMES" \
        --dynamic_dirname "$DYNAMIC_DIRNAME" \
        --input_model_type "$INPUT_MODEL_TYPE" \
        --saved_pseudo_folder_path "$SAVED_PSEUDO_FOLDER_PATH" \
        --split_dirname "$SPLIT_DIRNAME" \
        --json_foldername "$JSON_FOLDERNAME" \
        --output_labelname "$OUTPUT_LABELNAME" \
        --training_split "$TRAINING_SPLIT" \
        --testing_split "$TESTING_SPLIT" \
        ${OUTPUT_FOLDER:+--output_folder "$OUTPUT_FOLDER"} \
        --run_all
}

Run_Evaluation_Pipeline
