#!/bin/bash

# Unified evaluation pipeline script for VSRD++: Step 1-4
# This script runs all evaluation steps in sequence

Run_Evaluation_Pipeline(){
    cd ..
    cd tools
    
    # Configuration
    ROOT_DIRNAME="/media/zliu/data12/dataset/KITTI/KITTI360_For_Upload"
    CKPT_DIRNAME="/home/Journals2025/VSRD_plus_plus/trainer/ckpts/with_pseudo_depth_ssl_igevstereo"
    CKPT_FILENAME="step_2999.pt"
    DYNAMIC_DIRNAME="/media/zliu/data12/dataset/KITTI/KITTI360_For_Upload/dynamic_attributes_est/"
    INPUT_MODEL_TYPE="velocity_with_init"
    SAVED_PSEUDO_FOLDER_PATH="predictions"
    SPLIT_DIRNAME="R50-N16-M128-B16"
    JSON_FOLDERNAME="predictions"
    OUTPUT_LABELNAME="perfect_prediction"
    DYNAMIC_THRESHOLD=0.01
    TRAINING_SPLIT="03,07"
    TESTING_SPLIT="03,07"
    NUM_WORKERS=4
    CLASS_NAMES="car"
    # Output folder for Step4 (KITTI3D dataset structure)
    # If empty, will auto-generate to {ROOT_DIRNAME}/KITTI3D_Dataset/{CKPT_BASENAME}
    # Example: "/media/zliu/data12/dataset/KITTI/KITTI360_For_Upload/KITTI3D_Dataset/with_pseudo_depth_ssl_igevstereo"
    OUTPUT_FOLDER="/home/Journals2025/VSRD_plus_plus/Round1/Seg_Ablations/prefect_seg"
    
    # Run the unified pipeline
    python evaluation_pipeline.py \
        --root_dirname $ROOT_DIRNAME \
        --ckpt_dirname $CKPT_DIRNAME \
        --ckpt_filename $CKPT_FILENAME \
        --num_workers $NUM_WORKERS \
        --class_names $CLASS_NAMES \
        --dynamic_dirname $DYNAMIC_DIRNAME \
        --input_model_type $INPUT_MODEL_TYPE \
        --saved_pseudo_folder_path $SAVED_PSEUDO_FOLDER_PATH \
        --split_dirname $SPLIT_DIRNAME \
        --json_foldername $JSON_FOLDERNAME \
        --output_labelname $OUTPUT_LABELNAME \
        --dynamic_threshold $DYNAMIC_THRESHOLD \
        --training_split $TRAINING_SPLIT \
        --testing_split $TESTING_SPLIT \
        --output_folder "$OUTPUT_FOLDER" \
        --run_all
}

Run_Evaluation_Pipeline

