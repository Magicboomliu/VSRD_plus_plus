# Stage1 and Stage 2 Evaluations  

## Stage1: Multi-View AutoLabeling Evaluations.
#### Step1: Using the pretrained Models to Generated Psudo Labels in Json Format  
- Prediction Generation
```
cd Evaluations/make_predictions_scripts
sh make_prediction.sh 

# where inside it is:  

python make_predictions.py \
    --root_dirname $ROOT_DIRNAME \
    --ckpt_dirname $CKPT_DIRNAME \
    --num_workers $NUM_WORKERS \
    --dyanmic_root_filename $DYNAMIC_DIRNAME \
    --input_model_type $INPUT_MODEL_TYPE

```

Then you will find a directories named `predictions` which contains the all pseudo labels in the `.json` format.  
- Ground Truth Generation.  
```
cd Evaluations/make_predictions_scripts
sh make_gt_prediction.sh

# where inside it is:  

python make_gt_predictions.py \
    --root_dirname $ROOT_DIRNAME \
    --ckpt_dirname $CKPT_DIRNAME \
    --num_workers $NUM_WORKERS \
    --dyanmic_root_filename $DYNAMIC_DIRNAME \
    --input_model_type $INPUT_MODEL_TYPE

```  
Then you will find a directories named `my_gts` which contains the all pseudo labels in the `.json` format.  

#### Step2: Convert it into KITTI3D `.txt` format 

```
cd Evaluations/make_predictions_scripts


python convert_prediction.py \
    --root_dirname $ROOT_DIRNAME \
    --ckpt_dirname $CKPT_DIRNAME \
    --num_workers $NUM_WORKERS \
    --json_foldername $JSON_FOLDERNAME \
    --output_labelname $OUTPUT_LABELNAME

```

### Step3: Dynamic Objects Assignment using GT Labels

```
cd Evaluations/make_predictions_scripts
sh dynamic_attribute.sh

python get_gt_with_dynamic_label.py \
    --root_dirname $ROOT_DIRNAME \
    --ckpt_dirname $CKPT_DIRNAME \
    --num_workers $NUM_WORKERS \
    --json_foldername $JSON_FOLDERNAME \
    --output_labelname $OUTPUT_LABELNAME \
    --dynamic_threshold $DYNAMIC_THRESHOLD

```

### Step4: Convert it into the KITTI3D One Folder

```
cd Evaluations/dataset_structure_configuration
sh conversion_kitti3d_structure.sh

python conversion_kitt3d_structure.py --root_dirname $ROOT_DIRNAME \
                                        --prediction_label_path $PREDICTION_LABEL_PATH \
                                        --gt_label_path $GT_LABEL_PATH \
                                        --training_split $TRAINING_SPLIT \
                                        --testing_split $TESTING_SPLIT \
                                        --output_folder $OUTPUT_FOLDER

```

### Step5: Get the mIOU for each sequences

```
cd Evaluations/stage1_evaluation_scripts
sh get_iou.sh

python get_IoU.py --prediction_folder $PREDICTION_FOLDER \
                  --gt_folder $GT_FOLDER \
                  --output_name $OUTPUT_NAME \
                  --options $OPTIONS

```



### Step6: Get the mAP for specific sequences

```
cd Evaluations/stage1_evaluation_scripts
sh get_mAP.sh 

python get_mAP.py --pd_dir_folder $pd_dir_folder \
                 --gt_dir_folder $gt_dir_folder \
                 --saved_mAP_folder $saved_mAP_folder

```