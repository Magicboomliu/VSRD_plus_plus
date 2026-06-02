Make_Json_Prediction(){
cd ..
cd tools/Predictions

ROOT_DIRNAME="${ROOT_DIRNAME:-/media/zliu/data12/dataset/KITTI/KITTI360_For_Upload}"
CKPT_DIRNAME="${CKPT_DIRNAME:-/home/zliu/IJCV/VSRD_plus_plus/trainer/ckpts}"
DYNAMIC_DIRNAME="${DYNAMIC_DIRNAME:-${ROOT_DIRNAME}/dynamic_attributes_est_gt}"
INPUT_MODEL_TYPE="${INPUT_MODEL_TYPE:-velocity_with_init}"
NUM_WORKERS="${NUM_WORKERS:-4}"

python make_gt_predictions.py \
    --root_dirname "$ROOT_DIRNAME" \
    --ckpt_dirname "$CKPT_DIRNAME" \
    --num_workers "$NUM_WORKERS" \
    --dyanmic_root_filename "$DYNAMIC_DIRNAME" \
    --input_model_type "$INPUT_MODEL_TYPE"
}

Make_Json_Prediction
