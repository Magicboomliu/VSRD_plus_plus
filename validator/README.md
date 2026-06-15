# Validator (Stage 1 & 2 evaluation)

Evaluation pipeline for multi-view auto-labeling quality (IoU, mAP, visualization).

**Defaults (aligned with preprocessing output):**

| Artifact | Path under `DATASET.ROOT` |
|----------|---------------------------|
| Pseudo depth | `pseudo_depth_ssl_waft_stereo/<sequence>/image_00/data_rect/` |
| Dynamic labels | `dynamic_attributes_est_gt/<sequence>/dynamic_mask.txt` |

Legacy zip layout (`pseudo_depth_ssl/`, `dynamic_attributes_est/syncXX/`) is no longer the default.

## Workflow

```bash
# 1. Depth
sh preprocessing/scripts/generate_pseudo_depth_waft.sh 0006

# 2. Dynamic labels
python -m preprocessing.Dynamic_Labels.pipeline --config sequence_07
# or: sh preprocessing/scripts/generate_dynamic_labels.sh

# 3. Verify
python scripts/compare_dynamic_mask_gt.py --config sequence_07

# 4. Validator
export ROOT_DIRNAME=/path/to/KITTI360_For_Upload
export CKPT_DIRNAME=/path/to/trainer/ckpts/your_run
export DYNAMIC_DIRNAME=${ROOT_DIRNAME}/dynamic_attributes_est_gt
sh validator/make_predictions_scripts/run_evaluation_pipeline.sh
```

Compare vs **legacy** labels: add `--dynamic-path dynamic_attributes_est/sync07/dynamic_mask.txt` to the compare script.

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ROOT_DIRNAME` | KITTI360 dataset root | |
| `CKPT_DIRNAME` | `trainer/ckpts` | Checkpoint directory for export |
| `DYNAMIC_DIRNAME` | `{ROOT}/dynamic_attributes_est_gt` | Parent of `<sequence>/dynamic_mask.txt` |
| `INPUT_MODEL_TYPE` | `velocity_with_init` | Model architecture for JSON export |
| `NUM_WORKERS` | `4` | Multiprocessing workers |

---

## Stage1: Multi-View AutoLabeling validator

### Step1: Generate pseudo labels in JSON format

**Prediction generation**

```bash
cd validator/make_predictions_scripts
sh make_prediction.sh
```

**Ground truth generation**

```bash
cd validator/make_predictions_scripts
sh make_gt_prediction.sh
```

Both read `{DYNAMIC_DIRNAME}/<sequence>/dynamic_mask.txt` (e.g. `.../2013_05_28_drive_0007_sync/dynamic_mask.txt`).

### Step2: Convert to KITTI3D `.txt` format

```bash
cd validator/make_predictions_scripts
python ../tools/Predictions/convert_prediction.py \
    --root_dirname $ROOT_DIRNAME \
    --ckpt_dirname $CKPT_DIRNAME \
    --json_foldername predictions \
    --output_labelname perfect_prediction
```

### Step3: Dynamic attribute assignment for GT KITTI labels

Reads the same `dynamic_mask.txt` as Step1.

```bash
cd validator/make_predictions_scripts
sh dynamic_attribute.sh
```

### Step4: Convert to KITTI3D folder structure

```bash
cd validator/dataset_structure_configuration
sh conversion_kitti3d_structure.sh
```

### Step5: mIoU / Step6: mAP

```bash
cd validator/stage1_evaluation_scripts
sh get_iou.sh
sh get_mAP.sh
```

### Unified pipeline (Step 1–4)

```bash
sh validator/make_predictions_scripts/run_evaluation_pipeline.sh
```
