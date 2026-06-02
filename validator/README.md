# Validator (Stage 1 & 2 evaluation)

Evaluation pipeline for multi-view auto-labeling quality (IoU, mAP, visualization).

**Dynamic labels:** Step1 and Step3 read `dynamic_mask.txt` from `DYNAMIC_DIRNAME`. Training uses online GT velocity; validator uses the exported txt file. See also the root [README.md](../README.md#3-dynamic--static-classification).

## Dynamic labels workflow

Recommended flow before running validator:

```bash
# 1. Generate dynamic_mask.txt (GT bbox velocity, 0.20 m/frame)
python -m preprocessing.Dynamic_Labels.pipeline --config sequence_07
# or all sequences:
sh preprocessing/scripts/generate_dynamic_labels.sh

# 2. Compare against legacy labels (quality gate)
python scripts/compare_dynamic_mask_gt.py --config sequence_07
python scripts/sweep_dynamic_threshold_global.py   # optional

# 3. Run validator with generated labels
export ROOT_DIRNAME=/path/to/KITTI360_For_Upload
export CKPT_DIRNAME=/path/to/trainer/ckpts/your_run
export DYNAMIC_DIRNAME=${ROOT_DIRNAME}/dynamic_attributes_est_gt
sh validator/make_predictions_scripts/run_evaluation_pipeline.sh
```

| Path | Role |
|------|------|
| `dynamic_attributes_est_gt/syncXX/` | Generated labels (recommended for validator) |
| `dynamic_attributes_est/syncXX/` | Legacy reference (compare script ground truth) |

To test against legacy labels directly: `export DYNAMIC_DIRNAME=${ROOT_DIRNAME}/dynamic_attributes_est`

### Environment variables

All shell scripts under `make_predictions_scripts/` accept overrides:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ROOT_DIRNAME` | KITTI360 dataset root | |
| `CKPT_DIRNAME` | `trainer/ckpts` | Checkpoint directory for export |
| `DYNAMIC_DIRNAME` | `{ROOT}/dynamic_attributes_est_gt` | Parent of `syncXX/dynamic_mask.txt` |
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

Both Step1 scripts use `--dyanmic_root_filename $DYNAMIC_DIRNAME` to load per-instance dynamic flags.

Output: `predictions/` (model pseudo labels) and checkpoint-derived GT JSON.

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

Reads the same `dynamic_mask.txt` as Step1 (not recomputed from neighbors).

```bash
cd validator/make_predictions_scripts
sh dynamic_attribute.sh
```

### Step4: Convert to KITTI3D folder structure

```bash
cd validator/dataset_structure_configuration
sh conversion_kitti3d_structure.sh
```

### Step5: mIoU

```bash
cd validator/stage1_evaluation_scripts
sh get_iou.sh
```

### Step6: mAP

```bash
cd validator/stage1_evaluation_scripts
sh get_mAP.sh
```

### Unified pipeline (Step 1–4)

```bash
sh validator/make_predictions_scripts/run_evaluation_pipeline.sh
```
