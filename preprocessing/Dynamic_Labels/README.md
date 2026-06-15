# Dynamic label generation

Offline pipeline to write `dynamic_mask.txt` from **3D bounding-box GT velocity** (same rule as training: `||v|| >= threshold` in **m/frame**).

**Default output:**

```text
{DATASET.ROOT}/dynamic_attributes_est_gt/<sequence>/dynamic_mask.txt
```

Example: `.../dynamic_attributes_est_gt/2013_05_28_drive_0007_sync/dynamic_mask.txt`

## CLI

```bash
python -m preprocessing.Dynamic_Labels.pipeline --config sequence_00
sh preprocessing/scripts/generate_dynamic_labels.sh
```

Explicit paths:

```bash
python -m preprocessing.Dynamic_Labels.pipeline \
  --dataset-root /path/to/KITTI360_For_Upload \
  --filenames filenames/R50-N16-M128-B16/2013_05_28_drive_0000_sync/sampled_image_filenames.txt \
  --output dynamic_attributes_est_gt/2013_05_28_drive_0000_sync/dynamic_mask.txt \
  --threshold 0.20
```

## Validation

```bash
python scripts/compare_dynamic_mask_gt.py --config sequence_00
```

Against legacy zip labels:

```bash
python scripts/compare_dynamic_mask_gt.py --config sequence_00 \
  --dynamic-path dynamic_attributes_est/sync00/dynamic_mask.txt
```

Global sweep: `python scripts/sweep_dynamic_threshold_global.py`

## Validator

```bash
export DYNAMIC_DIRNAME=${ROOT_DIRNAME}/dynamic_attributes_est_gt
sh validator/make_predictions_scripts/run_evaluation_pipeline.sh
```

Path constants: `preprocessing/dataset_paths.py`
