# Dynamic label generation

Offline pipeline to write `dynamic_mask.txt` from **3D bounding-box GT velocity** (same rule as training: `||v|| >= threshold` in **m/frame**).

Output format (one line per target frame):

```text
<id1,id2,...> <relative_image_path> <0.0|1.0,...>
```

Example:

```text
26005,26006,26028 data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000251.png 0.0,0.0,0.0
```

## CLI

From a trainer config (recommended):

```bash
python -m preprocessing.Dynamic_Labels.pipeline --config sequence_00
```

Default output: `{DATASET.ROOT}/dynamic_attributes_est_gt/{sync_name}/dynamic_mask.txt`

Explicit paths:

```bash
python -m preprocessing.Dynamic_Labels.pipeline \
  --dataset-root /path/to/KITTI360_For_Upload \
  --filenames filenames/R50-N16-M128-B16/2013_05_28_drive_0000_sync/sampled_image_filenames.txt \
  --output dynamic_attributes_est_gt/2013_05_28_drive_0000_sync/dynamic_mask.txt \
  --threshold 0.20
```

## Python API

```python
from preprocessing.Dynamic_Labels import generate_dynamic_labels

result = generate_dynamic_labels(
    dataset_root="/path/to/KITTI360_For_Upload",
    filenames_path=".../sampled_image_filenames.txt",
    output_path=".../dynamic_mask.txt",
    velocity_threshold=0.20,
    num_source_frames=16,
)
print(result.num_dynamic, result.num_instances)
```

## Modules

| File | Role |
|------|------|
| `pipeline.py` | Main loop over `sampled_image_filenames.txt` |
| `multi_inputs.py` | Build aligned multi-frame annotations (no images) |
| `format.py` | Parse/write `dynamic_mask.txt` lines |

Velocity logic lives in `preprocessing/Initial_Attributes/gt_attributes.py` (`DEFAULT_DYNAMIC_VELOCITY_THRESHOLD = 0.20`).

## Validation

Compare generated file against **legacy** labels (`dynamic_attributes_est/syncXX/`):

```bash
python scripts/compare_dynamic_mask_gt.py --config sequence_00 --threshold 0.20
```

Global threshold sweep (all 9 sequences):

```bash
python scripts/sweep_dynamic_threshold_global.py
```

## Use with validator

After compare passes, set `DYNAMIC_DIRNAME` and run the evaluation pipeline:

```bash
export ROOT_DIRNAME=/path/to/KITTI360_For_Upload
export DYNAMIC_DIRNAME=${ROOT_DIRNAME}/dynamic_attributes_est_gt
sh validator/make_predictions_scripts/run_evaluation_pipeline.sh
```

Validator Step1 (`make_predictions.py`) and Step3 (`get_gt_with_dynamic_label.py`) both read `{DYNAMIC_DIRNAME}/syncXX/dynamic_mask.txt`. See [validator/README.md](../../validator/README.md).
