# Preprocessing

Data preparation for VSRD++ Stage 1 training and validator evaluation.

## Pipeline overview

```
pseudo depth (WAFT-Stereo)  ──►  estimate_initial_attributes  ──►  trainer (attribute init)
                                  (depth + RoI LiDAR + ICP;
                                   dynamic/static from GT bbox velocity)

sampled_image_filenames.txt  ──►  generate_dynamic_labels  ──►  dynamic_mask.txt (validator)
                                  (GT bbox velocity, threshold 0.20 m/frame)
```

## Default paths (`preprocessing/dataset_paths.py`)

| Output | Path under `DATASET.ROOT` |
|--------|---------------------------|
| WAFT depth | `pseudo_depth_ssl_waft_stereo/<sequence>/image_00/data_rect/<frame>.png` |
| Dynamic labels | `dynamic_attributes_est_gt/<sequence>/dynamic_mask.txt` |

`<sequence>` example: `2013_05_28_drive_0007_sync`

- **Training**: reads WAFT depth; infers dynamic/static **online** (no txt).
- **Validator**: reads `dynamic_mask.txt` from `dynamic_attributes_est_gt/`.
- **Legacy** (`pseudo_depth_ssl/`, `dynamic_attributes_est/syncXX/`): optional reference only.

## Quick commands

```bash
# 1. Depth (WAFT env: cd preprocessing/disparity_estimation/WAFT-Stereo && pixi install)
sh preprocessing/scripts/generate_pseudo_depth_waft.sh 0006
pixi run gen-depth-seq 0006

# 2. Dynamic labels (validator)
python -m preprocessing.Dynamic_Labels.pipeline --config sequence_07
sh preprocessing/scripts/generate_dynamic_labels.sh

# 3. Verify
pixi run test-data
python scripts/compare_dynamic_mask_gt.py --config sequence_07
```

Compare vs legacy zip labels:

```bash
python scripts/compare_dynamic_mask_gt.py --config sequence_07 \
  --dynamic-path dynamic_attributes_est/sync07/dynamic_mask.txt
```

---

## Depth — WAFT-Stereo (recommended)

Pseudo depth: uint16 PNG, `depth_m = pixel / 256`.

API: `preprocessing/apis/depth_estimator.py`  
Model: `preprocessing/disparity_estimation/WAFT-Stereo/` (separate pixi env, PyTorch ≥ 2.0).

```bash
cd preprocessing/disparity_estimation/WAFT-Stereo && pixi install && cd ../../..
DATASET_ROOT=/path/to/KITTI360_For_Upload sh preprocessing/scripts/generate_pseudo_depth_waft.sh 0006
```

Training loads depth via `get_depth_filename()` → `pseudo_depth_ssl_waft_stereo/` (`file_io_utils.py`).

### Depth — LEAStereo (legacy)

Python entry: `preprocessing/disparity_estimation/Leastereo/sequential_depth_estimation.py`  
Default output folder: `pseudo_depth_ssl/` (not used unless you generate there manually).

---

## Initial attribute estimation

See [Initial_Attributes/README.md](Initial_Attributes/README.md).

```python
from preprocessing.Initial_Attributes import estimate_initial_attributes, extract_initial_attributes

multi_inputs = estimate_initial_attributes(multi_inputs, device="cuda:0")
init_attrs = extract_initial_attributes(multi_inputs, device="cuda:0")
```

Smoke test: `python -m preprocessing.Initial_Attributes.pipeline --input Debug_Examples/exampleV2.pkl`

---

## Dynamic label generation

See [Dynamic_Labels/README.md](Dynamic_Labels/README.md).

Default threshold: **0.20 m/frame** (`DEFAULT_DYNAMIC_VELOCITY_THRESHOLD` in `gt_attributes.py`).

Configs (`sequence_XX.json`) set `DYNAMIC_LABELS_PATH` to the generated file path under `dynamic_attributes_est_gt/`.

Then run validator: `export DYNAMIC_DIRNAME=${ROOT_DIRNAME}/dynamic_attributes_est_gt` — see [validator/README.md](../validator/README.md).

---

## Module index

| Directory | Role |
|-----------|------|
| `dataset_paths.py` | Canonical depth / dynamic path constants |
| `apis/depth_estimator.py` | WAFT depth batch API |
| `Initial_Attributes/` | Online attribute + dynamic inference at train time |
| `Dynamic_Labels/` | Offline `dynamic_mask.txt` export |
| `scripts/` | Shell entry points (`generate_pseudo_depth_waft.sh`, `generate_dynamic_labels.sh`) |
| `data_organization/` | Legacy split64 / ablation utilities (hardcoded paths; use with care) |

Optical flow (`optical_flow_estimation/`) and InternImage segmentation are **not** used in the current training pipeline.
