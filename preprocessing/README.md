# Preprocessing

Data preparation for VSRD++ Stage 1 training.

## Pipeline overview

```
pseudo depth (WAFT-Stereo)  ──►  estimate_initial_attributes  ──►  trainer (attribute init)
                                  (depth + RoI LiDAR + ICP;
                                   dynamic/static from GT bbox velocity)

sampled_image_filenames.txt  ──►  generate_dynamic_labels  ──►  dynamic_mask.txt (optional export)
                                  (GT bbox velocity only; no images/depth)
```

- **Depth**: required by the initial-attribute pipeline (pseudo RoI LiDAR from depth maps).
- **Dynamic / static (training)**: inferred online from annotated 3D bbox GT velocity (`||v|| >= 0.20 m/frame`); no `dynamic_mask.txt` required at train time.
- **Dynamic labels (offline)**: optional pipeline writes `dynamic_attributes_est_gt/<sync>/dynamic_mask.txt` for validation or legacy tooling — see [Dynamic_Labels](Dynamic_Labels/README.md).
- **Optical flow**: not used in the current preprocessing pipeline.

## Depth — WAFT-Stereo (recommended)

Pseudo depth is stored as uint16 PNG (`value / 256` = depth in metres). Default output directory: `pseudo_depth_ssl_waft_stereo/`.

API: `preprocessing/apis/depth_estimator.py`  
Model code: `preprocessing/disparity_estimation/WAFT-Stereo/` (separate pixi env, PyTorch ≥ 2.0).

**Setup**

```bash
cd preprocessing/disparity_estimation/WAFT-Stereo
pixi install
```

**Batch generate**

```bash
# Single sequence (recommended first)
sh preprocessing/scripts/generate_pseudo_depth_waft.sh 0006

# All sequences
sh preprocessing/scripts/generate_pseudo_depth_waft.sh
```

**Single-pair API**

```python
from preprocessing.apis.depth_estimator import Load_Depth_Model

model = Load_Depth_Model("WAFT-Stereo", device="cuda:0")
depth_m = model.infer_from_left("data_2d_raw/.../image_00/data_rect/0000000251.png")
```

See `preprocessing/apis/depth_estimator.py` for full API.

### Depth — LEAStereo (legacy)

```bash
sh preprocessing/scripts/generate_pseudo_depth.sh 0006
```

Requires Kitti15 weights at `Leastereo/run/Kitti15/best/best.pth`.

---

## Initial attribute estimation

Used during Stage 1 training when attribute initialization is enabled (`TRAIN_DDP_VSRDPP`).  
**Only uses pseudo depth** — no optical flow.

### Pipeline

```
pseudo depth  →  build_roi_lidar  →  infer is_dynamic (GT bbox velocity)
                                        ↓
                              estimate_velocity_icp  →  estimate_location_orientation
                                        ↓
                              compute_gt_attributes (optional, from 3D bbox)
```

Entry point:

```python
from preprocessing.Initial_Attributes import estimate_initial_attributes

multi_inputs = estimate_initial_attributes(
    multi_inputs,
    device="cuda:0",
)
```

Or with explicit config:

```python
from preprocessing.Initial_Attributes import InitialAttributesPipeline, InitialAttributesConfig

pipeline = InitialAttributesPipeline(InitialAttributesConfig(min_roi_points=120))
multi_inputs = pipeline.run(multi_inputs)
```

Local smoke test:

```bash
python -m preprocessing.Initial_Attributes.pipeline --input Debug_Examples/exampleV2.pkl
```

### Modules

| File | Role |
|------|------|
| `pipeline.py` | Orchestrates all steps |
| `roi_lidar.py` | Step 1 — back-project depth to RoI point clouds |
| `velocity.py` | Step 2 — ICP velocity per instance |
| `location_orientation.py` | Step 3 — centroid location + velocity-based orientation |
| `gt_attributes.py` | GT from 3D bounding boxes (debug / trainer fallback) |

Training scripts use `estimate_initial_attributes` + `extract_initial_attributes`:

```python
from preprocessing.Initial_Attributes import estimate_initial_attributes, extract_initial_attributes

multi_inputs = estimate_initial_attributes(multi_inputs, device="cuda:0")
init_attrs = extract_initial_attributes(multi_inputs, device="cuda:0")
# init_attrs.is_dynamic — per-instance dynamic flags (||v|| >= 0.20 m/frame)
```

Compare against legacy labels: `python scripts/compare_dynamic_mask_gt.py --config sequence_00`

Depth path is resolved via `file_io_utils.get_depth_filename()` (`data_2d_raw` → `pseudo_depth_ssl` by default).

---

## Dynamic label generation (optional)

Offline export of per-instance dynamic flags in legacy `dynamic_mask.txt` format. Uses the same GT bbox velocity rule as training (`DEFAULT_DYNAMIC_VELOCITY_THRESHOLD = 0.20 m/frame`).

**Purpose:** reproduce legacy-compatible labels for [validator](../validator/README.md) (Step1 + Step3 read txt; training does not).

```bash
# One sequence → {DATASET.ROOT}/dynamic_attributes_est_gt/<sync>/dynamic_mask.txt
python -m preprocessing.Dynamic_Labels.pipeline --config sequence_00

# All sequences
sh preprocessing/scripts/generate_dynamic_labels.sh

# Quality gate (compare vs legacy dynamic_attributes_est/)
python scripts/compare_dynamic_mask_gt.py --config sequence_00
python scripts/sweep_dynamic_threshold_global.py
```

```python
from preprocessing.Dynamic_Labels import generate_dynamic_labels

result = generate_dynamic_labels(
    dataset_root="/path/to/KITTI360_For_Upload",
    filenames_path="filenames/.../sampled_image_filenames.txt",
    output_path="dynamic_attributes_est_gt/.../dynamic_mask.txt",
)
```

Then point validator at the output:

```bash
export DYNAMIC_DIRNAME=${ROOT_DIRNAME}/dynamic_attributes_est_gt
sh validator/make_predictions_scripts/run_evaluation_pipeline.sh
```

See [Dynamic_Labels/README.md](Dynamic_Labels/README.md) for CLI options.
