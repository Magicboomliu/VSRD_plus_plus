# Initial attribute estimation pipeline

Three-step estimator: **RoI LiDAR → GT-based dynamic mask → ICP velocity → location/orientation**.

## Usage

```python
from preprocessing.Initial_Attributes import (
    estimate_initial_attributes,
    extract_initial_attributes,
    AttributeKeys,
)

multi_inputs = estimate_initial_attributes(
    multi_inputs,
    device="cuda:0",
)

init = extract_initial_attributes(multi_inputs, device="cuda:0")
# init.est_location, init.est_velocity, init.est_orientation, init.is_dynamic, init.roi_lidar_valid
```

Smoke test:

```bash
python -m preprocessing.Initial_Attributes.pipeline --input Debug_Examples/exampleV2.pkl
```

## Modules

| File | Role |
|------|------|
| `pipeline.py` | Orchestration |
| `roi_lidar.py` | Step 1 — pseudo-depth RoI point clouds |
| `velocity.py` | Step 2 — ICP velocity |
| `location_orientation.py` | Step 3 — location & orientation |
| `gt_attributes.py` | GT from 3D bounding boxes (debug / fallback) |
| `keys.py` | Canonical field names on `multi_inputs` |
| `extract.py` | Read results for trainer initialization |

## `multi_inputs` fields (target frame `0`)

| Key | Description |
|-----|-------------|
| `pseudo_depth` | Loaded depth tensor |
| `roi_lidar` | `{instance_id: point_cloud}` |
| `roi_lidar_valid` | Whether RoI LiDAR is usable |
| `est_velocity` | `[N, 3]` per-instance velocity |
| `est_location` | `[1, N, 3]` centroids |
| `est_orientation` | `[1, N, 3, 3]` rotation matrices |
| `is_dynamic` | Per-instance dynamic flags (GT bbox speed ≥ 0.20 m/frame) |
| `gt_location`, `gt_velocity`, … | Optional bbox GT |

## Dynamic / static threshold

Speed is **m per relative frame index** (displacement / frame gap), not m/s.

Default: `DEFAULT_DYNAMIC_VELOCITY_THRESHOLD = 0.20` in `gt_attributes.py`, chosen by sweeping all 9 training sequences against legacy `dynamic_mask.txt` (99.98% accuracy, recall 100%).

```python
multi_inputs = estimate_initial_attributes(
    multi_inputs,
    device="cuda:0",
    dynamic_velocity_threshold=0.20,  # optional override
)
```

Re-run the global sweep: `python scripts/sweep_dynamic_threshold_global.py`

To export labels for validator: [Dynamic_Labels/README.md](../Dynamic_Labels/README.md)
