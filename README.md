# VSRD++: Autolabeling for 3D Object Detection via Instance-Aware Volumetric Silhouette Rendering

[![arXiv](https://img.shields.io/badge/arXiv-2512.01178-b31b1b.svg)](https://arxiv.org/abs/2512.01178)

VSRD++ is an advanced weakly supervised 3D object detection framework that extends the original VSRD (Volumetric Silhouette Rendering) method with dynamic object modeling capabilities. The system operates in a two-stage pipeline: **Multi-View 3D Auto-Labeling** followed by **Monocular 3D Detection Training**.

## Overall Pipeline

<div align="center">
  <img src="figs/teaser.png" width="100%">
</div>

---

## Data Preprocessing: nuScenes

This repository currently supports a simple two-step preprocessing pipeline for nuScenes,
implemented under `data_preprocessing/nuscenes/`:

1. **Step 1: build temporal infos**
2. **Step 2: add dynamic / static flags**

### Step 1: build temporal infos

Scripts:
- `data_preprocessing/nuscenes/step1_create_infos.py`
- backend: `data_preprocessing/nuscenes/nuscenes_converter.py`

This step scans the raw nuScenes data and produces temporal info files:

- `*_infos_temporal_train.pkl`
- `*_infos_temporal_val.pkl`

Each entry in `infos` (one per nuScenes sample / keyframe) contains:

- key-frame LiDAR path (`lidar_path`)
- historical LiDAR sweeps and transforms (`sweeps`)
- 6 camera image paths + cam–LiDAR extrinsics + intrinsics (`cams`)
- ego and global poses (`lidar2ego_*`, `ego2global_*`)
- CAN bus signals (`can_bus`)
- 3D ground truth (`gt_boxes`, `gt_names`, `gt_velocity`, `valid_flag`)
- temporal meta-data (`scene_token`, `frame_idx`, `prev`, `next`, `timestamp`)

Example:

```bash
python data_preprocessing/nuscenes/step1_create_infos.py \
  --root_path /path/to/nuscenes \
  --can_bus_root_path /path/to/nuscenes \
  --out_path ./data/nuscenes \
  --info_prefix nuscenes \
  --version v1.0-trainval \
  --max_sweeps 10
```

The resulting `infos` are ordered by scene and time (same `scene_token` grouped together,
`frame_idx` increasing within each scene).

### Step 2: add dynamic / static flags

Script:
- `data_preprocessing/nuscenes/step2_add_dynamic_flags.py`

This step augments each `info` with a boolean mask `gt_is_dynamic` (per instance), using:

- class ∈ a configurable “potentially dynamic” set
  (car, truck, trailer, bus, construction_vehicle, bicycle, motorcycle, pedestrian, …)
- LiDAR-frame speed \(\sqrt{v_x^2 + v_y^2}\) > `speed_thresh` (default 0.5 m/s)
- `valid_flag == True`

Example:

```bash
python data_preprocessing/nuscenes/step2_add_dynamic_flags.py \
  --info_path ./data/nuscenes/nuscenes_infos_temporal_train.pkl \
  --speed_thresh 0.5
```

Output:

- `./data/nuscenes/nuscenes_infos_temporal_train_dyn.pkl`

which keeps all original fields and adds:

- `gt_is_dynamic`: boolean array aligned with `gt_names`.

You can further customize:

- `--out_path` to control the output filename.
- `--dynamic_classes` to redefine the set of potentially dynamic categories, e.g.:

  ```bash
  --dynamic_classes car,truck,trailer,bus,construction_vehicle
  ```

These dynamic-aware infos (`*_dyn.pkl`) can then be consumed by downstream
auto-labeling, dynamic-object modeling, or analysis code.
