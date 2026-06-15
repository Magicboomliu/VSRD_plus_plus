# VSRD++: Autolabeling for 3D Object Detection via Instance-Aware Volumetric Silhouette Rendering

[![arXiv](https://img.shields.io/badge/arXiv-2512.01178-b31b1b.svg)](https://arxiv.org/abs/2512.01178)

VSRD++ is an advanced weakly supervised 3D object detection framework that extends the original VSRD (Volumetric Silhouette Rendering) method with dynamic object modeling capabilities. The system operates in a two-stage pipeline: **Multi-View 3D Auto-Labeling** followed by **Monocular 3D Detection Training**.

## Overall Pipeline

<div align="center">
  <img src="figs/teaser.png" width="100%">
</div>

---

## 🚀 Key Features

- **Dynamic Object Modeling**: Three modeling approaches for handling dynamic objects
  - Instance Residual Field (via MLP)
  - Vector Velocity Modeling
  - Scalar Velocity Modeling
- **Multi-View Optimization**: Scene-wise 3D bounding box rendering using multiple camera views
- **Weak Supervision**: Only requires 2D segmentation masks and camera poses
- **Robustness Enhancement**: MaskEroder functionality for simulating imperfect mask quality

---

## 📋 Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [Detailed Workflow](#detailed-workflow)
- [Configuration](#configuration)
- [Evaluation](#evaluation)

---

## 🔧 Installation

This project uses [pixi](https://pixi.sh) for environment management. Pixi handles all conda and PyPI dependencies in a single lockfile — no manual conda/pip steps needed.

### 1. Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

### 2. Clone and set up the environment

```bash
git clone git@github.com:Magicboomliu/VSRD_plus_plus.git
cd VSRD_plus_plus

# Resolve and install all dependencies (conda + PyPI)
pixi install
```

This automatically installs Python 3.10, PyTorch 1.13 (CUDA 11.6), and all required packages into an isolated environment at `.pixi/envs/default/`.

### 3. (Optional) Install nerfacc

`nerfacc` is only needed for the NerfAcc-based rendering path. It requires a special pre-built wheel:

```bash
pixi run install-nerfacc
```

### 4. Verify installation

```bash
pixi run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pixi run test-data   # check dataset paths (after setting DATASET.ROOT)
```

> **Note:** Large model weights (`.pth`, ~GB each) are listed in `.gitignore` and must be downloaded separately — WAFT-Stereo, LEAStereo, etc.

---

## 📦 Data Preparation

### Dataset layout

Point `trainer/configs/base.json` → `TRAIN.DATASET.ROOT` to your KITTI360 root. Expected structure:

```
KITTI360_For_Upload/
├── data_2d_raw/              # RGB images (KITTI360)
├── annotations/              # 2D instance masks
├── filenames/                # Sampled frame lists (.txt, relative paths inside)
├── pseudo_depth_ssl_waft_stereo/  # WAFT-Stereo pseudo depth (default for training & tests)
├── dynamic_attributes_est/        # Legacy dynamic labels (optional reference)
└── dynamic_attributes_est_gt/     # Generated dynamic labels (default in configs & validator)
```

Filenames inside `.txt` files use **relative paths**; only `DATASET.ROOT` in config needs to change when moving machines.

Verify paths before training:

```bash
pixi run test-data
pytest tests/test_dataset_paths.py -v
```

### Path conventions (single source of truth)

All default on-disk paths are defined in `preprocessing/dataset_paths.py`:

| Artifact | Default relative path | Used by |
|----------|----------------------|---------|
| Pseudo depth | `pseudo_depth_ssl_waft_stereo/<sequence>/image_00/data_rect/<frame>.png` | WAFT script, training attribute init, `pixi run test-data` |
| Dynamic labels | `dynamic_attributes_est_gt/<sequence>/dynamic_mask.txt` | `Dynamic_Labels` pipeline, validator, `sequence_XX.json` → `DYNAMIC_LABELS_PATH` |

`<sequence>` is the KITTI360 folder name (e.g. `2013_05_28_drive_0007_sync`).

Legacy zip folders (`pseudo_depth_ssl/`, `dynamic_attributes_est/syncXX/`) remain optional for comparison only.

### Data preparation checklist

| Step | Required for | Command |
|------|--------------|---------|
| Set `DATASET.ROOT` | Everything | Edit `trainer/configs/base.json` |
| WAFT pseudo depth | Attribute-init training | `sh preprocessing/scripts/generate_pseudo_depth_waft.sh 0006` or `pixi run gen-depth-seq 0006` |
| Dynamic labels | Validator only | `python -m preprocessing.Dynamic_Labels.pipeline --config sequence_00` or `pixi run gen-dynamic` |
| Verify paths | Before train/eval | `pixi run test-data` |
| Compare dynamic | After generating labels | `python scripts/compare_dynamic_mask_gt.py --config sequence_00` |

---

### 1. Download KITTI360 Dataset

Download from [Google Drive](https://drive.google.com/file/d/1syBPCdU0Hs2AWgQfsPohqMFXNEpM3eWV/view?usp=sharing) or use curl:

```bash
curl -H "Authorization: Bearer <YOUR_TOKEN>" \
     https://www.googleapis.com/drive/v3/files/1syBPCdU0Hs2AWgQfsPohqMFXNEpM3eWV?alt=media \
     -o KITTI360_For_Upload.zip
unzip KITTI360_For_Upload.zip
```

Set the root in config:

```json
// trainer/configs/base.json
"DATASET": {
  "ROOT": "/path/to/KITTI360_For_Upload",
  ...
}
```

Per-sequence overrides live in `trainer/configs/sequence_XX.json` (relative paths only).

### 2. Generate pseudo depth (WAFT-Stereo, recommended)

Model weights (~2 GB) are **not** in the repo — auto-downloaded on first run from [HuggingFace](https://huggingface.co/MemorySlices/WAFT-Stereo), or place manually at `preprocessing/disparity_estimation/WAFT-Stereo/ckpts/Real/DAv2L-5.pth`.

WAFT uses a **separate pixi env** (PyTorch ≥ 2.0):

```bash
cd preprocessing/disparity_estimation/WAFT-Stereo
pixi install
cd ../..

# Single sequence (recommended for first run)
sh preprocessing/scripts/generate_pseudo_depth_waft.sh 0006

# All sequences (slow — tens of hours)
sh preprocessing/scripts/generate_pseudo_depth_waft.sh

# Custom dataset root
DATASET_ROOT=/path/to/KITTI360_For_Upload sh preprocessing/scripts/generate_pseudo_depth_waft.sh 0006
```

**Output path (default):**

```text
{DATASET_ROOT}/pseudo_depth_ssl_waft_stereo/<sequence>/image_00/data_rect/<frame>.png
```

Training, attribute init, and `pixi run test-data` all read this folder (`preprocessing/dataset_paths.py`).

Pixi shortcuts (from project root, after WAFT `pixi install`):

```bash
pixi run gen-depth          # all sequences
pixi run gen-depth-seq 0006 # one sequence
pixi run gen-dynamic        # all dynamic_mask.txt files
```

See [preprocessing/README.md](preprocessing/README.md) for API usage (`preprocessing/apis/depth_estimator.py`).

Legacy LEAStereo can write to `pseudo_depth_ssl/` via `preprocessing/disparity_estimation/Leastereo/` (optional).

### 3. Dynamic / static classification

| Stage | How dynamic/static is determined |
|-------|----------------------------------|
| **Training** | Online from 3D bbox GT velocity (`\|\|v\|\| >= 0.20 m/frame`); no `dynamic_mask.txt` at train time |
| **Validator** | Reads `dynamic_mask.txt` from `DYNAMIC_DIRNAME` (Step1 + Step3) |
| **Offline export** | `preprocessing/Dynamic_Labels` writes txt for validator use |

**Recommended workflow** (generate → verify → evaluate):

```bash
# 1. Export dynamic_mask.txt (same rule as training: 0.20 m/frame)
python -m preprocessing.Dynamic_Labels.pipeline --config sequence_00
# or all 9 sequences:
sh preprocessing/scripts/generate_dynamic_labels.sh

# Optional: custom threshold / dataset root (uses trainer/configs/base.json → DATASET.ROOT)
THRESHOLD=0.20 sh preprocessing/scripts/generate_dynamic_labels.sh sequence_07
```

**Output path (default):**

```text
{DATASET.ROOT}/dynamic_attributes_est_gt/2013_05_28_drive_0007_sync/dynamic_mask.txt
```

(`<sequence>` = folder name under `filenames/R50-N16-M128-B16/`, same as in config `DYNAMIC_LABELS_PATH`.)

```bash
# 2. Sanity check (online rule vs generated txt in config path)
python scripts/compare_dynamic_mask_gt.py --config sequence_00

# Compare vs legacy zip labels instead:
python scripts/compare_dynamic_mask_gt.py --config sequence_00 \
  --dynamic-path dynamic_attributes_est/sync00/dynamic_mask.txt

# 3. Run validator (reads {ROOT}/dynamic_attributes_est_gt/<sequence>/)
export ROOT_DIRNAME=/path/to/KITTI360_For_Upload
export CKPT_DIRNAME=/path/to/trainer/ckpts/your_run
export DYNAMIC_DIRNAME=${ROOT_DIRNAME}/dynamic_attributes_est_gt
sh validator/make_predictions_scripts/run_evaluation_pipeline.sh
```

See [preprocessing/Dynamic_Labels/](preprocessing/Dynamic_Labels/), [preprocessing/Initial_Attributes/](preprocessing/Initial_Attributes/), and [validator/](validator/README.md).

---

## 🏗️ System Architecture

VSRD++ follows a **two-stage pipeline**:

### Stage 1: Multi-View 3D Auto-Labeling
- **Input**: Sequential 2D images, segmentation masks, camera poses
- **Process**: Optimization-based volumetric rendering with dynamic object modeling
- **Output**: 3D bounding box pseudo-labels in KITTI format

### Stage 2: Monocular 3D Detection Training
- **Input**: Pseudo-labels from Stage 1
- **Process**: Train monocular 3D detectors (WeakM3D, MonoFlex, MonoDeTR, etc.)
- **Output**: Trained monocular 3D detection models

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Preprocessing  │ --> │  Stage 1:        │ --> │  Stage 2:       │
│  - Depth        │     │  Multi-View      │     │  Monocular 3D   │
│  - Dynamic txt  │     │  Auto-Labeling   │     │  Detection      │
│  - Attributes   │     │  (VSRD++)        │     │  Training       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                        ▲
         │  dynamic_mask.txt      │  online GT velocity (train)
         └────────────────────────┘  txt file (validator)
```

---

## 🚀 Quick Start

### Stage 1: Multi-View 3D Auto-Labeling Training

Training entry points live under `trainer/`:

| Script | Purpose |
|--------|---------|
| `train.py` | Standard VSRD++ (attribute init + online dynamic/static) |
| `train_no_init.py` | Ablation: skip attribute initialization |
| `train_ablation.py` | Mask-erode robustness (`--erode_ratio`) |
| `train_sharded.py` | 64-way split training |
| `train_legacy.py` | Deprecated; do not use for new runs |

**Recommended (single GPU):**

```bash
cd trainer
CUDA_VISIBLE_DEVICES=0 torchrun \
    --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    --nnodes 1 --nproc_per_node 1 \
    train.py --config_path sequence_07 --device_id 0
```

From repo root via pixi:

```bash
pixi run train -- --config_path sequence_07 --device_id 0
pixi run train-no-init -- --config_path sequence_07 --device_id 0
pixi run train-ablation -- --config_path ablation_selective --device_id 0 --erode_ratio 0.03
```

Shell helpers in `trainer/scripts/` (mirror the Python entry points):

| Script | Calls |
|--------|--------|
| `train.sh` | `train.py` (default) or `train_no_init.py` via `run_with_init` / `run_no_init` |
| `train_smoke.sh` | Single-GPU smoke test with `train.py` |
| `train_ablation.sh` | `train_ablation.py` + `--erode_ratio` |
| `train_sharded.sh` | `train_sharded.py` for SPLITS64 |
| `train_tsubame.sh` | Tsubame cluster job wrapper for `train.py` |

```bash
cd trainer/scripts
sh train.sh              # default: with attribute init
# or switch mode inside train.sh: run_no_init

ERODE_RATIO=0.05 sh train_ablation.sh
pixi run train-shell     # same as train.sh from repo root
pixi run train-smoke
```

`--config_path` accepts a config stem (`sequence_07`, `ablation_selective`) or bare sequence id (`07` → `sequence_07.json`).

### Stage 1: Ablation Studies

For mask-quality ablations with custom erode ratios:

```bash
cd trainer/scripts
ERODE_RATIO=0.03 CONFIG_PATH=ablation_selective sh train_ablation.sh

# from repo root:
ERODE_RATIO=0.03 CONFIG_PATH=ablation_selective bash trainer/scripts/train_ablation.sh
```

Optional environment variables:
- `CONFIG_PATH`: e.g. `ablation_selective`
- `ERODE_RATIO`: mask erode ratio (0.0–1.0)
- `CKPT_DIRNAME`, `LOG_DIRNAME`, `OUT_DIRNAME`: custom output paths

---

## 📖 Detailed Workflow

### Phase 1: Preprocessing

#### 1.1 Pseudo depth (required for attribute init)

Generate pseudo depth with WAFT-Stereo (see [Quick Start §2](#2-generate-pseudo-depth-waft-stereo-recommended) and [preprocessing/README.md](preprocessing/README.md)).

**Required models (not shipped in repo — see `.gitignore`):**
- **Depth**: [WAFT-Stereo](https://github.com/MemorySlices/WAFT-Stereo) (recommended) or [LEAStereo](https://github.com/XuelianCheng/LEAStereo) (legacy)

#### 1.2 Initial attribute estimation

Used at training time via `estimate_initial_attributes()` (depth + RoI LiDAR + ICP). Smoke test:

```bash
python -m preprocessing.Initial_Attributes.pipeline
```

This step provides:
- Initial ROI LiDAR point clouds (from `pseudo_depth_ssl_waft_stereo/`)
- Initial velocity from ICP
- Location and orientation from ROI LiDAR + velocity
- Dynamic/static flags online (`||v|| >= 0.20 m/frame`; no txt at train time)

See [preprocessing/Initial_Attributes/README.md](preprocessing/Initial_Attributes/README.md).

#### 1.3 Dynamic label export (for validator)

Training infers dynamic/static online; validator reads `dynamic_mask.txt`. Generate and verify before evaluation:

```bash
sh preprocessing/scripts/generate_dynamic_labels.sh
python scripts/compare_dynamic_mask_gt.py --config sequence_00
```

See [preprocessing/Dynamic_Labels/README.md](preprocessing/Dynamic_Labels/README.md).

### Phase 2: Stage 1 - Multi-View 3D Auto-Labeling

#### 2.1 Training Configuration

Configs are JSON under `trainer/configs/`:

```
trainer/configs/
├── base.json              # defaults + DATASET.ROOT (edit this first)
├── sequence_XX.json       # one sequence per file (FILENAMES only)
├── smoke.json             # smoke test config (same training settings as base; use a short FILENAMES list)
├── ablation_selective.json
├── ablation_full.json
├── inference.json
└── SPLITS64/split_sub.json
```

`DYNAMIC_LABELS_PATH` is **auto-derived** from `FILENAMES` for standard sequences (`dynamic_attributes_est_gt/<sequence>/dynamic_mask.txt`). Custom ablation/split configs set it explicitly.

Load in code: `from trainer.configs import load_config; cfg = load_config("07")`

Key fields in `base.json`:

```json
{
  "TRAIN": {
    "DATASET": { "ROOT": "/path/to/KITTI360_For_Upload" },
    "USE_RDF_MODELING": true,
    "USE_DYNAMIC_MASK": true,
    "USE_DYNAMIC_MODELING": true,
    "DYNAMIC_MODELING_TYPE": "vector_velocity",
    "OPTIMIZATION_NUM_STEPS": 3000
  }
}
```

Per-sequence paths (`FILENAMES`, `DYNAMIC_LABELS_PATH`) are relative to `DATASET.ROOT` in `sequence_XX.json`. See [preprocessing/dataset_paths.py](preprocessing/dataset_paths.py).

#### 2.2 Training

Standard command (from `trainer/`):

```bash
cd trainer
CUDA_VISIBLE_DEVICES=0 torchrun \
    --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    --nnodes 1 --nproc_per_node 1 \
    train.py --config_path sequence_07 --device_id 0
```

Notes:
- **Pseudo depth** under `pseudo_depth_ssl_waft_stereo/` is required when attribute initialization is enabled (default in `train.py`).
- **Dynamic labels** are **not** read at train time; motion is inferred online from 3D bbox GT velocity. Generate `dynamic_attributes_est_gt/` only before validator evaluation.


### Phase 3: Evaluation Pipeline

> **Prerequisite:** generate `dynamic_attributes_est_gt/` and pass `compare_dynamic_mask_gt.py` before pointing validator at the new labels. See [validator/README.md](validator/README.md).

#### 3.1 Unified Evaluation Pipeline (Recommended)

```bash
export ROOT_DIRNAME=/path/to/KITTI360_For_Upload
export CKPT_DIRNAME=/path/to/trainer/ckpts/your_run
export DYNAMIC_DIRNAME=${ROOT_DIRNAME}/dynamic_attributes_est_gt

cd validator/make_predictions_scripts
sh run_evaluation_pipeline.sh
```

Environment variables (all optional — defaults in shell scripts):

| Variable | Default | Purpose |
|----------|---------|---------|
| `ROOT_DIRNAME` | KITTI360 root | Dataset root |
| `CKPT_DIRNAME` | `trainer/ckpts` | Trained checkpoint directory |
| `DYNAMIC_DIRNAME` | `{ROOT}/dynamic_attributes_est_gt` | Parent of `<sequence>/dynamic_mask.txt` |
| `INPUT_MODEL_TYPE` | `velocity_with_init` | Model type for prediction export |

This executes:
1. **Step 1**: Generate predictions and GT in JSON format (reads `dynamic_mask.txt`)
2. **Step 2**: Convert to KITTI3D `.txt` format
3. **Step 3**: Assign dynamic flags to GT KITTI labels (same `dynamic_mask.txt`)
4. **Step 4**: Organize into KITTI3D dataset structure

#### 3.2 Manual Evaluation Steps

**Step 1: Generate Predictions**

```bash
cd validator/make_predictions_scripts
sh make_prediction.sh
```

**Step 2: Convert to KITTI3D Format**

```bash
sh convert_into_kitti_format.sh
```

**Step 3: Dynamic Attribute Assignment** (reads `dynamic_mask.txt`, same as Step 1)

```bash
export DYNAMIC_DIRNAME=${ROOT_DIRNAME}/dynamic_attributes_est_gt
sh dynamic_attribute.sh
```

**Step 4: Organize Dataset Structure**

```bash
cd ../dataset_structure_configuration
sh conversion_kitti3d_structure.sh
```

**Step 5: Calculate IoU**

```bash
cd ../stage1_evaluation_scripts
sh get_iou.sh
```

**Step 6: Calculate mAP**

```bash
sh get_mAP.sh
```

---

## ⚙️ Configuration

### Training scripts & pixi tasks

| Entry | Command |
|-------|---------|
| Standard train | `pixi run train -- --config_path sequence_07 --device_id 0` |
| No attribute init | `pixi run train-no-init -- --config_path sequence_07 --device_id 0` |
| Mask erode ablation | `pixi run train-ablation -- --config_path ablation_selective --device_id 0 --erode_ratio 0.03` |
| Shell launcher | `pixi run train-shell` → `trainer/scripts/train.sh` |
| Smoke test | `pixi run train-smoke` → `trainer/scripts/train_smoke.sh` |

### Training Script Arguments

Run from `trainer/` (or prefix paths with `trainer/` when using pixi from repo root):

```bash
cd trainer
CUDA_VISIBLE_DEVICES=0 torchrun \
    --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    --nnodes 1 --nproc_per_node 1 \
    train.py \
    --config_path sequence_07 \
    --device_id 0 \
    --ckpt_dirname "/path/to/ckpts" \
    --log_dirname "/path/to/logs" \
    --out_dirname "/path/to/outputs"
```

| Argument | Description |
|----------|-------------|
| `--config_path` | Config stem: `sequence_07`, `ablation_selective`, or bare id `07` |
| `--device_id` | CUDA device index |
| `--ckpt_dirname` | Override checkpoint directory (optional) |
| `--log_dirname` | Override log directory (optional) |
| `--out_dirname` | Override output directory (optional) |
| `--erode_ratio` | (`train_ablation.py` only) mask erode ratio for robustness tests |

### Dynamic Modeling Types

1. **MLP-based Residual Field**
   - Uses MLP to learn instance-specific residual fields
   - Good for complex motion patterns

2. **Vector Velocity Modeling**
   - Models velocity as 3D vector
   - Suitable for objects with consistent motion direction

3. **Scalar Velocity Modeling**
   - Models velocity as scalar magnitude
   - Simpler, faster convergence

---

## 📊 Evaluation

### Dynamic label scripts

| Script | Purpose |
|--------|---------|
| `preprocessing/scripts/generate_pseudo_depth_waft.sh` | Batch WAFT depth → `pseudo_depth_ssl_waft_stereo/` |
| `preprocessing/scripts/generate_dynamic_labels.sh` | Batch dynamic txt → `dynamic_attributes_est_gt/` |
| `scripts/compare_dynamic_mask_gt.py` | Online rule vs reference `dynamic_mask.txt` (default: config path) |
| `scripts/sweep_dynamic_threshold_global.py` | Global threshold sweep (9 sequences) |
| `pixi run gen-depth` / `gen-depth-seq` / `gen-dynamic` | Pixi wrappers for preprocessing |
| `pixi run train` / `train-no-init` / `train-ablation` / `train-shell` / `train-smoke` | Pixi wrappers for Stage 1 training |

### Visualization

#### Projected 3D Boxes Visualization

```bash
cd validator/stage2_visualization_scripts
sh visualization_projected3d.sh
```

Edit the script to set:
- `ERODE_RATIO`: Apply erode to masks during visualization
- `OPTIONS`: `"pd_only"` or `"pd_gt"`

#### BEV Visualization

```bash
sh visualization_bev.sh
```

### Metrics

- **IoU (3D/BEV)**: Intersection over Union for 3D and BEV boxes
- **mAP**: Mean Average Precision at different IoU thresholds
- **Accuracy**: Percentage of boxes above IoU thresholds (0.25, 0.50)

---


### Custom Output Directories

```bash
cd trainer
CUDA_VISIBLE_DEVICES=0 torchrun \
    --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    --nnodes 1 --nproc_per_node 1 \
    train.py \
    --config_path ablation_selective \
    --device_id 0 \
    --ckpt_dirname "/path/to/ckpts" \
    --log_dirname "/path/to/logs" \
    --out_dirname "/path/to/outputs"
```

---

## 📁 Sub-modules

Detailed documentation for each module:

- **[preprocessing/](preprocessing/README.md)**: Data preparation
  - Pseudo depth (WAFT-Stereo / LEAStereo)
  - Initial attribute estimation (online dynamic/static)
  - Dynamic label export (`Dynamic_Labels/`)

- **[trainer/](trainer/README.md)**: Core training code for Stage 1
  - `train.py`, `train_no_init.py`, `train_ablation.py`, `train_sharded.py`
  - Multi-view 3D auto-labeling, dynamic object modeling, volumetric rendering

- **[validator/](validator/README.md)**: Evaluation tools and metrics
  - Prediction generation (reads `dynamic_mask.txt`)
  - KITTI format conversion
  - IoU and mAP calculation
  - Visualization tools

---

## 🔬 Ablation Studies

### Main Ablation Settings

1. **Dynamic Modeling**: Enable/disable dynamic object modeling
2. **Pseudo Attribute Initialization**: Use/ignore initial attributes from LiDAR
3. **Mask Quality**: Test with different erode ratios (0.0, 0.05, 0.10)

### Configuration Example

Edit `trainer/configs/base.json` or a per-sequence file such as `sequence_07.json`:

```json
{
  "TRAIN": {
    "USE_RDF_MODELING": true,
    "USE_DYNAMIC_MASK": true,
    "USE_DYNAMIC_MODELING": true,
    "DYNAMIC_MODELING_TYPE": "vector_velocity"
  }
}
```

`DYNAMIC_MODELING_TYPE`: `mlp`, `vector_velocity`, or `scalar_velocity`.

---


## 🙏 Acknowledgments

- [IGEVStereo](https://github.com/gangweix/IGEV) for depth estimation
- [InternImage](https://github.com/OpenGVLab/InternImage) for 2D detection
- [VSRD](https://github.com/Magicboomliu/VSRD) for providing the base of this code.


---

