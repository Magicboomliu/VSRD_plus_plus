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

> **Note:** Large model weights (`.pth`, ~GB each) are listed in `.gitignore` and must be downloaded separately — WAFT-Stereo, MeMFlow, LEAStereo, etc.

---

## 📦 Data Preparation

### Dataset layout

Point `trainer/configs/base.json` → `TRAIN.DATASET.ROOT` to your KITTI360 root. Expected structure:

```
KITTI360_For_Upload/
├── data_2d_raw/              # RGB images (KITTI360)
├── annotations/              # 2D instance masks
├── filenames/                # Sampled frame lists (.txt, relative paths inside)
├── pseudo_depth_ssl/         # Legacy pseudo depth (LEAStereo, optional)
├── pseudo_depth_ssl_waft_stereo/  # WAFT-Stereo pseudo depth (recommended)
└── dynamic_attributes_est/   # Dynamic object labels (generated)
```

Filenames inside `.txt` files use **relative paths**; only `DATASET.ROOT` in config needs to change when moving machines.

Verify paths before training:

```bash
pixi run test-data
```

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
```

Output: `{DATASET_ROOT}/pseudo_depth_ssl_waft_stereo/<sequence>/image_00/data_rect/*.png`  
Format: uint16 PNG, depth in metres = pixel value / 256.

See [preprocessing/README.md](preprocessing/README.md) for API usage (`preprocessing/apis/depth_estimator.py`).

Legacy LEAStereo path: `sh preprocessing/scripts/generate_pseudo_depth.sh` (requires manual Kitti15 weight download).

### 3. Generate dynamic labels

Requires MeMFlow optical flow weights (also not in repo). See [preprocessing/README.md](preprocessing/README.md) → Dynamic Static Filtering.

Output: `{DATASET_ROOT}/dynamic_attributes_est/syncXX/dynamic_mask.txt`

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
│  - Flow/Depth   │     │  Multi-View      │     │  Monocular 3D   │
│  - Dynamic Mask │     │  Auto-Labeling    │     │  Detection      │
│  - Attributes   │     │  (VSRD++)         │     │  Training       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

---

## 🚀 Quick Start

### Stage 1: Multi-View 3D Auto-Labeling Training

```bash
cd trainer/scripts
sh DDP_RUN.sh
```

The script contains two training modes:
- **`TRAIN_DDP_VSRDPP`**: With attribute initialization (IGEVStereo pretrained)
- **`TRAIN_DDP_VSRD_SIMPLE`**: Without attribute initialization

### Stage 1: Ablation Studies

For ablation studies with custom configurations:

```bash
cd trainer/scripts
sh DDP_RUN_ROUND1_ABLATION.sh
```

Edit the script to configure:
- `CONFIG_PATH`: Configuration identifier
- `ERODE_RATIO`: Mask erode ratio (0.0-1.0) for robustness testing
- `CKPT_DIRNAME`, `LOG_DIRNAME`, `OUT_DIRNAME`: Custom output paths

---

## 📖 Detailed Workflow

### Phase 1: Preprocessing

#### 1.1 Dynamic/Static Classification

Generate dynamic masks using optical flow and depth consistency:

```bash

# Step 1: Generate GT dynamic labels
python dynamic_mask_gt_generataion.py \
    --seed 1234 \
    --neighbour_sample 16 \
    --image_folder $image_folder \
    --filename_folder $filename_folder \
    --saved_folder $saved_folder \
    --use_multi_thread

# Step 2: Estimate dynamic masks using flow/depth
python preprocess.py \
    --seed 1234 \
    --neighbour_sample 16 \
    --image_folder $image_folder \
    --filename_folder $filename_folder \
    --saved_folder $saved_folder \
    --optical_flow_model_path $flow_model_path \
    --use_multi_thread
```

**Required Models (not shipped in repo — see `.gitignore`):**
- **Optical Flow**: [MeMFlow (CVPR 2024)](https://github.com/DQiaole/MemFlow) — download `.pth` to `preprocessing/optical_flow_estimation/MeMFlow/ckpts/`
- **Depth**: [WAFT-Stereo](https://github.com/MemorySlices/WAFT-Stereo) (recommended) or [LEAStereo](https://github.com/XuelianCheng/LEAStereo) (legacy)

#### 1.2 Initial Attribute Estimation

Get initial 3D attributes (location, orientation, velocity) from LiDAR:

```bash
python Get_Initial_Attributes.py
```

This step provides:
- Initial ROI LiDAR point clouds
- Initial velocity from ICP
- Location and orientation from ROI LiDAR + velocity

#### 1.3 Depth Estimation

Generate pseudo depth with WAFT-Stereo (recommended):

```bash
sh preprocessing/scripts/generate_pseudo_depth_waft.sh 0006
```

Legacy LEAStereo:

```bash
sh preprocessing/scripts/generate_pseudo_depth.sh 0006
```

Details: [preprocessing/README.md](preprocessing/README.md).

### Phase 2: Stage 1 - Multi-View 3D Auto-Labeling

#### 2.1 Training Configuration

Configs are JSON under `trainer/configs/`:

```
trainer/configs/
├── base.json           # defaults + DATASET.ROOT
├── sequence_00.json    # per-sequence overrides (relative paths)
└── ...
```

Load in code: `from trainer.configs import load_config; cfg = load_config("00")`

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

Per-sequence paths (`FILENAMES`, `DYNAMIC_LABELS_PATH`) are relative to `DATASET.ROOT` in `sequence_XX.json`.

#### 2.2 Training

Standard training command:

```bash
python train_sequence_ddp.py \
    --config_path "00" \
    --device_id 0
```


### Phase 3: Evaluation Pipeline

#### 3.1 Unified Evaluation Pipeline (Recommended)

Run all evaluation steps in one command:

```bash
cd validator/make_predictions_scripts
sh run_evaluation_pipeline.sh
```

This executes:
1. **Step 1**: Generate predictions and GT in JSON format
2. **Step 2**: Convert to KITTI3D `.txt` format
3. **Step 3**: Assign dynamic/static labels
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

**Step 3: Dynamic Attribute Assignment**

```bash
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

### Training Script Arguments

```bash
python train_sequence_ddp.py \
    --config_path "00" \                     # Config identifier
    --device_id 0 \                          # CUDA device ID
    --ckpt_dirname "<HOME>/ckpts" \         # Custom checkpoint directory
    --log_dirname "<HOME>/logs" \            # Custom log directory
    --out_dirname "<HOME>/outputs"           # Custom output directory
```

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

Specify custom paths for checkpoints, logs, and outputs:

```bash
python train_sequence_ddp.py \
    --config_path "ablation_selective" \
    --device_id 0 \
    --ckpt_dirname "<HOME>/ckpts" \
    --log_dirname "<HOME>/logs" \
    --out_dirname "<HOME>/outputs"
```

---

## 📁 Sub-modules

Detailed documentation for each module:

- **[trainer/](trainer/README.md)**: Core training code for Stage 1
  - Multi-view 3D auto-labeling
  - Dynamic object modeling
  - Volumetric rendering

  - Dynamic/static classification
  - Optical flow and depth estimation
  - Initial attribute estimation

- **[validator/](validator/README.md)**: Evaluation tools and metrics
  - Prediction generation
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

```python
# In trainer/configs/train_config_*.py

# Dynamic modeling settings
_C.TRAIN.USE_RDF_MODELING = True
_C.TRAIN.USE_DYNAMIC_MASK = True
_C.TRAIN.USE_DYNAMIC_MODELING = True
_C.TRAIN.DYNAMIC_MODELING_TYPE = 'vector_velocity'  # or 'mlp', 'scalar_velocity'

# Dynamic labels path
_C.TRAIN.DYNAMIC_LABELS_PATH = "<dataset_path>/dynamic_mask.txt"
```

---


## 🙏 Acknowledgments

- [MeMFlow](https://github.com/DQiaole/MemFlow) for optical flow estimation
- [IGEVStereo](https://github.com/gangweix/IGEV) for depth estimation
- [InternImage](https://github.com/OpenGVLab/InternImage) for 2D detection
- [VSRD](https://github.com/Magicboomliu/VSRD) for providing the base of this code.


---

