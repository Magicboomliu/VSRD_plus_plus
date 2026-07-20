# Stage 1: Multi-View 3D Auto-Labeling (VSRD++)

Optimization-based scene-wise multi-view 3D bounding box rendering with dynamic object modeling.

Dynamic modeling options (set `DYNAMIC_MODELING_TYPE` in `configs/base.json`):

- **MLP** — instance residual field
- **vector_velocity** — 3D velocity vector (default)
- **scalar_velocity** — scalar speed

[Slides](https://docs.google.com/presentation/d/1B2l-yRS63q4lu8Qb-4qCMdWHLMMMLU5L/edit?usp=sharing&ouid=112605403951022205460&rtpof=true&sd=true)

![MLP](figures/mlp.png)
![Velocity](figures/velocity.png)

---

## Prerequisites

1. Set `TRAIN.DATASET.ROOT` in `configs/base.json` to your KITTI360 root.
2. Generate **WAFT pseudo depth** under `pseudo_depth_ssl_waft_stereo/` (required for attribute initialization in `train.py`).
3. **Do not** need `dynamic_mask.txt` for training — dynamic/static is inferred online from GT 3D bbox velocity (`||v|| >= 0.20 m/frame`, same rule as [preprocessing/Dynamic_Labels](../preprocessing/Dynamic_Labels/)).

Verify paths from repo root:

```bash
pixi run test-data
```

---

## Training scripts

| Script | When to use |
|--------|-------------|
| **`train.py`** | Standard training and **all ablation modes** (recommended single entry) |
| **`train_sharded.py`** | 64-way data split for cluster jobs (`SPLITS64/sub_XX`) |

Legacy split scripts (`train_no_init.py`, `train_ablation.py`) are merged into `train.py`; use YAML flags (`SKIP_ATTRIBUTE_INIT`, `MASK_ERODE_RATIO`) or CLI (`--skip_attribute_init`, `--erode_ratio`).

### Paper ablation ladder (Table I)

Five studies (projection → silhouette → RDF → VSRD++ no-init → VSRD++ full) share **`train.py`**; switch via **`trainer/scripts/ablation.sh`**.

**Full doc:** [configs/experiment/ablations/README.md](configs/experiment/ablations/README.md)

```bash
# Edit STUDY= in ablation.sh, or pass on CLI:
bash trainer/scripts/ablation.sh ablations/vsrdpp_full
bash trainer/scripts/ablation.sh vsrd_projection_only
```

Outputs default to `SAVED_ROOT_PATH` (see ablation README). wandb run name follows `STUDY` unless `WANDB_USE_ENV_NAME=1`.

### Standard training (single GPU)

```bash
cd trainer
CUDA_VISIBLE_DEVICES=0 torchrun \
  --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
  --nnodes 1 --nproc_per_node 1 \
  train.py --config_path sequence_07 --device_id 0
```

### Weights & Biases (optional)

Enable wandb logging (losses/metrics; optional PD-vs-GT 3D box overlays):

Recommended: set your `WANDB_API_KEY` locally via `.env` (never commit):

```bash
cd /home/zliu/IJCV/VSRD_plus_plus
cp .env.sample .env
# edit .env and set WANDB_API_KEY=...
```

```bash
pixi run train -- --config_path sequence_07 --device_id 0 --wandb --wandb_entity "liuzihua1004" --wandb_project "VSRD-plus-plus" --wandb_log_images
```

From repo root:

```bash
pixi run train -- --config_path sequence_07 --device_id 0
```

`--config_path` accepts `sequence_07`, bare id `07`, `smoke`, `ablation_selective`, etc.

### Custom output directories

```bash
train.py \
  --config_path sequence_07 \
  --device_id 0 \
  --ckpt_dirname /path/to/ckpts \
  --log_dirname  /path/to/logs \
  --out_dirname  /path/to/outs
```

Default (when flags omitted): `trainer/ckpts/{MODEL_TYPE}/`, `trainer/logs/`, `trainer/outs/` — each frame gets a subfolder under the dataset-relative image path.

### Smoke test

Same code path as `train.py`, config `smoke.json` (sequence_00 by default). Hyperparameters come from `base.json`; use a **short** `FILENAMES` list for a quick run.

```bash
pixi run train-smoke
# or
cd trainer/scripts && sh train_smoke.sh
```

Optional env vars: `CKPT_DIRNAME`, `LOG_DIRNAME`, `OUT_DIRNAME`, `CONFIG_PATH`, `CUDA_VISIBLE_DEVICES`.

### Other ablation examples

```bash
# Skip attribute init (any config with SKIP_ATTRIBUTE_INIT or CLI)
pixi run train -- --config_path ablations/vsrdpp_velocity_no_init --device_id 0

# Mask erode robustness (full VSRD++ + eroded masks)
ERODE_RATIO=0.03 bash trainer/scripts/train_enrode_mask.sh
```

### Sharded training (cluster)

```bash
train_sharded.py --config_path 48 --device_id 0 \
  --saved_ckpt_path /path/to/output_models
```

Uses `configs/SPLITS64/split_sub.json` and `train_tsubame_filenames/filename_split64/sub_XX.txt`.

---

## Shell scripts (`trainer/scripts/`)

| Script | Purpose |
|--------|---------|
| **`ablation.sh`** | **Table I ablation ladder** → `launch_train.py` + study yaml |
| `launch_train.py` | torchrun launcher (used by `ablation.sh`, `train.sh`, …) |
| `train.sh` | Generic launcher → `launch_train.py` |
| `train_enrode_mask.sh` | Mask erode ablation → `train.py` + `--erode_ratio` |
| `train_smoke.sh` | Single-GPU smoke test |
| `train_sharded.sh` | `train_sharded.py` |
| `train_tsubame.sh` | Tsubame cluster job template |
| `lib.sh` | Shared helpers (`ensure_pixi`, wandb flags, …) |

Pixi: `pixi run train`, `pixi run train-smoke`, etc.

**wandb via shell env** (handled by `lib.sh`):

| Variable | Effect |
|----------|--------|
| `USE_WANDB=1` or `WANDB=1` | Enable `--wandb` |
| `WANDB_LOG_IMAGES=1` | Add `--wandb_log_images` (`train.py` / `train.sh` init mode only) |
| `WANDB_PROJECT` | Project name |
| `WANDB_ENTITY` | Team/entity |
| `WANDB_NAME` | Run name |
| `WANDB_TAGS` | Comma-separated tags |

Default run name (if you don't set `WANDB_NAME` / `--wandb_name`):  
`{config_path}-{hostname}-{YYYYMMDD-HHMMSS}` (e.g. `ablation_selective-megumi-20260702-160512`).

```bash
USE_WANDB=1 WANDB_LOG_IMAGES=1 WANDB_ENTITY=liuzihua1004 WANDB_PROJECT=VSRD-plus-plus \
pixi run bash trainer/scripts/train_smoke.sh
```

Example with explicit output directories (recommended layout):

```bash
CKPT_DIRNAME=/media/zliu/data12/IJCV/vsrdpp/ckpts \
LOG_DIRNAME=/media/zliu/data12/IJCV/vsrdpp/logs \
OUT_DIRNAME=/media/zliu/data12/IJCV/vsrdpp/outs \
USE_WANDB=1 WANDB_LOG_IMAGES=1 \
pixi run bash trainer/scripts/train.sh
```

You can also set a readable run name (two equivalent ways):

```bash
# 1) via env var
WANDB_NAME="seq10-debug" USE_WANDB=1 pixi run bash trainer/scripts/train.sh

# 2) pass-through CLI args to python entrypoint
USE_WANDB=1 pixi run bash trainer/scripts/train.sh --wandb_name "seq10-debug"
```

Tip: you may also run `bash trainer/scripts/train.sh` directly — the script will auto re-exec itself under `pixi run` (unless `VSRD_SKIP_PIXI=1`).

---

## Configuration

Experiment configs under `trainer/configs/experiment/` (Hydra-style YAML layers):

```
configs/
├── _defaults/           # train, data, model, optim, output, launch
├── experiment/
│   ├── ablations/       # Table I ladder — see ablations/README.md
│   ├── vsrdpp_sequentials/
│   ├── smoke.yaml
│   └── inference.yaml
└── paths.py, train_modes.py, …
```

Load in Python:

```python
from trainer.configs import load_config
cfg = load_config("ablations/vsrdpp_full")   # or "05", sequence id, etc.
```

Set `TRAIN.DATASET.ROOT` in `_defaults/data.yaml` or per-experiment override.

Key flags (see `_defaults/model.yaml`, per-ablation yaml):

```yaml
TRAIN:
  USE_RDF_MODELING: true
  USE_DYNAMIC_MASK: true
  USE_DYNAMIC_MODELING: true
  DYNAMIC_MODELING_TYPE: vector_velocity
  SKIP_ATTRIBUTE_INIT: false
  OPTIMIZATION_NUM_STEPS: 3000
```

Legacy JSON configs may still exist for older workflows; new runs should use `experiment/*.yaml`.

Path constants: [preprocessing/dataset_paths.py](../preprocessing/dataset_paths.py).

---

## Inference & evaluation

```bash
cd trainer
python inference.py
python evaluation.py
```

Use config `inference.json` (loaded as `conf_val` in those scripts). See [validator/README.md](../validator/README.md) for Stage 1 metrics (IoU, mAP) after training.

---

## Quick GT visualization (projected 3D boxes + BEV)

Use this to sanity-check camera conventions / GT orientation against the raw image.

```bash
pixi run python trainer/visualize_gt.py --config_path ablation_selective --index 0 --draw_masks
```

---

## Training vs validator: dynamic labels

| Stage | Dynamic/static source |
|-------|------------------------|
| **Training** | Online from GT 3D bbox velocity (threshold 0.20 m/frame) |
| **Validator** | Reads `dynamic_attributes_est_gt/<sequence>/dynamic_mask.txt` |

Generate labels before evaluation:

```bash
pixi run gen-dynamic
python scripts/compare_dynamic_mask_gt.py --config sequence_07
```
