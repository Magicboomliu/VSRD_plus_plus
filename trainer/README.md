# Optimization-Based Scene-Wise  Multi-view 3D Bounding Box Rendering.

Following the Vanila VSRD, we propsoed the VSRD++ consider the dynamic object modeling. We using the following three method for dynamic modeling: 

- Instance Residual Field (via MLP)
- Vector Velocity Modeling 
- Scalar Velocity Modeling 

[Detailed Slide Slide](https://docs.google.com/presentation/d/1B2l-yRS63q4lu8Qb-4qCMdWHLMMMLU5L/edit?usp=sharing&ouid=112605403951022205460&rtpof=true&sd=true)

![image](figures/mlp.png)
![image](figures/velocity.png)


## Training 

| Script | Purpose |
|--------|---------|
| `train.py` | Standard VSRD++ training (attribute init + online dynamic) |
| `train_no_init.py` | Ablation: skip attribute initialization |
| `train_ablation.py` | Mask-erode ablations (`--erode_ratio`) |
| `train_sharded.py` | 64-way split training (`SPLITS64/split_sub`) |
| `train_legacy.py` | Deprecated legacy entry point |

Recommended (single GPU):

```bash
cd trainer
CUDA_VISIBLE_DEVICES=0 torchrun \
    --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    --nnodes 1 --nproc_per_node 1 \
    train.py --config_path sequence_07 --device_id 0
```

Or via pixi from repo root: `pixi run train -- --config_path sequence_07 --device_id 0`

Shell helpers: `trainer/scripts/train.sh`, `train_smoke.sh`, `train_ablation.sh`, `train_sharded.sh`, `train_tsubame.sh`.

## Inference

```
python infernece.py

```

## evaluations

```
python evaluation.py
```

Configs are JSON under `trainer/configs/` (see `base.json` + `sequence_XX.json`).

Example `sequence_07.json` (only `FILENAMES` is required; paths are relative to `DATASET.ROOT`):

```json
{
  "TRAIN": {
    "DATASET": {
      "FILENAMES": [
        "filenames/R50-N16-M128-B16/2013_05_28_drive_0007_sync/sampled_image_filenames.txt"
      ]
    }
  }
}
```

`load_config("sequence_07")` auto-sets `DYNAMIC_LABELS_PATH` to `dynamic_attributes_est_gt/2013_05_28_drive_0007_sync/dynamic_mask.txt` (validator / compare scripts only — training infers dynamic/static online).

Smoke test uses `smoke.json`: training hyperparameters come from `base.json` (same as formal runs). To keep it fast, point `FILENAMES` at a short frame list (e.g. copy the first few lines of `sampled_image_filenames.txt` into a separate `.txt`).

Key training flags in `base.json`:

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

Load config: `from trainer.configs import load_config; cfg = load_config("07")`

Path constants: [preprocessing/dataset_paths.py](../preprocessing/dataset_paths.py)