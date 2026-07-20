# Ablation Studies（Table I ladder）

论文 Table I 的 ablation ladder，统一走 **`train.py`** + **`trainer/scripts/ablation.sh`**。  
五个 study 共用同一训练入口，通过 **experiment yaml** 切换模型/loss/init 行为；**不是**五个不同的 Python 脚本。

---

## 快速开始

```bash
# 推荐：仓库根目录，用 bash（不要用 sh）
cd /path/to/VSRD_plus_plus
bash trainer/scripts/ablation.sh

# 或指定 study（覆盖 sh 里 STUDY=）
bash trainer/scripts/ablation.sh ablations/vsrdpp_full
bash trainer/scripts/ablation.sh vsrd_projection_only   # 短名也行
```

脚本会自动 `pixi run` 重入（除非 `VSRD_SKIP_PIXI=1`）。

---

## STUDY → 调用链

改 `ablation.sh` 顶部的 `STUDY=`，或命令行第一个参数，会触发：

```text
STUDY
  → case 匹配某个 vsrd_*() shell 函数
  → launch_train.py --config_path ablations/<name>
  → experiment/ablations/0X_*.yaml
  → train.py（同一文件，不同 TRAIN.* flag 分支）
```

| STUDY（`STUDY=` 或 CLI） | shell 函数 | config | 输出根目录（under `SAVED_ROOT_PATH`） |
|--------------------------|------------|--------|----------------------------------------|
| `ablations/vsrd_projection_only` | `vsrd_projection_only` | `01_vsrd_projection_only` | `projection_only/{ckpts,logs,outs}` |
| `ablations/vsrd_projection_silhouette` | `vsrd_projection_silhouette` | `02_vsrd_projection_silhouette` | `projection_silhouette/...` |
| `ablations/vsrd_projection_silhouette_rdf` | `vsrd_projection_silhouette_rdf` | `03_vsrd_projection_silhouette_rdf` | `projection_silhouette_rdf/...` |
| `ablations/vsrdpp_velocity_no_init` | `vsrdpp_velocity_no_init` | `04_vsrdpp_velocity_no_init` | `vsrdpp_velocity_no_init/...` |
| `ablations/vsrdpp_full` | `vsrdpp_full` | `05_vsrdpp_full` | `vsrdpp_full/...` |

默认 `SAVED_ROOT_PATH=/media/zliu/data12/IJCV/ablations`（在 `ablation.sh` 里改）。

每帧输出仍在数据集相对路径下，例如：

```text
.../vsrdpp_full/ckpts/data_2d_raw/2013_05_28_drive_0007_sync/image_00/data_rect/0000001991/
```

---

## 五个 study 的训练差异（yaml）

数据层共用 `_dataset_casual_ablation.yaml` → **ablations_small**（**96 帧**，仅 **seq03 + seq07**）。

| 文件 | 路径 |
|------|------|
| 帧列表 | `/media/zliu/data12/IJCV/ablations/filenames/ablations_small/train_ablation_filenames.txt` |
| dynamic 列表 | `/media/zliu/data12/IJCV/ablations/filenames/ablations_small/train_ablation_dynamic_mask.txt` |

txt 内图像路径为相对 `TRAIN.DATASET.ROOT` 的 `data_2d_raw/...`。从旧机器拷来若带 `/gs/bs/...` 绝对路径，需先 normalize：

```bash
python preprocessing/data_organization/normalize_ablation_filenames.py \
  --root /media/zliu/data12/dataset/KITTI/KITTI360_For_Upload \
  --filenames /media/zliu/data12/IJCV/ablations/filenames/ablations_small/train_ablation_filenames.txt \
  --dynamic /media/zliu/data12/IJCV/ablations/filenames/ablations_small/train_ablation_dynamic_mask.txt \
  --in-place
```

| Config | Dynamic | RDF | Silhouette | Attribute init | Dynamic labels 文件 |
|--------|---------|-----|------------|----------------|---------------------|
| 01 projection only | off | off | off (w=0) | on | off |
| 02 + silhouette | off | off | on | on | off |
| 03 + RDF | off | on | on | on | off |
| 04 VSRD++ no init | on | on | on | **skip** | on |
| 05 VSRD++ full | on | on | on | on | on |

对应 yaml：`trainer/configs/experiment/ablations/0X_*.yaml`。

---

## `ablation.sh` 可调变量

编辑 `trainer/scripts/ablation.sh` 顶部：

| 变量 | 默认 | 说明 |
|------|------|------|
| `STUDY` | `ablations/vsrdpp_velocity_no_init` | 跑哪个 ladder |
| `SAVED_ROOT_PATH` | `/media/zliu/data12/IJCV/ablations` | ckpt/log/out 根目录 |
| `DEVICE_ID` | `0` | `--device_id` |
| `CUDA_DEVICES` | `0` | `CUDA_VISIBLE_DEVICES` |
| `NPROC_PER_NODE` | `1` | torchrun 进程数 |
| `RDZV_ENDPOINT` | `localhost:22500` | torchrun rendezvous |
| `USE_WANDB` | `1` | 是否 `--wandb` |
| `WANDB_LOG_IMAGES` | `1` | 是否 `--wandb_log_images`（红=GT，绿=PD 3D box） |

`.env`（仓库根，勿提交）常用项：

```bash
WANDB_API_KEY=...
WANDB_PROJECT=VSRD++
# WANDB_NAME 默认由 ablation.sh 按 STUDY 自动设置（如 vsrdpp_full）
# 若要用 .env 固定名：启动前 WANDB_USE_ENV_NAME=1
```

**wandb run name**：默认 `STUDY` 短名（`vsrd_projection_only` 等），在解析 CLI 参数之后写入，避免与 study 不一致。

**图像上传**：`WANDB_LOG_IMAGES=1` 时，每 `TRAIN.LOGGING.IMAGE_INTERVALS`（默认 500）step 上传 `viz/pd_vs_gt_3d` 等。

---

## 命令行透传

```bash
# 额外 train.py 参数（在 -- 之后）
bash trainer/scripts/ablation.sh ablations/vsrdpp_full -- --erode_ratio 0.03

# 临时关 wandb 图像
WANDB_LOG_IMAGES=0 bash trainer/scripts/ablation.sh

# 跳过 pixi 重入（已在外部 pixi shell 内时）
VSRD_SKIP_PIXI=1 bash trainer/scripts/ablation.sh
```

---

## 数据准备

见上文 **cascual_splits** 两文件；04/05 需要 pseudo depth（完整 VSRD++ attribute init 时）。

---

## 其他 ablation

### Mask 腐蚀鲁棒性（非 Table I ladder）

完整 VSRD++ + mask erode，单独脚本：

```bash
ERODE_RATIO=0.03 bash trainer/scripts/train_enrode_mask.sh
# config: ablations/erode_seg_mask_degradation_vsrdpp
```

---

## 配置文件索引

```
trainer/configs/experiment/ablations/
├── README.md                          # 本文件
├── _dataset_casual_ablation.yaml      # ablations_small（seq03+07，96 帧）
├── _dataset_seq10.yaml                # 备选：仅 seq10（当前 ladder 未用）
├── 01_vsrd_projection_only.yaml
├── 02_vsrd_projection_silhouette.yaml
├── 03_vsrd_projection_silhouette_rdf.yaml
├── 04_vsrdpp_velocity_no_init.yaml
├── 05_vsrdpp_full.yaml
└── erode_seg_mask_degradation_vsrdpp.yaml
```

底层 defaults：`trainer/configs/_defaults/{train,data,model,optim,output,launch}.yaml`。

---

## 直接调 launch_train.py（不用 ablation.sh）

```bash
cd /path/to/VSRD_plus_plus
python trainer/scripts/launch_train.py \
  --config_path ablations/vsrdpp_full \
  --device_id 0 \
  -- \
  --ckpt_dirname /path/to/ckpts \
  --log_dirname /path/to/logs \
  --out_dirname /path/to/outs \
  --wandb --wandb_log_images
```

注意：`launch_train.py` 后必须加 `--` 再写 train.py 的参数。
