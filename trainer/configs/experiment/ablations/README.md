# Ablation 实验（Table I Ladder）

论文 Table I 的 ablation ladder，统一走 **`train.py`** + **`trainer/scripts/ablation.sh`**。  
五个 study 共用同一 Python 入口，通过 **experiment yaml** 切换模型 / loss / init；**不是**五个不同的训练脚本。

---

## 快速开始

```bash
# 推荐：在仓库根目录，用 bash（不要用 sh）
cd /path/to/VSRD_plus_plus
bash trainer/scripts/ablation.sh

# 或命令行指定 study（覆盖 sh 里 STUDY=）
bash trainer/scripts/ablation.sh ablations/vsrdpp_full
bash trainer/scripts/ablation.sh vsrd_projection_only   # 短名也可
```

脚本会自动 `pixi run` 重入（除非 `VSRD_SKIP_PIXI=1`）。

---

## STUDY → 调用链

修改 `ablation.sh` 顶部 `STUDY=`，或把 study 作为第一个命令行参数：

```text
STUDY
  → case 匹配某个 vsrd_*() shell 函数
  → launch_train.py --config_path ablations/<name>
  → experiment/ablations/0X_*.yaml
  → train.py（同一文件，不同 TRAIN.* 分支）
```

| STUDY（`STUDY=` 或 CLI） | shell 函数 | config | 输出根目录（在 `SAVED_ROOT_PATH` 下） |
|--------------------------|------------|--------|----------------------------------------|
| `ablations/vsrd_projection_only` | `vsrd_projection_only` | `01_vsrd_projection_only` | `projection_only/{ckpts,logs,outs}` |
| `ablations/vsrd_projection_silhouette` | `vsrd_projection_silhouette` | `02_vsrd_projection_silhouette` | `projection_silhouette/...` |
| `ablations/vsrd_projection_silhouette_rdf` | `vsrd_projection_silhouette_rdf` | `03_vsrd_projection_silhouette_rdf` | `projection_silhouette_rdf/...` |
| `ablations/vsrdpp_velocity_no_init` | `vsrdpp_velocity_no_init` | `04_vsrdpp_velocity_no_init` | `vsrdpp_velocity_no_init/...` |
| `ablations/vsrdpp_full` | `vsrdpp_full` | `05_vsrdpp_full` | `vsrdpp_full/...` |

默认 `SAVED_ROOT_PATH=/media/zliu/data12/IJCV/ablations`（在 `ablation.sh` 里改）。

每帧 checkpoint / log 仍按**数据集相对路径**分子目录，例如：

```text
.../vsrdpp_full/ckpts/data_2d_raw/2013_05_28_drive_0007_sync/image_00/data_rect/0000001991/
```

---

## 五个 study 的差异（yaml）

### 数据

数据层共用 `_dataset_casual_ablation.yaml` → **[stage1_trainfiles/_dataset_casual](../stage1_trainfiles/_dataset_casual.yaml)**（**4143 帧**，`cascual_splits`）。

| 文件 | 路径 |
|------|------|
| 帧列表 | `/media/zliu/data12/IJCV/ablations/filenames/cascual_splits/train_all_filenames.txt` |
| dynamic 列表 | `/media/zliu/data12/IJCV/ablations/filenames/cascual_splits/train_all_dynamic_mask.txt` |

其他列表（供参考，非 ablation 默认）：

| 列表 | 帧数 | 说明 |
|------|------|------|
| `ablations_small` | 96 | 仅 seq03+07 |
| `vsrd24_splits` | 6901 | VSRD24 Stage1，见 [stage1_trainfiles/README.md](../stage1_trainfiles/README.md) |

txt 内路径为相对 `TRAIN.DATASET.ROOT` 的 `data_2d_raw/...`。从旧机器拷来若含 `/gs/bs/...` 绝对路径，需先 normalize：

```bash
python preprocessing/data_organization/normalize_ablation_filenames.py \
  --root /media/zliu/data12/dataset/KITTI/KITTI360_For_Upload \
  --filenames /media/zliu/data12/IJCV/ablations/filenames/cascual_splits/train_all_filenames.txt \
  --dynamic /media/zliu/data12/IJCV/ablations/filenames/cascual_splits/train_all_dynamic_mask.txt \
  --in-place
```

### 模型 / loss

| Config | Dynamic | RDF | Silhouette | Attribute init | 读 dynamic 文件 |
|--------|---------|-----|------------|----------------|-----------------|
| 01 projection only | 关 | 关 | 关（w=0） | 开 | 关 |
| 02 + silhouette | 关 | 关 | 开 | 开 | 关 |
| 03 + RDF | 关 | 开 | 开 | 开 | 关 |
| 04 VSRD++ 无 init | 开 | 开 | 开 | **跳过** | 开 |
| 05 VSRD++ full | 开 | 开 | 开 | 开 | 开 |

对应 yaml：`trainer/configs/experiment/ablations/0X_*.yaml`。

---

## `ablation.sh` 可调变量

编辑 `trainer/scripts/ablation.sh` 顶部：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `STUDY` | `ablations/vsrdpp_full` | 跑哪个 ladder |
| `SAVED_ROOT_PATH` | `/media/zliu/data12/IJCV/ablations` | ckpt / log / out 根目录 |
| `DEVICE_ID` | `0` | `--device_id` |
| `CUDA_DEVICES` | `0` | `CUDA_VISIBLE_DEVICES` |
| `NPROC_PER_NODE` | `1` | torchrun 进程数 |
| `RDZV_ENDPOINT` | `localhost:22500` | torchrun rendezvous |
| `USE_WANDB` | `1` | 是否加 `--wandb` |
| `WANDB_LOG_IMAGES` | `1` | 是否加 `--wandb_log_images`（红=GT，绿=PD 3D box） |

`.env`（仓库根，勿提交）常用项：

```bash
WANDB_API_KEY=...
WANDB_PROJECT=VSRD++
# WANDB_NAME 默认由 ablation.sh 按 STUDY 自动设置（如 vsrdpp_full）
# 若要坚持用 .env 里的固定名：启动前设 WANDB_USE_ENV_NAME=1
```

**wandb run name**：默认取 `STUDY` 短名（如 `vsrd_projection_only`），在解析 CLI 参数之后写入，避免名实不符。

**图像上传**：`WANDB_LOG_IMAGES=1` 时，每 `TRAIN.LOGGING.IMAGE_INTERVALS`（默认 500）step 上传 `viz/pd_vs_gt_3d` 等。

---

## 命令行透传

```bash
# 额外 train.py 参数（写在 -- 之后）
bash trainer/scripts/ablation.sh ablations/vsrdpp_full -- --erode_ratio 0.03

# 临时关闭 wandb 图像
WANDB_LOG_IMAGES=0 bash trainer/scripts/ablation.sh

# 已在 pixi shell 内、跳过重入
VSRD_SKIP_PIXI=1 bash trainer/scripts/ablation.sh
```

---

## 数据准备

- 默认使用 **cascual_splits** 两文件（见上文）。
- study **04 / 05** 需要 pseudo depth（完整 VSRD++ attribute init）。
- Stage1 全量 Casual / VSRD24 训练见 [stage1_trainfiles/README.md](../stage1_trainfiles/README.md) 与 `trainer/scripts/train.sh`。

---

## 其他 ablation

### Mask 腐蚀鲁棒性（不在 Table I ladder 内）

完整 VSRD++ + mask 腐蚀，单独脚本：

```bash
ERODE_RATIO=0.03 bash trainer/scripts/train_enrode_mask.sh
# config: ablations/erode_seg_mask_degradation_vsrdpp
```

---

## 配置文件索引

```
trainer/configs/experiment/ablations/
├── README.md                          # 本文件
├── _dataset_casual_ablation.yaml      # → stage1_trainfiles/_dataset_casual
├── _dataset_seq10.yaml                # 备选：仅 seq10（当前 ladder 未使用）
├── 01_vsrd_projection_only.yaml
├── 02_vsrd_projection_silhouette.yaml
├── 03_vsrd_projection_silhouette_rdf.yaml
├── 04_vsrdpp_velocity_no_init.yaml
├── 05_vsrdpp_full.yaml
└── erode_seg_mask_degradation_vsrdpp.yaml
```

底层 defaults：`trainer/configs/_defaults/{train,data,model,optim,output,launch}.yaml`。

---

## 不用 ablation.sh，直接调 launch_train.py

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

注意：`launch_train.py` 的参数写完后必须加 **`--`**，再写 `train.py` 的参数。
