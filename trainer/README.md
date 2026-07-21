# Stage 1：多视角 3D 自动标注（VSRD++）

基于优化的 scene-wise 多视角 3D 包围盒渲染，支持动态物体建模。

动态建模方式（在配置里设 `DYNAMIC_MODELING_TYPE`）：

- **MLP** — 实例残差场
- **vector_velocity** — 3D 速度向量（默认）
- **scalar_velocity** — 标量速度

[Slides](https://docs.google.com/presentation/d/1B2l-yRS63q4lu8Qb-4qCMdWHLMMMLU5L/edit?usp=sharing&ouid=112605403951022205460&rtpof=true&sd=true)

![MLP](figures/mlp.png)
![Velocity](figures/velocity.png)

---

## 环境准备

1. 在 `configs/_defaults/data.yaml`（或 experiment yaml）里设置 `TRAIN.DATASET.ROOT` 为你的 KITTI360 根目录。
2. 生成 **WAFT pseudo depth**（路径 `pseudo_depth_ssl_waft_stereo/`，`train.py` 做 attribute init 时需要）。
3. 训练**不强制**依赖 `dynamic_mask.txt` — 默认在线用 GT 3D bbox 速度推断 dynamic/static（`||v|| >= 0.20 m/frame`，规则见 [preprocessing/Dynamic_Labels](../preprocessing/Dynamic_Labels/)）。

在仓库根目录检查数据路径：

```bash
pixi run test-data
```

---

## 训练入口

| 脚本 | 用途 |
|------|------|
| **`train.py`** | 标准训练 + **所有 ablation 模式**（推荐唯一 Python 入口） |
| **`train_sharded.py`** | 64 分片集群训练（`SPLITS64/sub_XX`） |

旧脚本 `train_no_init.py`、`train_ablation.py` 已合并进 `train.py`；用 YAML（`SKIP_ATTRIBUTE_INIT`、`MASK_ERODE_RATIO`）或 CLI（`--skip_attribute_init`、`--erode_ratio`）切换行为。

### 论文 Ablation Ladder（Table I）

五个 study（projection → silhouette → RDF → VSRD++ 无 init → VSRD++ full）共用 **`train.py`**，通过 **`trainer/scripts/ablation.sh`** 切换。

**详细文档：** [configs/experiment/ablations/README.md](configs/experiment/ablations/README.md)

```bash
# 改 ablation.sh 里的 STUDY=，或命令行指定：
bash trainer/scripts/ablation.sh ablations/vsrdpp_full
bash trainer/scripts/ablation.sh vsrd_projection_only
```

输出默认在 `SAVED_ROOT_PATH` 下（见 ablation README）。wandb run name 默认跟随 `STUDY`，除非设 `WANDB_USE_ENV_NAME=1`。

### Stage1 全量训练（Casual / VSRD24）

两套 **VSRD++ full** 配置，模型相同，**帧列表不同**，通过 **`trainer/scripts/train.sh`** 切换。

**详细文档：** [configs/experiment/stage1_trainfiles/README.md](configs/experiment/stage1_trainfiles/README.md)

```bash
bash trainer/scripts/train.sh                  # 默认 vsrdpp_casual（4143 帧）
bash trainer/scripts/train.sh vsrdpp_vsrd24    # VSRD24（6901 帧）
```

| STUDY | 数据 | 输出（在 `SAVED_ROOT_PATH` 下） |
|-------|------|-----------------------------------|
| `vsrdpp_casual` | `cascual_splits` | `vsrdpp_casual/` |
| `vsrdpp_vsrd24` | `vsrd24_splits` | `vsrdpp_vsrd24/` |

### 单 sequence 训练（单卡）

```bash
cd trainer
CUDA_VISIBLE_DEVICES=0 torchrun \
  --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
  --nnodes 1 --nproc_per_node 1 \
  train.py --config_path sequence_07 --device_id 0
```

### Weights & Biases（可选）

记录 loss/metric；可选上传 PD vs GT 3D box 可视化。

建议在仓库根目录用 `.env` 配置 `WANDB_API_KEY`（勿提交）：

```bash
cd /path/to/VSRD_plus_plus
cp .env.sample .env
# 编辑 .env，设置 WANDB_API_KEY=...
```

```bash
pixi run train -- --config_path sequence_07 --device_id 0 \
  --wandb --wandb_entity "your_entity" --wandb_project "VSRD++" --wandb_log_images
```

`--config_path` 支持 `sequence_07`、简写 `07`、`smoke`、`ablations/vsrdpp_full` 等。

### 自定义输出目录

```bash
train.py \
  --config_path sequence_07 \
  --device_id 0 \
  --ckpt_dirname /path/to/ckpts \
  --log_dirname  /path/to/logs \
  --out_dirname  /path/to/outs
```

未指定时默认：`trainer/ckpts/{MODEL_TYPE}/`、`trainer/logs/`、`trainer/outs/`，每帧再按数据集相对图像路径分子目录。

### Smoke 测试

与 `train.py` 同一代码路径，配置 `smoke`（默认 seq00）。超参与正式训练一致；可用**短** `FILENAMES` 列表快速跑通。

```bash
pixi run train-smoke
# 或
cd trainer/scripts && bash train_smoke.sh
```

可选环境变量：`CKPT_DIRNAME`、`LOG_DIRNAME`、`OUT_DIRNAME`、`CONFIG_PATH`、`CUDA_VISIBLE_DEVICES`。

### 其他 ablation 示例

```bash
# 跳过 attribute init
pixi run train -- --config_path ablations/vsrdpp_velocity_no_init --device_id 0

# Mask 腐蚀鲁棒性（完整 VSRD++ + 腐蚀 mask）
ERODE_RATIO=0.03 bash trainer/scripts/train_enrode_mask.sh
```

### 分片训练（集群）

```bash
train_sharded.py --config_path 48 --device_id 0 \
  --saved_ckpt_path /path/to/output_models
```

使用 `configs/SPLITS64/split_sub.json` 与 `train_tsubame_filenames/filename_split64/sub_XX.txt`。

---

## Shell 脚本（`trainer/scripts/`）

| 脚本 | 用途 |
|------|------|
| **`ablation.sh`** | **Table I ablation ladder** → `launch_train.py` + study yaml |
| **`train.sh`** | **Stage1 VSRD++ full** → Casual 或 VSRD24 帧列表 |
| `launch_train.py` | torchrun 启动器（`ablation.sh`、`train.sh` 等共用） |
| `train_enrode_mask.sh` | Mask 腐蚀 ablation → `train.py` + `--erode_ratio` |
| `train_smoke.sh` | 单卡 smoke 测试 |
| `train_sharded.sh` | `train_sharded.py` |
| `train_tsubame.sh` | Tsubame 集群任务模板 |
| `lib.sh` | 公共函数（`ensure_pixi`、wandb 参数等） |

Pixi 快捷命令：`pixi run train`、`pixi run train-smoke` 等。

**通过 shell 环境变量控制 wandb**（部分脚本经 `lib.sh` 处理）：

| 变量 | 作用 |
|------|------|
| `USE_WANDB=1` 或 `WANDB=1` | 启用 `--wandb` |
| `WANDB_LOG_IMAGES=1` | 添加 `--wandb_log_images` |
| `WANDB_PROJECT` | 项目名 |
| `WANDB_ENTITY` | 团队/实体 |
| `WANDB_NAME` | run 名称 |
| `WANDB_TAGS` | 逗号分隔标签 |

未设置 `WANDB_NAME` / `--wandb_name` 时的默认名：  
`{config_path}-{hostname}-{YYYYMMDD-HHMMSS}`。

```bash
USE_WANDB=1 WANDB_LOG_IMAGES=1 WANDB_PROJECT=VSRD++ \
pixi run bash trainer/scripts/train_smoke.sh
```

指定输出目录示例：

```bash
USE_WANDB=1 WANDB_LOG_IMAGES=1 \
pixi run bash trainer/scripts/train.sh vsrdpp_casual
```

自定义 wandb run name（两种方式等价）：

```bash
# 1) 环境变量
WANDB_NAME="vsrd24-run" USE_WANDB=1 pixi run bash trainer/scripts/train.sh vsrdpp_vsrd24

# 2) 在 -- 之后透传给 train.py
USE_WANDB=1 pixi run bash trainer/scripts/train.sh vsrdpp_casual -- --wandb_name "casual-debug"
```

可直接 `bash trainer/scripts/train.sh` — 脚本会自动 `pixi run` 重入（除非 `VSRD_SKIP_PIXI=1`）。**请用 bash，不要用 sh。**

---

## 配置结构

Experiment 配置在 `trainer/configs/experiment/`（Hydra 风格分层 YAML）：

```
configs/
├── _defaults/              # train、data、model、optim、output、launch
├── experiment/
│   ├── ablations/          # Table I ladder — 见 ablations/README.md
│   ├── stage1_trainfiles/  # Casual / VSRD24 帧列表 — 见 stage1_trainfiles/README.md
│   ├── vsrdpp_sequentials/
│   ├── smoke.yaml
│   └── inference.yaml
└── paths.py, train_modes.py, …
```

Python 加载：

```python
from trainer.configs import load_config
cfg = load_config("stage1_trainfiles/vsrdpp_full_casual")
cfg = load_config("ablations/vsrdpp_full")   # 或 sequence id、smoke 等
```

在 `_defaults/data.yaml` 或各 experiment yaml 里设置 `TRAIN.DATASET.ROOT`。

常用训练开关（见 `_defaults/model.yaml` 与各 ablation yaml）：

```yaml
TRAIN:
  USE_RDF_MODELING: true
  USE_DYNAMIC_MASK: true
  USE_DYNAMIC_MODELING: true
  DYNAMIC_MODELING_TYPE: vector_velocity
  SKIP_ATTRIBUTE_INIT: false
  OPTIMIZATION_NUM_STEPS: 3000
```

新实验请用 `experiment/*.yaml`；旧 JSON 配置可能仍存在，仅供兼容。

路径常量：[preprocessing/dataset_paths.py](../preprocessing/dataset_paths.py)。

---

## 推理与评估

```bash
cd trainer
python inference.py
python evaluation.py
```

使用 `inference` 配置。Stage 1 指标（IoU、mAP 等）见 [validator/README.md](../validator/README.md)。

---

## 快速 GT 可视化（投影 3D box + BEV）

用于检查相机坐标系 / GT 朝向是否与图像一致。

```bash
pixi run python trainer/visualize_gt.py --config_path ablation_selective --index 0 --draw_masks
```

---

## 训练 vs 验证：dynamic 标签

| 阶段 | dynamic/static 来源 |
|------|---------------------|
| **训练** | 默认在线：GT 3D bbox 速度（阈值 0.20 m/frame） |
| **验证** | 读取 `dynamic_attributes_est_gt/<sequence>/dynamic_mask.txt` |

评估前可先生成标签：

```bash
pixi run gen-dynamic
python scripts/compare_dynamic_mask_gt.py --config sequence_07
```
