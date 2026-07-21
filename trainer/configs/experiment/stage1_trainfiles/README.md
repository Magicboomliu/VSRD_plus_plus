# Stage1 训练帧列表（Casual / VSRD24）

Stage1 全量训练用的两套合并帧列表，供 `train.sh` 或 experiment yaml 引用。

---

## 两套数据

| 名称 | 配置文件 | 帧列表 | 帧数 | Sequence |
|------|----------|--------|------|----------|
| **Casual** | `_dataset_casual.yaml` | `cascual_splits/train_all_*.txt` | 4143 | 00/02/03/04/05/06/07/09 |
| **VSRD24** | `_dataset_vsrd24.yaml` | `vsrd24_splits/train_all_*.txt` | 6901 | 00/02/04/05/06/09（不含 03/07/10） |

列表文件在 `DATASET.ROOT` **之外**（绝对路径）：

```text
/media/zliu/data12/IJCV/ablations/filenames/cascual_splits/train_all_filenames.txt
/media/zliu/data12/IJCV/ablations/filenames/cascual_splits/train_all_dynamic_mask.txt

/media/zliu/data12/IJCV/ablations/filenames/vsrd24_splits/train_all_filenames.txt
/media/zliu/data12/IJCV/ablations/filenames/vsrd24_splits/train_all_dynamic_mask.txt
```

txt 内图像路径为相对 `TRAIN.DATASET.ROOT` 的 `data_2d_raw/...`。

**VSRD24 生成方式**：从  
`filenames/R50-N16-M128-B16/<seq>/sampled_image_filenames.txt`  
与  
`dynamic_attributes_est_gt/<seq>/dynamic_mask.txt`  
按 sequence 合并得到。

**格式说明**：

- `train_all_filenames.txt` 第 3 列：source 帧索引（整数，逗号分隔）
- `train_all_dynamic_mask.txt` 第 3 列：dynamic 标签（`0.0` / `1.0`）
- **不要**把 dynamic 文件填进 `FILENAMES`（否则会报 `invalid literal for int() '0.0'`）

---

## VSRD++ Full 训练（`train.sh`）

两个 study **模型配置相同**（完整 VSRD++），仅训练帧范围不同：

| STUDY | shell 函数 | config | 输出（在 `SAVED_ROOT_PATH` 下） |
|-------|------------|--------|----------------------------------|
| `vsrdpp_casual`（默认） | `vsrdpp_casual()` | `vsrdpp_full_casual` | `vsrdpp_casual/{ckpts,logs,outs}` |
| `vsrdpp_vsrd24` | `vsrdpp_vsrd24()` | `vsrdpp_full_vsrd24` | `vsrdpp_vsrd24/{ckpts,logs,outs}` |

```bash
bash trainer/scripts/train.sh                  # Casual，4143 帧
bash trainer/scripts/train.sh vsrdpp_vsrd24    # VSRD24，6901 帧
```

默认 `SAVED_ROOT_PATH=/media/zliu/data12/IJCV/ablations`（在 `train.sh` 顶部修改）。

wandb run name 默认跟随 `STUDY`（如 `vsrdpp_casual`），除非设 `WANDB_USE_ENV_NAME=1`。

---

## 在 yaml 里引用

仅引用帧列表（partial config）：

```yaml
defaults:
  - ../stage1_trainfiles/_dataset_casual    # 或 _dataset_vsrd24
```

完整 VSRD++ full + 读 dynamic 文件：

```yaml
defaults:
  - ../../_defaults/train
  - ../../_defaults/launch
  - ../../_defaults/data
  - ../../_defaults/model
  - ../../_defaults/optim
  - ../../_defaults/output
  - _dataset_casual   # 同目录下可换 _dataset_vsrd24

TRAIN:
  MODEL_TYPE: vsrdpp_full_casual
  USE_DYNAMIC_LABELS_FILE: true
```

Ablation ladder（01~05）通过 `ablations/_dataset_casual_ablation.yaml` 默认 include `_dataset_casual`。

---

## 与 ablation 数据的关系

| 列表 | 帧数 | 用途 |
|------|------|------|
| `cascual_splits` | 4143 | Casual / 当前 ablation 默认数据 |
| `vsrd24_splits` | 6901 | VSRD24 Stage1 |
| `ablations_small` | 96 | 仅 seq03+07 的小子集（Table I 早期实验，非当前 ablation 默认） |

---

## 文件索引

```
stage1_trainfiles/
├── README.md                 # 本文件
├── _dataset_casual.yaml      # 帧列表片段（Casual）
├── _dataset_vsrd24.yaml      # 帧列表片段（VSRD24）
├── vsrdpp_full_casual.yaml   # VSRD++ full + Casual
└── vsrdpp_full_vsrd24.yaml   # VSRD++ full + VSRD24
```
