# Validator — Ckpt infer

从 Stage1 训练 ckpt 推理，导出 **PD + GT** JSON（legacy Step1a + Step1b）。

## Split 配置

| split | 帧列表 | 帧数 |
|-------|--------|------|
| `casual` | `cascual_splits/train_all_filenames.txt` | 4143 |
| `vsrd24` | `vsrd24_splits/train_all_filenames.txt` | 6901 |

配置在 `validator/configs/{casual,vsrd24}.yaml`（帧列表 / dynamic）；ckpt 路径在 `validator/scripts/ckpt_infer.sh` 里设 `CKPT_DIRNAME`。

## 运行

```bash
bash validator/scripts/ckpt_infer.sh
bash validator/scripts/ckpt_infer.sh ckpt_infer_vsrd24
bash validator/scripts/ckpt_infer.sh ckpt_infer_casual -- --ckpt_dirname /path/to/ckpts
```

脚本顶部改这几项：

```bash
DATASET_ROOT=...          # KITTI360 根目录（读图像/annotation）
CKPT_DIRNAME=...          # 训练 ckpt 根目录
CKPT_FILENAME=step_2499.pt
OUTPUT_ROOT=...           # PD: ${OUTPUT_ROOT}/${STUDY}/predictions/json/...
                          # GT: ${OUTPUT_ROOT}/${STUDY}/predictions/gt/...
NUM_WORKERS=4
```

示例（casual）：

```text
.../ckpt_infer_casual/predictions/json/data_2d_raw/.../0000000250.json   # PD
.../ckpt_infer_casual/predictions/gt/data_2d_raw/.../0000000250.json       # GT
```

帧列表 / dynamic 读 `validator/configs/splits/_*.yaml`。不使用 wandb。
