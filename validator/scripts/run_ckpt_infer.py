#!/usr/bin/env python3
"""Ckpt infer Step1: export PD + GT JSON for a split."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from validator.configs import load_validator_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ckpt infer — export PD and GT JSON using split config.",
    )
    parser.add_argument("split", choices=["casual", "vsrd24"], help="Split config name")
    parser.add_argument("--ckpt_dirname", type=str, default="", help="Override VALIDATOR.CKPT.DIRNAME")
    parser.add_argument("--ckpt_filename", type=str, default="", help="Override VALIDATOR.CKPT.FILENAME")
    parser.add_argument("--json_pd_out_dirname", type=str, default="", help="Absolute PD JSON output root")
    parser.add_argument("--json_gt_out_dirname", type=str, default="", help="Absolute GT JSON output root")
    parser.add_argument("--num_workers", type=int, default=-1, help="Override VALIDATOR.RUN.NUM_WORKERS")
    return parser.parse_args()


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("[launch]", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(_PROJECT_ROOT), env=env)


def main() -> int:
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["USE_WANDB"] = "0"

    args = parse_args()
    cfg = load_validator_config(args.split)
    v = cfg.VALIDATOR

    ckpt_dir = args.ckpt_dirname or os.environ.get("CKPT_DIRNAME", v.CKPT.DIRNAME)
    ckpt_file = args.ckpt_filename or os.environ.get("CKPT_FILENAME", v.CKPT.FILENAME)
    num_workers = args.num_workers if args.num_workers >= 0 else int(
        os.environ.get("NUM_WORKERS", v.RUN.NUM_WORKERS)
    )

    json_pd_dir = args.json_pd_out_dirname or os.environ.get("JSON_PD_OUT_DIR", "")
    json_gt_dir = args.json_gt_out_dirname or os.environ.get("JSON_GT_OUT_DIR", "")
    if not json_pd_dir:
        json_pd_dir = os.path.join(
            v.DATASET.ROOT,
            v.OUTPUT.JSON_FOLDER,
            os.path.basename(os.path.normpath(ckpt_dir)),
        )
    if not json_gt_dir:
        json_gt_dir = json_pd_dir.replace("/json", "/gt") if "/json" in json_pd_dir else f"{json_pd_dir}_gt"

    py = sys.executable
    pred_script = _PROJECT_ROOT / "validator/tools/Predictions/make_predictions.py"
    gt_script = _PROJECT_ROOT / "validator/tools/Predictions/export_gt_json.py"
    env = {**os.environ, "WANDB_MODE": "disabled", "USE_WANDB": "0"}

    common = [
        "--root_dirname", v.DATASET.ROOT,
        "--filenames_list", v.DATASET.FILENAMES,
        "--dynamic_labels_path", v.DATASET.DYNAMIC_LABELS_PATH,
        "--num_workers", str(num_workers),
        "--class_names", *v.MODEL.CLASS_NAMES,
    ]

    print("[ckpt_infer] step1a PD json")
    _run([
        py, str(pred_script),
        *common,
        "--ckpt_dirname", ckpt_dir,
        "--ckpt_filename", ckpt_file,
        "--input_model_type", v.MODEL.INPUT_MODEL_TYPE,
        "--json_out_dirname", json_pd_dir,
    ], env)

    print("[ckpt_infer] step1b GT json")
    _run([
        py, str(gt_script),
        *common,
        "--json_out_dirname", json_gt_dir,
    ], env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
