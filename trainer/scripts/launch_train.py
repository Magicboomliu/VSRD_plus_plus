#!/usr/bin/env python3
"""Launch torchrun from experiment config."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

TRAINER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(TRAINER_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trainer.configs import load_launch_settings  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch torchrun for trainer/train.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python launch_train.py --config_path vsrdpp_full\n"
            "  python launch_train.py --config_path vsrdpp_full --device_id 0 -- --wandb\n"
            "  python launch_train.py vsrdpp_sequentials/vsrd_plus_full_seq_10\n"
        ),
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        help="experiment name (shortcut for --config_path)",
    )
    parser.add_argument(
        "--config_path",
        "-c",
        dest="config_path",
        help="experiment config name/path (same as train.py --config_path)",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=None,
        help="override LAUNCH.DEVICE_ID from yaml",
    )
    parser.add_argument(
        "--cuda_devices",
        default=None,
        help="override LAUNCH.CUDA_DEVICES from yaml",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=None,
        help="override LAUNCH.NPROC_PER_NODE from yaml",
    )
    parser.add_argument(
        "--rdzv_endpoint",
        default=None,
        help="override LAUNCH.RDZV_ENDPOINT from yaml",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="extra args forwarded to train.py (prefix with -- if needed)",
    )
    args = parser.parse_args(argv)

    if args.train_args and args.train_args[0] == "--":
        args.train_args = args.train_args[1:]

    # argparse quirk: optional positional `experiment` can consume the first token
    # after `--`, e.g. `--ckpt_dirname` becomes experiment instead of train arg.
    if args.experiment and str(args.experiment).startswith("-"):
        args.train_args = [args.experiment, *args.train_args]
        args.experiment = None

    config_path = args.config_path or args.experiment
    if not config_path:
        parser.error("missing experiment: pass --config_path NAME or positional NAME")
    args.config_path = config_path
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    launch = load_launch_settings(args.config_path)

    if args.device_id is not None:
        launch["device_id"] = args.device_id
    if args.cuda_devices is not None:
        launch["cuda_devices"] = args.cuda_devices
    if args.nproc_per_node is not None:
        launch["nproc_per_node"] = args.nproc_per_node
    if args.rdzv_endpoint is not None:
        launch["rdzv_endpoint"] = args.rdzv_endpoint

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = launch["cuda_devices"]

    cmd = [
        "torchrun",
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint",
        launch["rdzv_endpoint"],
        "--nnodes",
        "1",
        "--nproc_per_node",
        str(launch["nproc_per_node"]),
        launch["train_script"],
        "--config_path",
        launch["config_path"],
        "--device_id",
        str(launch["device_id"]),
        *args.train_args,
    ]

    print("[launch]", " ".join(cmd))
    os.chdir(TRAINER_DIR)
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
