"""Output path helpers for training scripts."""

from __future__ import annotations

import os
from typing import Any


def _get(obj: Any, *keys: str, default: Any = ""):
    cur = obj
    for key in keys:
        if cur is None:
            return default
        cur = getattr(cur, key, None)
    return default if cur is None else cur


def resolve_output_roots(cfg: Any, args: Any | None = None) -> tuple[str, str, str]:
    """
    Return (ckpt_root, log_root, out_root) before per-sample subdirs.

    Priority: CLI args > TRAIN.OUTPUT.* > built-in defaults under trainer/.
    """
    train = cfg.TRAIN
    configs_dir = train.CONFIG
    model_type = train.MODEL_TYPE
    output = getattr(train, "OUTPUT", None)

    ckpt_cfg = _get(output, "CKPT_ROOT")
    log_cfg = _get(output, "LOG_ROOT")
    out_cfg = _get(output, "OUT_ROOT")

    ckpt_cli = getattr(args, "ckpt_dirname", None) if args else None
    log_cli = getattr(args, "log_dirname", None) if args else None
    out_cli = getattr(args, "out_dirname", None) if args else None

    saved_root = getattr(args, "saved_ckpt_path", None) if args else None
    if saved_root:
        return (
            os.path.join(saved_root, "ckpts"),
            os.path.join(saved_root, "logs"),
            os.path.join(saved_root, "outs"),
        )

    base_ckpt = ckpt_cli or ckpt_cfg or configs_dir.replace("configs", f"ckpts/{model_type}")
    base_log = log_cli or log_cfg or configs_dir.replace("configs", "logs")
    base_out = out_cli or out_cfg or configs_dir.replace("configs", "outs")
    return base_ckpt, base_log, base_out
