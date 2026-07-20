from __future__ import annotations

from dataclasses import dataclass
import os
import socket
from datetime import datetime
from typing import Any, Optional


def _try_load_dotenv() -> None:
    """
    Best-effort load of repo-root `.env` for local secrets (WANDB_API_KEY, etc).
    No-op if python-dotenv isn't installed or `.env` is missing.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    # Load from current working directory upward; supports running from trainer/ or repo root.
    load_dotenv(override=False)


@dataclass
class WandbConfig:
    enabled: bool
    project: str
    entity: str | None
    name: str | None
    tags: list[str]
    log_images: bool


def _safe_import_wandb():
    try:
        import wandb  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "wandb is enabled but import failed. "
            "Install it (e.g. `pixi add --pypi wandb` / `pixi install`) and try again."
        ) from e
    return wandb


def build_wandb_config(args: Any, cfg: Any = None) -> WandbConfig:
    wb = getattr(getattr(cfg, "TRAIN", None), "WANDB", None) if cfg is not None else None

    enabled = bool(getattr(args, "wandb", False))
    if not enabled and wb is not None:
        enabled = bool(getattr(wb, "ENABLED", False))

    project_raw = getattr(args, "wandb_project", None) or None
    entity_raw = getattr(args, "wandb_entity", None) or None

    project = project_raw or (getattr(wb, "PROJECT", None) if wb else None) or "VSRD-plus-plus"
    entity = entity_raw or (getattr(wb, "ENTITY", None) if wb else None) or "liuzihua1004"

    # Backward/robust: if user provided "entity/project" in wandb_project, split it.
    if project_raw and "/" in project_raw and not entity_raw:
        maybe_entity, maybe_project = project_raw.split("/", 1)
        if maybe_entity and maybe_project:
            entity = maybe_entity
            project = maybe_project

    name = getattr(args, "wandb_name", None) or None
    if not name and wb is not None:
        cfg_name = getattr(wb, "NAME", "") or ""
        name = cfg_name or None
    if not name:
        # Default to a readable, unique-ish name (user can override via --wandb_name / WANDB_NAME)
        # Example: ablation_selective-megumi-20260702-160512
        cfg_name = str(getattr(args, "config_path", "run"))
        host = socket.gethostname()
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{cfg_name}-{host}-{ts}"
    tags_raw = getattr(args, "wandb_tags", "") or ""
    if not tags_raw and wb is not None:
        tags_raw = getattr(wb, "TAGS", "") or ""
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
    log_images = bool(getattr(args, "wandb_log_images", False))
    if not log_images and wb is not None:
        log_images = bool(getattr(wb, "LOG_IMAGES", False))
    return WandbConfig(
        enabled=enabled,
        project=project,
        entity=entity,
        name=name,
        tags=tags,
        log_images=log_images,
    )


def maybe_init_wandb(
    *,
    cfg: Any,
    args: Any,
    rank: int,
    extra_config: Optional[dict[str, Any]] = None,
):
    """
    Initialize wandb only on rank 0.
    Returns (wandb_module, run) or (None, None) when disabled.
    """
    wb_cfg = build_wandb_config(args, cfg)
    if (not wb_cfg.enabled) or rank != 0:
        return None, None

    _try_load_dotenv()
    wandb = _safe_import_wandb()

    # Make wandb quieter by default (training already logs a lot).
    os.environ.setdefault("WANDB_SILENT", "true")

    config = {
        "config_path": getattr(args, "config_path", None),
        "model_type": getattr(getattr(cfg, "TRAIN", None), "MODEL_TYPE", None),
        "dynamic_modeling_type": getattr(getattr(cfg, "TRAIN", None), "DYNAMIC_MODELING_TYPE", None),
        "optimization_num_steps": getattr(getattr(cfg, "TRAIN", None), "OPTIMIZATION_NUM_STEPS", None),
    }
    if extra_config:
        config.update(extra_config)

    run = wandb.init(
        project=wb_cfg.project,
        entity=wb_cfg.entity,
        name=wb_cfg.name,
        tags=wb_cfg.tags,
        config=config,
    )
    return wandb, run


def wandb_log_scalars(wandb: Any, run: Any, scalars: dict[str, Any], step: int, **extra: Any) -> None:
    if wandb is None or run is None:
        return
    payload = dict(scalars)
    payload.update(extra)
    wandb.log(payload, step=step)


def wandb_image(wandb: Any, image_chw, caption: str | None = None):
    """
    Convert a CHW torch/numpy image in [0,1] (or uint8) to wandb.Image.
    Returns None if wandb is None.
    """
    if wandb is None:
        return None
    try:
        import numpy as np
        import torch

        if isinstance(image_chw, torch.Tensor):
            img = image_chw.detach().float().cpu()
            if img.ndim == 3:
                img = img.permute(1, 2, 0)  # HWC
            img = img.numpy()
        else:
            img = image_chw
            if hasattr(img, "transpose") and getattr(img, "ndim", 0) == 3 and img.shape[0] in (1, 3, 4):
                img = img.transpose(1, 2, 0)
        img = np.asarray(img)
        return wandb.Image(img, caption=caption)
    except Exception:  # pragma: no cover
        return None

