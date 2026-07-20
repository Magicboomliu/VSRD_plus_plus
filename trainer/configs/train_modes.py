"""Runtime train-mode helpers (init / erode) for unified train.py."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def _get(obj: Any, *keys: str, default: Any = None):
    cur = obj
    for key in keys:
        if cur is None:
            return default
        cur = getattr(cur, key, None)
    return default if cur is None else cur


def skip_attribute_init(cfg: Any, args: Any | None = None) -> bool:
    if args is not None and getattr(args, "skip_attribute_init", False):
        return True
    return bool(_get(cfg.TRAIN, "SKIP_ATTRIBUTE_INIT", default=False))


def resolve_mask_erode_ratio(cfg: Any, args: Any | None = None) -> float:
    if args is not None and getattr(args, "erode_ratio", None) is not None:
        return float(args.erode_ratio)
    return float(_get(cfg.TRAIN, "MASK_ERODE_RATIO", default=0.0) or 0.0)


def apply_train_runtime_overrides(cfg: Any, args: Any | None = None) -> SimpleNamespace:
    """Return a small namespace consumed by train.py."""
    return SimpleNamespace(
        skip_init=skip_attribute_init(cfg, args),
        erode_ratio=resolve_mask_erode_ratio(cfg, args),
    )
