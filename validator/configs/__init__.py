"""Load validator YAML configs (Hydra-style defaults merge)."""

from __future__ import annotations

import os
import types
from typing import Any

import yaml

_CONFIGS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_file(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    out = base.copy()
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _compose_yaml(path: str, *, _stack: set[str] | None = None) -> dict:
    abs_path = os.path.abspath(path)
    if _stack is None:
        _stack = set()
    if abs_path in _stack:
        raise ValueError(f"Circular validator config defaults: {abs_path}")
    _stack.add(abs_path)

    raw = _load_file(abs_path)
    defaults = raw.pop("defaults", []) or []
    cfg: dict = {}
    base_dir = os.path.dirname(abs_path)
    for ref in defaults:
        ref_path = os.path.normpath(os.path.join(base_dir, str(ref)))
        if not ref_path.endswith((".yaml", ".yml")):
            ref_path += ".yaml"
        if not os.path.isfile(ref_path):
            raise FileNotFoundError(f"Validator config default not found: {ref} -> {ref_path}")
        cfg = _deep_merge(cfg, _compose_yaml(ref_path, _stack=_stack))
    cfg = _deep_merge(cfg, raw)
    _stack.remove(abs_path)
    return cfg


def _to_namespace(d: Any):
    if isinstance(d, dict):
        return types.SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_namespace(v) for v in d]
    return d


def load_validator_config(name: str):
    """Load ``validator/configs/{name}.yaml`` (e.g. ``casual``, ``vsrd24``)."""
    norm = name.strip().replace("\\", "/")
    if norm.startswith("validator/configs/"):
        norm = norm[len("validator/configs/") :]
    if norm.endswith((".yaml", ".yml")):
        norm = os.path.splitext(norm)[0]
    path = os.path.join(_CONFIGS_DIR, f"{norm}.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Validator config not found: {path}")
    return _to_namespace(_compose_yaml(path))
