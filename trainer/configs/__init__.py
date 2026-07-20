import json
import os
import types
from typing import Any

import yaml

from preprocessing.dataset_paths import (
    DEFAULT_DYNAMIC_LABELS_DIRNAME,
    dynamic_mask_path_from_filenames_txt,
)

_CONFIGS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENT_DIR = os.path.join(_CONFIGS_DIR, "experiment")
_SEQUENTIALS_DIR = "vsrdpp_sequentials"

# 旧名 → experiment 下的相对路径
_LEGACY_ALIASES = {
    "ablation_selective": "ablations/erode_seg_mask_degradation_vsrdpp",
    "no_init": "ablations/04_vsrdpp_velocity_no_init",
    "ablation_01": "ablations/01_vsrd_projection_only",
    "ablation_02": "ablations/02_vsrd_projection_silhouette",
    "ablation_03": "ablations/03_vsrd_projection_silhouette_rdf",
    "ablation_04": "ablations/04_vsrdpp_velocity_no_init",
    "ablation_05": "ablations/05_vsrdpp_full",
    "vsrd_projection_only": "ablations/01_vsrd_projection_only",
    "vsrd_projection_silhouette": "ablations/02_vsrd_projection_silhouette",
    "vsrd_projection_silhouette_rdf": "ablations/03_vsrd_projection_silhouette_rdf",
    "vsrdpp_velocity_no_init": "ablations/04_vsrdpp_velocity_no_init",
    "vsrdpp_full": "ablations/05_vsrdpp_full",
    "ablations/vsrd_projection_only": "ablations/01_vsrd_projection_only",
    "ablations/vsrd_projection_silhouette": "ablations/02_vsrd_projection_silhouette",
    "ablations/vsrd_projection_silhouette_rdf": "ablations/03_vsrd_projection_silhouette_rdf",
    "ablations/vsrdpp_velocity_no_init": "ablations/04_vsrdpp_velocity_no_init",
    "ablations/vsrdpp_full": "ablations/05_vsrdpp_full",
    # 旧路径兼容
    "ablation/01_vsrd_projection_only": "ablations/01_vsrd_projection_only",
    "ablation/02_vsrd_projection_silhouette": "ablations/02_vsrd_projection_silhouette",
    "ablation/03_vsrd_projection_silhouette_rdf": "ablations/03_vsrd_projection_silhouette_rdf",
    "ablation/04_vsrdpp_velocity_no_init": "ablations/04_vsrdpp_velocity_no_init",
    "ablation/05_vsrdpp_full": "ablations/05_vsrdpp_full",
    "ablation/erode_seg_mask_degradation_vsrdpp": "ablations/erode_seg_mask_degradation_vsrdpp",
}


def _experiment_config_path(norm: str, ext: str = ".yaml") -> str:
    parts = norm.replace("\\", "/").split("/")
    return os.path.join(_EXPERIMENT_DIR, *parts) + ext


def _config_id_from_path(abs_path: str) -> str:
    rel = os.path.relpath(abs_path, _CONFIGS_DIR).replace("\\", "/")
    return os.path.splitext(rel)[0]


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins on conflict)."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _to_namespace(d):
    """Recursively convert a dict to a SimpleNamespace for dot-access."""
    if isinstance(d, dict):
        return types.SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_namespace(v) for v in d]
    return d


def _load_file(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f) or {}
        return json.load(f)


def _compose_yaml(path: str, *, _stack: set[str] | None = None) -> dict:
    """Hydra-style: merge `defaults:` list then local keys."""
    abs_path = os.path.abspath(path)
    if _stack is None:
        _stack = set()
    if abs_path in _stack:
        raise ValueError(f"Circular config defaults: {abs_path}")
    _stack.add(abs_path)

    raw = _load_file(abs_path)
    defaults = raw.pop("defaults", []) or []
    cfg: dict = {}

    base_dir = os.path.dirname(abs_path)
    for ref in defaults:
        if isinstance(ref, dict):
            continue
        ref_path = os.path.normpath(os.path.join(base_dir, str(ref)))
        if not ref_path.endswith((".yaml", ".yml", ".json")):
            ref_path += ".yaml"
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Config default not found: {ref} -> {ref_path}")
        if ref_path.endswith((".yaml", ".yml")):
            merged = _compose_yaml(ref_path, _stack=_stack)
        else:
            merged = _load_file(ref_path)
        cfg = _deep_merge(cfg, merged)

    cfg = _deep_merge(cfg, raw)
    _stack.remove(abs_path)
    return cfg


def _sequential_config_name(seq_id: str) -> str:
    return f"{_SEQUENTIALS_DIR}/vsrd_plus_full_seq_{seq_id}"


def _normalize_name(name: str) -> str:
    name = name.strip().replace("\\", "/")
    if name.startswith("experiment/"):
        name = name[len("experiment/") :]
    if name.endswith((".yaml", ".yml", ".json")):
        name = os.path.splitext(name)[0]
    if name == "debug":
        name = "smoke"
    # 07 / 10 → vsrdpp_sequentials/vsrd_plus_full_seq_07
    if len(name) == 2 and name.isdigit():
        return _sequential_config_name(name)
    # sequence_07（旧名）
    if name.startswith("sequence_") and name[len("sequence_") :].isdigit():
        return _sequential_config_name(name[len("sequence_") :])
    # vsrd_plus_full_seq_07（短名）→ vsrdpp_sequentials/...
    if (
        name.startswith("vsrd_plus_full_seq_")
        and name[len("vsrd_plus_full_seq_") :].isdigit()
        and not name.startswith(f"{_SEQUENTIALS_DIR}/")
    ):
        return f"{_SEQUENTIALS_DIR}/{name}"
    if name in _LEGACY_ALIASES:
        return _LEGACY_ALIASES[name]
    return name


def _experiment_yaml_candidates(norm: str) -> list[str]:
    paths = [
        _experiment_config_path(norm, ".yaml"),
        _experiment_config_path(norm, ".yml"),
        os.path.join(_EXPERIMENT_DIR, f"{norm}.yaml"),
        os.path.join(_EXPERIMENT_DIR, f"{norm}.yml"),
    ]
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _resolve_config_path(name: str) -> tuple[str, str]:
    """
    Return (absolute_path, config_id) where config_id is passed to train.py --config_path.
    """
    norm = _normalize_name(name)

    for path in _experiment_yaml_candidates(norm):
        if os.path.exists(path):
            return path, _config_id_from_path(path)

    candidates = [
        os.path.join(_CONFIGS_DIR, f"{norm}.yaml"),
        os.path.join(_CONFIGS_DIR, f"{norm}.yml"),
        os.path.join(_CONFIGS_DIR, f"{norm}.json"),
        os.path.join(_CONFIGS_DIR, f"{name}.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path, _config_id_from_path(path)

    available: list[str] = []
    if os.path.isdir(_EXPERIMENT_DIR):
        for root, _, files in os.walk(_EXPERIMENT_DIR):
            for fn in files:
                if fn.endswith((".yaml", ".yml")) and not fn.startswith("_"):
                    rel = os.path.relpath(os.path.join(root, fn), _EXPERIMENT_DIR)
                    available.append(os.path.splitext(rel.replace("\\", "/"))[0])
    raise FileNotFoundError(
        f"Config '{name}' not found.\n"
        f"Try one of: {sorted(available) or ['vsrdpp_sequentials/vsrd_plus_full_seq_10', 'ablations/05_vsrdpp_full']}"
    )


def _finalize_paths(cfg: dict) -> dict:
    cfg["TRAIN"]["CONFIG"] = _CONFIGS_DIR

    train = cfg["TRAIN"]
    dataset_root = train["DATASET"].get("ROOT", "") or os.environ.get("VSRD_DATASET_ROOT", "")
    if dataset_root:
        train["DATASET"]["ROOT"] = dataset_root

    rel_filenames = list(train["DATASET"].get("FILENAMES", []))
    dynamic_path = (train.get("DYNAMIC_LABELS_PATH") or "").strip()

    if not dynamic_path and rel_filenames:
        rel_path = rel_filenames[0]
        if os.path.isabs(rel_path) and dataset_root:
            rel_path = os.path.relpath(rel_path, dataset_root)
        if not os.path.isabs(rel_path):
            sequence_folder = os.path.basename(os.path.dirname(rel_path))
            if sequence_folder.startswith("2013_05_28_drive_"):
                if dataset_root:
                    train["DYNAMIC_LABELS_PATH"] = dynamic_mask_path_from_filenames_txt(
                        rel_path, dataset_root
                    )
                else:
                    train["DYNAMIC_LABELS_PATH"] = os.path.join(
                        DEFAULT_DYNAMIC_LABELS_DIRNAME,
                        sequence_folder,
                        "dynamic_mask.txt",
                    )

    if dataset_root:
        train["DATASET"]["FILENAMES"] = [
            p if os.path.isabs(p) else os.path.join(dataset_root, p)
            for p in rel_filenames
        ]
        dynamic_path = train.get("DYNAMIC_LABELS_PATH", "")
        if dynamic_path and not os.path.isabs(dynamic_path):
            train["DYNAMIC_LABELS_PATH"] = os.path.join(dataset_root, dynamic_path)

    return cfg


def load_config(name: str):
    """
    Load layered experiment YAML (Hydra-style defaults merge).

    Examples:
        load_config("vsrdpp_sequentials/vsrd_plus_full_seq_10")
        load_config("vsrd_plus_full_seq_10")     # 短名 alias
        load_config("sequence_10")               # legacy alias
        load_config("10")                        # shorthand
        load_config("ablations/erode_seg_mask_degradation_vsrdpp")
        load_config("no_init")                   # SKIP_ATTRIBUTE_INIT=true
    """
    norm = _normalize_name(name)

    for path in _experiment_yaml_candidates(norm):
        if os.path.exists(path):
            cfg = _compose_yaml(path)
            return _to_namespace(_finalize_paths(cfg))

    root_yaml = os.path.join(_CONFIGS_DIR, f"{norm}.yaml")
    if os.path.exists(root_yaml):
        cfg = _compose_yaml(root_yaml)
        return _to_namespace(_finalize_paths(cfg))

    _resolve_config_path(name)


def load_launch_settings(name: str) -> dict[str, Any]:
    """Return LAUNCH dict (GPU / train script) from merged experiment config."""
    _, config_id = _resolve_config_path(name)
    cfg = load_config(config_id)
    launch = getattr(cfg, "LAUNCH", None)
    if launch is None:
        return {
            "train_script": "train.py",
            "device_id": 0,
            "cuda_devices": "0",
            "nproc_per_node": 1,
            "rdzv_endpoint": "localhost:22500",
            "config_path": config_id,
        }
    return {
        "train_script": "train.py",
        "device_id": int(getattr(launch, "DEVICE_ID", 0)),
        "cuda_devices": str(getattr(launch, "CUDA_DEVICES", "0")),
        "nproc_per_node": int(getattr(launch, "NPROC_PER_NODE", 1)),
        "rdzv_endpoint": str(getattr(launch, "RDZV_ENDPOINT", "localhost:22500")),
        "config_path": config_id,
    }


# Pre-loaded config for inference.py / evaluation.py (legacy import pattern)
conf_val = load_config("inference")
