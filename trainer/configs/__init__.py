import json
import os
import types

from preprocessing.dataset_paths import (
    DEFAULT_DYNAMIC_LABELS_DIRNAME,
    dynamic_mask_path_from_filenames_txt,
)

_CONFIGS_DIR = os.path.dirname(os.path.abspath(__file__))


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


def load_config(name: str):
    """
    Load a merged config: base.json deep-merged with <name>.json.

    Args:
        name: config name without extension, e.g. "sequence_00", "smoke",
              "ablation_full", "inference".  Also accepts bare sequence IDs
              like "00", "02" etc. for convenience.

    Returns:
        A nested SimpleNamespace so cfg.TRAIN.DATASET.FILENAMES etc. work.
        TRAIN.CONFIG is auto-set to the configs directory path.
    """
    # Allow bare sequence IDs like "00" as shorthand for "sequence_00"
    if len(name) == 2 and name.isdigit():
        name = f"sequence_{name}"
    if name == "debug":
        name = "smoke"

    base_path = os.path.join(_CONFIGS_DIR, "base.json")
    # Support sub-directory paths like "SPLITS64/split_sub"
    override_path = os.path.join(_CONFIGS_DIR, f"{name}.json")

    with open(base_path) as f:
        cfg = json.load(f)

    if os.path.exists(override_path):
        with open(override_path) as f:
            overrides = json.load(f)
        cfg = _deep_merge(cfg, overrides)
    else:
        raise FileNotFoundError(
            f"Config '{name}' not found at {override_path}.\n"
            f"Available configs: {[p[:-5] for p in os.listdir(_CONFIGS_DIR) if p.endswith('.json') and p != 'base.json']}"
        )

    # Auto-set TRAIN.CONFIG so ckpt/log paths are derived correctly
    cfg["TRAIN"]["CONFIG"] = _CONFIGS_DIR

    train = cfg["TRAIN"]
    dataset_root = train["DATASET"].get("ROOT", "") or os.environ.get("VSRD_DATASET_ROOT", "")
    if dataset_root:
        train["DATASET"]["ROOT"] = dataset_root

    rel_filenames = list(train["DATASET"].get("FILENAMES", []))
    dynamic_path = (train.get("DYNAMIC_LABELS_PATH") or "").strip()

    # Default dynamic_mask.txt from first FILENAMES (standard sequence layout).
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

    # Resolve relative paths using DATASET.ROOT so callers always get absolute paths.
    if dataset_root:
        train["DATASET"]["FILENAMES"] = [
            p if os.path.isabs(p) else os.path.join(dataset_root, p)
            for p in rel_filenames
        ]
        dynamic_path = train.get("DYNAMIC_LABELS_PATH", "")
        if dynamic_path and not os.path.isabs(dynamic_path):
            train["DYNAMIC_LABELS_PATH"] = os.path.join(dataset_root, dynamic_path)

    return _to_namespace(cfg)


# Pre-loaded config for inference.py / evaluation.py (legacy import pattern)
conf_val = load_config("inference")
