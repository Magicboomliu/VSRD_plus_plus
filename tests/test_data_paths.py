"""
Tests: data file existence and readability.

Checks that:
  1. load_config() works for every named config.
  2. DATASET.ROOT directory exists.
  3. Every FILENAMES txt file exists and is non-empty.
  4. Every path inside a FILENAMES txt resolves to a real image file
     (sampled, not exhaustive – first and last line per file).
  5. DYNAMIC_LABELS_PATH exists and is non-empty.
  6. Every path inside dynamic_mask.txt resolves to a real image file
     (first and last line per file).

Run from project root:
    pixi run pytest tests/test_data_paths.py -v
"""

import os
import pytest
from trainer.configs import load_config

# All named configs that should be loadable
ALL_CONFIGS = [
    "sequence_00",
    "sequence_02",
    "sequence_03",
    "sequence_04",
    "sequence_05",
    "sequence_06",
    "sequence_07",
    "sequence_09",
    "sequence_10",
    "ablation_full",
    "debug",
    "inference",
]

# Configs that have a real FILENAMES file on disk (ablation_selective skipped if
# its data doesn't exist yet)
SEQUENCE_CONFIGS = [c for c in ALL_CONFIGS if c.startswith("sequence_")]


def _filenames_exist(name: str) -> bool:
    """Return True only if every FILENAMES txt for this config is present on disk."""
    try:
        cfg = load_config(name)
        return all(os.path.isfile(p) for p in cfg.TRAIN.DATASET.FILENAMES)
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _first_and_last(filepath):
    """Return (first_line, last_line) stripped, skipping blank lines."""
    lines = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    assert lines, f"{filepath} is empty"
    return lines[0], lines[-1]


def _parse_filenames_line(line, dataset_root):
    """Parse a sampled_image_filenames.txt line → absolute image path."""
    # Format: <instance_ids> <image_path> <relative_indices>
    parts = line.split(" ")
    assert len(parts) >= 2, f"Unexpected line format: {line!r}"
    img_path = parts[1]
    if not os.path.isabs(img_path):
        img_path = os.path.join(dataset_root, img_path)
    return img_path


def _parse_dynamic_mask_line(line, dataset_root):
    """Parse a dynamic_mask.txt line → absolute image path."""
    # Format: <instance_ids> <image_path> <labels>
    parts = line.split(" ")
    assert len(parts) >= 2, f"Unexpected line format: {line!r}"
    img_path = parts[1]
    if not os.path.isabs(img_path):
        img_path = os.path.join(dataset_root, img_path)
    return img_path


# ─────────────────────────────────────────────────────────────────────────────
# 1. load_config() smoke test
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_load_config(name):
    """load_config() should return a valid namespace with required fields."""
    cfg = load_config(name)
    assert hasattr(cfg, "TRAIN")
    assert hasattr(cfg.TRAIN, "DATASET")
    assert hasattr(cfg.TRAIN.DATASET, "ROOT")
    assert hasattr(cfg.TRAIN.DATASET, "FILENAMES")
    assert isinstance(cfg.TRAIN.DATASET.FILENAMES, list)
    assert len(cfg.TRAIN.DATASET.FILENAMES) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET.ROOT exists
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_dataset_root_exists(name):
    cfg = load_config(name)
    root = cfg.TRAIN.DATASET.ROOT
    assert os.path.isdir(root), f"DATASET.ROOT does not exist: {root}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. FILENAMES txt files exist and are non-empty
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_filenames_txt_exist(name):
    cfg = load_config(name)
    for txt_path in cfg.TRAIN.DATASET.FILENAMES:
        if not os.path.isfile(txt_path):
            pytest.skip(f"Data not yet generated — FILENAMES txt missing: {txt_path}")
        assert os.path.getsize(txt_path) > 0, f"FILENAMES txt is empty: {txt_path}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Image paths inside FILENAMES txt resolve to real files (first + last line)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", SEQUENCE_CONFIGS)
def test_filenames_image_paths_exist(name):
    cfg = load_config(name)
    root = cfg.TRAIN.DATASET.ROOT
    for txt_path in cfg.TRAIN.DATASET.FILENAMES:
        if not os.path.isfile(txt_path):
            pytest.skip(f"Data not yet generated — txt missing: {txt_path}")
        first, last = _first_and_last(txt_path)
        for line in (first, last):
            img_path = _parse_filenames_line(line, root)
            assert os.path.isfile(img_path), (
                f"Image referenced in {txt_path} not found:\n  {img_path}\n"
                f"  (line: {line[:80]})"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 5. DYNAMIC_LABELS_PATH exists and is non-empty
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", SEQUENCE_CONFIGS)
def test_dynamic_labels_path_exists(name):
    cfg = load_config(name)
    dyn_path = cfg.TRAIN.DYNAMIC_LABELS_PATH
    if not os.path.isfile(dyn_path):
        pytest.skip(f"Data not yet generated — dynamic_mask missing: {dyn_path}")
    assert os.path.getsize(dyn_path) > 0, f"DYNAMIC_LABELS_PATH is empty: {dyn_path}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Image paths inside dynamic_mask.txt resolve to real files (first + last)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 7. pseudo_depth_ssl coverage: every image in FILENAMES must have a depth PNG
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", SEQUENCE_CONFIGS)
def test_pseudo_depth_exists_for_sampled_frames(name):
    """Every frame listed in sampled_image_filenames.txt must have a depth map."""
    cfg = load_config(name)
    root = cfg.TRAIN.DATASET.ROOT
    for txt_path in cfg.TRAIN.DATASET.FILENAMES:
        if not os.path.isfile(txt_path):
            pytest.skip(f"Data not yet generated — txt missing: {txt_path}")
        with open(txt_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            img_path = _parse_filenames_line(line, root)
            depth_path = img_path.replace("data_2d_raw", "pseudo_depth_ssl")
            assert os.path.isfile(depth_path), (
                f"Depth map missing for sampled frame:\n"
                f"  image: {img_path}\n"
                f"  depth: {depth_path}"
            )


@pytest.mark.parametrize("name", SEQUENCE_CONFIGS)
def test_dynamic_mask_image_paths_exist(name):
    cfg = load_config(name)
    root = cfg.TRAIN.DATASET.ROOT
    dyn_path = cfg.TRAIN.DYNAMIC_LABELS_PATH
    if not os.path.isfile(dyn_path):
        pytest.skip(f"Data not yet generated — dynamic_mask missing: {dyn_path}")
    first, last = _first_and_last(dyn_path)
    for line in (first, last):
        img_path = _parse_dynamic_mask_line(line, root)
        assert os.path.isfile(img_path), (
            f"Image referenced in dynamic_mask.txt not found:\n  {img_path}\n"
            f"  (line: {line[:80]})"
        )
