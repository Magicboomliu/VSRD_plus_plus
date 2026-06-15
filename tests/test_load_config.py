"""Tests for config loading and path auto-resolution."""

import os

from preprocessing.dataset_paths import DEFAULT_DYNAMIC_LABELS_DIRNAME
from trainer.configs import load_config


def test_sequence_config_auto_dynamic_labels_path():
    cfg = load_config("sequence_07")
    root = cfg.TRAIN.DATASET.ROOT
    expected = os.path.join(
        root,
        DEFAULT_DYNAMIC_LABELS_DIRNAME,
        "2013_05_28_drive_0007_sync",
        "dynamic_mask.txt",
    )
    assert cfg.TRAIN.DYNAMIC_LABELS_PATH == expected


def test_smoke_config_has_short_optimization():
    cfg = load_config("smoke")
    assert cfg.TRAIN.OPTIMIZATION_NUM_STEPS == 50


def test_debug_alias_loads_smoke():
    assert load_config("debug").TRAIN.OPTIMIZATION_NUM_STEPS == 50


def test_ablation_full_keeps_custom_dynamic_path():
    cfg = load_config("ablation_full")
    root = cfg.TRAIN.DATASET.ROOT
    assert cfg.TRAIN.DYNAMIC_LABELS_PATH == os.path.join(
        root, "train_ablation_filenames/train_ablation_dynamic_mask.txt"
    )
