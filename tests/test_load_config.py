"""Tests for config loading and path auto-resolution."""

import os

from preprocessing.dataset_paths import DEFAULT_DYNAMIC_LABELS_DIRNAME
from trainer.configs import load_config, load_launch_settings


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


def test_smoke_config_matches_formal_training_defaults():
    smoke = load_config("smoke")
    seq00 = load_config("vsrd_plus_full_seq_00")
    assert smoke.TRAIN.OPTIMIZATION_NUM_STEPS == seq00.TRAIN.OPTIMIZATION_NUM_STEPS
    assert smoke.TRAIN.LOGGING.CKPT_INTERVALS == seq00.TRAIN.LOGGING.CKPT_INTERVALS


def test_debug_alias_loads_smoke():
    assert load_config("debug").TRAIN.OPTIMIZATION_NUM_STEPS == load_config("smoke").TRAIN.OPTIMIZATION_NUM_STEPS


def test_legacy_sequence_alias():
    assert load_config("sequence_07").TRAIN.OPTIMIZATION_NUM_STEPS == load_config(
        "vsrd_plus_full_seq_07"
    ).TRAIN.OPTIMIZATION_NUM_STEPS


def test_shorthand_sequence_id():
    assert load_config("07").TRAIN.OPTIMIZATION_NUM_STEPS == load_config(
        "vsrd_plus_full_seq_07"
    ).TRAIN.OPTIMIZATION_NUM_STEPS


def test_ablation_ladder_configs():
    c1 = load_config("ablations/01_vsrd_projection_only")
    assert c1.TRAIN.USE_DYNAMIC_MODELING is False
    assert c1.TRAIN.USE_RDF_MODELING is False
    assert c1.TRAIN.LOSS_WEIGHT.SILHOUETTE_LOSS == 0.0

    c2 = load_config("ablations/02_vsrd_projection_silhouette")
    assert c2.TRAIN.LOSS_WEIGHT.SILHOUETTE_LOSS == 1.0
    assert c2.TRAIN.USE_RDF_MODELING is False

    c3 = load_config("ablations/03_vsrd_projection_silhouette_rdf")
    assert c3.TRAIN.USE_RDF_MODELING is True
    assert c3.TRAIN.USE_DYNAMIC_MODELING is False

    c4 = load_config("no_init")
    assert load_launch_settings("no_init")["train_script"] == "train.py"
    assert c4.TRAIN.SKIP_ATTRIBUTE_INIT is True
    assert c4.TRAIN.USE_DYNAMIC_MODELING is True

    erode = load_config("ablations/erode_seg_mask_degradation_vsrdpp")
    assert erode.TRAIN.MASK_ERODE_RATIO == 0.03
    assert load_launch_settings("ablations/erode_seg_mask_degradation_vsrdpp")["train_script"] == "train.py"

    c5 = load_config("ablations/05_vsrdpp_full")
    assert load_launch_settings("ablations/05_vsrdpp_full")["train_script"] == "train.py"
    assert c5.TRAIN.USE_DYNAMIC_MODELING is True
    assert c5.TRAIN.USE_RDF_MODELING is True

    casual = load_config("vsrd_projection_only")
    assert casual.TRAIN.DATASET.FILENAMES[0].endswith(
        "filenames/cascual_splits/train_all_filenames.txt"
    )
    assert casual.TRAIN.DYNAMIC_LABELS_PATH.endswith(
        "filenames/cascual_splits/train_all_dynamic_mask.txt"
    )
    assert casual.TRAIN.USE_DYNAMIC_LABELS_FILE is False


def test_stage1_trainfile_dataset_configs():
    casual = load_config("stage1_trainfiles/_dataset_casual")
    vsrd24 = load_config("stage1_trainfiles/_dataset_vsrd24")
    assert casual.TRAIN.DATASET.FILENAMES[0].endswith(
        "filenames/cascual_splits/train_all_filenames.txt"
    )
    assert vsrd24.TRAIN.DATASET.FILENAMES[0].endswith(
        "filenames/vsrd24_splits/train_all_filenames.txt"
    )
    assert casual.TRAIN.DYNAMIC_LABELS_PATH.endswith(
        "filenames/cascual_splits/train_all_dynamic_mask.txt"
    )
    assert vsrd24.TRAIN.DYNAMIC_LABELS_PATH.endswith(
        "filenames/vsrd24_splits/train_all_dynamic_mask.txt"
    )


def test_stage1_vsrdpp_full_train_configs():
    casual = load_config("stage1_trainfiles/vsrdpp_full_casual")
    vsrd24 = load_config("stage1_trainfiles/vsrdpp_full_vsrd24")
    assert casual.TRAIN.MODEL_TYPE == "vsrdpp_full_casual"
    assert vsrd24.TRAIN.MODEL_TYPE == "vsrdpp_full_vsrd24"
    assert casual.TRAIN.USE_DYNAMIC_LABELS_FILE is True
    assert vsrd24.TRAIN.USE_DYNAMIC_LABELS_FILE is True
    assert casual.TRAIN.USE_RDF_MODELING is True
    assert vsrd24.TRAIN.USE_DYNAMIC_MODELING is True
    assert casual.TRAIN.DATASET.FILENAMES[0].endswith("cascual_splits/train_all_filenames.txt")
    assert vsrd24.TRAIN.DATASET.FILENAMES[0].endswith("vsrd24_splits/train_all_filenames.txt")


def test_ablation_name_aliases():
    assert load_config("vsrdpp_full").TRAIN.MODEL_TYPE == "ablation_05_vsrdpp_full"
    assert load_config("ablations/vsrdpp_full").TRAIN.MODEL_TYPE == "ablation_05_vsrdpp_full"
    assert load_config("vsrd_projection_only").TRAIN.USE_RDF_MODELING is False
    assert load_config("ablations/vsrd_projection_only").TRAIN.USE_RDF_MODELING is False


def test_ablation_selective_legacy_alias():
    assert load_config("ablation_selective").TRAIN.MASK_ERODE_RATIO == load_config(
        "ablations/erode_seg_mask_degradation_vsrdpp"
    ).TRAIN.MASK_ERODE_RATIO


def test_sequential_short_name_alias():
    assert load_config("vsrd_plus_full_seq_07").TRAIN.OPTIMIZATION_NUM_STEPS == load_config(
        "vsrdpp_sequentials/vsrd_plus_full_seq_07"
    ).TRAIN.OPTIMIZATION_NUM_STEPS


def test_sequential_model_type_and_train_modes():
    cfg = load_config("07")
    assert cfg.TRAIN.MODEL_TYPE == "vsrd_plus_full_seq_07"
    assert cfg.TRAIN.SKIP_ATTRIBUTE_INIT is False
    assert cfg.TRAIN.MASK_ERODE_RATIO == 0.0


def test_erode_inherits_full_ablation():
    erode = load_config("ablations/erode_seg_mask_degradation_vsrdpp")
    full = load_config("ablations/05_vsrdpp_full")
    assert erode.TRAIN.USE_DYNAMIC_MODELING == full.TRAIN.USE_DYNAMIC_MODELING
    assert erode.TRAIN.USE_RDF_MODELING == full.TRAIN.USE_RDF_MODELING
    assert erode.TRAIN.MASK_ERODE_RATIO == 0.03
