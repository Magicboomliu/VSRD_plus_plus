"""Tests for preprocessing output path conventions."""

from preprocessing.dataset_paths import (
    DEFAULT_DYNAMIC_LABELS_DIRNAME,
    DEFAULT_PSEUDO_DEPTH_DIRNAME,
    dynamic_mask_path,
    dynamic_mask_path_from_filenames_txt,
    pseudo_depth_path_from_image,
)


def test_pseudo_depth_path():
    img = "data_2d_raw/2013_05_28_drive_0006_sync/image_00/data_rect/0001.png"
    out = pseudo_depth_path_from_image(img)
    assert DEFAULT_PSEUDO_DEPTH_DIRNAME in out
    assert out.endswith("2013_05_28_drive_0006_sync/image_00/data_rect/0001.png")


def test_dynamic_mask_path():
    root = "/data/KITTI360"
    seq = "2013_05_28_drive_0007_sync"
    path = dynamic_mask_path(root, seq)
    assert path == f"/data/KITTI360/{DEFAULT_DYNAMIC_LABELS_DIRNAME}/{seq}/dynamic_mask.txt"


def test_dynamic_mask_path_from_filenames_txt():
    rel = "filenames/R50-N16-M128-B16/2013_05_28_drive_0000_sync/sampled_image_filenames.txt"
    path = dynamic_mask_path_from_filenames_txt(rel, "/data/KITTI360")
    assert path.endswith(
        f"{DEFAULT_DYNAMIC_LABELS_DIRNAME}/2013_05_28_drive_0000_sync/dynamic_mask.txt"
    )
