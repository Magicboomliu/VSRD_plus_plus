"""Default on-disk layout for preprocessing outputs (relative to ``DATASET.ROOT``)."""

from __future__ import annotations

import os

# WAFT-Stereo default from preprocessing/scripts/generate_pseudo_depth_waft.sh
DEFAULT_PSEUDO_DEPTH_DIRNAME = "pseudo_depth_ssl_waft_stereo"

# Dynamic_Labels.pipeline default output root
DEFAULT_DYNAMIC_LABELS_DIRNAME = "dynamic_attributes_est_gt"

# Legacy layouts (reference / manual migration only)
LEGACY_PSEUDO_DEPTH_DIRNAME = "pseudo_depth_ssl"
LEGACY_DYNAMIC_LABELS_DIRNAME = "dynamic_attributes_est"


def pseudo_depth_path_from_image(image_path: str) -> str:
    """Map ``data_2d_raw/...`` → ``pseudo_depth_ssl_waft_stereo/...``."""
    return image_path.replace("data_2d_raw", DEFAULT_PSEUDO_DEPTH_DIRNAME)


def dynamic_mask_path(dataset_root: str, sequence_folder: str) -> str:
    """Default generated ``dynamic_mask.txt`` for a KITTI360 sync folder name."""
    return os.path.join(
        dataset_root,
        DEFAULT_DYNAMIC_LABELS_DIRNAME,
        sequence_folder,
        "dynamic_mask.txt",
    )


def dynamic_mask_path_from_filenames_txt(filenames_rel_path: str, dataset_root: str) -> str:
    """Derive default dynamic mask path from a ``sampled_image_filenames.txt`` config path."""
    sequence_folder = os.path.basename(os.path.dirname(filenames_rel_path))
    return dynamic_mask_path(dataset_root, sequence_folder)
