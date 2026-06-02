"""Canonical ``multi_inputs`` field names for the initial-attribute pipeline."""


class AttributeKeys:
    PSEUDO_DEPTH = "pseudo_depth"
    ROI_LIDAR = "roi_lidar"
    PROJECTED_VALID_MASK = "projected_valid_mask"
    ROI_LIDAR_VALID = "roi_lidar_valid"

    EST_VELOCITY = "est_velocity"
    EST_LOCATION = "est_location"
    EST_ORIENTATION = "est_orientation"

    GT_LOCATION = "gt_location"
    GT_VELOCITY = "gt_velocity"
    GT_ORIENTATION = "gt_orientation"
    GT_DIMENSION = "gt_dimension"

    # Per-instance dynamic flags inferred from GT bbox velocity (``True`` = dynamic).
    IS_DYNAMIC = "is_dynamic"
