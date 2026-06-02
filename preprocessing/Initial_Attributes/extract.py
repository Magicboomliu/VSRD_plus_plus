"""Helpers for reading pipeline outputs in training scripts."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from preprocessing.Initial_Attributes.keys import AttributeKeys


@dataclass
class InitialAttributesResult:
    """Estimated (and optional GT) attributes at the target frame."""

    est_location: torch.Tensor
    est_velocity: torch.Tensor
    est_orientation: torch.Tensor
    roi_lidar_valid: bool
    gt_location: torch.Tensor | None = None
    gt_velocity: torch.Tensor | None = None
    gt_orientation: torch.Tensor | None = None
    gt_dimension: torch.Tensor | None = None
    is_dynamic: list[bool] | None = None


def extract_initial_attributes(
    multi_inputs: dict,
    *,
    frame_key: int = 0,
    device=None,
    gt_fallback_threshold_m: float = 4.0,
) -> InitialAttributesResult:
    """Read initialization tensors from ``multi_inputs`` after ``estimate_initial_attributes``."""
    frame = multi_inputs[frame_key]

    est_location = frame[AttributeKeys.EST_LOCATION].float().contiguous().to(device)
    est_velocity = frame[AttributeKeys.EST_VELOCITY].float().contiguous().to(device)
    est_orientation = frame[AttributeKeys.EST_ORIENTATION].float().contiguous().to(device)
    roi_lidar_valid = bool(frame[AttributeKeys.ROI_LIDAR_VALID])

    gt_location = frame.get(AttributeKeys.GT_LOCATION)
    gt_velocity = frame.get(AttributeKeys.GT_VELOCITY)
    gt_orientation = frame.get(AttributeKeys.GT_ORIENTATION)
    gt_dimension = frame.get(AttributeKeys.GT_DIMENSION)
    is_dynamic = frame.get(AttributeKeys.IS_DYNAMIC)

    if gt_location is not None:
        gt_location = gt_location.float().contiguous().to(device)
        if torch.max(torch.abs(gt_location - est_location)) > gt_fallback_threshold_m:
            est_location = gt_location

    if gt_velocity is not None:
        gt_velocity = gt_velocity.float().contiguous().to(device)
    if gt_orientation is not None:
        gt_orientation = gt_orientation.float().contiguous().to(device)
    if gt_dimension is not None:
        gt_dimension = gt_dimension.float().contiguous().to(device)

    return InitialAttributesResult(
        est_location=est_location,
        est_velocity=est_velocity,
        est_orientation=est_orientation,
        roi_lidar_valid=roi_lidar_valid,
        gt_location=gt_location,
        gt_velocity=gt_velocity,
        gt_orientation=gt_orientation,
        gt_dimension=gt_dimension,
        is_dynamic=list(is_dynamic) if is_dynamic is not None else None,
    )
