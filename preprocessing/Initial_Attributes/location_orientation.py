"""Estimate location and orientation from RoI LiDAR and velocity."""

import torch
import torch.nn as nn

from preprocessing.Initial_Attributes.geometry_op import rotation_matrix_y
from preprocessing.Initial_Attributes.keys import AttributeKeys


def _roi_lidar_valid_mask(roi_lidar: dict, min_points: int) -> list[bool]:
    valid = []
    for instance_id in roi_lidar:
        points = roi_lidar[instance_id]
        if points is None or points.shape[0] < min_points:
            valid.append(False)
        else:
            valid.append(True)
    return valid


def _bidirectional_search_roi_lidar(
    multi_inputs: dict,
    current_frame_idx: int,
    instance_idx: int,
    instance_ids: list[int],
    *,
    min_roi_points: int,
    device: torch.device | str | None,
) -> tuple[torch.Tensor, int]:
    """Find a nearby frame with enough RoI points; project back with velocity."""
    frame_keys = sorted(multi_inputs.keys())
    current_pos = frame_keys.index(current_frame_idx)
    forward_cloud = None
    backward_cloud = None
    time_gap = 0

    for next_key in frame_keys[current_pos + 1 :]:
        roi_lidar = multi_inputs[next_key][AttributeKeys.ROI_LIDAR]
        valid = _roi_lidar_valid_mask(roi_lidar, min_roi_points)
        if valid[instance_idx]:
            forward_cloud = roi_lidar[instance_ids[instance_idx]]
            time_gap = next_key - current_frame_idx
            break

    for prev_key in frame_keys[:current_pos][::-1]:
        roi_lidar = multi_inputs[prev_key][AttributeKeys.ROI_LIDAR]
        valid = _roi_lidar_valid_mask(roi_lidar, min_roi_points)
        if valid[instance_idx]:
            backward_cloud = roi_lidar[instance_ids[instance_idx]]
            time_gap = prev_key - current_frame_idx
            break

    if backward_cloud is not None and forward_cloud is not None:
        selected = backward_cloud if backward_cloud.shape[0] > forward_cloud.shape[0] else forward_cloud
        return selected, time_gap
    if forward_cloud is not None:
        return forward_cloud, time_gap
    if backward_cloud is not None:
        return backward_cloud, time_gap

    return torch.randn((1, 3), device=device), time_gap


def estimate_location_orientation(
    multi_inputs: dict,
    dynamic_mask_list: list[bool],
    *,
    min_roi_points: int = 120,
    device: torch.device | str | None = None,
) -> dict:
    """Compute per-instance location (point-cloud centroid) and orientation."""
    for frame_key, frame_inputs in multi_inputs.items():
        frame_inputs[AttributeKeys.EST_VELOCITY] = frame_inputs[AttributeKeys.EST_VELOCITY].unsqueeze(0)

        roi_lidar = frame_inputs[AttributeKeys.ROI_LIDAR]
        velocity = frame_inputs[AttributeKeys.EST_VELOCITY]
        instance_ids = list(roi_lidar.keys())
        valid_mask = _roi_lidar_valid_mask(roi_lidar, min_roi_points)

        locations = []
        orientations_from_velocity = nn.functional.normalize(velocity[..., [2, 0]], dim=-1)
        orientations_from_velocity = rotation_matrix_y(
            *torch.unbind(orientations_from_velocity, dim=-1)
        )

        for idx, is_valid in enumerate(valid_mask):
            if is_valid:
                instance_points = roi_lidar[instance_ids[idx]]
            else:
                borrowed_points, time_gap = _bidirectional_search_roi_lidar(
                    multi_inputs=multi_inputs,
                    current_frame_idx=frame_key,
                    instance_idx=idx,
                    instance_ids=instance_ids,
                    min_roi_points=min_roi_points,
                    device=device,
                )
                instance_velocity = velocity[0][idx].unsqueeze(0)
                instance_points = borrowed_points.to(device) + time_gap * instance_velocity

            instance_points = instance_points.to(device)
            locations.append(torch.mean(instance_points, dim=0, keepdim=True))

            if dynamic_mask_list[idx]:
                orientations_from_velocity = orientations_from_velocity.float()

            if frame_key == 0:
                frame_inputs[AttributeKeys.EST_ORIENTATION] = orientations_from_velocity.float()

        frame_inputs[AttributeKeys.EST_LOCATION] = torch.cat(locations, dim=0).unsqueeze(0)

    return multi_inputs
