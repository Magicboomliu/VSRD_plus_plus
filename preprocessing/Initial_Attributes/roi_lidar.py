"""Build per-instance RoI point clouds from pseudo depth."""

import torch

from preprocessing.Initial_Attributes.file_io_utils import (
    merge_point_clouds_for_vis,
    read_image_tensor,
    visualize_point_cloud_with_axis,
)
from preprocessing.Initial_Attributes.geometry_op import (
    disparity_to_depth,
    disp_warp,
    generate_point_cloud,
    match_instance_masks_with_iou,
    point_cloud_to_2d_mask,
    transform_bounding_boxes_to_world,
)
from preprocessing.Initial_Attributes.keys import AttributeKeys
from preprocessing.Initial_Attributes.post_processing import (
    cluster_point_cloud,
    compute_point_density,
    shrink_masks_torch,
    update_instance_masks,
)


def build_roi_lidar(
    frame_inputs: dict,
    target_instance_ids: list[int],
    *,
    mask_shrink_ratio: float = 0.25,
    warp_error_keep_ratio: float = 0.40,
    density_radius: float = 0.5,
    min_density_quantile: float = 0.01,
    visualize: bool = False,
) -> tuple[dict, torch.Tensor | None]:
    """Back-project pseudo depth inside instance masks to 3D RoI point clouds."""
    roi_lidar: dict = {}
    filename = frame_inputs["filenames"][0]
    image = frame_inputs["images"]
    boxes_2d = frame_inputs["boxes_2d"]
    boxes_3d = frame_inputs["boxes_3d"]
    masks = frame_inputs["masks"]
    extrinsic = frame_inputs["extrinsic_matrices"]
    intrinsic = frame_inputs["intrinsic_matrices"]
    visible = frame_inputs["visible_masks"][0].cpu().numpy().tolist()
    depth = frame_inputs[AttributeKeys.PSEUDO_DEPTH]

    masks = match_instance_masks_with_iou(
        instance_masks=masks[0],
        bounding_boxes=boxes_2d[0],
    ).unsqueeze(0)

    masks[0] = shrink_masks_torch(instance_masks=masks[0], shrink_ratio=mask_shrink_ratio)

    left_path = filename
    right_path = left_path.replace("image_00", "image_01")
    left_image = read_image_tensor(left_path).to(image.device)
    right_image = read_image_tensor(right_path).to(image.device)
    depth_map = disparity_to_depth(disparity=depth)
    warped_right, valid_mask = disp_warp(img=right_image, disp=depth_map)
    warped_right = warped_right * valid_mask
    warp_error = (
        torch.sum(torch.abs(warped_right - left_image), dim=1, keepdim=True)
        * valid_mask[:, 0:1, :, :]
    )

    for idx in range(masks[0].shape[0]):
        masks[0][idx] = valid_mask[:, 0, :, :].squeeze(0) * masks[0][idx]

    masks = update_instance_masks(
        warp_error=warp_error,
        instance_masks=masks,
        ratio=warp_error_keep_ratio,
    )

    boxes_3d_world = transform_bounding_boxes_to_world(
        extrinsic_matrix=torch.inverse(extrinsic),
        bounding_boxes_camera=boxes_3d[0],
    )
    point_cloud = generate_point_cloud(
        image=image,
        intrinsics=intrinsic,
        extrinsics=extrinsic,
        depth_map=depth,
    )

    num_instances = masks[0].shape[0]
    for instance_idx in range(num_instances):
        instance_mask = masks[0][instance_idx].bool().repeat(1, 3, 1, 1)
        instance_points = point_cloud[instance_mask]
        instance_points = (
            instance_points.squeeze(0).view(3, -1).permute(1, 0).reshape(-1, 3)
        )

        density = compute_point_density(point_cloud=instance_points, radius=density_radius)
        if density.numel() > 0:
            cutoff = torch.quantile(density.float(), min_density_quantile)
            instance_points = instance_points[density > cutoff]

        if len(instance_points) > 11:
            instance_points = cluster_point_cloud(cam_points=instance_points)

        instance_id = target_instance_ids[instance_idx]
        if (
            instance_points is not None
            and instance_points.shape[0] > 1
            and visible[instance_idx]
        ):
            roi_lidar[instance_id] = instance_points
        else:
            roi_lidar[instance_id] = None

    merged_points = merge_point_clouds_for_vis(pcds=roi_lidar)
    if len(merged_points) == 0:
        return roi_lidar, None

    merged_points = [tensor.to(image.device) for tensor in merged_points]
    merged_points = torch.cat(merged_points, dim=0)
    projected_mask = point_cloud_to_2d_mask(
        merged_points.type_as(extrinsic),
        K=intrinsic,
        world_to_camera=extrinsic,
        H=image.shape[-2],
        W=image.shape[-1],
    )

    if visualize:
        visualize_point_cloud_with_axis(
            point_cloud=merged_points,
            boxes_3d=boxes_3d_world.cpu().numpy(),
        )

    return roi_lidar, projected_mask
