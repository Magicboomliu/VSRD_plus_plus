import operator
import functools

import torch
import torch.nn as nn

# from .. import utils


def expand_to_4x4(matrices):
    matrices_4x4 = torch.eye(4).to(matrices)
    matrices_4x4 = matrices_4x4.reshape(*[1] * len(matrices.shape[:-2]), 4, 4)
    matrices_4x4 = matrices_4x4.repeat(*matrices.shape[:-2], 1, 1)
    matrices_4x4[..., :matrices.shape[-2], :matrices.shape[-1]] = matrices
    return matrices_4x4


def skew_symmetric_matrix(vectors):
    x, y, z = torch.unbind(vectors, dim=-1)
    zero = torch.zeros_like(x)
    cross_matrices = torch.stack([
        torch.stack([zero,   -z,    y], dim=-1),
        torch.stack([   z, zero,   -x], dim=-1),
        torch.stack([  -y,    x, zero], dim=-1),
    ], dim=-2)
    return cross_matrices


def rotation_matrix_x(angles):
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    one = torch.ones_like(angles)
    zero = torch.zeros_like(angles)
    rotation_matrices = torch.stack([
        torch.stack([ one, zero,  zero], dim=-1),
        torch.stack([zero,  cos,  -sin], dim=-1),
        torch.stack([zero,  sin,   cos], dim=-1),
    ], dim=-2)
    return rotation_matrices


def rotation_matrix_y(angles):
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    one = torch.ones_like(angles)
    zero = torch.zeros_like(angles)
    rotation_matrices = torch.stack([
        torch.stack([ cos, zero,  sin], dim=-1),
        torch.stack([zero,  one, zero], dim=-1),
        torch.stack([-sin, zero,  cos], dim=-1),
    ], dim=-2)
    return rotation_matrices


def rotation_matrix_z(angles):
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    one = torch.ones_like(angles)
    zero = torch.zeros_like(angles)
    rotation_matrices = torch.stack([
        torch.stack([ cos, -sin, zero], dim=-1),
        torch.stack([ sin,  cos, zero], dim=-1),
        torch.stack([zero, zero,  one], dim=-1),
    ], dim=-2)
    return rotation_matrices


def rotation_matrix(rotation_axes, rotation_angles):
    cos = torch.cos(rotation_angles).unsqueeze(-1).unsqueeze(-2)
    sin = torch.sin(rotation_angles).unsqueeze(-1).unsqueeze(-2)
    rotation_matrices = (
        (1.0 - cos) * torch.einsum("...m,...n->...mn", rotation_axes, rotation_axes) +
        sin * skew_symmetric_matrix(rotation_axes) +
        cos * torch.eye(3).to(rotation_axes)
    )
    return rotation_matrices


def translation_matrix(translation_vectors):
    translation_matrices = torch.eye(4).to(translation_vectors)
    translation_matrices = translation_matrices.reshape(*[1] * len(translation_vectors.shape[:-1]), 4, 4)
    translation_matrices = translation_matrices.repeat(*translation_vectors.shape[:-1], 1, 1)
    translation_matrices[..., :3, 3] = translation_vectors
    return translation_matrices


def essential_matrix(rotation_matrices, translation_vectors):
    essential_matrices = skew_symmetric_matrix(translation_vectors) @ rotation_matrices
    return essential_matrices


def fundamental_matrix(essential_matrices, intrinsic_matrices_1, intrinsic_matrices_2):
    fundamental_matrces = torch.linalg.inv(intrinsic_matrices_2).transpose(-2, -1) @ essential_matrices @ torch.linalg.inv(intrinsic_matrices_1)
    return fundamental_matrces


def projection(coord_maps, intrinsic_matrices, extrinsic_matrices=None):
    # vector to matrix
    coord_maps = coord_maps.unsqueeze(-1)
    if extrinsic_matrices is not None:
        # broadcast to the image plane
        if extrinsic_matrices.ndim == 3:
            extrinsic_matrices = extrinsic_matrices.unsqueeze(-3).unsqueeze(-4)
        # (x, y, z, w) <- (R | t) @ (x, y, z, w)
        coord_maps = extrinsic_matrices @ coord_maps
    # (x, y, z) <- (x / w, y / w, z / w)
    coord_maps = coord_maps[..., :-1, :] / coord_maps[..., -1:, :]
    # broadcast to the image plane
    if intrinsic_matrices.ndim == 3:
        intrinsic_matrices = intrinsic_matrices.unsqueeze(-3).unsqueeze(-4)
    # (x, y, z) <- K @ (x, y, z)
    coord_maps = intrinsic_matrices @ coord_maps
    # matrix to vector
    coord_maps = coord_maps.squeeze(-1)
    return coord_maps


def backprojection(depth_maps, intrinsic_matrices, extrinsic_matrices=None):
    coords_y = torch.arange(depth_maps.shape[-2]).to(depth_maps)
    coords_x = torch.arange(depth_maps.shape[-1]).to(depth_maps)
    coord_maps_y, coord_maps_x = torch.meshgrid(coords_y, coords_x, indexing="ij")
    coord_maps = torch.stack([coord_maps_x, coord_maps_y], dim=-1)
    # homogeneous coordinates
    coord_maps = nn.functional.pad(coord_maps, (0, 1), mode="constant", value=1.0)
    # (x, y, z) <- d × (x, y, 1)
    coord_maps = coord_maps * depth_maps.permute(0, 2, 3, 1)
    # vector to matrix
    coord_maps = coord_maps.unsqueeze(-1)
    # broadcast to the image plane
    if intrinsic_matrices.ndim == 3:
        intrinsic_matrices = intrinsic_matrices.unsqueeze(-3).unsqueeze(-4)
    # (x, y, z) <- inv(K) @ (x, y, z)
    coord_maps = torch.linalg.inv(intrinsic_matrices) @ coord_maps
    # (x, y, z, w) <- (x, y, z, 1)
    coord_maps = nn.functional.pad(coord_maps, (0, 0, 0, 1), mode="constant", value=1.0)
    if extrinsic_matrices is not None:
        # broadcast to the image plane
        if extrinsic_matrices.ndim == 3:
            extrinsic_matrices = extrinsic_matrices.unsqueeze(-3).unsqueeze(-4)
        # (x, y, z, w) <- inv(R | t) @ (x, y, z, w)
        coord_maps = torch.linalg.inv(extrinsic_matrices) @ coord_maps
    # matrix to vector
    coord_maps = coord_maps.squeeze(-1)
    return coord_maps
