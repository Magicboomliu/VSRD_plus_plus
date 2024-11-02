from __future__ import division
from genericpath import samefile

import numpy as np
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as F
import random
import cv2 as cv
import skimage

import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import utils

import matplotlib.pyplot as plt


class Resizer(nn.Module):
    
    def __init__(
        self,
        image_size,
        image_interp_mode="bilinear",
        masks_interp_mode="nearest",):
        super().__init__()
        self.image_size = image_size
        self.image_interp_mode = image_interp_mode
        self.masks_interp_mode = masks_interp_mode

    def forward(self, inputs):
        
        masks = inputs['masks']
        image = inputs['image']
        intrinsic_matrix = inputs['intrinsic_matrix']
        
        assert masks is None or image.shape[-2:] == masks.shape[-2:]
        scale_factor = np.divide(self.image_size, image.shape[-2:])
        image = utils.unvectorize(nn.functional.interpolate)(
            input=image,
            size=self.image_size,
            mode=self.image_interp_mode,
        )

        if masks is not None:
            if len(masks):
                masks = utils.unvectorize(nn.functional.interpolate)(
                    input=masks,
                    size=self.image_size,
                    mode=self.masks_interp_mode,
                )
            else:
                masks = masks.new_empty(*masks.shape[:-2], *self.image_size)

        if intrinsic_matrix is not None:
            intrinsic_matrix = intrinsic_matrix.new_tensor([
                [scale_factor[-1], 0.0, 0.0],
                [0.0, scale_factor[-2], 0.0],
                [0.0, 0.0, 1.0],
            ]) @ intrinsic_matrix
        
        inputs['intrinsic_matrix'] = intrinsic_matrix
        inputs['image'] = image
        inputs['masks'] = masks

        return inputs

class MaskAreaFilter(nn.Module):

    def __init__(self, min_mask_area, threshold=0.5):
        super().__init__()
        self.min_mask_area = min_mask_area
        self.threshold = threshold

    def forward(self, inputs):
        '''
        Filter out the masks smaller than one threshold, including the masks,boxes 3d and instances
        '''
        
        masks = inputs['masks']
        labels = inputs['labels']
        boxes_3d = inputs['boxes_3d']
        instance_ids = inputs['instance_ids']
        
        mask_areas = torch.sum(masks > self.threshold, dim=(-2, -1)) # should hvae somesthings
        instance_mask = mask_areas >= self.min_mask_area

        # all filtered outs
        masks = masks[instance_mask, ...]
        labels = labels[instance_mask, ...]
        boxes_3d = boxes_3d[instance_mask, ...]
        instance_ids = instance_ids[instance_mask, ...]
        
        # update
        inputs['masks'] = masks
        inputs['labels'] = labels
        inputs['boxes_3d'] = boxes_3d
        inputs['instance_ids'] = instance_ids 
        
        return inputs




class MaskRefiner(nn.Module):

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @utils.torch_function
    def make_polygon(self, mask):
        mask = mask > self.threshold
        mask = skimage.img_as_ubyte(mask)
        polygons, _ = cv.findContours(
            image=mask,
            mode=cv.RETR_EXTERNAL,
            method=cv.CHAIN_APPROX_SIMPLE,
        )
        
        polygon = max(polygons, key=cv.contourArea)
        polygon = polygon.squeeze(-2)
        return polygon

    @utils.torch_function
    def make_mask(self, polygon, image_size):
        mask = np.zeros(image_size, dtype=np.uint8)
        mask = cv.fillPoly(
            img=mask,
            pts=[polygon.astype(np.int64)],
            color=(1 << 8) - 1,
            lineType=cv.LINE_8,
        )
        mask = skimage.img_as_float32(mask)
        return mask

    @utils.vectorize
    def refine_mask(self, mask):
        
        polygon = self.make_polygon(mask)
        mask = self.make_mask(polygon, mask.shape[-2:])
        return mask

    def forward(self, inputs):
        
        masks = inputs['masks']
    
        if masks.numel():
            masks = self.refine_mask(masks)
        
        inputs['masks'] = masks
        
        return inputs


class BoxGenerator(nn.Module):

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, inputs):
        
        masks = inputs['masks']

        @utils.torch_function
        def where(*args, **kwargs):
            return np.where(*args, **kwargs)

        if len(masks):
            binary_masks = masks > self.threshold
            boxes_2d = torch.stack([
                torch.stack([
                    torch.stack(list(map(torch.min, where(binary_mask)))),
                    torch.stack(list(map(torch.max, where(binary_mask)))),
                ], dim=0)
                for binary_mask in binary_masks
            ], dim=0).flip(-1).to(masks)
        else:
            boxes_2d = masks.new_empty(*masks.shape[:-2], 2, 2)
            
        # 这个 BoxGenerator 类的作用是生成2D边界框（bounding boxes），
        # 用于描述给定掩码（masks）的最小包围矩形。它利用一个阈值将掩码二值化，然后计算这些二值掩码的边界框。
        inputs['masks'] = masks
        inputs['boxes_2d'] = boxes_2d #(N,2,2)
        return inputs


class BoxSizeFilter(nn.Module):
    '''
    这个 BoxSizeFilter 类用于根据2D边界框的大小过滤掩码及其对应的标签、3D边界框和实例ID。
    具体来说，它过滤掉那些边界框尺寸小于指定最小尺寸的实例。这有助于清除噪声数据或不感兴趣的小目标，从而提高后续处理的质量。
    
    '''
    def __init__(self, min_box_size):
        super().__init__()
        self.min_box_size = min_box_size

    def forward(self, inputs):
        
        boxes_2d = inputs['boxes_2d']
        masks = inputs['masks']
        labels = inputs['labels']
        boxes_3d = inputs['boxes_3d']
        instance_ids = inputs['instance_ids']
        

        box_sizes = torch.min(-torch.sub(*torch.unbind(boxes_2d, dim=-2)), dim=-1).values
        instance_mask = box_sizes >= self.min_box_size

        masks = masks[instance_mask, ...]
        labels = labels[instance_mask, ...]
        boxes_3d = boxes_3d[instance_mask, ...]
        boxes_2d = boxes_2d[instance_mask, ...]
        instance_ids = instance_ids[instance_mask, ...]

        inputs['masks'] = masks
        inputs['labels'] = labels
        inputs['boxes_3d'] = boxes_3d
        inputs['boxes_2d'] = boxes_2d
        inputs['instance_ids'] = instance_ids
        
        return inputs


class SoftRasterizer(nn.Module):

    def __init__(self, threshold=0.5, temperature=10.0):
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature

    @utils.torch_function
    def make_polygon(self, mask):
        mask = mask > self.threshold
        mask = skimage.img_as_ubyte(mask)
        polygons, _ = cv.findContours(
            image=mask,
            mode=cv.RETR_EXTERNAL,
            method=cv.CHAIN_APPROX_SIMPLE,
        )
        # 每个轮廓是一个形状为(N, 1, 2)的NumPy数组
        polygon = max(polygons, key=cv.contourArea)
        polygon = polygon.squeeze(-2)
        return polygon

    @utils.torch_function
    def make_binary_mask(self, polygon, image_size):
        mask = np.zeros(image_size, dtype=np.uint8)
        mask = cv.fillPoly(
            img=mask,
            pts=[polygon],
            color=(1 << 8) - 1,
            lineType=cv.LINE_8,
        )
        mask = skimage.img_as_bool(mask)
        return mask

    def make_distance_map(self, polygons, image_size):
        # [HW, 2]
        # # 创建一个网格，表示图像中每个像素的位置 [HW, 2]
        positions = list(reversed(torch.meshgrid(*map(torch.arange, image_size), indexing="ij")))
        positions = torch.stack(list(map(torch.flatten, positions)), dim=-1)
        
        # [B, N, 2]
        #获取每个多边形的顶点和下一个顶点 [B, N, 2]
        prev_vertices, next_vertices = polygons, torch.roll(polygons, shifts=-1, dims=-2)
        
        # [B, 1, N, 2]
        polysides = next_vertices.unsqueeze(-3) - prev_vertices.unsqueeze(-3)
        # [B, HW, N, 2]
        positions = positions.unsqueeze(-2) - prev_vertices.unsqueeze(-3)
        # [B, HW, N, 1]
        ratios = (
            torch.sum(polysides * positions, dim=-1, keepdim=True) /
            (torch.sum(polysides * polysides, dim=-1, keepdim=True) + 1e-6)
        )
        # [B, HW, N, 2]
        normals = positions - polysides * torch.clamp(ratios, 0.0, 1.0)
        # [B, HW, N]
        distances = torch.linalg.norm(normals, dim=-1)
        # [B, HW]
        distances = torch.min(distances, dim=-1).values
        # [B, H, W]
        distance_maps = distances.unflatten(-1, image_size)
        return distance_maps

    def forward(self, inputs):
        
        masks = inputs['masks']

        if len(masks):
            
            # get the polygons of such instances
            polygons = list(map(self.make_polygon, masks))

            binary_masks = torch.stack([
                self.make_binary_mask(polygon, masks.shape[-2:])
                for polygon in polygons
            ], dim=0)
            
  

            distance_maps = torch.stack([
                self.make_distance_map(polygon, masks.shape[-2:])
                for polygon in polygons
            ], dim=0)
            
            # 2d sdf map: inside the surface is the positive, outside the surface is negative.
            sdf_maps = torch.where(binary_masks, distance_maps, -distance_maps)
            
            
            soft_masks = torch.sigmoid(sdf_maps / self.temperature)

        else:
            soft_masks = torch.empty_like(masks)
            
        inputs['masks'] = masks
        inputs['hard_masks'] = masks
        inputs['soft_masks'] = soft_masks
        

        

        return inputs