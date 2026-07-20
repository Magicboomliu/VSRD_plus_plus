"""Mask erosion transform for segmentation-robustness ablations."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import cv2 as cv

from vsrd_plus_plus import utils


class MaskEroder(nn.Module):
  """Erode segmentation masks to simulate imperfect mask quality."""

  def __init__(self, erode_ratio: float = 0.05, threshold: float = 0.5):
    super().__init__()
    self.erode_ratio = erode_ratio
    self.threshold = threshold

  def erode_single_mask(self, mask_np: np.ndarray) -> np.ndarray:
    h, w = mask_np.shape
    kernel_size = max(1, int(min(h, w) * self.erode_ratio))
    if kernel_size % 2 == 0:
      kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv.erode(mask_np, kernel, iterations=1)

  @utils.vectorize
  def erode_mask(self, mask: torch.Tensor) -> torch.Tensor:
    original_device = mask.device
    original_dtype = mask.dtype
    mask_np = (mask > self.threshold).cpu().numpy().astype(np.uint8)
    eroded_mask_np = self.erode_single_mask(mask_np)
    if isinstance(eroded_mask_np, torch.Tensor):
      eroded_mask_np = eroded_mask_np.cpu().numpy()
    eroded_mask = torch.from_numpy(eroded_mask_np).to(device=original_device)
    if original_dtype.is_floating_point:
      eroded_mask = eroded_mask.float()
    else:
      eroded_mask = eroded_mask.to(dtype=original_dtype)
    return eroded_mask

  def forward(self, inputs):
    masks = inputs["masks"]
    if masks.numel() and len(masks) > 0:
      masks = self.erode_mask(masks)
    inputs["masks"] = masks
    return inputs
