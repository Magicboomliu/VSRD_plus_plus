"""Visualisation helpers used by infer.py."""

import cv2
import numpy as np
import torch


def create_color_bar(height, width, color_map):
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.repeat(gradient[np.newaxis, :], height, axis=0)
    return cv2.applyColorMap(gradient, color_map)


def vis_heatmap(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Overlay a [0,1] heatmap on an RGB image; returns a BGR image."""
    heatmap = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = (image * 0.3 + colored_heatmap * 0.7).astype(np.uint8)
    h, w = image.shape[:2]
    color_bar = create_color_bar(50, w, cv2.COLORMAP_JET)
    return cv2.vconcat([overlay, color_bar])


def get_heatmap(info: torch.Tensor) -> torch.Tensor:
    """Extract per-pixel uncertainty from the delta-iter info tensor."""
    weight = info[:, :2].softmax(dim=1)
    heatmap = weight[:, 0]
    h, w = heatmap.shape[-2:]
    return heatmap.view(h, w)
