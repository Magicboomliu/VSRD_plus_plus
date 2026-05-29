"""Programmatic depth-estimation pipeline for WAFT-Stereo."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Union

import imageio.v2 as imageio
import numpy as np
import torch
from peft import PeftModel

from algorithms.waft import WAFT
from bridgedepth.config import get_cfg

ArrayLike = Union[np.ndarray, str, os.PathLike]


HF_REPO = "MemorySlices/WAFT-Stereo"
CKPT_REGISTRY = {
    "ckpts/Real/DAv2L-5.pth": "Real/kitti/DAv2L-5.pth",
    "ckpts/SynLarge/DAv2L-5.pth": "SynLarge/DAv2L-5.pth",
    "ckpts/SynLarge/DAv2B-4.pth": "SynLarge/DAv2B-4.pth",
    "ckpts/SynLarge/DAv2S-4.pth": "SynLarge/DAv2S-4.pth",
    "ckpts/Real/middlebury/DAv2L-5.pth": "Real/middlebury/DAv2L-5.pth",
}


@dataclass
class CameraParams:
    fx: float
    baseline: float
    depth_min: float = 0.0
    depth_max: float = 80.0


def ensure_checkpoint(ckpt_path: str) -> None:
    if os.path.exists(ckpt_path):
        return

    hf_filename = CKPT_REGISTRY.get(ckpt_path)
    if hf_filename is None:
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "It is not in the auto-download registry. Please download it manually."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for auto-download. "
            "Install it with: pixi run pip install huggingface_hub"
        ) from exc

    print(f"Downloading checkpoint from HuggingFace: {HF_REPO}/{hf_filename}")
    cached = hf_hub_download(repo_id=HF_REPO, filename=hf_filename)
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    shutil.copy2(cached, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")


def load_image(image: ArrayLike) -> np.ndarray:
    if isinstance(image, (str, os.PathLike)):
        image = imageio.imread(image)

    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[..., :3]
    elif image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape {image.shape}")
    return image


class DepthEstimationPipeline:
    """WAFT-Stereo depth-estimation pipeline.

    Example
    -------
    >>> pipeline = DepthEstimationPipeline(
    ...     fx=552.554261,
    ...     baseline=0.5942,
    ...     depth_min=0.0,
    ...     depth_max=80.0,
    ... )
    >>> depth = pipeline.infer(left_img, right_img)  # (H, W), float32, metres
    """

    def __init__(
        self,
        config_file: str = "configs/Real/kitti.yaml",
        ckpt_path: str = "ckpts/Real/DAv2L-5.pth",
        fx: float = 552.554261,
        baseline: float = 0.5942,
        depth_min: float = 0.0,
        depth_max: float = 80.0,
        device: str = "cuda",
        auto_download: bool = True,
    ) -> None:
        if depth_max <= depth_min:
            raise ValueError(f"depth_max ({depth_max}) must be greater than depth_min ({depth_min})")

        self.config_file = config_file
        self.ckpt_path = ckpt_path
        self.device = torch.device(device)
        self.camera = CameraParams(
            fx=fx,
            baseline=baseline,
            depth_min=depth_min,
            depth_max=depth_max,
        )

        if auto_download:
            ensure_checkpoint(ckpt_path)

        self.model = self._build_model(config_file, ckpt_path)

    def set_camera(
        self,
        fx: float | None = None,
        baseline: float | None = None,
        depth_min: float | None = None,
        depth_max: float | None = None,
    ) -> None:
        if fx is not None:
            self.camera.fx = fx
        if baseline is not None:
            self.camera.baseline = baseline
        if depth_min is not None:
            self.camera.depth_min = depth_min
        if depth_max is not None:
            self.camera.depth_max = depth_max

        if self.camera.depth_max <= self.camera.depth_min:
            raise ValueError(
                f"depth_max ({self.camera.depth_max}) must be greater than "
                f"depth_min ({self.camera.depth_min})"
            )

    def infer(self, left: ArrayLike, right: ArrayLike) -> np.ndarray:
        """Run stereo inference and return a depth map in metres.

        Parameters
        ----------
        left, right
            RGB images as ``HxWx3`` numpy arrays or file paths.
            Any resolution is supported; no resizing is applied.

        Returns
        -------
        depth
            ``float32`` array of shape ``(H, W)``, clipped to
            ``[depth_min, depth_max]``.
        """
        left_img = load_image(left)
        right_img = load_image(right)

        if left_img.shape != right_img.shape:
            raise ValueError(
                f"Left/right image shapes must match: {left_img.shape} vs {right_img.shape}"
            )

        disp = self._predict_disparity(left_img, right_img)
        return self._disparity_to_depth(disp)

    def infer_with_disparity(
        self, left: ArrayLike, right: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return both depth (metres) and raw disparity (pixels)."""
        left_img = load_image(left)
        right_img = load_image(right)

        if left_img.shape != right_img.shape:
            raise ValueError(
                f"Left/right image shapes must match: {left_img.shape} vs {right_img.shape}"
            )

        disp = self._predict_disparity(left_img, right_img)
        depth = self._disparity_to_depth(disp)
        return depth, disp

    def _build_model(self, config_file: str, ckpt_path: str) -> WAFT:
        cfg = get_cfg()
        if config_file:
            cfg.merge_from_file(config_file)
        cfg.freeze()

        model = WAFT(cfg)
        model.eval()
        model = model.to(self.device)

        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        weights = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(weights, strict=False)

        for module in model.modules():
            if isinstance(module, PeftModel):
                module.merge_and_unload()

        return model

    def _predict_disparity(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        h, w = left_img.shape[:2]
        img1 = torch.as_tensor(left_img, device=self.device).float()[None].permute(0, 3, 1, 2)
        img2 = torch.as_tensor(right_img, device=self.device).float()[None].permute(0, 3, 1, 2)
        sample = {"img1": img1, "img2": img2}

        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                results = self.model(sample)

        return results["disp_pred"].detach().cpu().numpy().reshape(h, w)

    def _disparity_to_depth(self, disp: np.ndarray) -> np.ndarray:
        depth = np.zeros_like(disp, dtype=np.float32)
        valid = disp > 0
        depth[valid] = self.camera.fx * self.camera.baseline / disp[valid]
        return np.clip(depth, self.camera.depth_min, self.camera.depth_max)
