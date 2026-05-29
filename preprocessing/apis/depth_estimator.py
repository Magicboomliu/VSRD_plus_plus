"""Depth-estimation API for preprocessing.

Currently supports WAFT-Stereo (see ``disparity_estimation/WAFT-Stereo/pipeline.py``).

Run inside the WAFT pixi environment (PyTorch >= 2.0):

    cd preprocessing/disparity_estimation/WAFT-Stereo
    PYTHONPATH=<project_root> pixi run python -c "
    from preprocessing.apis.depth_estimator import Load_Depth_Model
    model = Load_Depth_Model('WAFT-Stereo', device='cuda:0')
    "

Single-pair example:

    model = Load_Depth_Model('WAFT-Stereo', device='cuda:0')
    depth_m = model.infer(left_path, right_path)          # float32 (H, W), metres
    depth_m = model.infer_from_left(left_image_path)      # auto image_00 -> image_01

Batch pseudo-depth for KITTI360 layout:

    generate_pseudo_depth_sequence(
        sequence_dir='/path/to/data_2d_raw/2013_05_28_drive_0000_sync',
        dataset_root='/path/to/KITTI360_For_Upload',
    )
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Union

import numpy as np

ArrayLike = Union[np.ndarray, str, os.PathLike]

# Project + WAFT roots
_APIS_DIR = Path(__file__).resolve().parent
_PREPROCESSING_DIR = _APIS_DIR.parent
WAFT_STEREO_ROOT = _PREPROCESSING_DIR / "disparity_estimation" / "WAFT-Stereo"

# KITTI360 rectified camera defaults (perspective camera 0)
KITTI360_FX = 552.554261
KITTI360_BASELINE = 0.5942
KITTI360_DEPTH_MIN = 0.0
KITTI360_DEPTH_MAX = 80.0

WAFT_KITTI360_CONFIG = "configs/Real/kitti360.yaml"
WAFT_KITTI360_CKPT = "ckpts/Real/DAv2L-5.pth"

SUPPORTED_MODELS = {"WAFT-Stereo", "WAFT"}

DEFAULT_DEPTH_MODEL = "WAFT-Stereo"


def output_name_for_model(model_name: str) -> str:
    """e.g. ``WAFT-Stereo`` -> ``pseudo_depth_ssl_waft_stereo``"""
    slug = model_name.lower().replace("-", "_").replace(" ", "_")
    return f"pseudo_depth_ssl_{slug}"


DEFAULT_OUTPUT_NAME = output_name_for_model(DEFAULT_DEPTH_MODEL)


def _normalize_device(device: str) -> str:
    if device.startswith("cuda"):
        return "cuda"
    return device


def _resolve_waft_path(path: str | os.PathLike) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(WAFT_STEREO_ROOT / p)


@contextmanager
def _waft_import_context():
    """Temporarily expose WAFT-Stereo on ``sys.path`` and as cwd."""
    waft_root = str(WAFT_STEREO_ROOT)
    if not WAFT_STEREO_ROOT.is_dir():
        raise FileNotFoundError(f"WAFT-Stereo not found at {WAFT_STEREO_ROOT}")

    old_cwd = os.getcwd()
    old_path = sys.path.copy()
    sys.path.insert(0, waft_root)
    os.chdir(waft_root)
    try:
        yield waft_root
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def _load_waft_pipeline(
    config_file: str,
    ckpt_path: str,
    device: str,
    fx: float,
    baseline: float,
    depth_min: float,
    depth_max: float,
    auto_download: bool,
):
    with _waft_import_context():
        from pipeline import DepthEstimationPipeline

        return DepthEstimationPipeline(
            config_file=_resolve_waft_path(config_file),
            ckpt_path=_resolve_waft_path(ckpt_path),
            fx=fx,
            baseline=baseline,
            depth_min=depth_min,
            depth_max=depth_max,
            device=_normalize_device(device),
            auto_download=auto_download,
        )


class DepthModel:
    """Thin wrapper around WAFT ``DepthEstimationPipeline``."""

    def __init__(self, pipeline, device: str = "cuda") -> None:
        self.pipeline = pipeline
        self.device = device

    def infer(self, left: ArrayLike, right: ArrayLike) -> np.ndarray:
        """Return depth map in metres, shape ``(H, W)`` float32."""
        return self.pipeline.infer(left, right)

    def infer_with_disparity(
        self, left: ArrayLike, right: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(depth_m, disparity_px)``."""
        return self.pipeline.infer_with_disparity(left, right)

    def infer_from_left(self, left_path: str | os.PathLike) -> np.ndarray:
        """Infer depth from a KITTI360 left image path (image_00 -> image_01)."""
        right_path = kitti360_left_to_right(left_path)
        return self.infer(left_path, right_path)

    def set_camera(
        self,
        fx: float | None = None,
        baseline: float | None = None,
        depth_min: float | None = None,
        depth_max: float | None = None,
    ) -> None:
        self.pipeline.set_camera(fx=fx, baseline=baseline, depth_min=depth_min, depth_max=depth_max)


def Load_Depth_Model(
    model_name: str,
    device: str = "cuda:0",
    config_file: str = WAFT_KITTI360_CONFIG,
    ckpt_path: str = WAFT_KITTI360_CKPT,
    fx: float = KITTI360_FX,
    baseline: float = KITTI360_BASELINE,
    depth_min: float = KITTI360_DEPTH_MIN,
    depth_max: float = KITTI360_DEPTH_MAX,
    auto_download: bool = True,
) -> DepthModel:
    """Load a depth-estimation model.

    Parameters
    ----------
    model_name
        ``"WAFT-Stereo"`` or ``"WAFT"``.
    device
        ``"cuda:0"``, ``"cuda"``, or ``"cpu"``.
    config_file, ckpt_path
        Paths relative to ``WAFT-Stereo/`` unless absolute.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model_name={model_name!r}. "
            f"Supported: {sorted(SUPPORTED_MODELS)}"
        )

    pipeline = _load_waft_pipeline(
        config_file=config_file,
        ckpt_path=ckpt_path,
        device=device,
        fx=fx,
        baseline=baseline,
        depth_min=depth_min,
        depth_max=depth_max,
        auto_download=auto_download,
    )
    return DepthModel(pipeline, device=device)


def convert_disparity_to_depth(
    disparity: np.ndarray,
    fx: float = KITTI360_FX,
    baseline: float = KITTI360_BASELINE,
    depth_min: float = KITTI360_DEPTH_MIN,
    depth_max: float = KITTI360_DEPTH_MAX,
) -> np.ndarray:
    """Convert disparity (pixels) to depth (metres)."""
    disp = np.asarray(disparity, dtype=np.float32)
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = disp > 0
    depth[valid] = fx * baseline / disp[valid]
    return np.clip(depth, depth_min, depth_max)


def depth_to_uint16(depth_m: np.ndarray) -> np.ndarray:
    """Encode depth in metres as uint16 (value / 256 = metres), matching ``pseudo_depth_ssl``."""
    return (np.asarray(depth_m, dtype=np.float32) * 256).astype(np.uint16)


def save_depth_png(depth_m: np.ndarray, output_path: str | os.PathLike) -> None:
    """Save depth map as uint16 PNG compatible with ``read_depth()`` in Initial_Attributes."""
    import imageio.v2 as imageio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(output_path), depth_to_uint16(depth_m))


def kitti360_left_to_right(left_path: str | os.PathLike) -> str:
    """Map ``image_00/...`` left path to matching ``image_01/...`` right path."""
    left_path = str(left_path)
    if "image_00" not in left_path:
        raise ValueError(f"Expected image_00 in path, got: {left_path}")
    return left_path.replace("image_00", "image_01")


def left_path_to_depth_output(
    left_path: str | os.PathLike,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> str:
    """Derive ``pseudo_depth_ssl/...`` output path from a ``data_2d_raw/...`` left image."""
    return str(left_path).replace("data_2d_raw", output_name)


def generate_pseudo_depth_sequence(
    sequence_dir: str | os.PathLike,
    model: DepthModel | None = None,
    dataset_root: str | os.PathLike | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
    skip_existing: bool = True,
    device: str = "cuda:0",
) -> dict:
    """Generate pseudo depth PNGs for one KITTI360 sequence.

    Parameters
    ----------
    sequence_dir
        Path to ``.../data_2d_raw/2013_05_28_drive_XXXX_sync`` (or just the sync folder name
        if ``dataset_root`` is given).
    model
        Pre-loaded :class:`DepthModel`. Loaded automatically if ``None``.
    dataset_root
        Required when ``sequence_dir`` is only a folder name like ``2013_05_28_drive_0000_sync``.
    output_name
        Output directory name replacing ``data_2d_raw``
        (default ``pseudo_depth_ssl_<model>``, e.g. ``pseudo_depth_ssl_waft_stereo``).
    skip_existing
        Skip frames whose output PNG already exists.

    Returns
    -------
    dict with keys ``processed``, ``skipped``, ``missing_right``, ``total``.
    """
    sequence_dir = Path(sequence_dir)
    if not sequence_dir.is_absolute() and dataset_root is not None:
        sequence_dir = Path(dataset_root) / "data_2d_raw" / sequence_dir.name

    left_dir = sequence_dir / "image_00" / "data_rect"
    right_dir = sequence_dir / "image_01" / "data_rect"
    if not left_dir.is_dir():
        raise FileNotFoundError(f"Left image folder not found: {left_dir}")

    if model is None:
        model = Load_Depth_Model("WAFT-Stereo", device=device)

    stats = {"processed": 0, "skipped": 0, "missing_right": 0, "total": 0}

    left_files = sorted(left_dir.glob("*.png"))
    stats["total"] = len(left_files)

    # Pre-scan so we can show progress (inference is slow; no bar looks like a hang).
    todo = []
    for left_path in left_files:
        out_path = Path(left_path_to_depth_output(left_path, output_name=output_name))
        if skip_existing and out_path.is_file():
            stats["skipped"] += 1
            continue
        right_path = left_dir.parent.parent / "image_01" / "data_rect" / left_path.name
        if not right_path.is_file():
            stats["missing_right"] += 1
            continue
        todo.append((left_path, right_path, out_path))

    n_todo = len(todo)
    print(
        f"   frames={stats['total']}  to_process={n_todo}  "
        f"skip={stats['skipped']}  missing_right={stats['missing_right']}",
        flush=True,
    )
    if n_todo == 0:
        return stats

    from tqdm import tqdm

    for left_path, right_path, out_path in tqdm(todo, desc=f"   {sequence_dir.name}", unit="frame"):
        depth_m = model.infer(str(left_path), str(right_path))
        save_depth_png(depth_m, out_path)
        stats["processed"] += 1

    return stats


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Generate pseudo depth maps with WAFT-Stereo")
    parser.add_argument(
        "--dataset-root",
        default="/media/zliu/data12/dataset/KITTI/KITTI360_For_Upload",
        help="KITTI360 dataset root",
    )
    parser.add_argument(
        "--seq",
        default=None,
        help="Sequence id, e.g. 0000. If omitted, process all sequences under data_2d_raw.",
    )
    parser.add_argument("--model-name", default=DEFAULT_DEPTH_MODEL, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument(
        "--output-name",
        default=None,
        help=f"Output dir under dataset root (default: pseudo_depth_ssl_<model>, "
        f"currently {DEFAULT_OUTPUT_NAME!r})",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_name = args.output_name or output_name_for_model(args.model_name)
    data_2d_raw = Path(args.dataset_root) / "data_2d_raw"

    if args.seq is not None:
        seq_name = f"2013_05_28_drive_{args.seq}_sync"
        sequences = [data_2d_raw / seq_name]
    else:
        sequences = sorted(p for p in data_2d_raw.glob("2013_05_28_drive_*_sync") if p.is_dir())

    print(f"Loading {args.model_name} on {args.device} (may take ~10s) ...", flush=True)
    model = Load_Depth_Model(args.model_name, device=args.device)
    print(f"Model ready.", flush=True)
    print(f"Output directory: {output_name}/", flush=True)
    print(f"Processing {len(sequences)} sequence(s)", flush=True)
    if args.seq is None:
        print(
            "NOTE: running ALL sequences — this can take many hours. "
            "Tip: sh preprocessing/scripts/generate_pseudo_depth_waft.sh 0006",
            flush=True,
        )

    for seq_dir in sequences:
        print(f"── {seq_dir.name}", flush=True)
        stats = generate_pseudo_depth_sequence(
            sequence_dir=seq_dir,
            model=model,
            output_name=output_name,
            skip_existing=not args.no_skip_existing,
        )
        print(
            f"   done: processed={stats['processed']}  skipped={stats['skipped']}  "
            f"missing_right={stats['missing_right']}  total={stats['total']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
