"""Generate ``dynamic_mask.txt`` from GT bbox velocity."""

from __future__ import annotations

from dataclasses import dataclass
import os

from tqdm import tqdm

from preprocessing.Dynamic_Labels.format import format_dynamic_line, parse_filenames_line
from preprocessing.Dynamic_Labels.multi_inputs import build_multi_from_annotations
from preprocessing.Initial_Attributes.gt_attributes import (
    DEFAULT_DYNAMIC_VELOCITY_THRESHOLD,
    infer_dynamic_mask_from_gt_velocity,
    compute_gt_velocity,
)
from vsrd_plus_plus.datasets.kitti360_dataset import KITTI360Dataset


@dataclass
class DynamicLabelsConfig:
    """Settings for offline dynamic-label generation."""

    dataset_root: str
    filenames_path: str
    output_path: str
    class_names: list[str] | None = None
    num_source_frames: int = 16
    velocity_threshold: float = DEFAULT_DYNAMIC_VELOCITY_THRESHOLD
    show_progress: bool = True


@dataclass
class DynamicLabelsResult:
    """Summary statistics from a generation run."""

    num_frames: int
    num_instances: int
    num_dynamic: int
    num_skipped: int
    output_path: str


class DynamicLabelsPipeline:
    """Scan ``sampled_image_filenames.txt`` and write per-frame dynamic flags."""

    def __init__(self, config: DynamicLabelsConfig):
        self.config = config
        class_names = config.class_names or ["car"]
        self._ds = KITTI360Dataset(
            filenames=[config.filenames_path],
            class_names=class_names,
            dataset_root=config.dataset_root,
        )

    def run(self) -> DynamicLabelsResult:
        cfg = self.config
        os.makedirs(os.path.dirname(os.path.abspath(cfg.output_path)), exist_ok=True)

        with open(cfg.filenames_path) as f:
            lines = [line.strip() for line in f if line.strip()]

        iterator = tqdm(lines, desc="dynamic labels") if cfg.show_progress else lines
        output_lines: list[str] = []
        num_instances = 0
        num_dynamic = 0
        num_skipped = 0

        for line in iterator:
            instance_ids, target_img, rel_img, source_rels = parse_filenames_line(
                line, cfg.dataset_root,
            )
            multi = build_multi_from_annotations(
                self._ds,
                target_img,
                source_rels,
                cfg.num_source_frames,
            )
            if multi is None:
                labels = [False] * len(instance_ids)
                num_skipped += 1
            else:
                gt_velocity = compute_gt_velocity(multi)
                pred = infer_dynamic_mask_from_gt_velocity(
                    gt_velocity,
                    threshold=cfg.velocity_threshold,
                )
                ann_ids = multi[0]["instance_ids"][0].tolist()
                pred_by_id = dict(zip(ann_ids, pred))
                labels = [pred_by_id.get(iid, False) for iid in instance_ids]

            num_instances += len(labels)
            num_dynamic += sum(labels)
            output_lines.append(format_dynamic_line(instance_ids, rel_img, labels))

        with open(cfg.output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
            if output_lines:
                f.write("\n")

        return DynamicLabelsResult(
            num_frames=len(output_lines),
            num_instances=num_instances,
            num_dynamic=num_dynamic,
            num_skipped=num_skipped,
            output_path=cfg.output_path,
        )


def generate_dynamic_labels(
    *,
    dataset_root: str,
    filenames_path: str,
    output_path: str,
    class_names: list[str] | None = None,
    num_source_frames: int = 16,
    velocity_threshold: float = DEFAULT_DYNAMIC_VELOCITY_THRESHOLD,
    show_progress: bool = True,
) -> DynamicLabelsResult:
    """Generate ``dynamic_mask.txt`` from 3D bbox GT velocity."""
    config = DynamicLabelsConfig(
        dataset_root=dataset_root,
        filenames_path=filenames_path,
        output_path=output_path,
        class_names=class_names,
        num_source_frames=num_source_frames,
        velocity_threshold=velocity_threshold,
        show_progress=show_progress,
    )
    return DynamicLabelsPipeline(config).run()


if __name__ == "__main__":
    import argparse

    from trainer.configs import load_config

    parser = argparse.ArgumentParser(description="Generate dynamic_mask.txt from GT bbox velocity.")
    parser.add_argument("--config", type=str, default="", help="Trainer config name, e.g. sequence_00")
    parser.add_argument("--dataset-root", type=str, default="")
    parser.add_argument("--filenames", type=str, default="", help="sampled_image_filenames.txt")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output dynamic_mask.txt (default: dynamic_attributes_est_gt/<sync>/dynamic_mask.txt)",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_DYNAMIC_VELOCITY_THRESHOLD)
    parser.add_argument("--num-source-frames", type=int, default=0)
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        dataset_root = cfg.TRAIN.DATASET.ROOT
        filenames_path = cfg.TRAIN.DATASET.FILENAMES[0]
        class_names = list(cfg.TRAIN.DATASET.CLASS_NAMES)
        num_source_frames = cfg.TRAIN.DATASET.NUM_SOURCE_FRAMES
        if args.output:
            output_path = args.output
        else:
            sync_name = os.path.basename(os.path.dirname(filenames_path))
            output_path = os.path.join(
                dataset_root,
                "dynamic_attributes_est_gt",
                sync_name,
                "dynamic_mask.txt",
            )
    else:
        if not args.dataset_root or not args.filenames or not args.output:
            parser.error("Provide --config or (--dataset-root, --filenames, --output)")
        dataset_root = args.dataset_root
        filenames_path = args.filenames
        output_path = args.output
        class_names = ["car"]
        num_source_frames = args.num_source_frames or 16

    if args.dataset_root:
        dataset_root = args.dataset_root
    if args.filenames:
        filenames_path = args.filenames
    if args.output:
        output_path = args.output
    if args.num_source_frames:
        num_source_frames = args.num_source_frames

    result = generate_dynamic_labels(
        dataset_root=dataset_root,
        filenames_path=filenames_path,
        output_path=output_path,
        class_names=class_names,
        num_source_frames=num_source_frames,
        velocity_threshold=args.threshold,
    )
    print(
        f"Wrote {result.output_path}: "
        f"{result.num_frames} frames, {result.num_instances} instances, "
        f"{result.num_dynamic} dynamic ({100*result.num_dynamic/max(result.num_instances,1):.2f}%), "
        f"{result.num_skipped} frames fallback-all-static"
    )
