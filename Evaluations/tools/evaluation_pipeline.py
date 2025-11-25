#!/usr/bin/env python
"""
Unified evaluation pipeline for VSRD++: Step1-Step4
This script combines all evaluation steps into one unified workflow.
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd, description, cwd=None):
    """Run a command and handle errors."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"✓ {description} completed successfully")
        if result.stdout:
            logger.debug(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with exit code {e.returncode}")
        if e.stderr:
            logger.error(f"Error: {e.stderr}")
        if e.stdout:
            logger.error(f"Output: {e.stdout}")
        return False


def step1_make_predictions(args):
    """Step1: Generate predictions and GT in JSON format."""
    logger.info("=" * 80)
    logger.info("Step 1: Generating predictions and GT in JSON format")
    logger.info("=" * 80)
    
    # Get the script directory
    script_dir = Path(__file__).parent
    predictions_dir = script_dir / "Predictions"
    
    # Step 1a: Generate predictions
    logger.info("Step 1a: Generating predictions...")
    cmd = [
        sys.executable, "make_predictions.py",
        "--root_dirname", args.root_dirname,
        "--ckpt_dirname", args.ckpt_dirname,
        "--ckpt_filename", args.ckpt_filename,
        "--num_workers", str(args.num_workers),
        "--dyanmic_root_filename", args.dynamic_dirname,
        "--input_model_type", args.input_model_type,
        "--saved_pseudo_folder_path", args.saved_pseudo_folder_path,
        "--split_dirname", args.split_dirname,
        "--class_names"] + args.class_names
    
    if not run_command(cmd, "Generate predictions", cwd=str(predictions_dir)):
        return False
    
    # Step 1b: Generate GT predictions
    logger.info("Step 1b: Generating GT predictions...")
    cmd = [
        sys.executable, "make_gt_predictions.py",
        "--root_dirname", args.root_dirname,
        "--ckpt_dirname", args.ckpt_dirname,
        "--ckpt_filename", args.ckpt_filename,
        "--num_workers", str(args.num_workers),
        "--dyanmic_root_filename", args.dynamic_dirname,
        "--input_model_type", args.input_model_type,
        "--split_dirname", args.split_dirname,
        "--class_names"] + args.class_names
    
    if not run_command(cmd, "Generate GT predictions", cwd=str(predictions_dir)):
        return False
    
    logger.info("✓ Step 1 completed: JSON predictions and GT generated")
    return True


def step2_convert_to_kitti_format(args):
    """Step2: Convert JSON to KITTI3D .txt format."""
    logger.info("=" * 80)
    logger.info("Step 2: Converting JSON to KITTI3D .txt format")
    logger.info("=" * 80)
    
    script_dir = Path(__file__).parent
    predictions_dir = script_dir / "Predictions"
    
    cmd = [
        sys.executable, "convert_prediction.py",
        "--root_dirname", args.root_dirname,
        "--ckpt_dirname", args.ckpt_dirname,
        "--num_workers", str(args.num_workers),
        "--json_foldername", args.json_foldername,
        "--output_labelname", args.output_labelname,
        "--class_names"] + args.class_names
    
    if not run_command(cmd, "Convert to KITTI3D format", cwd=str(predictions_dir)):
        return False
    
    logger.info("✓ Step 2 completed: KITTI3D .txt files generated")
    return True


def step3_dynamic_attribute(args):
    """Step3: Dynamic objects assignment using GT labels."""
    logger.info("=" * 80)
    logger.info("Step 3: Dynamic objects assignment")
    logger.info("=" * 80)
    
    script_dir = Path(__file__).parent
    dynamic_dir = script_dir / "Dyanmic_Attribute"
    
    cmd = [
        sys.executable, "get_gt_with_dynamic_label.py",
        "--root_dirname", args.root_dirname,
        "--ckpt_dirname", args.ckpt_dirname,
        "--num_workers", str(args.num_workers),
        "--json_foldername", args.json_foldername,
        "--output_labelname", args.output_labelname,
        "--dynamic_threshold", str(args.dynamic_threshold),
        "--class_names"] + args.class_names
    
    if not run_command(cmd, "Dynamic attribute assignment", cwd=str(dynamic_dir)):
        return False
    
    # Verify that Step3 output exists
    ckpt_basename = os.path.basename(args.ckpt_dirname.rstrip('/'))
    expected_gt_path = os.path.join(
        args.root_dirname,
        args.output_labelname,
        ckpt_basename
    )
    if not os.path.exists(expected_gt_path):
        logger.warning(f"Step3 output path not found: {expected_gt_path}")
        logger.warning("This might cause Step4 to fail. Please check Step3 execution.")
    else:
        logger.info(f"Step3 output verified at: {expected_gt_path}")
    
    logger.info("✓ Step 3 completed: Dynamic attributes assigned")
    return True


def step4_convert_to_kitti3d_structure(args):
    """Step4: Convert to KITTI3D dataset structure."""
    logger.info("=" * 80)
    logger.info("Step 4: Converting to KITTI3D dataset structure")
    logger.info("=" * 80)
    
    script_dir = Path(__file__).parent
    organize_dir = script_dir / "organize_to_kitti3d_dataset_structure"
    
    # Build prediction and GT label paths
    # Step2 now saves:
    # - Predictions to: {output_labelname}/{ckpt_basename}/predictions/{sequence}/...
    # - GT to: {output_labelname}/{ckpt_basename}/gt/{sequence}/...
    # Step3 saves GT with dynamic attributes to: {output_labelname}/{ckpt_basename}/{sequence}/...
    # (Step3 overwrites the directory structure, so we need to use Step3's output for GT)
    ckpt_basename = os.path.basename(args.ckpt_dirname.rstrip('/'))
    
    # Prediction label path: Step2 saves predictions to predictions subdirectory
    prediction_label_path = os.path.join(
        args.root_dirname,
        args.output_labelname,
        ckpt_basename,
        "predictions"
    )
    
    # GT label path: Step3 saves GT with dynamic attributes to the main directory
    gt_label_path = os.path.join(
        args.root_dirname,
        args.output_labelname,
        ckpt_basename
    )
    
    if not os.path.exists(gt_label_path):
        logger.error(f"GT label path does not exist: {gt_label_path}")
        logger.error("Please ensure Step3 completed successfully before running Step4.")
        return False
    
    # Validate paths exist
    if not os.path.exists(prediction_label_path):
        logger.error(f"Prediction label path does not exist: {prediction_label_path}")
        logger.error("Please ensure Step2 completed successfully before running Step4.")
        return False
    
    # Check if there are any label files in the GT path
    training_splits = args.training_split.split(",")
    if training_splits:
        # Check first training sequence
        sample_seq = f"2013_05_28_drive_{training_splits[0].zfill(4)}_sync"
        sample_gt_path = os.path.join(gt_label_path, sample_seq, "image_00", "data_rect")
        if os.path.exists(sample_gt_path):
            label_files = [f for f in os.listdir(sample_gt_path) if f.endswith('.txt')]
            if label_files:
                logger.info(f"Found {len(label_files)} label files in sample sequence: {sample_seq}")
            else:
                logger.warning(f"No .txt files found in: {sample_gt_path}")
        else:
            logger.warning(f"Sample GT path not found: {sample_gt_path}")
            logger.warning("This might indicate Step3 did not complete successfully.")
    
    cmd = [
        sys.executable, "conversion_kitt3d_structure.py",
        "--root_dirname", args.root_dirname,
        "--prediction_label_path", prediction_label_path,
        "--gt_label_path", gt_label_path,
        "--training_split", args.training_split,
        "--testing_split", args.testing_split,
        "--output_folder", args.output_folder
    ]
    
    if not run_command(cmd, "Convert to KITTI3D structure", cwd=str(organize_dir)):
        return False
    
    logger.info("✓ Step 4 completed: KITTI3D dataset structure created")
    return True


def main(args):
    """Main pipeline execution."""
    logger.info("=" * 80)
    logger.info("VSRD++ Evaluation Pipeline: Step 1-4")
    logger.info("=" * 80)
    
    # Validate paths
    if not os.path.exists(args.root_dirname):
        logger.error(f"Root directory does not exist: {args.root_dirname}")
        return False
    
    if not os.path.exists(args.ckpt_dirname):
        logger.error(f"Checkpoint directory does not exist: {args.ckpt_dirname}")
        return False
    
    # Execute steps
    steps = []
    
    if args.run_step1:
        steps.append(("Step 1", step1_make_predictions))
    if args.run_step2:
        steps.append(("Step 2", step2_convert_to_kitti_format))
    if args.run_step3:
        steps.append(("Step 3", step3_dynamic_attribute))
    if args.run_step4:
        steps.append(("Step 4", step4_convert_to_kitti3d_structure))
    
    if not steps:
        logger.warning("No steps selected. Use --run_step1, --run_step2, etc.")
        return False
    
    # Run selected steps
    for step_name, step_func in steps:
        if not step_func(args):
            logger.error(f"Pipeline failed at {step_name}")
            return False
    
    logger.info("=" * 80)
    logger.info("✓ All selected steps completed successfully!")
    logger.info("=" * 80)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VSRD++ Unified Evaluation Pipeline (Step 1-4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all steps
  python evaluation_pipeline.py --root_dirname /path/to/data --ckpt_dirname /path/to/ckpts --run_all
  
  # Run specific steps
  python evaluation_pipeline.py --root_dirname /path/to/data --ckpt_dirname /path/to/ckpts --run_step1 --run_step2
        """
    )
    
    # Common arguments
    parser.add_argument("--root_dirname", type=str, required=True,
                        help="Root directory of the dataset")
    parser.add_argument("--ckpt_dirname", type=str, required=True,
                        help="Checkpoint directory path")
    parser.add_argument("--ckpt_filename", type=str, default="step_2999.pt",
                        help="Checkpoint filename (default: step_2999.pt)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers (default: 4)")
    parser.add_argument("--class_names", type=str, nargs="+", default=["car"],
                        help="Class names (default: ['car'])")
    
    # Step 1 arguments
    parser.add_argument("--dynamic_dirname", type=str, default="",
                        help="Dynamic mask directory path")
    parser.add_argument("--input_model_type", type=str, default="velocity_with_init",
                        help="Model type: vanilla, velocity, mlp, velocity_with_init")
    parser.add_argument("--saved_pseudo_folder_path", type=str, default="predictions",
                        help="Folder name for saved predictions (default: predictions)")
    parser.add_argument("--split_dirname", type=str, default="R50-N16-M128-B16",
                        help="Split directory name (default: R50-N16-M128-B16)")
    
    # Step 2 arguments
    parser.add_argument("--json_foldername", type=str, default="predictions",
                        help="JSON folder name (default: predictions)")
    parser.add_argument("--output_labelname", type=str, default="perfect_prediction",
                        help="Output label folder name (default: perfect_prediction)")
    
    # Step 3 arguments
    parser.add_argument("--dynamic_threshold", type=float, default=0.01,
                        help="Dynamic threshold (default: 0.01)")
    
    # Step 4 arguments
    parser.add_argument("--training_split", type=str, default="00,02,03,04,05,06,07,09",
                        help="Training split sequences (comma-separated)")
    parser.add_argument("--testing_split", type=str, default="10",
                        help="Testing split sequences (comma-separated)")
    parser.add_argument("--output_folder", type=str, default="",
                        help="Output folder for KITTI3D structure (auto-generated if empty)")
    
    # Step selection
    parser.add_argument("--run_step1", action="store_true",
                        help="Run Step 1: Generate JSON predictions")
    parser.add_argument("--run_step2", action="store_true",
                        help="Run Step 2: Convert to KITTI3D format")
    parser.add_argument("--run_step3", action="store_true",
                        help="Run Step 3: Dynamic attribute assignment")
    parser.add_argument("--run_step4", action="store_true",
                        help="Run Step 4: Convert to KITTI3D structure")
    parser.add_argument("--run_all", action="store_true",
                        help="Run all steps (Step 1-4)")
    
    args = parser.parse_args()
    
    # Auto-generate output_folder if not provided
    if not args.output_folder and args.run_step4:
        ckpt_basename = os.path.basename(args.ckpt_dirname.rstrip('/'))
        args.output_folder = os.path.join(
            args.root_dirname,
            "KITTI3D_Dataset",
            ckpt_basename
        )
        logger.info(f"Auto-generated output_folder: {args.output_folder}")
    
    # Set all steps if --run_all is specified
    if args.run_all:
        args.run_step1 = True
        args.run_step2 = True
        args.run_step3 = True
        args.run_step4 = True
    
    # Run pipeline
    success = main(args)
    sys.exit(0 if success else 1)

