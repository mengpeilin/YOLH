"""
YOLH Data Processing Pipeline

Usage:
    python3 run_voxelization_pipeline.py \
        --input-dir /path/to/ros2bags \
        --output-dir data/ \
        --task-name pick_cup
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    cmd = [str(c) for c in cmd]
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    assert result.returncode == 0


def conda_run(env, script, *extra_args):
    return [
        "conda", "run", "-n", env, "--no-capture-output",
        "python3", str(script),
        *[str(a) for a in extra_args],
    ]


def main():
    parser = argparse.ArgumentParser(description="YOLH Voxelization Pipeline")
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing ROS2 bag folders")
    parser.add_argument("--output-dir", required=True,
                        help="Root output directory for all pipeline data")
    parser.add_argument("--task-name", required=True,
                        help="Task name used for the final dataset (e.g. pick_cup)")
    parser.add_argument("--sam2-env",    default="sam2",
                        help="Conda env for SAM2 (default: sam2)")
    parser.add_argument("--phantom-env", default="phantom",
                        help="Conda env for DINO / HaMeR / action (default: phantom)")
    parser.add_argument("--hand-side",   default="right", choices=["left", "right"],
                        help="Which hand to track (default: right)")
    args = parser.parse_args()

    pipeline_dir = Path(__file__).resolve().parent / "voxelization_pipeline"
    input_dir  = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # # ------------------------------------------------------------------
    # print("############# 00  ros2bag_process ###########")
    # run([sys.executable, str(pipeline_dir / "00_ros2bag_process.py"),
    #      "--input-dir", str(input_dir),
    #      "--output-dir", str(output_dir)])

    # # ------------------------------------------------------------------
    # print("############# 01  hand_bbox (DINO) ###########")
    # run(conda_run(args.phantom_env, pipeline_dir / "01_hand_bbox.py",
    #               "--data-dir", output_dir))

    # # ------------------------------------------------------------------
    # print("############# 02  mask_generation (SAM2) ###########")
    # run(conda_run(args.sam2_env, pipeline_dir / "02_mask_generation.py",
    #               "--data-dir", output_dir))

    # # ------------------------------------------------------------------
    # print("############# 03  hand_state (HaMeR + ICP) ###########")
    # run(conda_run(args.phantom_env, pipeline_dir / "03_hand_state.py",
    #               "--data-dir", output_dir,
    #               "--hand-side", args.hand_side))

    # # ------------------------------------------------------------------
    # print("############# 04  gripper_action (smoothing) ###########")
    # run(conda_run(args.phantom_env, pipeline_dir / "04_gripper_action.py",
    #               "--data-dir", output_dir))

    # # ------------------------------------------------------------------
    # print("############# 05  keyframe_detection ###########")
    # run([sys.executable, str(pipeline_dir / "05_keyframe_detection.py"),
    #      "--data-dir", str(output_dir)])

    # ------------------------------------------------------------------
    print("############# 06  voxelization ###########")
    run([sys.executable, pipeline_dir / "06_voxelization.py",
         "--data-dir", output_dir, "--coord-bounds", -0.5, -0.1, 0.0, 0.5, 0.5, 1.0])

    # ------------------------------------------------------------------
    print("############# 07  generate_dataset ###########")
    dataset_path = output_dir / "train_dataset.npz"
    run([sys.executable, pipeline_dir / "07_generate_dataset.py",
         "--data-dir",     output_dir,
         "--output-path",  dataset_path,
         "--task-name",    args.task_name])

    print("############# Pipeline complete ###########")
    print(f"Training dataset: {dataset_path}")


if __name__ == "__main__":
    main()
