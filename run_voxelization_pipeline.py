"""
YOLH Data Processing Pipeline

Usage:
    # Step 00 requires ROS2; run this script from a shell where ROS2 is sourced
    # and no conda environment is active.
    python3 run_voxelization_pipeline.py \
        --input-dir /path/to/ros2bags \
        --output-dir data/ \
        --task-name pick_cup

Steps that require a specific conda environment use `conda run -n <env>`.
Override the environment names with --sam2-env, --handstate-env, --wilor-env.
All steps are idempotent; completed steps are skipped automatically.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print(" ".join(str(c) for c in cmd))
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
    parser.add_argument("--sam2-env",      default="sam2",
                        help="Conda env name for SAM2 (default: sam2)")
    parser.add_argument("--handstate-env", default="handstate",
                        help="Conda env name for hand_object_detector (default: handstate)")
    parser.add_argument("--wilor-env",     default="wilor",
                        help="Conda env name for WiLoR (default: wilor)")
    args = parser.parse_args()

    pipeline_dir = Path(__file__).resolve().parent / "voxelization_pipeline"
    input_dir  = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    print("############# 00_ros2bag_process ###########")
    # Must run with the Python that has ROS2 / rosbag2_py available.
    # Do NOT activate any conda environment before running this script.
    script_path = pipeline_dir / "00_ros2bag_process.py"
    assert script_path.is_file()
    run([sys.executable, str(script_path),
         "--input-dir", str(input_dir),
         "--output-dir", str(output_dir)])

    # ------------------------------------------------------------------
    print("############# 01_mask_generation ###########")
    # Interactive: for each session you will draw a bounding box on the
    # first frame to initialise SAM2 tracking.
    script_path = pipeline_dir / "01_mask_generation.py"
    assert script_path.is_file()
    run(conda_run(args.sam2_env, script_path,
                  "--data-dir", output_dir))

    # ------------------------------------------------------------------
    print("############# 02_hand_openclose ###########")
    script_path = pipeline_dir / "02_hand_openclose.py"
    assert script_path.is_file()
    run(conda_run(args.handstate_env, script_path,
                  "--data-dir", output_dir))

    # ------------------------------------------------------------------
    print("############# 03_hand_pose ###########")
    script_path = pipeline_dir / "03_hand_pose.py"
    assert script_path.is_file()
    run(conda_run(args.wilor_env, script_path,
                  "--data-dir", output_dir))

    # ------------------------------------------------------------------
    print("############# 04_keyframe_detection ###########")
    script_path = pipeline_dir / "04_keyframe_detection.py"
    assert script_path.is_file()
    run([sys.executable, str(script_path),
         "--data-dir", str(output_dir)])

    # ------------------------------------------------------------------
    print("############# 05_voxelization ###########")
    script_path = pipeline_dir / "05_voxelization.py"
    assert script_path.is_file()
    run([sys.executable, str(script_path),
         "--data-dir", str(output_dir)])

    # ------------------------------------------------------------------
    # print("############# 06_generate_dataset ###########")
    # script_path = pipeline_dir / "06_generate_dataset.py"
    # assert script_path.is_file()
    # dataset_path = output_dir / "train_dataset.npz"
    # run([sys.executable, str(script_path),
    #      "--data-dir",     str(output_dir),
    #      "--output-path",  str(dataset_path),
    #      "--task-name",    args.task_name])

    # ------------------------------------------------------------------
    # print("############# Pipeline complete ###########")
    # print(f"Training dataset: {dataset_path}")


if __name__ == "__main__":
    main()
