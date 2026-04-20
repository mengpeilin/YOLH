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
        "conda",
        "run",
        "-n",
        env,
        "--no-capture-output",
        "python3",
        str(script),
        *[str(a) for a in extra_args],
    ]


def main():
    parser = argparse.ArgumentParser(description="YOLH Data Pipeline")
    parser.add_argument("--input-dir", required=True, help="Input dataset directory")
    parser.add_argument("--output-dir", required=True, help="Root output directory for all pipeline data")
    parser.add_argument("--task-name", required=True, help="Task name (e.g. pick_cup)")
    parser.add_argument(
        "--input-format",
        default="rosbag",
        choices=["rosbag", "benchmark"],
        help="Input format: rosbag (ROS2 bags) or benchmark (LeRobot-style human demos)",
    )
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Pipeline config file")
    parser.add_argument("--sam2-env", default="sam2", help="Conda env for SAM2")
    parser.add_argument("--phantom-env", default="phantom", help="Conda env for DINO / HaMeR / action")
    parser.add_argument("--hand-side", default="right", choices=["left", "right"], help="Hand to track")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    pipeline_dir = project_root / "yolh_pipeline"
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_format == "rosbag":
        print("############# 00  ros2bag_process ###########")
        step00_script = pipeline_dir / "00_ros2bag_process.py"
    else:
        print("############# 00  benchmark_process ###########")
        step00_script = pipeline_dir / "00_benchmark_process.py"

    run(
        conda_run(
            args.phantom_env, step00_script,
            "--input-dir",
            input_dir,
            "--output-dir",
            output_dir,
            "--config",
            config_path,
        )
    )

    print("############# 01  hand_bbox ###########")
    run(
        conda_run(
            args.phantom_env,
            pipeline_dir / "01_hand_bbox.py",
            "--data-dir",
            output_dir,
            "--config",
            config_path,
        )
    )

    print("############# 02  mask_generation ###########")
    run(
        conda_run(
            args.sam2_env,
            pipeline_dir / "02_mask_generation.py",
            "--data-dir",
            output_dir,
            "--config",
            config_path,
        )
    )

    print("############# 03  hand_state ###########")
    run(
        conda_run(
            args.phantom_env,
            pipeline_dir / "03_hand_state.py",
            "--data-dir",
            output_dir,
            "--config",
            config_path,
            "--hand-side",
            args.hand_side,
        )
    )

    print("############# 04  gripper_action ###########")
    run(
        conda_run(
            args.phantom_env,
            pipeline_dir / "04_gripper_action.py",
            "--data-dir",
            output_dir,
            "--config",
            config_path,
        )
    )

    print("############# 05  gripper_insertion ###########")
    run([
        sys.executable,
        pipeline_dir / "05_point_cloud.py",
        "--data-dir",
        output_dir,
        "--config",
        config_path,
    ])

    print("############# 06  generate_dataset ###########")
    dataset_path = output_dir / "train_dataset.npz"
    run([
        sys.executable,
        pipeline_dir / "06_generate_dataset.py",
        "--data-dir",
        output_dir,
        "--output-path",
        dataset_path,
        "--task-name",
        args.task_name,
        "--config",
        config_path,
    ])

    print("############# Pipeline complete ###########")
    print(f"Training dataset: {dataset_path}")


if __name__ == "__main__":
    main()