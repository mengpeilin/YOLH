#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ros2bag_to_npz import export_ros2_bag_to_npz
from yolh_pipeline.config_utils import load_pipeline_config, get_step_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    step_cfg = get_step_cfg(cfg, "step00")

    color_topic = step_cfg.get("color_topic", "/camera/camera/color/image_raw")
    depth_topic = step_cfg.get("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
    info_topic = step_cfg.get("info_topic", "/camera/camera/color/camera_info")
    max_frames = step_cfg.get("max_frames")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    bag_dirs = sorted(
        d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("rosbag")
    )
    if not bag_dirs:
        if input_dir.name.startswith("rosbag") and (input_dir / "metadata.yaml").exists():
            bag_dirs = [input_dir]
        else:
            print(f"No ROS2 bag directories found in {input_dir}")
            sys.exit(1)

    print(f"Found {len(bag_dirs)} bag(s)")
    for i, bag_dir in enumerate(bag_dirs):
        session_dir = output_dir / bag_dir.name
        session_dir.mkdir(parents=True, exist_ok=True)
        out = session_dir / "raw.npz"
        if out.exists():
            print(f"[{i+1}/{len(bag_dirs)}] {bag_dir.name}: skip (exists)")
            continue
        print(f"[{i+1}/{len(bag_dirs)}] {bag_dir.name}")
        export_ros2_bag_to_npz(
            str(bag_dir),
            str(out),
            color_topic,
            depth_topic,
            info_topic,
            max_frames,
        )


if __name__ == "__main__":
    main()
