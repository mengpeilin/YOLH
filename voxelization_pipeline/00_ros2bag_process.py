#!/usr/bin/env python3
"""
Step 00: Batch convert all ROS2 bags to NPZ.
Run WITHOUT any conda environment (system Python with ROS2).

Usage:
    python voxelization_pipeline/00_ros2bag_process.py \
        --input-dir /path/to/ros2bags --output-dir data/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ros2bag_to_npz import export_ros2_bag_to_npz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--color-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--info-topic", default="/camera/camera/color/camera_info")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    bag_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not bag_dirs:
        if (input_dir / "metadata.yaml").exists():
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
            str(bag_dir), str(out),
            args.color_topic, args.depth_topic, args.info_topic,
            args.max_frames)


if __name__ == "__main__":
    main()
