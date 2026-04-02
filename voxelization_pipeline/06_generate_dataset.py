"""
Step 06: Merge all session episodes into a single training dataset.
No special conda environment needed.

Usage:
    python voxelization_pipeline/06_gripper_insertion.py \
        --data-dir data/ --output-path data/train_dataset.npz --task-name pick_cup
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.merge_episodes import merge_episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--task-name", default="default_task")
    args = parser.parse_args()

    merge_episodes(args.data_dir, args.output_path, args.task_name)


if __name__ == "__main__":
    main()
