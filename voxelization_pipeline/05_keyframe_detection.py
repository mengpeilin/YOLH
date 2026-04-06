"""
Step 05: Keyframe detection from gripper actions.
No special conda environment needed.

Usage:
    python voxelization_pipeline/05_keyframe_detection.py --data-dir data/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.keyframe_detection import detect_keyframes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--stopping-delta", type=float, default=0.005)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sessions = sorted(
        d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("rosbag")
    )

    for i, sess in enumerate(sessions):
        action = sess / "gripper_action.npz"
        out = sess / "keyframes.npy"
        if not action.exists():
            print(f"[{i+1}/{len(sessions)}] {sess.name}: skip (no gripper_action.npz)")
            continue
        if out.exists():
            print(f"[{i+1}/{len(sessions)}] {sess.name}: skip (exists)")
            continue
        print(f"\n[{i+1}/{len(sessions)}] {sess.name}")
        detect_keyframes(str(action), str(out), args.stopping_delta)


if __name__ == "__main__":
    main()
