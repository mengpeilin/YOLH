"""
Step 04: Gripper action computation + trajectory smoothing.
Requires: phantom conda environment (scikit-learn, scipy).

Usage:
    conda run -n phantom python voxelization_pipeline/04_gripper_action.py --data-dir data/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.gripper_action import compute_gripper_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--min-open-ratio", type=float, default=0.1,
                        help="Widths below max_width * this ratio are forced to 0 (closed)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sessions = sorted(
        d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("rosbag")
    )

    for i, sess in enumerate(sessions):
        hand_state = sess / "hand_state.npz"
        out = sess / "gripper_action.npz"
        if not hand_state.exists():
            print(f"[{i+1}/{len(sessions)}] {sess.name}: skip (no hand_state.npz)")
            continue
        if out.exists():
            print(f"[{i+1}/{len(sessions)}] {sess.name}: skip (exists)")
            continue
        print(f"\n[{i+1}/{len(sessions)}] {sess.name}")
        compute_gripper_actions(
            str(hand_state), str(out), args.min_open_ratio
        )


if __name__ == "__main__":
    main()
