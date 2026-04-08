"""
Step 06: Voxelization + gripper insertion from gripper actions.
No special conda environment needed.

Usage:
    python voxelization_pipeline/06_voxelization.py --data-dir data/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.voxelization import build_training_episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--voxel-size", type=int, default=100)
    parser.add_argument("--coord-bounds", type=float, nargs=6, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sessions = sorted(
        d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("rosbag")
    )

    for i, sess in enumerate(sessions):
        raw = sess / "raw.npz"
        masks = sess / "masks.npz"
        action = sess / "gripper_action.npz"
        kf = sess / "keyframes.npy"
        out = sess / "episodes.npz"

        required = [raw, masks, action, kf]
        if not all(f.exists() for f in required):
            missing = [f.name for f in required if not f.exists()]
            print(f"[{i+1}/{len(sessions)}] {sess.name}: skip (missing {missing})")
            continue
        if out.exists():
            print(f"[{i+1}/{len(sessions)}] {sess.name}: skip (exists)")
            continue
        print(f"\n[{i+1}/{len(sessions)}] {sess.name}")
        build_training_episodes(
            str(raw), str(masks), str(action), str(kf),
            str(out), args.voxel_size, args.coord_bounds, gripper_offset=[0.05, 0, 0]
        )


if __name__ == "__main__":
    main()
