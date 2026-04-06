"""
Step 03: Hand state estimation (HaMeR + ICP depth alignment).
Requires: phantom conda environment.

Usage:
    conda run -n phantom python voxelization_pipeline/03_hand_state.py --data-dir data/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.hand_state import estimate_hand_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--hand-side", default="right", choices=["left", "right"])
    parser.add_argument("--hamer-checkpoint", default=None,
                        help="Path to HaMeR checkpoint (default: auto-detect)")
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sessions = sorted(
        d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("rosbag")
    )

    for i, sess in enumerate(sessions):
        raw = sess / "raw.npz"
        bboxes = sess / "hand_bboxes.npz"
        masks = sess / "masks.npz"
        out = sess / "hand_state.npz"

        required = [raw, bboxes, masks]
        if not all(f.exists() for f in required):
            missing = [f.name for f in required if not f.exists()]
            print(f"[{i+1}/{len(sessions)}] {sess.name}: skip (missing {missing})")
            continue
        if out.exists():
            print(f"[{i+1}/{len(sessions)}] {sess.name}: skip (exists)")
            continue
        print(f"\n[{i+1}/{len(sessions)}] {sess.name}")
        estimate_hand_states(
            str(raw), str(bboxes), str(masks), str(out),
            hand_side=args.hand_side,
            hamer_checkpoint=args.hamer_checkpoint,
            rescale_factor=args.rescale_factor,
        )


if __name__ == "__main__":
    main()
