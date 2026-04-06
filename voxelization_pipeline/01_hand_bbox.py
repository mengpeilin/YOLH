"""
Step 01: Hand bounding-box detection with GroundingDINO.
Requires: phantom conda environment.

Usage:
    conda run -n phantom python voxelization_pipeline/01_hand_bbox.py --data-dir data/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.hand_bbox import detect_hand_bboxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sessions = sorted(
        d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("rosbag")
    )

    for i, sess in enumerate(sessions):
        raw = sess / "raw.npz"
        out = sess / "hand_bboxes.npz"
        if not raw.exists():
            print(f"[{i+1}/{len(sessions)}] {sess.name}: skip (no raw.npz)")
            continue
        if out.exists():
            print(f"[{i+1}/{len(sessions)}] {sess.name}: skip (exists)")
            continue
        print(f"\n[{i+1}/{len(sessions)}] {sess.name}")
        detect_hand_bboxes(str(raw), str(out), args.dino_model, args.threshold)


if __name__ == "__main__":
    main()
