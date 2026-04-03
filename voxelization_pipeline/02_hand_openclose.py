"""
Step 02: Batch hand open/close detection.
Requires: handstate conda environment.

Usage:
    conda activate handstate
    python voxelization_pipeline/02_hand_openclose.py --data-dir data/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.hand_openclose import detect_hand_states_for_video

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-path", default=str(
        PROJECT_ROOT / "hand_object_detector/models/res101_handobj_100K/pascal_voc/"
                       "faster_rcnn_1_8_89999.pth"))
    parser.add_argument("--cfg-file", default=str(
        PROJECT_ROOT / "hand_object_detector/cfgs/res101.yml"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--thresh-hand", type=float, default=0.5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sessions = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("rosbag")
    ])

    for i, sess in enumerate(sessions):
        raw = sess / "raw.npz"
        out = sess / "hand_states.npy"
        if not raw.exists():
            continue
        if out.exists():
            print(f"[{i+1}/{len(sessions)}] {sess.name}: skip (exists)")
            continue
        print(f"\n[{i+1}/{len(sessions)}] {sess.name}")
        detect_hand_states_for_video(
            str(raw), str(out), args.model_path, args.cfg_file,
            args.device, args.thresh_hand)


if __name__ == "__main__":
    main()
