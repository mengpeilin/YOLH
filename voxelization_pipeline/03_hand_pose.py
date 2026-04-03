"""
Step 03: Batch hand pose extraction.
Requires: wilor conda environment.

Usage:
    conda activate wilor
    python voxelization_pipeline/03_hand_pose.py --data-dir data/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.hand_pose import extract_hand_poses

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WILOR_ROOT = PROJECT_ROOT / "WiLoR"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint-path",
                        default=str(WILOR_ROOT / "pretrained_models/wilor_final.ckpt"))
    parser.add_argument("--cfg-path",
                        default=str(WILOR_ROOT / "pretrained_models/model_config.yaml"))
    parser.add_argument("--detector-path",
                        default=str(WILOR_ROOT / "pretrained_models/detector.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--fast", action="store_true", default=False)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sessions = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("rosbag")
    ])

    for i, sess in enumerate(sessions):
        raw = sess / "raw.npz"
        out = sess / "hand_pose.npz"
        if not raw.exists():
            continue
        if out.exists():
            print(f"[{i+1}/{len(sessions)}] {sess.name}: skip (exists)")
            continue
        print(f"\n[{i+1}/{len(sessions)}] {sess.name}")
        extract_hand_poses(
            str(raw), str(out),
            args.checkpoint_path, args.cfg_path, args.detector_path,
            args.device, args.conf, args.rescale_factor, args.fast)


if __name__ == "__main__":
    main()
