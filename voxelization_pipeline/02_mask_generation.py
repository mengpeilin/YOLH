"""
Step 02: Dual mask generation using SAM2.
Requires: sam2 conda environment.

Usage:
    conda run -n sam2 python voxelization_pipeline/02_mask_generation.py --data-dir data/

Outputs one masks.npz containing:
    - arm_hand_masks: user-initialized hand+arm mask for voxel-grid removal
    - hand_masks: bbox-initialized hand-only mask for hand pose refinement
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.mask_generation import generate_masks

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--sam2-checkpoint",
        default=str(PROJECT_ROOT / "sam2/checkpoints/sam2.1_hiera_large.pt"),
    )
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sessions = sorted(
        d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("rosbag")
    )

    for i, sess in enumerate(sessions):
        raw = sess / "raw.npz"
        bbox_file = sess / "hand_bboxes.npz"
        out = sess / "masks.npz"
        if not raw.exists():
            print(f"[{i+1}/{len(sessions)}] {sess.name}: skip (no raw.npz)")
            continue
        if out.exists():
            print(f"[{i+1}/{len(sessions)}] {sess.name}: skip (masks exist)")
            continue
        hand_bboxes_path = str(bbox_file) if bbox_file.exists() else None
        print(f"\n[{i+1}/{len(sessions)}] {sess.name}")
        generate_masks(
            str(raw),
            str(out),
            args.sam2_checkpoint,
            args.sam2_config,
            args.device,
            hand_bboxes_path=hand_bboxes_path,
        )


if __name__ == "__main__":
    main()
