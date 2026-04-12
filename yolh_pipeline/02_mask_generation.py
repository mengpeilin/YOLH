import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.mask_generation import generate_masks, _select_user_bbox
from yolh_pipeline.config_utils import load_pipeline_config, get_step_cfg

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    step_cfg = get_step_cfg(cfg, "step02")

    default_ckpt = step_cfg.get("sam2_checkpoint", "dependencies/sam2/checkpoints/sam2.1_hiera_large.pt")
    sam2_checkpoint = Path(default_ckpt)
    if not sam2_checkpoint.is_absolute():
        sam2_checkpoint = (PROJECT_ROOT / sam2_checkpoint).resolve()

    sam2_config = step_cfg.get("sam2_config", "configs/sam2.1/sam2.1_hiera_l.yaml")
    device = step_cfg.get("device", "auto")

    data_dir = Path(args.data_dir)
    sessions = sorted(
        d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("rosbag")
    )
    for i, sess in enumerate(sessions):
        raw = sess / "raw.npz"
        out = sess / "arm_bboxes.npz"
        if not raw.exists():
            print(f"[{i+1}/{len(sessions)}] {sess.name}: skip (no raw.npz)")
            continue
        if out.exists():
            print(f"[{i+1}/{len(sessions)}] {sess.name}: skip (arm bboxes exist)")
            continue
        arm_bbox = _select_user_bbox(str(raw))
        np.savez_compressed(out, **{arm_bbox: arm_bbox})
        print(f"[{i+1}/{len(sessions)}] {sess.name}: saved arm bbox to {out}")

    for i, sess in enumerate(sessions):
        raw = sess / "raw.npz"
        out = sess / "masks.npz"
        handbbox_file = sess / "hand_bboxes.npz"
        armbbox_file = sess / "arm_bboxes.npz"
        if not raw.exists():
            print(f"[{i+1}/{len(sessions)}] {sess.name}: skip (no raw.npz)")
            continue
        if out.exists():
            print(f"[{i+1}/{len(sessions)}] {sess.name}: skip (masks exist)")
            continue

        hand_bboxes_path = str(handbbox_file) if handbbox_file.exists() else None
        arm_bboxes_path = str(armbbox_file) if armbbox_file.exists() else None
        print(f"\n[{i+1}/{len(sessions)}] {sess.name}")
        generate_masks(
            str(raw),
            str(out),
            str(sam2_checkpoint),
            sam2_config,
            device,
            hand_bboxes_path=hand_bboxes_path,
            arm_bboxes_path=arm_bboxes_path
        )


if __name__ == "__main__":
    main()
