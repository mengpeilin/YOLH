import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.hand_state import estimate_hand_states
from yolh_pipeline.config_utils import load_pipeline_config, get_step_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--hand-side", default=None, choices=["left", "right"])
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    step_cfg = get_step_cfg(cfg, "step03")

    hand_side = args.hand_side or step_cfg.get("hand_side", "right")
    hamer_checkpoint = step_cfg.get("hamer_checkpoint")
    rescale_factor = float(step_cfg.get("rescale_factor", 2.0))

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
            str(raw),
            str(bboxes),
            str(masks),
            str(out),
            hand_side=hand_side,
            hamer_checkpoint=hamer_checkpoint,
            rescale_factor=rescale_factor,
        )


if __name__ == "__main__":
    main()
