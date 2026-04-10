import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.hand_bbox import detect_hand_bboxes
from yolh_pipeline.config_utils import load_pipeline_config, get_step_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    step_cfg = get_step_cfg(cfg, "step01")

    dino_model = step_cfg.get("dino_model", "IDEA-Research/grounding-dino-tiny")
    threshold = float(step_cfg.get("threshold", 0.2))
    max_jump = float(step_cfg.get("max_jump", 200.0))
    max_gap = int(step_cfg.get("max_gap", 10))

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
        detect_hand_bboxes(str(raw), str(out), dino_model, threshold, max_jump, max_gap)


if __name__ == "__main__":
    main()
