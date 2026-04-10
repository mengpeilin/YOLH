import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.gripper_action import compute_gripper_actions
from yolh_pipeline.config_utils import load_pipeline_config, get_step_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default=None)
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
        compute_gripper_actions(str(hand_state), str(out))


if __name__ == "__main__":
    main()
