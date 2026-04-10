import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.merge_episodes import merge_episodes
from yolh_pipeline.config_utils import load_pipeline_config, get_step_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--task-name", default="default_task")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    step_cfg = get_step_cfg(cfg, "step06")

    num_action = int(step_cfg.get("num_action", 20))
    trans_min = step_cfg.get("trans_min", [-0.5, -0.1, 0.0])
    trans_max = step_cfg.get("trans_max", [0.5, 0.5, 1.0])
    max_gripper_width = step_cfg.get("max_gripper_width")

    merge_episodes(
        data_dir=args.data_dir,
        output_path=args.output_path,
        num_action=num_action,
        trans_min=trans_min,
        trans_max=trans_max,
        max_gripper_width=max_gripper_width,
        task_name=args.task_name,
    )


if __name__ == "__main__":
    main()
