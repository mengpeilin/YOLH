import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.point_cloud import build_point_clouds
from yolh_pipeline.config_utils import load_pipeline_config, get_step_cfg

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    step_cfg = get_step_cfg(cfg, "step05")

    voxel_size = float(step_cfg.get("voxel_size", 0.005))
    workspace_min = step_cfg.get("workspace_min", [-0.5, -0.1, 0.0])
    workspace_max = step_cfg.get("workspace_max", [0.5, 0.5, 1.0])
    gripper_offset = step_cfg.get("gripper_offset", [0.05, 0.0, 0.0])
    gripper_type = step_cfg.get("gripper_type", "so101")
    gripper_urdf_path = step_cfg.get("gripper_urdf_path")
    if gripper_urdf_path is not None:
        gripper_urdf_path = Path(gripper_urdf_path)
        if not gripper_urdf_path.is_absolute():
            gripper_urdf_path = (PROJECT_ROOT / gripper_urdf_path).resolve()
        gripper_urdf_path = str(gripper_urdf_path)
    gripper_tcp_local = step_cfg.get("gripper_tcp_local")
    gripper_num_points = int(step_cfg.get("gripper_num_points", 1200))
    tip_sample_points = int(step_cfg.get("tip_sample_points", 20000))
    contact_offset_z = float(step_cfg.get("contact_offset_z", 0.05))
    z_band = float(step_cfg.get("z_band", 0.005))

    data_dir = Path(args.data_dir)
    sessions = sorted(
        d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("rosbag")
    )

    for i, sess in enumerate(sessions):
        raw = sess / "raw.npz"
        masks = sess / "masks.npz"
        action = sess / "gripper_action.npz"
        out = sess / "episodes.npz"

        required = [raw, masks, action]
        if not all(f.exists() for f in required):
            missing = [f.name for f in required if not f.exists()]
            print(f"[{i+1}/{len(sessions)}] {sess.name}: skip (missing {missing})")
            continue
        if out.exists():
            print(f"[{i+1}/{len(sessions)}] {sess.name}: skip (exists)")
            continue
        print(f"\n[{i+1}/{len(sessions)}] {sess.name}")
        build_point_clouds(
            str(raw),
            str(masks),
            str(action),
            str(out),
            voxel_size=voxel_size,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            gripper_offset=gripper_offset,
            gripper_type=gripper_type,
            gripper_urdf_path=gripper_urdf_path,
            gripper_tcp_local=gripper_tcp_local,
            gripper_num_points=gripper_num_points,
            tip_sample_points=tip_sample_points,
            contact_offset_z=contact_offset_z,
            z_band=z_band,
        )


if __name__ == "__main__":
    main()
