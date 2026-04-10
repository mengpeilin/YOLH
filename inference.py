#!/usr/bin/env python3
"""
YOLH Inference Script
=====================

Captures RGB-D from RealSense, removes arm from point cloud via FK-based
capsule filter, runs YOLH policy, denormalises actions, projects camera-frame
predictions to base frame, and sends them to the robot via IK.

Usage:
    python inference.py --ckpt logs/my_task/policy_last.ckpt \
                        --config configs/camera_calibration.yaml \
                        --dataset-meta train_dataset.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from lerobot_policy_yolh import YolhPolicy, YolhConfig
from scripts.arm_filter import ArmFilter

from policy.utils.config import load_config, load_action_norm_stats
from policy.utils.observation import build_observation_cloud
from policy.utils.action import project_action_to_base, discretize_rotation
from policy.utils.ensemble import EnsembleBuffer
from interface.so101_interface import RealSenseSO101Interface

PROJECT_ROOT = Path(__file__).resolve().parent
SO101_URDF = str(PROJECT_ROOT / "URDF/SO-ARM100/Simulation/SO101/so101_new_calib.urdf")


def run_inference(
    ckpt_path: str,
    cfg: dict,
    robot,
):
    """Main inference loop.

    Parameters
    ----------
    ckpt_path : str
        Path to a YOLH policy checkpoint.
    cfg : dict
        Merged deployment config (camera_calibration.yaml + dataset stats).
    robot : RobotInterface
        Concrete robot interface (e.g. ``RealSenseSO101Interface``).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mcfg = cfg["model"]
    yolh_cfg = YolhConfig(
        chunk_size=cfg["num_action"],
        n_action_steps=cfg.get("num_inference_step", cfg["num_action"]),
        input_dim=6,
        obs_feature_dim=mcfg["obs_feature_dim"],
        action_dim=10,
        hidden_dim=mcfg["hidden_dim"],
        nheads=mcfg["nheads"],
        num_encoder_layers=mcfg["num_encoder_layers"],
        num_decoder_layers=mcfg["num_decoder_layers"],
        dropout=mcfg["dropout"],
        voxel_size=cfg["voxel_size"],
        trans_min=cfg["trans_min"].tolist(),
        trans_max=cfg["trans_max"].tolist(),
        max_gripper_width=float(cfg["max_gripper_width"]),
    )
    policy = YolhPolicy(yolh_cfg)

    # Load checkpoint
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if any(k.startswith("model.") for k in state.keys()):
        policy.load_state_dict(state, strict=False)
    else:
        policy.model.load_state_dict(state, strict=False)
    policy.to(device)
    policy.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # ── Arm filter ──
    capsule_radii = cfg.get("arm_filter", {}).get("capsule_radius", None)
    arm_filter = ArmFilter(
        urdf_path=SO101_URDF,
        cam_to_base=cfg["cam_to_base"],
        capsule_radii=capsule_radii,
    )

    # ── Ensemble ──
    ensemble = EnsembleBuffer(mode=cfg.get("ensemble_mode", "act"))

    intrinsic = cfg["camera_intrinsic"]
    num_inf_step = cfg.get("num_inference_step", 20)
    max_steps = cfg.get("max_steps", 300)

    use_disc_rot = cfg.get("discretize_rotation", False)
    rot_step = cfg.get("rot_step_size", np.pi / 16)

    last_rot6d = None

    print("Starting inference loop ...")
    with torch.inference_mode():
        for t in range(max_steps):
            if t % num_inf_step == 0:
                # ── Capture & preprocess ──
                rgb, depth = robot.get_observation()
                joint_angles = robot.get_joint_angles()

                cloud = build_observation_cloud(
                    rgb, depth, intrinsic, arm_filter, joint_angles, cfg,
                )

                if len(cloud) == 0:
                    print(f"  [t={t}] Empty point cloud, skipping")
                    continue

                # ── Forward pass ──
                cloud_tensor = torch.from_numpy(cloud).unsqueeze(0).to(device)
                batch = {"observation.point_cloud": cloud_tensor}
                action_chunk = policy.predict_action_chunk(batch)
                action_cam = action_chunk.squeeze(0).cpu().numpy()

                # ── Camera → base frame ──
                action_base = project_action_to_base(action_cam, cfg["cam_to_base"])

                # ── Safety clamp (base frame position) ──
                safe_min = cfg["safe_workspace_min"] + cfg["safe_eps"]
                safe_max = cfg["safe_workspace_max"] - cfg["safe_eps"]
                action_base[..., :3] = np.clip(
                    action_base[..., :3], safe_min, safe_max,
                )

                # ── Add to ensemble ──
                ensemble.add_action(action_base, t)

            # ── Get ensembled step action ──
            step = ensemble.get_action()
            if step is None:
                continue

            step_pos = step[:3]
            step_rot6d = step[3:9]
            step_width = float(step[9])

            # ── Send to robot ──
            if use_disc_rot and last_rot6d is not None:
                rot_steps = discretize_rotation(last_rot6d, step_rot6d, rot_step)
                for rot in rot_steps:
                    robot.send_action(step_pos, rot, step_width)
            else:
                robot.send_action(step_pos, step_rot6d, step_width)

            last_rot6d = step_rot6d

    robot.stop()
    print("Inference complete.")


def main():
    parser = argparse.ArgumentParser(description="YOLH inference with arm self-filter")
    parser.add_argument("--ckpt", required=True, help="Path to policy checkpoint")
    parser.add_argument("--config", required=True, help="Path to camera_calibration.yaml")
    parser.add_argument("--dataset-meta", default=None)
    parser.add_argument("--serial-port", default="/dev/ttyACM0", help="Dynamixel serial port")
    args = parser.parse_args()

    cfg = load_config(args.config)

    dataset_meta_path = args.dataset_meta or cfg.get("dataset_meta")
    if dataset_meta_path is not None:
        stats = load_action_norm_stats(str(dataset_meta_path))
        cfg["trans_min"] = stats["trans_min"]
        cfg["trans_max"] = stats["trans_max"]
        cfg["max_gripper_width"] = stats["max_gripper_width"]
        if "num_action" not in cfg:
            cfg["num_action"] = stats["num_action"]
        print(
            f"Loaded action normalization from: {dataset_meta_path} "
            f"(max_gripper_width={float(cfg['max_gripper_width']):.4f}m)"
        )
    else:
        raise KeyError("Please pass --dataset-meta train_dataset.npz.")

    robot = RealSenseSO101Interface(cfg, serial_port=args.serial_port)

    run_inference(args.ckpt, cfg, robot)


if __name__ == "__main__":
    main()
