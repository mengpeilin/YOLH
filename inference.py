#!/usr/bin/env python3
"""YOLH ZMQ inference server.

Receives observations (RGB-D + joint angles) from a remote control machine,
runs policy inference with ACT-style temporal ensemble, and sends back
per-step actions via ZMQ.

Robot-agnostic: all hardware-specific parameters (URDF, frame transforms,
workspace bounds, etc.) come from the config file.

Usage:
    python inference.py --ckpt policy.ckpt --config configs/camera_calibration.yaml \
                        --dataset-meta train_dataset.npz [--urdf path/to/robot.urdf]
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from lerobot_policy_yolh import YolhPolicy, YolhConfig
from scripts.arm_filter import ArmFilter
from policy.utils.config import load_config, load_action_norm_stats
from policy.utils.observation import build_observation_cloud
from policy.utils.action import project_action_to_base
from policy.utils.ensemble import EnsembleBuffer
from interface.zmq_interface import ZmqSender, ZmqReceiver, OBS_PORT, ACT_PORT


def load_policy(ckpt_path: str, cfg: dict, device: torch.device) -> YolhPolicy:
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
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if any(k.startswith("model.") for k in state.keys()):
        policy.load_state_dict(state, strict=False)
    else:
        policy.model.load_state_dict(state, strict=False)
    policy.to(device)
    policy.eval()
    return policy


def build_arm_filter(cfg: dict, urdf_override: str = None) -> ArmFilter | None:
    af_cfg = cfg.get("arm_filter", {})
    if not af_cfg.get("enabled", True):
        return None
    urdf_path = urdf_override or cfg.get("urdf_path")
    if urdf_path is None:
        print("[WARN] No urdf_path provided, arm filter disabled")
        return None
    return ArmFilter(
        urdf_path=urdf_path,
        cam_to_base=cfg["cam_to_base"],
        capsule_radii=af_cfg.get("capsule_radius"),
    )


def serve(policy, cfg, arm_filter, device, obs_port: int, act_port: int):
    obs_receiver = ZmqReceiver(obs_port)
    act_sender = ZmqSender(act_port)

    ensemble = EnsembleBuffer(mode=cfg.get("ensemble_mode", "act"))
    intrinsic = cfg["camera_intrinsic"]
    num_inf_step = cfg.get("num_inference_step", 20)
    cam_to_base = cfg["cam_to_base"]

    safe_eps = cfg.get("safe_eps", 0.002)
    safe_min = cfg.get("safe_workspace_min")
    safe_max = cfg.get("safe_workspace_max")
    do_clamp = safe_min is not None and safe_max is not None
    if do_clamp:
        safe_min = safe_min + safe_eps
        safe_max = safe_max - safe_eps

    # disable arm filter in observation builder if filter object unavailable
    if arm_filter is None:
        cfg.setdefault("arm_filter", {})["enabled"] = False

    print(f"Inference server ready  obs_port={obs_port}  act_port={act_port}")

    with torch.inference_mode():
        while True:
            obs = obs_receiver.recv()
            rgb = obs["rgb"]
            depth = obs["depth"]
            joint_angles = obs["joint_angles"]
            timestamp = obs["timestamp"]

            cloud = build_observation_cloud(
                rgb, depth, intrinsic, arm_filter, joint_angles, cfg,
            )

            if len(cloud) == 0:
                print(f"[t={timestamp}] Empty cloud, sending None x{num_inf_step}")
                for _ in range(num_inf_step):
                    act_sender.send({"action": None})
                continue

            cloud_t = torch.from_numpy(cloud).unsqueeze(0).to(device)
            batch = {"observation.point_cloud": cloud_t}
            action_chunk = policy.predict_action_chunk(batch)
            action_cam = action_chunk.squeeze(0).cpu().numpy()

            action_base = project_action_to_base(action_cam, cam_to_base)
            if do_clamp:
                action_base[..., :3] = np.clip(
                    action_base[..., :3], safe_min, safe_max,
                )

            ensemble.add_action(action_base, timestamp)

            for _ in range(num_inf_step):
                step = ensemble.get_action()
                act_sender.send({"action": step})


def main():
    parser = argparse.ArgumentParser(description="YOLH ZMQ inference server")
    parser.add_argument("--ckpt", required=True, help="Policy checkpoint path")
    parser.add_argument("--config", required=True, help="Deployment YAML config")
    parser.add_argument("--dataset-meta", default=None, help="Action norm stats (.npz)")
    parser.add_argument("--urdf", default=None, help="Robot URDF for arm filter")
    parser.add_argument("--obs-port", type=int, default=OBS_PORT)
    parser.add_argument("--act-port", type=int, default=ACT_PORT)
    args = parser.parse_args()

    cfg = load_config(args.config)

    meta_path = args.dataset_meta or cfg.get("dataset_meta")
    if meta_path is None:
        raise ValueError("Provide --dataset-meta or set dataset_meta in config")
    stats = load_action_norm_stats(str(meta_path))
    cfg.update({k: v for k, v in stats.items() if v is not None})
    print(f"Action stats loaded: max_gripper_width={float(cfg['max_gripper_width']):.4f}m")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.ckpt, cfg, device)
    print(f"Policy loaded from {args.ckpt} on {device}")

    arm_filter = build_arm_filter(cfg, urdf_override=args.urdf)

    try:
        serve(policy, cfg, arm_filter, device, args.obs_port, args.act_port)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
