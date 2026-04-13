#!/usr/bin/env python3
"""Debug inference server that replays FK poses instead of policy output."""

import argparse
import os
import threading
import time
from typing import Optional

import numpy as np
import torch

from policy.utils.arm_filter import ArmFilter
from policy.utils.ensemble import EnsembleBuffer
from policy.utils.inference_state import InferenceState
from policy.utils.transformation import matrix_to_rot6d
from policy.utils.config import load_config, load_action_norm_stats
from policy.utils.observation import build_observation_cloud
from interface.ik_solver import SO101IKSolver
from interface.zmq_interface import ZmqSender, ZmqReceiver, OBS_PORT, ACT_PORT


def build_arm_filter(cfg: dict, urdf_override: str = None) -> Optional[ArmFilter]:
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


def _run_inference_loop(
    policy,
    cfg,
    arm_filter,
    device,
    obs_receiver,
    chunk_buffer: EnsembleBuffer,
    state: InferenceState,
    inference_hz: float,
    skip_chunk_steps: int,
):
    interval = 1.0 / max(inference_hz, 1e-6)
    next_run_time = time.monotonic()
    intrinsic = cfg["camera_intrinsic"]
    cam_to_base = cfg["cam_to_base"]

    safe_eps = cfg.get("safe_eps", 0.002)
    safe_min = cfg.get("safe_workspace_min")
    safe_max = cfg.get("safe_workspace_max")
    do_clamp = safe_min is not None and safe_max is not None
    if do_clamp:
        safe_min = safe_min + safe_eps
        safe_max = safe_max - safe_eps

    if arm_filter is None:
        cfg.setdefault("arm_filter", {})["enabled"] = False

    fk_solver = None
    fk_solver = SO101IKSolver(str(urdf_path))
    num_action = int(cfg.get("num_action", 1))

    cloud_dir = os.path.expanduser("~/yolh_data/cloud")
    os.makedirs(cloud_dir, exist_ok=True)
    cloud_counter = 0

    while state.is_running():
        obs = obs_receiver.recv(timeout_ms=50)
        if obs is not None:
            state.update_observation(obs)
            latest = obs_receiver.recv_latest()
            if latest is not None:
                state.update_observation(latest)

        now = time.monotonic()
        if now < next_run_time:
            continue

        current_obs = state.get_latest_observation()
        if current_obs is None:
            next_run_time = now + interval
            continue

        rgb = current_obs["rgb"]
        depth = current_obs["depth"]
        joint_angles = current_obs["joint_angles"]
        timestamp = current_obs.get("timestamp", -1)

        cloud = build_observation_cloud(
            rgb, depth, intrinsic, arm_filter, joint_angles, cfg,
        )

        next_run_time = now + interval

        if len(cloud) == 0:
            print(f"[t={timestamp}] Empty cloud, skip inference")
            continue

        cloud_file = os.path.join(cloud_dir, f"cloud_{cloud_counter:06d}.npz")
        np.savez_compressed(cloud_file, cloud=cloud)
        cloud_counter += 1

        if fk_solver is None:
            continue

        q = np.asarray(joint_angles[:5], dtype=np.float64)
        tcp_pose = fk_solver.fk(q)
        tcp_pos = tcp_pose[:3, 3].astype(np.float32)
        tcp_rot6d = matrix_to_rot6d(tcp_pose[:3, :3]).astype(np.float32)

        single_action = np.zeros(10, dtype=np.float32)
        single_action[:3] = tcp_pos
        single_action[3:9] = tcp_rot6d
        single_action[9] = 10.0
        action_base = np.repeat(single_action[None, :], num_action, axis=0)

        if do_clamp:
            action_base[..., :3] = np.clip(action_base[..., :3], safe_min, safe_max)

        trimmed_chunk = action_base[skip_chunk_steps:]
        start_step = state.get_action_step()
        chunk_buffer.add_chunk(trimmed_chunk, start_step)
        print(
            f"[t={timestamp}] chunk ready size={len(trimmed_chunk)} start_step={start_step}"
        )


def _run_action_loop(
    act_sender,
    chunk_buffer: EnsembleBuffer,
    state: InferenceState,
    action_hz: float,
):
    interval = 1.0 / max(action_hz, 1e-6)
    next_send_time = time.monotonic()

    while state.is_running():
        step = state.get_action_step()
        action = chunk_buffer.get_action(step)
        if action is None:
            next_send_time += interval
            sleep_time = next_send_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_send_time = time.monotonic()
            continue

        act_sender.send({"action": action, "step": step})
        state.next_action_step()

        next_send_time += interval
        sleep_time = next_send_time - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_send_time = time.monotonic()


def serve(policy, cfg, arm_filter, device, obs_port: int, act_port: int):
    obs_receiver = ZmqReceiver(obs_port)
    act_sender = ZmqSender(act_port)
    state = InferenceState()
    chunk_buffer = EnsembleBuffer(
        mode=cfg.get("ensemble_mode", "act"),
        k=cfg.get("act_ensemble_k", 0.01),
        hold_last=cfg.get("hold_last_action", True),
    )

    inference_hz = float(cfg.get("inference_hz", 1.0))
    action_hz = float(cfg.get("action_hz", 20.0))
    skip_chunk_steps = int(cfg.get("ignore_chunk_steps", 2))

    print(
        f"Inference server ready obs_port={obs_port} act_port={act_port} "
        f"inference_hz={inference_hz:.2f} action_hz={action_hz:.2f} "
        f"ignore_chunk_steps={skip_chunk_steps}"
    )

    infer_thread = threading.Thread(
        target=_run_inference_loop,
        args=(
            policy,
            cfg,
            arm_filter,
            device,
            obs_receiver,
            chunk_buffer,
            state,
            inference_hz,
            skip_chunk_steps,
        ),
        daemon=True,
    )
    action_thread = threading.Thread(
        target=_run_action_loop,
        args=(act_sender, chunk_buffer, state, action_hz),
        daemon=True,
    )

    infer_thread.start()
    action_thread.start()

    try:
        while infer_thread.is_alive() and action_thread.is_alive():
            time.sleep(0.2)
    finally:
        state.stop()
        infer_thread.join(timeout=1.0)
        action_thread.join(timeout=1.0)
        obs_receiver.close()
        act_sender.close()


def main():
    parser = argparse.ArgumentParser(description="YOLH ZMQ inference server")
    parser.add_argument("--ckpt", required=True, help="Policy checkpoint path")
    parser.add_argument("--config", required=True, help="Deployment YAML config")
    parser.add_argument("--dataset-meta", default=None, help="Action norm stats (.npz)")
    parser.add_argument("--urdf", default=None, help="Robot URDF for arm filter")
    parser.add_argument("--obs-port", type=int, default=OBS_PORT)
    parser.add_argument("--act-port", type=int, default=ACT_PORT)
    parser.add_argument("--inference-hz", type=float, default=None)
    parser.add_argument("--action-hz", type=float, default=None)
    parser.add_argument("--ignore-chunk-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    meta_path = args.dataset_meta or cfg.get("dataset_meta")
    if meta_path is None:
        raise ValueError("Provide --dataset-meta or set dataset_meta in config")
    stats = load_action_norm_stats(str(meta_path))
    cfg.update({k: v for k, v in stats.items() if v is not None})
    if args.inference_hz is not None:
        cfg["inference_hz"] = args.inference_hz
    if args.action_hz is not None:
        cfg["action_hz"] = args.action_hz
    if args.ignore_chunk_steps is not None:
        cfg["ignore_chunk_steps"] = args.ignore_chunk_steps
    print(f"Action stats loaded: max_gripper_width={float(cfg['max_gripper_width']):.4f}m")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = None
    print(f"Policy loaded from {args.ckpt} on {device}")
    global urdf_path
    urdf_path = args.urdf
    arm_filter = build_arm_filter(cfg, urdf_override=args.urdf)

    try:
        serve(policy, cfg, arm_filter, device, args.obs_port, args.act_port)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
