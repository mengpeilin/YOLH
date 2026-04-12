#!/usr/bin/env python3
"""
Test inference script with fake policy for visualization.

Receives observations from a remote control machine (same ZMQ topics as real inference),
builds observation point clouds, and visualizes them in real time using Open3D.
A fake policy outputs constant or random actions to test the full pipeline.
"""

import argparse
import threading
import time
import numpy as np
import torch
import open3d as o3d
import cv2
from typing import Optional, Dict, Any

from policy.utils.observation import build_observation_cloud
from policy.utils.config import load_config
from policy.utils.arm_filter import ArmFilter
from interface.zmq_interface import ZmqReceiver, ZmqSender, OBS_PORT, ACT_PORT
from policy.utils.inference_state import InferenceState
from policy.utils.ensemble import EnsembleBuffer


# --------------------------- Fake Policy ---------------------------
class FakePolicy:
    """A fake policy that outputs constant or random actions."""
    def __init__(self, action_dim: int = 10, action_type: str = "constant", constant_value: float = 0.0):
        self.action_dim = action_dim
        self.action_type = action_type
        self.constant_value = constant_value

    def __call__(self, cloud_sparse, actions=None, batch_size=1):
        """
        Mimics the real policy's forward method.
        Returns a tensor of shape (1, num_action, action_dim).
        """
        # Here we assume the policy outputs a chunk of actions (num_action steps)
        # For testing, we just produce a constant or random chunk.
        num_action = 20  # default, could be from config
        if self.action_type == "constant":
            action_chunk = np.full((num_action, self.action_dim), self.constant_value, dtype=np.float32)
        elif self.action_type == "random":
            action_chunk = np.random.uniform(-1.0, 1.0, (num_action, self.action_dim)).astype(np.float32)
        else:
            raise ValueError(f"Unknown action_type: {self.action_type}")
        return torch.from_numpy(action_chunk).unsqueeze(0)  # (1, num_action, dim)


# --------------------------- Visualization ---------------------------
class PointCloudVisualizer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="YOLH Observation Point Cloud", width=1024, height=768)
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        self.is_running = True

    def update(self, cloud: np.ndarray):
        """cloud: (N, 6) array (x,y,z,r,g,b) with rgb in [0,1]"""
        if cloud.shape[0] == 0:
            return
        points = cloud[:, :3]
        colors = cloud[:, 3:6]  # already in [0,1]
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()
        self.is_running = False


def visualize_rgb_image(rgb: np.ndarray, window_name="RGB Image"):
    """Show RGB image using OpenCV (blocking for a short time)."""
    # rgb is (H,W,3) in [0,255]
    cv2.imshow(window_name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)  # non-blocking


# --------------------------- Main Test Loop ---------------------------
def _run_fake_inference_loop(
    policy,
    cfg,
    arm_filter,
    device,
    obs_receiver,
    chunk_buffer: EnsembleBuffer,
    state: InferenceState,
    inference_hz: float,
    skip_chunk_steps: int,
    visualizer: Optional[PointCloudVisualizer] = None,
    show_rgb: bool = False,
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

        # Build point cloud (removes arm if arm_filter enabled)
        cloud = build_observation_cloud(
            rgb, depth, intrinsic, arm_filter, joint_angles, cfg,
        )
        next_run_time = now + interval

        # Visualize
        if visualizer is not None and cloud.shape[0] > 0:
            visualizer.update(cloud)
        if show_rgb:
            visualize_rgb_image(rgb)

        if cloud.shape[0] == 0:
            print(f"[t={timestamp}] Empty cloud, skip inference")
            continue

        # Fake policy inference
        # We need to convert cloud to sparse tensor (same as real inference)
        from policy.utils.observation import _cloud_to_sparse  # reuse function
        cloud_sparse = _cloud_to_sparse(cloud, cfg["voxel_size"], device)
        action_chunk = policy(cloud_sparse, actions=None, batch_size=1)
        action_cam = action_chunk.squeeze(0).cpu().numpy()

        # Project action from camera to base frame
        from policy.utils.transformation import project_action_to_base
        action_base = project_action_to_base(action_cam, cam_to_base)
        if do_clamp:
            action_base[..., :3] = np.clip(action_base[..., :3], safe_min, safe_max)

        trimmed_chunk = action_base[skip_chunk_steps:]
        start_step = state.get_action_step()
        chunk_buffer.add_chunk(trimmed_chunk, start_step)
        print(f"[t={timestamp}] Fake chunk ready size={len(trimmed_chunk)} start_step={start_step}")


def _run_action_loop(
    act_sender,
    chunk_buffer: EnsembleBuffer,
    state: InferenceState,
    action_hz: float,
):
    interval = 1.0 / max(action_hz, 1e-6)
    next_send_time = time.monotonic()

    while state.is_running():
        step = state.next_action_step()
        action = chunk_buffer.get_action(step)
        act_sender.send({"action": action, "step": step})
        # Optionally print sent action (reduce frequency)
        if step % 50 == 0:
            print(f"Sent action step {step}: {action[:3]}...")

        next_send_time += interval
        sleep_time = next_send_time - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_send_time = time.monotonic()


def serve_test(
    policy,
    cfg,
    arm_filter,
    device,
    obs_port: int,
    act_port: int,
    visualizer: PointCloudVisualizer,
    show_rgb: bool,
):
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

    print(f"Test server ready obs_port={obs_port} act_port={act_port}")
    print(f"  inference_hz={inference_hz:.2f} action_hz={action_hz:.2f} ignore_chunk_steps={skip_chunk_steps}")

    infer_thread = threading.Thread(
        target=_run_fake_inference_loop,
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
            visualizer,
            show_rgb,
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
        visualizer.close()
        cv2.destroyAllWindows()


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


def main():
    parser = argparse.ArgumentParser(description="YOLH test inference with fake policy")
    parser.add_argument("--config", required=True, help="Deployment YAML config")
    parser.add_argument("--urdf", default=None, help="Robot URDF for arm filter")
    parser.add_argument("--obs-port", type=int, default=OBS_PORT)
    parser.add_argument("--act-port", type=int, default=ACT_PORT)
    parser.add_argument("--action-type", default="constant", choices=["constant", "random"], help="Fake policy output type")
    parser.add_argument("--action-const", type=float, default=0.0, help="Constant action value (for all dims)")
    parser.add_argument("--no-vis", action="store_true", help="Disable point cloud visualization")
    parser.add_argument("--show-rgb", action="store_true", help="Show RGB image in OpenCV window")
    parser.add_argument("--action-dim", type=int, default=10, help="Action dimension (usually 10: 7 pose + 3 gripper?)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Fake policy does not need dataset stats, but we keep them for consistency
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = FakePolicy(action_dim=args.action_dim, action_type=args.action_type, constant_value=args.action_const)
    print(f"Fake policy loaded: type={args.action_type}, dim={args.action_dim}")

    arm_filter = build_arm_filter(cfg, urdf_override=args.urdf)

    visualizer = None
    if not args.no_vis:
        visualizer = PointCloudVisualizer()

    try:
        serve_test(
            policy,
            cfg,
            arm_filter,
            device,
            args.obs_port,
            args.act_port,
            visualizer,
            args.show_rgb,
        )
    except KeyboardInterrupt:
        print("\nTest server stopped.")


if __name__ == "__main__":
    main()