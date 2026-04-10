#!/usr/bin/env python3
"""
YOLH Inference Script with Robot Arm Self-Filtering
====================================================

Captures RGB-D from RealSense, removes arm from point cloud via FK-based
capsule filter, runs YOLH policy, denormalises actions, projects camera-frame
predictions to base frame, and sends them to the robot.

The observed gripper is kept (not filtered or re-inserted), which aligns with
training data where a synthetic gripper was inserted at the same pose.

Usage:
    python inference.py --ckpt logs/my_task/policy_last.ckpt \
                        --config configs/camera_calibration.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
from lerobot_policy_yolh import YolhPolicy, YolhConfig
from scripts.arm_filter import ArmFilter
from scripts.urdf_reader import rgbd_to_points

# ImageNet stats
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SO101_URDF = str(PROJECT_ROOT / "URDF/SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["cam_to_base"] = np.array(cfg["cam_to_base"], dtype=np.float64).reshape(4, 4)
    cfg["workspace_min"] = np.array(cfg["workspace_min"], dtype=np.float32)
    cfg["workspace_max"] = np.array(cfg["workspace_max"], dtype=np.float32)
    cfg["safe_workspace_min"] = np.array(cfg["safe_workspace_min"], dtype=np.float32)
    cfg["safe_workspace_max"] = np.array(cfg["safe_workspace_max"], dtype=np.float32)
    cfg["gripper_offset"] = cfg.get("gripper_offset", [0.05, 0.0, 0.0])
    cfg["camera_intrinsic"] = np.array(cfg["camera_intrinsic"], dtype=np.float32)
    return cfg


def load_action_norm_stats(dataset_path: str) -> dict:
    data = np.load(dataset_path, allow_pickle=True)
    required = ["trans_min", "trans_max", "max_gripper_width"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(
            f"Dataset meta missing keys {missing}: {dataset_path}. "
            "Expected output from scripts/merge_episodes.py"
        )
    max_w = float(data["max_gripper_width"])
    if max_w <= 0:
        raise ValueError(
            f"Invalid max_gripper_width={max_w} from dataset meta: {dataset_path}"
        )
    return {
        "trans_min": np.asarray(data["trans_min"], dtype=np.float32),
        "trans_max": np.asarray(data["trans_max"], dtype=np.float32),
        "max_gripper_width": np.float32(max_w),
        "num_action": int(data["num_action"]) if "num_action" in data else None,
        "task_name": str(data["task_name"]) if "task_name" in data else None,
    }

def build_observation_cloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    arm_filter: ArmFilter,
    joint_angles: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    voxel_size = cfg["voxel_size"]
    ws_min = cfg["workspace_min"]
    ws_max = cfg["workspace_max"]

    # 1. RGB-D → points (no mask – gripper is preserved, arm filtered below)
    coords, colors = rgbd_to_points(rgb, depth, intrinsic, mask=None)
    if len(coords) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # 2. Arm self-filter (removes arm body, keeps gripper)
    if cfg.get("arm_filter", {}).get("enabled", True):
        coords, colors = arm_filter.filter(coords, colors, joint_angles)
    if len(coords) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # 3. Workspace filter
    in_ws = np.all((coords >= ws_min) & (coords <= ws_max), axis=1)
    coords = coords[in_ws]
    colors = colors[in_ws]
    if len(coords) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # 4. Voxel downsample
    vi = np.floor(coords / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(vi, axis=0, return_index=True)
    coords = coords[unique_idx]
    colors = colors[unique_idx]

    # 5. ImageNet normalise
    colors = (colors - IMG_MEAN) / IMG_STD

    cloud = np.concatenate([coords, colors], axis=-1).astype(np.float32)
    return cloud

def cloud_to_me_input(cloud: np.ndarray, voxel_size: float, device: torch.device):
    coords = np.ascontiguousarray(cloud[:, :3] / voxel_size, dtype=np.int32)
    feats = cloud.astype(np.float32)
    coords_batch, feats_batch = ME.utils.sparse_collate([coords], [feats])
    return ME.SparseTensor(feats_batch.to(device), coords_batch.to(device))

def unnormalize_action(action: np.ndarray, cfg: dict) -> np.ndarray:
    action = action.copy()
    trans_min = cfg["trans_min"]
    trans_max = cfg["trans_max"]
    max_w = float(cfg["max_gripper_width"])
    if max_w <= 0:
        raise ValueError(f"max_gripper_width must be > 0, got {max_w}")

    action[..., :3] = (action[..., :3] + 1) / 2.0 * (trans_max - trans_min) + trans_min
    action[..., -1] = (action[..., -1] + 1) / 2.0 * max_w
    return action

def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    # Normalise first vector
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    # Second vector: orthogonalise then normalise
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    # Third vector: cross product
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def project_action_to_base(
    action_cam: np.ndarray, cam_to_base: np.ndarray
) -> np.ndarray:
    T = cam_to_base.astype(np.float64)
    R_cb = T[:3, :3]
    t_cb = T[:3, 3]

    out = action_cam.copy()
    num = action_cam.shape[0]

    for i in range(num):
        # Position
        pos_cam = action_cam[i, :3].astype(np.float64)
        pos_base = R_cb @ pos_cam + t_cb
        out[i, :3] = pos_base

        # Rotation 6d → matrix → rotate to base → back to 6d
        rot6d = action_cam[i, 3:9].astype(np.float64)
        R_ee_cam = rot6d_to_matrix(rot6d)
        R_ee_base = R_cb @ R_ee_cam
        out[i, 3:9] = R_ee_base[:2, :].flatten()  # first two rows

    return out.astype(np.float32)

class EnsembleBuffer:
    def __init__(self, mode: str = "new"):
        self.mode = mode
        self.buffer = []   # list of (start_t, actions_array)
        self.t = 0

    def add(self, actions: np.ndarray, timestep: int):
        self.buffer.append((timestep, actions))

    def get(self) -> np.ndarray | None:
        candidates = []
        for start_t, acts in self.buffer:
            idx = self.t - start_t
            if 0 <= idx < len(acts):
                candidates.append((start_t, acts[idx]))
        self.t += 1
        if not candidates:
            return None
        if self.mode == "new":
            return candidates[-1][1]
        elif self.mode == "avg":
            return np.mean([c[1] for c in candidates], axis=0)
        return candidates[-1][1]

def rot6d_angular_distance(r1: np.ndarray, r2: np.ndarray) -> float:
    """Angular distance between two rot6d vectors (in radians)."""
    m1 = rot6d_to_matrix(r1)
    m2 = rot6d_to_matrix(r2)
    diff = m1 @ m2.T
    cos_val = np.clip((np.trace(diff) - 1) / 2.0, -1, 1)
    return float(np.arccos(cos_val))


def discretize_rotation(
    rot_begin: np.ndarray, rot_end: np.ndarray, step_size: float
) -> list:
    """Interpolate rot6d from begin to end in angular steps."""
    angle = rot6d_angular_distance(rot_begin, rot_end)
    n_steps = max(1, int(angle / step_size) + 1)
    steps = []
    for i in range(n_steps):
        alpha = (i + 1) / n_steps
        rot_i = rot_begin * (1 - alpha) + rot_end * alpha
        steps.append(rot_i)
    return steps

class RobotInterface:
    """
    Abstract interface for robot communication.
    Subclass and implement for your specific robot (e.g. URDF/SO-ARM100 via LeRobot).
    """

    def get_observation(self) -> tuple:
        """Returns (rgb, depth) from the camera."""
        raise NotImplementedError

    def get_joint_angles(self) -> np.ndarray:
        """Returns (5,) array: shoulder_pan through wrist_roll."""
        raise NotImplementedError

    def send_action(self, pos: np.ndarray, rot6d: np.ndarray, width: float):
        """Send a single action step to the robot (base frame)."""
        raise NotImplementedError

    def stop(self):
        """Stop the robot."""
        pass


class RealSenseSO101Interface(RobotInterface):
    """
    Reference implementation using pyrealsense2 for camera
    and URDF/SO-ARM100 serial interface for the robot.

    Users should adapt this for their specific hardware setup.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.cam_to_base = cfg["cam_to_base"]
        self.base_to_cam = np.linalg.inv(self.cam_to_base)
        self._init_camera(cfg)

    def _init_camera(self, cfg: dict):
        import pyrealsense2 as rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        serial = cfg.get("camera_serial", "")
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.pipeline.start(config)
        # Warm up
        for _ in range(30):
            self.pipeline.wait_for_frames()

    def get_observation(self) -> tuple:
        import pyrealsense2 as rs
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        rgb = np.asanyarray(color_frame.get_data())          # (H, W, 3) uint8
        depth = np.asanyarray(depth_frame.get_data())         # (H, W) uint16 mm
        return rgb, depth

    def get_joint_angles(self) -> np.ndarray:
        """Override with real robot joint readings."""
        raise NotImplementedError(
            "Implement get_joint_angles() to read from your URDF/SO-ARM100 controller."
        )

    def send_action(self, pos: np.ndarray, rot6d: np.ndarray, width: float):
        raise NotImplementedError(
            "Implement send_action() for your URDF/SO-ARM100 controller."
        )

    def stop(self):
        self.pipeline.stop()

def run_inference(
    ckpt_path: str,
    cfg: dict,
    robot: RobotInterface,
):
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

    # Load checkpoint – handle both raw YOLH and wrapped YolhPolicy state dicts
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
    ensemble = EnsembleBuffer(mode=cfg.get("ensemble_mode", "new"))

    intrinsic = cfg["camera_intrinsic"]
    num_inf_step = cfg.get("num_inference_step", 20)
    max_steps = cfg.get("max_steps", 300)

    use_disc_rot = cfg.get("discretize_rotation", False)
    rot_step = cfg.get("rot_step_size", np.pi / 16)
    gripper_thresh = cfg.get("gripper_threshold", 0.02)

    last_rot6d = None
    prev_width = None

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

                # ── Forward pass (via LeRobot YolhPolicy) ──
                cloud_tensor = torch.from_numpy(cloud).unsqueeze(0).to(device)  # (1, N, 6)
                batch = {"observation.point_cloud": cloud_tensor}
                action_chunk = policy.predict_action_chunk(batch)  # (1, chunk, 10) physical
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
                ensemble.add(action_base, t)

            # ── Get ensembled step action ──
            step = ensemble.get()
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

            if prev_width is None or abs(prev_width - step_width) > gripper_thresh:
                prev_width = step_width

    robot.stop()
    print("Inference complete.")

def main():
    parser = argparse.ArgumentParser(description="YOLH inference with arm self-filter")
    parser.add_argument("--ckpt", required=True, help="Path to policy checkpoint")
    parser.add_argument("--config", required=True, help="Path to camera_calibration.yaml")
    parser.add_argument("--dataset-meta", default=None)
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
            "Loaded action normalization from dataset meta: "
            f"{dataset_meta_path} (max_gripper_width={float(cfg['max_gripper_width']):.4f}m)"
        )
    else :
        raise KeyError(
            "Please pass --dataset-meta train_dataset.npz."
        )

    robot = RealSenseSO101Interface(cfg)

    run_inference(args.ckpt, cfg, robot)


if __name__ == "__main__":
    main()
