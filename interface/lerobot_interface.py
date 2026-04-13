"""SO-101 + RealSense interface used by YOLH control scripts."""

import json
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pyrealsense2 as rs

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.utils import ensure_safe_goal_position

from policy.robot_interface import RobotInterface
from policy.utils.transformation import rot6d_to_matrix
from scripts.urdf_reader import _get_urdf_data, width_to_jaw_angle

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TICKS_PER_REV = 4096
TWO_PI = 2.0 * np.pi

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _normalize_joint_delta_limit(limit_cfg) -> Optional[float | dict[str, float]]:
    if limit_cfg is None:
        return None
    if isinstance(limit_cfg, (int, float)):
        return float(limit_cfg)
    if isinstance(limit_cfg, dict):
        joint_limit = {}
        for jname in JOINT_NAMES[:5]:
            if jname in limit_cfg:
                joint_limit[jname] = float(limit_cfg[jname])
        return joint_limit or None
    raise TypeError(f"Unsupported joint delta limit config: {type(limit_cfg)!r}")


def _load_motor_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _build_motor_calibration(motor_cfg: dict) -> dict[str, MotorCalibration]:
    calibration = {}
    for jname in JOINT_NAMES:
        cfg = motor_cfg[jname]
        calibration[jname] = MotorCalibration(
            id=cfg["id"],
            drive_mode=cfg["drive_mode"],
            homing_offset=cfg["homing_offset"],
            range_min=cfg["range_min"],
            range_max=cfg["range_max"],
        )
    return calibration


def _tick_to_rad(tick: int, homing_offset: int, drive_mode: int = 0) -> float:
    """Convert a motor tick value to radians."""
    if drive_mode == 0:
        return (tick - homing_offset) * TWO_PI / TICKS_PER_REV
    else:
        return -(tick - homing_offset) * TWO_PI / TICKS_PER_REV


def _rad_to_tick(rad: float, homing_offset: int, drive_mode: int = 0) -> int:
    """Convert radians to a motor tick value."""
    if drive_mode == 0:
        return int(round(rad * TICKS_PER_REV / TWO_PI + homing_offset))
    else:
        return int(round(-rad * TICKS_PER_REV / TWO_PI + homing_offset))


class RealSenseSO101Interface(RobotInterface):
    """Robot interface for the SO-101 arm and a RealSense camera."""

    def __init__(
        self,
        cfg: dict,
        motor_config_path: Optional[str] = None,
        urdf_path: Optional[str] = None,
        serial_port: str = "/dev/ttyACM0",
    ):
        self.cfg = cfg
        self.cam_to_base = cfg["cam_to_base"]
        self.base_to_cam = np.linalg.inv(self.cam_to_base)

        if motor_config_path is None:
            motor_config_path = str(PROJECT_ROOT / "configs" / "lerobot.json")
        self.motor_cfg = _load_motor_config(motor_config_path)
        self._motor_calibration = _build_motor_calibration(self.motor_cfg)

        if urdf_path is None:
            urdf_path = str(
                PROJECT_ROOT / "URDF" / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
            )
        self.ik = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name="gripper_frame_link",
            joint_names=JOINT_NAMES[:5],
        )

        self.serial_port = serial_port
        self._bus = None
        self._bus_lock = threading.RLock()
        self._init_motors()

        self._joint_delta_limit = _normalize_joint_delta_limit(
            cfg.get("joint_delta_limit", cfg.get("max_relative_target"))
        )

        self._init_camera(cfg)
        self._last_q = None

    def _init_motors(self):
        """Initialise the Feetech motor bus."""
        motor_dict = {
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        }
        for jname, motor in motor_dict.items():
            motor.id = self.motor_cfg[jname]["id"]

        self._bus = FeetechMotorsBus(
            port=self.serial_port,
            motors=motor_dict,
            calibration=self._motor_calibration,
        )
        self._bus.connect()
        print(f"[SO101] Connected to Feetech bus on {self.serial_port}")

    def _init_camera(self, cfg: dict):
        self._rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        serial = cfg.get("camera_serial", "")
        if serial:
            rs_config.enable_device(serial)
        rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self._rs_align = rs.align(rs.stream.color)
        self._rs_pipeline.start(rs_config)
        # Warm up
        for _ in range(30):
            self._rs_pipeline.wait_for_frames()
        print("[SO101] RealSense camera started")

    def get_observation(self) -> tuple:
        frames = self._rs_pipeline.wait_for_frames()
        aligned = self._rs_align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        return rgb, depth

    def get_joint_angles(self) -> np.ndarray:
        """Read the current arm joint angles in radians."""
        if self._bus is None:
            raise NotImplementedError(
                "Motor bus is not connected. Install lerobot and check serial port."
            )
        with self._bus_lock:
            positions = self._bus.sync_read("Present_Position", JOINT_NAMES, normalize=False)
        q = np.zeros(5, dtype=np.float64)
        for i, jname in enumerate(JOINT_NAMES[:5]):
            mc = self.motor_cfg[jname]
            tick = int(positions[jname])
            q[i] = _tick_to_rad(tick, mc["homing_offset"], mc["drive_mode"])
        return q

    def get_gripper_width(self) -> float:
        """Read the current gripper width in metres."""
        if self._bus is None:
            raise NotImplementedError("Motor bus not connected.")
        with self._bus_lock:
            positions = self._bus.sync_read("Present_Position", JOINT_NAMES, normalize=False)
        mc = self.motor_cfg["gripper"]
        tick = int(positions["gripper"])
        angle = _tick_to_rad(tick, mc["homing_offset"], mc["drive_mode"])
        d = _get_urdf_data()
        lower, upper = d["jaw_lower"], d["jaw_upper"]
        ratio = float(np.clip((angle - lower) / (upper - lower + 1e-8), 0.0, 1.0))
        max_w = float(self.cfg.get("max_gripper_width", 0.11))
        return ratio * max_w

    def send_action(self, pos: np.ndarray, rot6d: np.ndarray, width: float):
        """Solve IK and send a target pose to the arm."""
        target_rot = rot6d_to_matrix(np.asarray(rot6d, dtype=np.float64)).T
        target_pos = np.asarray(pos, dtype=np.float64)
        target_pose = np.eye(4, dtype=np.float64)
        target_pose[:3, :3] = target_rot
        target_pose[:3, 3] = target_pos

        if self._last_q is not None:
            q_init = self._last_q
        else:
            try:
                q_init = self.get_joint_angles()
            except (NotImplementedError, Exception):
                q_init = None

        if q_init is None:
            q_init_deg = np.zeros(5, dtype=np.float64)
        else:
            q_init_deg = np.rad2deg(np.asarray(q_init, dtype=np.float64))
        q_deg = self.ik.inverse_kinematics(
            q_init_deg,
            target_pose,
            position_weight=1.0,
            orientation_weight=0.8,
        )
        q = np.deg2rad(np.asarray(q_deg[:5], dtype=np.float64))
        print("solved ik", q)

        if self._joint_delta_limit is not None:
            present_q = self.get_joint_angles()
            if isinstance(self._joint_delta_limit, dict):
                joint_limit = {
                    jname: self._joint_delta_limit[jname]
                    for jname in JOINT_NAMES[:5]
                    if jname in self._joint_delta_limit
                }
                goal_present_pos = {
                    jname: (float(q[i]), float(present_q[i]))
                    for i, jname in enumerate(JOINT_NAMES[:5])
                    if jname in joint_limit
                }
                safe_q = ensure_safe_goal_position(goal_present_pos, joint_limit)
                for i, jname in enumerate(JOINT_NAMES[:5]):
                    if jname in safe_q:
                        q[i] = safe_q[jname]
            else:
                goal_present_pos = {
                    jname: (float(q[i]), float(present_q[i]))
                    for i, jname in enumerate(JOINT_NAMES[:5])
                }
                safe_q = ensure_safe_goal_position(goal_present_pos, self._joint_delta_limit)
                q = np.array([safe_q[jname] for jname in JOINT_NAMES[:5]], dtype=np.float64)

        self._last_q = q.copy()
        max_w = float(self.cfg.get("max_gripper_width", 0.11))
        gripper_angle = width_to_jaw_angle(width, max_w)

        goal_positions = {}
        for i, jname in enumerate(JOINT_NAMES[:5]):
            mc = self.motor_cfg[jname]
            tick = _rad_to_tick(float(q[i]), mc["homing_offset"], mc["drive_mode"])
            tick = int(np.clip(tick, mc["range_min"], mc["range_max"]))
            goal_positions[jname] = tick

        mc_g = self.motor_cfg["gripper"]
        tick_g = _rad_to_tick(gripper_angle, mc_g["homing_offset"], mc_g["drive_mode"])
        tick_g = int(np.clip(tick_g, mc_g["range_min"], mc_g["range_max"]))
        goal_positions["gripper"] = tick_g

        if self._bus is None:
            raise NotImplementedError("Motor bus not connected.")
        with self._bus_lock:
            self._bus.sync_write("Goal_Position", goal_positions, normalize=False)

    def stop(self):
        if self._rs_pipeline is not None:
            try:
                self._rs_pipeline.stop()
            except Exception:
                pass
        if self._bus is not None:
            try:
                self._bus.disconnect()
            except Exception:
                pass
        print("[SO101] Stopped.")
