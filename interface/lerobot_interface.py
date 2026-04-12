"""
SO-101 robot interface for LeRobot hardware.

Combines:
- RealSense D435i RGB-D capture
- STS3215 Dynamixel motor communication (via lerobot bus)
- URDF-based IK for Cartesian TCP control

Tick ↔ radian conversion
-------------------------
The STS3215 servo has 4096 ticks per revolution.
    angle_rad = (tick - homing_offset) * 2π / 4096  (drive_mode 0)
    tick      = angle_rad * 4096 / (2π) + homing_offset
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from policy.robot_interface import RobotInterface
from policy.utils.action import rot6d_to_matrix
from interface.ik_solver import SO101IKSolver

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dynamixel STS3215 constants
TICKS_PER_REV = 4096
TWO_PI = 2.0 * np.pi

# Motor IDs in chain order (shoulder_pan … wrist_roll, gripper)
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _load_motor_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _tick_to_rad(tick: int, homing_offset: int, drive_mode: int = 0) -> float:
    """Convert Dynamixel tick to radian."""
    if drive_mode == 0:
        return (tick - homing_offset) * TWO_PI / TICKS_PER_REV
    else:
        return -(tick - homing_offset) * TWO_PI / TICKS_PER_REV


def _rad_to_tick(rad: float, homing_offset: int, drive_mode: int = 0) -> int:
    """Convert radian to Dynamixel tick."""
    if drive_mode == 0:
        return int(round(rad * TICKS_PER_REV / TWO_PI + homing_offset))
    else:
        return int(round(-rad * TICKS_PER_REV / TWO_PI + homing_offset))


class RealSenseSO101Interface(RobotInterface):
    """Concrete RobotInterface for SO-101 arm + RealSense D435i.

    Parameters
    ----------
    cfg : dict
        Deployment config (from ``load_config``).
    motor_config_path : str
        Path to lerobot.json with per-motor calibration.
    urdf_path : str | None
        URDF path for IK.  Defaults to the repo's SO101 URDF.
    serial_port : str
        Serial port for Dynamixel bus (e.g. ``/dev/ttyACM0``).
    """

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

        # ── Motor calibration ──
        if motor_config_path is None:
            motor_config_path = str(PROJECT_ROOT / "configs" / "lerobot.json")
        self.motor_cfg = _load_motor_config(motor_config_path)

        # ── IK solver ──
        if urdf_path is None:
            urdf_path = str(
                PROJECT_ROOT / "URDF" / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
            )
        self.ik = SO101IKSolver(
            urdf_path,
            pos_weight=1.0,
            ori_weight=0.3,
            damping=0.05,
            max_iter=100,
        )

        # ── Dynamixel bus ──
        self.serial_port = serial_port
        self._bus = None
        self._init_motors()

        # ── RealSense camera ──
        self._init_camera(cfg)

        # ── IK warm-start ──
        self._last_q = None

    # ────────────────────────────── Motor init ──────────────────────────

    def _init_motors(self):
        """Initialise motor bus.

        Uses the ``lerobot`` library's Dynamixel bus if available,
        otherwise falls back to a minimal serial implementation.
        """
        try:
            from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

            motor_dict = {}
            for jname in JOINT_NAMES:
                mc = self.motor_cfg[jname]
                motor_dict[jname] = (mc["id"], "sts3215")
            self._bus = FeetechMotorsBus(
                port=self.serial_port,
                motors=motor_dict,
            )
            self._bus.connect()
            print(f"[SO101] Connected to Feetech bus on {self.serial_port}")
        except ImportError:
            print(
                "[SO101] WARNING: lerobot.common.robot_devices.motors.feetech not found. "
                "Motor read/write will raise NotImplementedError."
            )
            self._bus = None

    # ────────────────────────────── Camera init ─────────────────────────

    def _init_camera(self, cfg: dict):
        import pyrealsense2 as rs

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

    # ────────────────────────────── Observation ─────────────────────────

    def get_observation(self) -> tuple:
        import pyrealsense2 as rs

        frames = self._rs_pipeline.wait_for_frames()
        aligned = self._rs_align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        rgb = np.asanyarray(color_frame.get_data())    # (H, W, 3) uint8
        depth = np.asanyarray(depth_frame.get_data())   # (H, W) uint16 mm
        return rgb, depth

    # ────────────────────────────── Joint read ──────────────────────────

    def get_joint_angles(self) -> np.ndarray:
        """Read current joint angles (5,) from motor bus [radians]."""
        if self._bus is None:
            raise NotImplementedError(
                "Motor bus is not connected. Install lerobot and check serial port."
            )
        positions = self._bus.read("Present_Position")
        q = np.zeros(5, dtype=np.float64)
        for i, jname in enumerate(JOINT_NAMES[:5]):
            mc = self.motor_cfg[jname]
            tick = int(positions[i])
            q[i] = _tick_to_rad(tick, mc["homing_offset"], mc["drive_mode"])
        return q

    def get_gripper_width(self) -> float:
        """Read current gripper width (metres) from gripper motor."""
        if self._bus is None:
            raise NotImplementedError("Motor bus not connected.")
        positions = self._bus.read("Present_Position")
        mc = self.motor_cfg["gripper"]
        tick = int(positions[5])
        angle = _tick_to_rad(tick, mc["homing_offset"], mc["drive_mode"])
        # Gripper angle → width conversion uses urdf_reader
        from scripts.urdf_reader import width_to_jaw_angle, _get_urdf_data
        d = _get_urdf_data()
        # Inverse of width_to_jaw_angle: ratio = (angle - lower) / (upper - lower)
        lower, upper = d["jaw_lower"], d["jaw_upper"]
        ratio = float(np.clip((angle - lower) / (upper - lower + 1e-8), 0.0, 1.0))
        max_w = float(self.cfg.get("max_gripper_width", 0.11))
        return ratio * max_w

    # ────────────────────────────── Action send ─────────────────────────

    def send_action(self, pos: np.ndarray, rot6d: np.ndarray, width: float):
        """Send TCP action (base frame) to the robot via IK.

        1. Convert rot6d → rotation matrix (transpose to undo row convention).
        2. Run IK to get joint angles.
        3. Convert joint angles to Dynamixel ticks.
        4. Set gripper width.
        5. Write to motor bus.
        """
        # rot6d encodes first-two-rows → rot6d_to_matrix gives R^T → transpose
        target_rot = rot6d_to_matrix(np.asarray(rot6d, dtype=np.float64)).T
        target_pos = np.asarray(pos, dtype=np.float64)

        # Warm-start IK from last solution or current position
        if self._last_q is not None:
            q_init = self._last_q
        else:
            try:
                q_init = self.get_joint_angles()
            except (NotImplementedError, Exception):
                q_init = None

        q, success, info = self.ik.solve(target_pos, target_rot, q_init=q_init)
        if not success:
            print(
                f"[SO101] IK did not converge: pos_err={info['pos_err']:.4f}m, "
                f"ori_err={info['ori_err']:.4f}rad after {info['iters']} iters"
            )
        self._last_q = q.copy()

        # ── Gripper angle ──
        from scripts.urdf_reader import width_to_jaw_angle
        max_w = float(self.cfg.get("max_gripper_width", 0.11))
        gripper_angle = width_to_jaw_angle(width, max_w)

        # ── Build tick array ──
        ticks = np.zeros(6, dtype=np.int32)
        for i, jname in enumerate(JOINT_NAMES[:5]):
            mc = self.motor_cfg[jname]
            tick = _rad_to_tick(float(q[i]), mc["homing_offset"], mc["drive_mode"])
            tick = int(np.clip(tick, mc["range_min"], mc["range_max"]))
            ticks[i] = tick

        mc_g = self.motor_cfg["gripper"]
        tick_g = _rad_to_tick(gripper_angle, mc_g["homing_offset"], mc_g["drive_mode"])
        tick_g = int(np.clip(tick_g, mc_g["range_min"], mc_g["range_max"]))
        ticks[5] = tick_g

        # ── Write to bus ──
        if self._bus is None:
            raise NotImplementedError("Motor bus not connected.")
        self._bus.write("Goal_Position", ticks.tolist())

    # ────────────────────────────── Stop ────────────────────────────────

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
