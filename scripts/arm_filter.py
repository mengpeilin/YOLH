import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

def _rpy_to_rot(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rpy_to_rot(rpy)
    T[:3, 3] = xyz
    return T


def _parse_origin(element) -> Tuple[np.ndarray, np.ndarray]:
    origin = element.find("origin") if element is not None else None
    if origin is None:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ", dtype=np.float64)
    rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ", dtype=np.float64)
    if xyz.size != 3:
        xyz = np.zeros(3, dtype=np.float64)
    if rpy.size != 3:
        rpy = np.zeros(3, dtype=np.float64)
    return xyz, rpy

# Ordered joint chain from base to end-effector (excluding gripper jaw).
ARM_JOINT_CHAIN = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# Links we want to filter (arm body, not gripper).
ARM_LINKS_TO_FILTER = {
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "lower_arm_link",
    "wrist_link",
}

# Gripper links are NOT filtered.
GRIPPER_LINKS = {"gripper_link", "gripper_frame_link", "moving_jaw_so101_v1_link"}


class ArmFilter:
    def __init__(
        self,
        urdf_path: str,
        cam_to_base: np.ndarray,
        capsule_radii: Optional[Dict[str, float]] = None,
    ):
        self.cam_to_base = np.asarray(cam_to_base, dtype=np.float64).reshape(4, 4)
        self.base_to_cam = np.linalg.inv(self.cam_to_base)

        # Parse URDF
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        self.joints = {}
        for jt in root.findall("joint"):
            name = jt.get("name")
            if name is None:
                continue
            jtype = jt.get("type", "fixed")
            parent = jt.find("parent").get("link") if jt.find("parent") is not None else None
            child = jt.find("child").get("link") if jt.find("child") is not None else None
            xyz, rpy = _parse_origin(jt)
            self.joints[name] = {
                "type": jtype,
                "parent": parent,
                "child": child,
                "origin_tf": _make_transform(xyz, rpy),
            }

        # Default capsule radii
        self.capsule_radii = {
            "base_link": 0.05,
            "shoulder_link": 0.04,
            "upper_arm_link": 0.035,
            "lower_arm_link": 0.03,
            "wrist_link": 0.03,
        }
        if capsule_radii is not None:
            self.capsule_radii.update(capsule_radii)

    def _fk_link_origins(
        self, joint_angles: np.ndarray
    ) -> Dict[str, np.ndarray]:
        assert len(joint_angles) == len(ARM_JOINT_CHAIN), (
            f"Expected {len(ARM_JOINT_CHAIN)} joint angles, got {len(joint_angles)}"
        )

        link_positions: Dict[str, np.ndarray] = {}
        # base_link origin is at world origin
        link_positions["base_link"] = np.zeros(3, dtype=np.float64)

        T_current = np.eye(4, dtype=np.float64)
        for i, jname in enumerate(ARM_JOINT_CHAIN):
            jdata = self.joints[jname]
            # Static transform from parent to joint frame
            T_static = jdata["origin_tf"]
            # Revolute rotation about Z
            q = float(joint_angles[i])
            c, s = np.cos(q), np.sin(q)
            Rz = np.eye(4, dtype=np.float64)
            Rz[0, 0] = c; Rz[0, 1] = -s
            Rz[1, 0] = s; Rz[1, 1] = c
            T_current = T_current @ T_static @ Rz
            child_link = jdata["child"]
            link_positions[child_link] = T_current[:3, 3].copy()

        return link_positions

    def _build_capsules_in_cam(
        self, joint_angles: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        link_pos_base = self._fk_link_origins(joint_angles)
        R = self.base_to_cam[:3, :3]
        t = self.base_to_cam[:3, 3]

        # Transform all positions to camera frame
        link_pos_cam = {
            name: R @ pos + t for name, pos in link_pos_base.items()
        }

        # Build capsules between consecutive links in the chain
        chain_links = ["base_link", "shoulder_link", "upper_arm_link",
                       "lower_arm_link", "wrist_link", "gripper_link"]
        capsules = []
        for i in range(len(chain_links) - 1):
            link_a = chain_links[i]
            link_b = chain_links[i + 1]
            if link_a not in ARM_LINKS_TO_FILTER:
                continue
            if link_a not in link_pos_cam or link_b not in link_pos_cam:
                continue
            r = self.capsule_radii.get(link_a, 0.03)
            capsules.append((link_pos_cam[link_a], link_pos_cam[link_b], r))

        return capsules

    def filter(
        self,
        coords: np.ndarray,
        colors: np.ndarray,
        joint_angles: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(coords) == 0:
            return coords, colors

        capsules = self._build_capsules_in_cam(joint_angles)
        if not capsules:
            return coords, colors

        mask = np.ones(len(coords), dtype=bool)
        pts = coords.astype(np.float64)

        for p0, p1, radius in capsules:
            # Point-to-segment distance
            seg = p1 - p0
            seg_len_sq = np.dot(seg, seg)
            if seg_len_sq < 1e-12:
                # Degenerate segment – use sphere
                dists_sq = np.sum((pts - p0) ** 2, axis=1)
            else:
                # Project each point onto the segment
                t_param = np.dot(pts - p0, seg) / seg_len_sq
                t_param = np.clip(t_param, 0.0, 1.0)
                closest = p0 + t_param[:, None] * seg
                dists_sq = np.sum((pts - closest) ** 2, axis=1)

            mask &= dists_sq > radius * radius

        return coords[mask], colors[mask]