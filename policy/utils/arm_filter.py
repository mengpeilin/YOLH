import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

from policy.utils.transformation import make_transform, parse_origin, rot_z_transform

ARM_JOINT_CHAIN = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

ARM_LINKS_TO_FILTER = {
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "lower_arm_link",
    "wrist_link",
}

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
            xyz, rpy = parse_origin(jt)
            self.joints[name] = {
                "type": jtype,
                "parent": parent,
                "child": child,
                "origin_tf": make_transform(xyz, rpy),
            }

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
        link_positions["base_link"] = np.zeros(3, dtype=np.float64)

        T_current = np.eye(4, dtype=np.float64)
        for i, jname in enumerate(ARM_JOINT_CHAIN):
            jdata = self.joints[jname]
            T_static = jdata["origin_tf"]
            q = float(joint_angles[i])
            T_current = T_current @ T_static @ rot_z_transform(q)
            child_link = jdata["child"]
            link_positions[child_link] = T_current[:3, 3].copy()

        return link_positions

    def _build_capsules_in_cam(
        self, joint_angles: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        link_pos_base = self._fk_link_origins(joint_angles)
        R = self.base_to_cam[:3, :3]
        t = self.base_to_cam[:3, 3]

        link_pos_cam = {
            name: R @ pos + t for name, pos in link_pos_base.items()
        }

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
            seg = p1 - p0
            seg_len_sq = np.dot(seg, seg)
            if seg_len_sq < 1e-12:
                dists_sq = np.sum((pts - p0) ** 2, axis=1)
            else:
                t_param = np.dot(pts - p0, seg) / seg_len_sq
                t_param = np.clip(t_param, 0.0, 1.0)
                closest = p0 + t_param[:, None] * seg
                dists_sq = np.sum((pts - closest) ** 2, axis=1)

            mask &= dists_sq > radius * radius

        return coords[mask], colors[mask]