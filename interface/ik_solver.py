"""
SO-101 inverse kinematics solver.

Uses damped-least-squares (Levenberg–Marquardt style) Jacobian IK.
The SO-101 has 5 revolute joints (all z-axis) so it can reach 3 position
DOF + 2 orientation DOF.  The solver minimises a combined position +
orientation error with configurable weights.

Coordinate conventions
----------------------
- All angles are in **radians**.
- The URDF kinematic chain is:
    base_link
      └─ shoulder_pan  → shoulder_link
          └─ shoulder_lift → upper_arm_link
              └─ elbow_flex   → lower_arm_link
                  └─ wrist_flex  → wrist_link
                      └─ wrist_roll → gripper_link
                          └─ gripper_frame_joint (fixed) → gripper_frame_link
- All revolute joints rotate about the **local z-axis**.
- The FK output is a 4×4 homogeneous matrix T_base_tcp.
"""

import numpy as np
import xml.etree.ElementTree as ET
from typing import Optional, Tuple

from policy.utils.transformation import make_transform, parse_origin, rot_z_transform

__all__ = ["SO101IKSolver"]

def _so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithmic map SO(3) → ℝ³  (axis-angle vector)."""
    cos_val = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_val)
    if abs(angle) < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if abs(angle - np.pi) < 1e-6:
        # Near π: extract axis from symmetric part
        # R + I = 2 * n n^T  →  n is the column with largest diagonal
        S = R + np.eye(3)
        col = np.argmax(np.diag(S))
        n = S[:, col]
        n = n / (np.linalg.norm(n) + 1e-12)
        return n * angle
    k = angle / (2.0 * np.sin(angle))
    return k * np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]], dtype=np.float64)


# ────────────────────────────────────────────────────────────────────────
# Solver
# ────────────────────────────────────────────────────────────────────────

# Kinematic chain order (matches Dynamixel motor IDs 1-5).
ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


class SO101IKSolver:
    """Damped-least-squares IK for the SO-101 5-DOF arm.

    Parameters
    ----------
    urdf_path : str
        Path to the SO-101 URDF file.
    pos_weight : float
        Weight for the position error (metres).
    ori_weight : float
        Weight for the orientation error (radians).
        For a 5-DOF arm set this lower than *pos_weight* because full
        6-DOF orientation cannot always be achieved.
    damping : float
        Damping factor λ for the DLS pseudo-inverse  (J^T J + λ²I)^{-1} J^T.
    max_iter : int
        Maximum iterations per solve.
    tol_pos : float
        Position convergence tolerance (metres).
    tol_ori : float
        Orientation convergence tolerance (radians).
    """

    def __init__(
        self,
        urdf_path: str,
        pos_weight: float = 1.0,
        ori_weight: float = 0.3,
        damping: float = 0.05,
        max_iter: int = 100,
        tol_pos: float = 1e-4,
        tol_ori: float = 5e-3,
    ):
        self.pos_weight = pos_weight
        self.ori_weight = ori_weight
        self.damping = damping
        self.max_iter = max_iter
        self.tol_pos = tol_pos
        self.tol_ori = tol_ori

        # Parse URDF
        self._parse_urdf(urdf_path)

    # ── URDF parsing ────────────────────────────────────────────────────

    def _parse_urdf(self, urdf_path: str):
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        joints = {}
        for jt in root.findall("joint"):
            name = jt.get("name")
            if name is None:
                continue
            jtype = jt.get("type", "fixed")
            parent = jt.find("parent").get("link") if jt.find("parent") is not None else None
            child = jt.find("child").get("link") if jt.find("child") is not None else None
            xyz, rpy = parse_origin(jt)
            limit_elem = jt.find("limit")
            lower = float(limit_elem.get("lower", "-3.14")) if limit_elem is not None else -np.pi
            upper = float(limit_elem.get("upper", "3.14")) if limit_elem is not None else np.pi
            joints[name] = {
                "type": jtype,
                "parent": parent,
                "child": child,
                "origin_tf": make_transform(xyz, rpy),
                "lower": lower,
                "upper": upper,
            }
        self._joints = joints

        # Collect static transforms of the 5-DOF arm chain
        self._chain_tf = []   # list of 4×4 static transforms
        self._joint_limits = np.zeros((5, 2), dtype=np.float64)
        for i, jname in enumerate(ARM_JOINT_NAMES):
            jd = joints[jname]
            self._chain_tf.append(jd["origin_tf"].copy())
            self._joint_limits[i, 0] = jd["lower"]
            self._joint_limits[i, 1] = jd["upper"]

        # Fixed transform from last joint child (gripper_link) to TCP
        # (gripper_frame_link via gripper_frame_joint).
        if "gripper_frame_joint" in joints:
            self._tcp_tf = joints["gripper_frame_joint"]["origin_tf"].copy()
        else:
            self._tcp_tf = np.eye(4, dtype=np.float64)

    # ── Forward kinematics ──────────────────────────────────────────────

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Compute TCP pose (4×4) in base frame from joint angles *q* (5,)."""
        T = np.eye(4, dtype=np.float64)
        for i in range(5):
            T = T @ self._chain_tf[i] @ rot_z_transform(float(q[i]))
        T = T @ self._tcp_tf
        return T

    def fk_chain(self, q: np.ndarray) -> list:
        """Return list of 5 joint-origin 4×4 transforms (for Jacobian)."""
        frames = []
        T = np.eye(4, dtype=np.float64)
        for i in range(5):
            T = T @ self._chain_tf[i]
            frames.append(T.copy())      # frame BEFORE rotation
            T = T @ rot_z_transform(float(q[i]))
        return frames

    # ── Numerical Jacobian ──────────────────────────────────────────────

    def _jacobian(self, q: np.ndarray) -> np.ndarray:
        """Compute 6×5 geometric Jacobian (linear + angular rows)."""
        frames = self.fk_chain(q)
        T_ee = self.fk(q)
        p_ee = T_ee[:3, 3]

        J = np.zeros((6, 5), dtype=np.float64)
        T_accum = np.eye(4, dtype=np.float64)
        for i in range(5):
            T_accum = T_accum @ self._chain_tf[i]
            # Joint axis in world frame (all joints rotate about local z)
            z_i = T_accum[:3, 2]
            p_i = T_accum[:3, 3]
            # Linear velocity: z_i × (p_ee - p_i)
            J[:3, i] = np.cross(z_i, p_ee - p_i)
            # Angular velocity: z_i
            J[3:, i] = z_i
            # Accumulate rotation for this joint
            T_accum = T_accum @ rot_z_transform(float(q[i]))

        return J

    # ── IK solve ────────────────────────────────────────────────────────

    def _solve_single(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q_init: np.ndarray,
    ) -> Tuple[np.ndarray, bool, dict]:
        """Run one DLS solve from a single initial guess."""
        q = q_init.copy()

        lam2 = self.damping ** 2
        w_p = self.pos_weight
        w_o = self.ori_weight

        for it in range(self.max_iter):
            T_cur = self.fk(q)
            p_cur = T_cur[:3, 3]
            R_cur = T_cur[:3, :3]

            dp = target_pos - p_cur
            pos_err = np.linalg.norm(dp)

            R_err = target_rot @ R_cur.T
            do = _so3_log(R_err)
            ori_err = np.linalg.norm(do)

            if pos_err < self.tol_pos and ori_err < self.tol_ori:
                return q, True, {"pos_err": pos_err, "ori_err": ori_err, "iters": it}

            e = np.concatenate([w_p * dp, w_o * do])
            J = self._jacobian(q)
            J_w = np.vstack([w_p * J[:3], w_o * J[3:]])

            JtJ = J_w.T @ J_w
            dq = np.linalg.solve(JtJ + lam2 * np.eye(5), J_w.T @ e)

            max_dq = 0.3
            scale = np.max(np.abs(dq)) / max_dq
            if scale > 1.0:
                dq /= scale

            q += dq
            q = np.clip(q, self._joint_limits[:, 0], self._joint_limits[:, 1])

        T_final = self.fk(q)
        pos_err = np.linalg.norm(target_pos - T_final[:3, 3])
        ori_err = np.linalg.norm(_so3_log(target_rot @ T_final[:3, :3].T))
        return q, False, {"pos_err": pos_err, "ori_err": ori_err, "iters": self.max_iter}

    def solve(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        num_restarts: int = 8,
    ) -> Tuple[np.ndarray, bool, dict]:
        """Solve IK for a desired TCP pose.

        Uses multiple random restarts if the primary initial guess fails
        to converge, then returns the best solution.

        Parameters
        ----------
        target_pos : (3,) target position in base frame [metres].
        target_rot : (3, 3) target rotation matrix in base frame.
        q_init : (5,) initial joint angles.  If None, uses mid-range.
        num_restarts : int
            Number of random restarts to attempt if q_init does not converge.

        Returns
        -------
        q : (5,) solution joint angles [radians].
        success : bool – True if converged within tolerance.
        info : dict with ``pos_err``, ``ori_err``, ``iters``.
        """
        target_pos = np.asarray(target_pos, dtype=np.float64)
        target_rot = np.asarray(target_rot, dtype=np.float64).reshape(3, 3)

        # Primary attempt with user-provided initial guess
        if q_init is not None:
            q0 = np.asarray(q_init, dtype=np.float64)
        else:
            q0 = (self._joint_limits[:, 0] + self._joint_limits[:, 1]) / 2.0

        best_q, best_ok, best_info = self._solve_single(target_pos, target_rot, q0)
        if best_ok:
            return best_q, True, best_info

        best_cost = best_info["pos_err"] + 0.1 * best_info["ori_err"]

        # Random restarts
        rng = np.random.default_rng()
        lo = self._joint_limits[:, 0]
        hi = self._joint_limits[:, 1]
        for _ in range(num_restarts):
            q_rand = rng.uniform(lo, hi)
            q_sol, ok, info = self._solve_single(target_pos, target_rot, q_rand)
            if ok:
                return q_sol, True, info
            cost = info["pos_err"] + 0.1 * info["ori_err"]
            if cost < best_cost:
                best_q, best_ok, best_info = q_sol, ok, info
                best_cost = cost

        return best_q, best_ok, best_info
