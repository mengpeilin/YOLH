"""Action post-processing: denormalisation, coordinate frame projection, rotation helpers."""

import numpy as np


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert rotation-6d representation to 3×3 rotation matrix (Gram-Schmidt)."""
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)  # (…, 3, 3) columns = basis


def matrix_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """Extract rotation-6d (first two rows) from a 3×3 rotation matrix.

    Matches the training-data convention in merge_episodes._mat_to_rot6d:
        rot6d = mat[:2, :].flatten()
    """
    return mat[..., :2, :].reshape(mat.shape[:-2] + (6,))


def unnormalize_action(action: np.ndarray, cfg: dict) -> np.ndarray:
    """Map action from [-1, 1] back to physical units.

    Expected action layout per step: [tx, ty, tz, rot6d(6), gripper_width] = 10.
    """
    action = action.copy()
    trans_min = cfg["trans_min"]
    trans_max = cfg["trans_max"]
    max_w = float(cfg["max_gripper_width"])
    if max_w <= 0:
        raise ValueError(f"max_gripper_width must be > 0, got {max_w}")

    action[..., :3] = (action[..., :3] + 1) / 2.0 * (trans_max - trans_min) + trans_min
    action[..., -1] = (action[..., -1] + 1) / 2.0 * max_w
    return action


def project_action_to_base(
    action_cam: np.ndarray, cam_to_base: np.ndarray
) -> np.ndarray:
    """Project action chunk from camera frame to robot base frame.

    Transforms position (first 3 dims) and rotation-6d (dims 3:9).
    Gripper width (dim 9) is preserved.

    Note on rotation convention:
        rot6d encodes the first two **rows** of the rotation matrix (training
        convention).  ``rot6d_to_matrix`` therefore gives R^T.  We transpose
        before applying the frame change R_cb, then re-encode as rows.
    """
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

        # Rotation: decode → transpose to get actual R → frame xform → re-encode
        r6 = action_cam[i, 3:9].astype(np.float64)
        R_cam = rot6d_to_matrix(r6).T          # actual rotation in camera frame
        R_base = R_cb @ R_cam                   # actual rotation in base frame
        out[i, 3:9] = R_base[:2, :].flatten()   # re-encode as first two rows

    return out.astype(np.float32)


def rot6d_angular_distance(r1: np.ndarray, r2: np.ndarray) -> float:
    """Angular distance (radians) between two rotation-6d vectors.

    Handles the row-encoding convention (rot6d_to_matrix gives R^T).
    Since angular distance is symmetric under transpose, no correction needed:
        trace(R1^T @ R2) == trace(R1 @ R2^T).
    """
    m1 = rot6d_to_matrix(r1)
    m2 = rot6d_to_matrix(r2)
    diff = m1 @ m2.T
    cos_val = np.clip((np.trace(diff) - 1) / 2.0, -1, 1)
    return float(np.arccos(cos_val))


def discretize_rotation(
    rot_begin: np.ndarray, rot_end: np.ndarray, step_size: float
) -> list:
    """Interpolate rotation-6d from *begin* to *end* in angular steps of *step_size*."""
    angle = rot6d_angular_distance(rot_begin, rot_end)
    n_steps = max(1, int(angle / step_size) + 1)
    steps = []
    for i in range(n_steps):
        alpha = (i + 1) / n_steps
        rot_i = rot_begin * (1 - alpha) + rot_end * alpha
        steps.append(rot_i)
    return steps
