import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
from sklearn.gaussian_process.kernels import RBF, WhiteKernel  # type: ignore

# MediaPipe / MANO landmark indices
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
INDEX_MCP = 5


def _get_gripper_orientation(thumb_tip, index_tip, vertices):
    gripper_dir = thumb_tip - index_tip
    palm_axis = vertices[INDEX_MCP] - (thumb_tip + index_tip) / 2.0
    x_axis = gripper_dir / max(np.linalg.norm(gripper_dir), 1e-10)
    z_axis = -palm_axis / max(np.linalg.norm(palm_axis), 1e-10)

    # Gram–Schmidt
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(np.linalg.norm(y_axis), 1e-10)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= max(np.linalg.norm(z_axis), 1e-10)

    if z_axis @ palm_axis > 0:
        x_axis, y_axis, z_axis = -x_axis, -y_axis, -z_axis

    ori = np.column_stack([x_axis, y_axis, z_axis])
    if np.linalg.det(ori) < 0:
        ori[:, 0] = -ori[:, 0]
    return ori


def _compute_frame_action(kpts_3d):
    thumb_tip = kpts_3d[THUMB_TIP]
    index_tip = kpts_3d[INDEX_TIP]
    middle_tip = kpts_3d[MIDDLE_TIP]
    grasp_pt = (thumb_tip + middle_tip) / 2.0
    gripper_ori = _get_gripper_orientation(thumb_tip, index_tip, kpts_3d)
    gripper_width = float(np.linalg.norm(thumb_tip - index_tip))
    return grasp_pt, gripper_ori, gripper_width


def gaussian_process_smoothing(pts):
    """GP regression with RBF + WhiteKernel on each dimension independently."""
    if len(pts) == 0:
        return pts
    time = np.arange(len(pts))[:, None]
    kernel = RBF(length_scale=1) + WhiteKernel(noise_level=1)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    if pts.ndim == 1:
        return gpr.fit(time, pts).predict(time)
    return np.column_stack(
        [gpr.fit(time, pts[:, i]).predict(time) for i in range(pts.shape[1])]
    )


def _gaussian_kernel(size, sigma):
    x = np.arange(size) - size // 2
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_slerp_smoothing(rot_mats, sigma=10.0, kernel_size=41):
    """Gaussian-weighted local SLERP smoothing for rotation matrices."""
    N = len(rot_mats)
    if N <= 1:
        return rot_mats

    half_k = kernel_size // 2
    quats = Rotation.from_matrix(rot_mats).as_quat()

    # hemisphere correction
    quats_fixed = [quats[0]]
    for i in range(1, N):
        q = quats[i]
        if np.dot(q, quats_fixed[-1]) < 0:
            q = -q
        quats_fixed.append(q)
    quats_fixed = np.array(quats_fixed)

    weights = _gaussian_kernel(kernel_size, sigma)
    smoothed = []
    for i in range(N):
        lo = max(0, i - half_k)
        hi = min(N, i + half_k + 1)
        local_q = quats_fixed[lo:hi]
        local_w = weights[half_k - (i - lo) : half_k + (hi - i)]
        local_w = local_w / local_w.sum()
        r_avg = Rotation.from_quat(local_q[0])
        for j in range(1, len(local_q)):
            r_next = Rotation.from_quat(local_q[j])
            w = local_w[j] / local_w[: j + 1].sum()
            r_avg = Slerp([0, 1], Rotation.concatenate([r_avg, r_next]))([w])[0]
        smoothed.append(r_avg.as_matrix())
    return np.stack(smoothed)


def compute_gripper_actions(
    hand_state_path: str,
    output_path: str,
    # min_open_ratio: float = 0.1,
):
    data = np.load(hand_state_path, allow_pickle=True)
    kpts_3d = data["kpts_3d"]
    hand_detected = data["hand_detected"]

    N = len(kpts_3d)
    print(f"     Processing {N} frames for gripper action computation")

    ee_pts = np.zeros((N, 3), dtype=np.float32)
    ee_oris = np.zeros((N, 3, 3), dtype=np.float32)
    ee_widths = np.zeros(N, dtype=np.float32)

    # per-frame raw actions
    for i in range(N):
        if not hand_detected[i]:
            continue
        pt, ori, width = _compute_frame_action(kpts_3d[i])
        ee_pts[i] = pt
        ee_oris[i] = ori
        ee_widths[i] = width

    # fill undetected frames by carry-forward
    det_idx = np.where(hand_detected)[0]
    if len(det_idx) == 0:
        print("     WARNING: no hand detected in any frame!")
        np.savez_compressed(
            output_path,
            ee_pts=ee_pts,
            ee_oris=ee_oris,
            ee_widths=ee_widths,
            hand_detected=hand_detected,
            max_width=np.float32(0.0),
        )
        return

    first = det_idx[0]
    last_pt, last_ori, last_w = ee_pts[first].copy(), ee_oris[first].copy(), ee_widths[first]
    for i in range(N):
        if hand_detected[i]:
            last_pt, last_ori, last_w = ee_pts[i].copy(), ee_oris[i].copy(), ee_widths[i]
        else:
            ee_pts[i], ee_oris[i], ee_widths[i] = last_pt, last_ori, last_w

    # GP smooth positions and widths
    ee_pts_s = gaussian_process_smoothing(ee_pts).astype(np.float32)
    ee_widths_s = gaussian_process_smoothing(ee_widths).astype(np.float32).ravel()

    # SLERP smooth orientations
    ee_oris_s = gaussian_slerp_smoothing(ee_oris, sigma=10.0, kernel_size=41).astype(np.float32)

    # threshold small widths to zero (disabled for now)
    # max_width = float(ee_widths_s.max()) if ee_widths_s.max() > 0 else 0.05
    # min_threshold = max_width * min_open_ratio
    # ee_widths_s[ee_widths_s < min_threshold] = 0.0

    np.savez_compressed(
        output_path,
        ee_pts=ee_pts_s,
        ee_oris=ee_oris_s,
        ee_widths=ee_widths_s,
        hand_detected=hand_detected,
        max_width=np.float32(ee_widths_s.max()),
    )
    print(f"     Saved gripper actions ({N} frames, max_width={ee_widths_s.max():.4f}m) to {output_path}")
