import os
import numpy as np
import struct
import xml.etree.ElementTree as ET

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SO101_URDF_PATH = os.path.join(
    PROJECT_ROOT, "SO-ARM100", "Simulation", "SO101", "so101_new_calib.urdf"
)


_GRIPPER_MODEL_CACHE = {}
_URDF_DATA_CACHE = None


def rgbd_to_points(rgb, depth, intrinsic, mask=None):
    """
    Convert RGB-D image to point cloud.
    
    Args:
        rgb: (H, W, 3) uint8
        depth: (H, W) uint16, millimeters
        intrinsic: [fx, fy, cx, cy]
        mask: (H, W) bool - True = hand/arm region to EXCLUDE
    
    Returns:
        coords: (N, 3) float32
        colors: (N, 3) float32 in [0, 1]
    """
    fx, fy, cx, cy = intrinsic
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    depth_m = depth.astype(np.float32) / 1000.0
    valid = depth_m > 0

    # Apply mask: exclude hand/arm region
    if mask is not None:
        valid = valid & (~mask)

    z = depth_m[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy

    coords = np.stack([x, y, z], axis=-1).astype(np.float32)
    colors = rgb[valid].astype(np.float32) / 255.0
    return coords, colors


def _rpy_to_rot(rpy):
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _make_transform(xyz, rpy):
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = _rpy_to_rot(rpy)
    t[:3, 3] = xyz
    return t


def _invert_transform(t):
    inv = np.eye(4, dtype=np.float64)
    r = t[:3, :3]
    p = t[:3, 3]
    inv[:3, :3] = r.T
    inv[:3, 3] = -(r.T @ p)
    return inv


def _transform_points(points, t):
    if points.size == 0:
        return points
    return (t[:3, :3] @ points.T).T + t[:3, 3]


def _axis_angle_rot(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / n
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c
    return np.array([
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
    ], dtype=np.float64)


def _parse_origin(element):
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


def _read_stl_triangles(stl_path):
    with open(stl_path, "rb") as f:
        header = f.read(80)
        tri_count_bytes = f.read(4)
        if len(tri_count_bytes) < 4:
            raise ValueError(f"Invalid STL: {stl_path}")
        tri_count = struct.unpack("<I", tri_count_bytes)[0]
        expected_size = 84 + tri_count * 50
        f.seek(0, os.SEEK_END)
        file_size = f.tell()

    # Binary STL fast path.
    if file_size == expected_size:
        triangles = np.empty((tri_count, 3, 3), dtype=np.float64)
        with open(stl_path, "rb") as f:
            f.seek(84)
            for i in range(tri_count):
                data = f.read(50)
                if len(data) < 50:
                    raise ValueError(f"Unexpected EOF while reading {stl_path}")
                vals = struct.unpack("<12fH", data)
                triangles[i, 0] = vals[3:6]
                triangles[i, 1] = vals[6:9]
                triangles[i, 2] = vals[9:12]
        return triangles

    # ASCII STL fallback.
    vertices = []
    with open(stl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip().lower()
            if s.startswith("vertex"):
                parts = s.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if len(vertices) % 3 != 0 or len(vertices) == 0:
        raise ValueError(f"Could not parse STL triangles from {stl_path}")

    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)


def _sample_points_from_triangles(triangles, num_points, rng):
    if len(triangles) == 0 or num_points <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total_area = float(areas.sum())
    if total_area <= 1e-12:
        idx = rng.integers(0, len(triangles), size=num_points)
    else:
        probs = areas / total_area
        idx = rng.choice(len(triangles), size=num_points, p=probs)

    tris = triangles[idx]
    r1 = np.sqrt(rng.random(num_points))
    r2 = rng.random(num_points)
    points = (1.0 - r1)[:, None] * tris[:, 0, :] + \
             (r1 * (1.0 - r2))[:, None] * tris[:, 1, :] + \
             (r1 * r2)[:, None] * tris[:, 2, :]
    return points.astype(np.float32)


def _resolve_mesh_path(urdf_dir, mesh_file):
    cand = os.path.join(urdf_dir, mesh_file)
    if os.path.isfile(cand):
        return cand

    # Fallback for meshes renamed in STL folders.
    basename = os.path.basename(mesh_file)
    stl_root = os.path.join(PROJECT_ROOT, "SO-ARM100", "STL")
    for root, _, files in os.walk(stl_root):
        for f in files:
            if f.lower() == basename.lower():
                return os.path.join(root, f)
    raise FileNotFoundError(f"Mesh not found for URDF entry: {mesh_file}")


def _collect_link_visual_meshes(link_elem):
    meshes = []
    for visual in link_elem.findall("visual"):
        geom = visual.find("geometry")
        if geom is None:
            continue
        mesh = geom.find("mesh")
        if mesh is None:
            continue
        filename = mesh.get("filename")
        if not filename:
            continue
        xyz, rpy = _parse_origin(visual)
        meshes.append((filename, _make_transform(xyz, rpy)))
    return meshes


def _sample_link_points(link_elem, urdf_dir, n_samples, rng):
    """Sample *n_samples* surface points from the visual meshes of a URDF link (link-local frame)."""
    mesh_specs = _collect_link_visual_meshes(link_elem)
    if not mesh_specs:
        return np.zeros((0, 3), dtype=np.float64)
    tri_counts, tri_list, tf_list = [], [], []
    for mesh_file, t_mesh in mesh_specs:
        mesh_path = _resolve_mesh_path(urdf_dir, mesh_file)
        tri = _read_stl_triangles(mesh_path)
        tri_counts.append(max(len(tri), 1))
        tri_list.append(tri)
        tf_list.append(t_mesh)
    total = float(sum(tri_counts))
    pts_all, assigned = [], 0
    for i, tri in enumerate(tri_list):
        cnt = (n_samples - assigned if i == len(tri_list) - 1
               else max(1, int(round(n_samples * tri_counts[i] / total))))
        assigned += cnt
        pts = _sample_points_from_triangles(tri, cnt, rng)
        pts_all.append(_transform_points(pts.astype(np.float64), tf_list[i]))
    return np.concatenate(pts_all, axis=0) if pts_all else np.zeros((0, 3), dtype=np.float64)


def _get_urdf_data():
    """Parse URDF once and cache gripper kinematic parameters + jaw tip references."""
    global _URDF_DATA_CACHE
    if _URDF_DATA_CACHE is not None:
        return _URDF_DATA_CACHE

    if not os.path.isfile(SO101_URDF_PATH):
        raise FileNotFoundError(f"Missing URDF: {SO101_URDF_PATH}")

    urdf_dir = os.path.dirname(SO101_URDF_PATH)
    root = ET.parse(SO101_URDF_PATH).getroot()
    links = {lk.get("name"): lk for lk in root.findall("link") if lk.get("name")}
    joints = {jt.get("name"): jt for jt in root.findall("joint") if jt.get("name")}

    if "gripper_link" not in links or "moving_jaw_so101_v1_link" not in links:
        raise ValueError("URDF is missing gripper links.")
    if "gripper_frame_joint" not in joints or "gripper" not in joints:
        raise ValueError("URDF is missing required gripper joints.")

    # gripper_link <-> gripper_frame_link (fixed joint)
    gf_joint = joints["gripper_frame_joint"]
    xyz_gf, rpy_gf = _parse_origin(gf_joint)
    t_gripper_to_frame = _make_transform(xyz_gf, rpy_gf)
    t_frame_to_gripper = _invert_transform(t_gripper_to_frame)

    # gripper_link -> moving_jaw_link (revolute joint)
    jaw_joint = joints["gripper"]
    xyz_j, rpy_j = _parse_origin(jaw_joint)
    t_gripper_to_jaw = _make_transform(xyz_j, rpy_j)

    axis_elem = jaw_joint.find("axis")
    jaw_axis = (np.fromstring(axis_elem.get("xyz", "0 0 1"), sep=" ", dtype=np.float64)
                if axis_elem is not None else np.array([0.0, 0.0, 1.0], dtype=np.float64))

    limit_elem = jaw_joint.find("limit")
    jaw_lower = float(limit_elem.get("lower", "0")) if limit_elem is not None else 0.0
    jaw_upper = float(limit_elem.get("upper", "0")) if limit_elem is not None else 0.0

    # --- Compute inner-surface contact reference points for TCP ---
    # TCP = midpoint of two points on the inner contact face of each finger,
    #       each located 2 cm from the respective fingertip.
    rng = np.random.default_rng(seed=999)
    N_TIP = 20000
    CONTACT_OFFSET_Z = 0.05   # 5 cm from fingertip toward gripper body
    Z_BAND = 0.005            # ±5 mm selection band

    # Static jaw (part of gripper_link) -> gripper_frame
    static_pts_link = _sample_link_points(links["gripper_link"], urdf_dir, N_TIP, rng)
    static_pts_frame = _transform_points(static_pts_link, t_frame_to_gripper)

    # Moving jaw in its local frame
    moving_pts_local = _sample_link_points(links["moving_jaw_so101_v1_link"], urdf_dir, N_TIP, rng)

    # Transform moving jaw to gripper_frame at closed angle
    r_closed = _axis_angle_rot(jaw_axis, jaw_lower)
    t_closed = np.eye(4, dtype=np.float64)
    t_closed[:3, :3] = r_closed
    moving_pts_frame_closed = _transform_points(
        _transform_points(moving_pts_local, t_gripper_to_jaw @ t_closed),
        t_frame_to_gripper,
    )

    def _inner_contact_point(pts, other_centroid_x, z_tip):
        """Inner-surface point at CONTACT_OFFSET_Z from fingertip."""
        z_target = z_tip + CONTACT_OFFSET_Z
        zmask = np.abs(pts[:, 2] - z_target) < Z_BAND
        if zmask.sum() < 5:  # widen band if too few points
            dists = np.abs(pts[:, 2] - z_target)
            zmask = dists < np.percentile(dists, 10)
        cands = pts[zmask]
        # Inner 30 %: side closest to the opposing jaw along x
        cx = cands[:, 0]
        if other_centroid_x > cx.mean():
            inner = cands[cx >= np.percentile(cx, 70)]
        else:
            inner = cands[cx <= np.percentile(cx, 30)]
        return inner.mean(axis=0) if len(inner) > 0 else cands.mean(axis=0)

    static_z_tip = static_pts_frame[:, 2].min()
    moving_z_tip = moving_pts_frame_closed[:, 2].min()
    static_cx = static_pts_frame[:, 0].mean()
    moving_cx = moving_pts_frame_closed[:, 0].mean()

    # Static jaw contact point (in gripper_frame)
    static_contact = _inner_contact_point(static_pts_frame, moving_cx, static_z_tip)

    # Moving jaw contact point — compute in gripper_frame then map to local frame
    moving_contact_frame = _inner_contact_point(
        moving_pts_frame_closed, static_cx, moving_z_tip)
    t_local_to_frame = t_frame_to_gripper @ t_gripper_to_jaw @ t_closed
    moving_contact_local = _transform_points(
        moving_contact_frame.reshape(1, 3),
        _invert_transform(t_local_to_frame),
    ).flatten()

    _URDF_DATA_CACHE = {
        "urdf_dir": urdf_dir,
        "links": links,
        "t_gripper_to_frame": t_gripper_to_frame,
        "t_frame_to_gripper": t_frame_to_gripper,
        "t_gripper_to_jaw": t_gripper_to_jaw,
        "jaw_axis": jaw_axis,
        "jaw_lower": jaw_lower,
        "jaw_upper": jaw_upper,
        "static_contact_in_frame": static_contact.astype(np.float64),
        "moving_contact_in_link": moving_contact_local.astype(np.float64),
    }
    return _URDF_DATA_CACHE


def _compute_tcp_in_frame(jaw_angle):
    """
    Compute TCP in gripper_frame coordinates for a given jaw angle (rad).

    TCP = midpoint of two inner-surface contact points, each located
    2 cm from the respective fingertip.
    """
    d = _get_urdf_data()
    static_pt = d["static_contact_in_frame"]

    # Transform moving contact point from local frame to gripper_frame
    pt_h = np.array([*d["moving_contact_in_link"], 1.0], dtype=np.float64)
    r_jaw = _axis_angle_rot(d["jaw_axis"], jaw_angle)
    t_rot = np.eye(4, dtype=np.float64)
    t_rot[:3, :3] = r_jaw
    pt_in_frame = d["t_frame_to_gripper"] @ d["t_gripper_to_jaw"] @ t_rot @ pt_h
    moving_pt = pt_in_frame[:3]

    return ((static_pt + moving_pt) / 2.0).astype(np.float64)


def width_to_jaw_angle(width, max_width, lower=None, upper=None):
    """
    Linear mapping from hand fingertip width (m) to jaw angle (rad).
    width=0 -> jaw=lower (fully closed), width=max_width -> jaw=upper (fully open).
    """
    if lower is None or upper is None:
        d = _get_urdf_data()
        if lower is None:
            lower = d["jaw_lower"]
        if upper is None:
            upper = d["jaw_upper"]
    if max_width <= 0:
        return float(lower)
    ratio = float(np.clip(width / max_width, 0.0, 1.0))
    return lower + ratio * (upper - lower)


def _load_so101_gripper_points_in_frame(jaw_angle, num_points):
    """
    Sample gripper surface points in gripper_frame at a given jaw angle (rad).
    Cached per quantized angle (0.02 rad ≈ 1°) and point count.
    """
    angle_q = round(float(jaw_angle), 2)
    cache_key = (angle_q, int(num_points))
    if cache_key in _GRIPPER_MODEL_CACHE:
        return _GRIPPER_MODEL_CACHE[cache_key]

    d = _get_urdf_data()

    r_jaw = _axis_angle_rot(d["jaw_axis"], jaw_angle)
    t_jaw_rot = np.eye(4, dtype=np.float64)
    t_jaw_rot[:3, :3] = r_jaw

    rng = np.random.default_rng(int(abs(angle_q * 100)) % (2**31))
    n_static = max(1, int(num_points * 0.6))
    n_moving = max(1, num_points - n_static)

    # Static gripper body in gripper_link frame
    pts_gripper = _sample_link_points(
        d["links"]["gripper_link"], d["urdf_dir"], n_static, rng
    ).astype(np.float64)

    # Moving jaw: apply joint rotation, then transform to gripper_link frame
    pts_jaw_child = _sample_link_points(
        d["links"]["moving_jaw_so101_v1_link"], d["urdf_dir"], n_moving, rng
    ).astype(np.float64)
    pts_jaw_in_gripper = _transform_points(
        pts_jaw_child, d["t_gripper_to_jaw"] @ t_jaw_rot
    )

    pts_link = np.concatenate([pts_gripper, pts_jaw_in_gripper], axis=0)
    pts_frame = _transform_points(pts_link, d["t_frame_to_gripper"]).astype(np.float32)
    _GRIPPER_MODEL_CACHE[cache_key] = pts_frame
    return pts_frame


def create_gripper_points(position, orientation, jaw_angle, num_points=1200, gripper_offset=None):
    """
    Create SO-ARM100 gripper point cloud at the given pose.

    The gripper is positioned so that its TCP (two-finger contact midpoint)
    aligns with *position*.  This accounts for the offset between the
    gripper_frame origin and the actual contact point at the given jaw angle.

    Args:
        position:    (3,)   target TCP / grasp-point position
        orientation: (3,3)  rotation matrix
        jaw_angle:   float  jaw joint angle in radians
        num_points:  int    number of sampled mesh surface points
        gripper_offset: (3,) optional position offset in world frame [m]

    Returns:
        coords: (N, 3) float32
        colors: (N, 3) float32 (all zeros = black)
    """
    gripper_local = _load_so101_gripper_points_in_frame(jaw_angle, int(num_points))

    # Compute TCP offset to align contact midpoint with target position
    tcp_local = _compute_tcp_in_frame(jaw_angle)

    # Shift gripper origin: position = R @ tcp_local + p_gripper
    # => p_gripper = position - R @ tcp_local
    p_gripper = position - orientation @ tcp_local
    
    # Apply additional gripper offset if provided
    if gripper_offset is not None:
        p_gripper = p_gripper + np.asarray(gripper_offset, dtype=np.float64)

    # Transform all gripper points to world / camera frame
    all_pts = (orientation @ gripper_local.T).T + p_gripper
    colors = np.zeros_like(all_pts)
    return all_pts.astype(np.float32), colors.astype(np.float32)


def voxelize_point_cloud(coords, colors, voxel_size=100, coord_bounds=None):
    """
    Voxelize point cloud into a 3D grid.
    
    Args:
        coords: (N, 3) point positions
        colors: (N, 3) RGB colors [0, 1]
        voxel_size: grid resolution per axis
        coord_bounds: [x_min, y_min, z_min, x_max, y_max, z_max]
    
    Returns:
        voxel_grid: (voxel_size, voxel_size, voxel_size, 4) - RGB + occupancy
    """
    if coord_bounds is None:
        # Auto-compute bounds with some padding
        if len(coords) == 0:
            return np.zeros((voxel_size, voxel_size, voxel_size, 4), dtype=np.float32)
        mins = coords.min(axis=0) - 0.05
        maxs = coords.max(axis=0) + 0.05
        coord_bounds = np.concatenate([mins, maxs])

    bb_mins = coord_bounds[:3]
    bb_maxs = coord_bounds[3:]

    # Ignore points outside coord_bounds instead of clamping them to boundary voxels.
    in_bounds = np.all((coords >= bb_mins) & (coords < bb_maxs), axis=1)
    if not np.any(in_bounds):
        return np.zeros((voxel_size, voxel_size, voxel_size, 4), dtype=np.float32)

    coords = coords[in_bounds]
    colors = colors[in_bounds]

    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / voxel_size

    # Compute voxel indices for in-bound points.
    indices = np.floor((coords - bb_mins) / (res + 1e-12)).astype(np.int32)
    indices = np.clip(indices, 0, voxel_size - 1)

    # Accumulate colors and counts
    voxel_grid = np.zeros((voxel_size, voxel_size, voxel_size, 4), dtype=np.float32)
    count_grid = np.zeros((voxel_size, voxel_size, voxel_size, 1), dtype=np.float32)

    for idx in range(len(coords)):
        xi, yi, zi = indices[idx]
        voxel_grid[xi, yi, zi, :3] += colors[idx]
        count_grid[xi, yi, zi, 0] += 1.0

    # Average colors where occupied
    occupied = count_grid[..., 0] > 0
    voxel_grid[occupied, :3] /= count_grid[occupied]
    voxel_grid[occupied, 3] = 1.0  # occupancy flag

    return voxel_grid


def build_training_episodes(
    raw_npz_path: str,
    masks_npz_path: str,
    gripper_action_path: str,
    keyframes_path: str,
    output_path: str,
    voxel_size: int = 100,
    coord_bounds: list = None,
    gripper_offset: list = None,
):
    """
    Build training episodes from pipeline outputs.

    Each episode = (voxel_grid, action) at a keyframe.
    Action = [x, y, z, r00..r22, width_norm]  (13 dimensions)
    where width_norm = ee_width / max_width ∈ [0, 1].
    
    Args:
        gripper_offset: [dx, dy, dz] offset to apply to gripper position (m)
    """
    # Load data
    raw = np.load(raw_npz_path, allow_pickle=True)
    rgb_frames = raw["rgb"]           # (N, H, W, 3)
    depth_frames = raw["depth"]       # (N, H, W)
    intrinsic = raw["intrinsic"]      # [fx, fy, cx, cy]

    masks_data = np.load(masks_npz_path, allow_pickle=True)
    masks = masks_data["arm_hand_masks"] if "arm_hand_masks" in masks_data else masks_data["masks"]

    action_data = np.load(gripper_action_path, allow_pickle=True)
    positions = action_data["ee_pts"]         # (N, 3)
    orientations = action_data["ee_oris"]     # (N, 3, 3)
    ee_widths = action_data["ee_widths"]      # (N,) continuous width in metres
    pose_valid = action_data["hand_detected"] # (N,) bool
    keyframes = np.load(keyframes_path)       # (K,) int

    # max_width for normalisation (saved by gripper_action step)
    max_width = float(action_data["max_width"]) if "max_width" in action_data else 0.0
    if max_width <= 0:
        valid_w = ee_widths[pose_valid] if pose_valid.any() else ee_widths
        max_width = float(np.percentile(valid_w[valid_w > 0], 95)) if (valid_w > 0).any() else 0.05

    # URDF jaw limits for width → angle mapping
    urdf_data = _get_urdf_data()
    jaw_lower = urdf_data["jaw_lower"]
    jaw_upper = urdf_data["jaw_upper"]

    if coord_bounds is not None:
        coord_bounds = np.array(coord_bounds, dtype=np.float32)

    print(f"[06] Building episodes for {len(keyframes)} keyframes")
    print(f"     Voxel size: {voxel_size}^3, max_width: {max_width:.4f}m")
    if gripper_offset is not None:
        print(f"     Gripper offset: {gripper_offset}")

    episode_voxel_grids = []
    episode_actions = []
    episode_proprios = []
    episode_keyframe_indices = []

    # We drop the first valid keyframe and use its action as the seed for
    # "previous keyframe action" so no zero-filled proprio is needed.
    prev_action = None
    dropped_first_valid = False

    # Auto-detect coordinate bounds from all keyframe point clouds if not specified
    if coord_bounds is None:
        all_coords = []
        for kf_idx in keyframes:
            c, _ = rgbd_to_points(rgb_frames[kf_idx], depth_frames[kf_idx],
                                   intrinsic, masks[kf_idx])
            if len(c) > 0:
                all_coords.append(c)
        if all_coords:
            all_coords = np.concatenate(all_coords, axis=0)
            mins = np.percentile(all_coords, 1, axis=0) - 0.05
            maxs = np.percentile(all_coords, 99, axis=0) + 0.05
            coord_bounds = np.concatenate([mins, maxs]).astype(np.float32)
            print(f"     Auto bounds: {coord_bounds}")
        else:
            coord_bounds = np.array([-0.5, -0.5, 0.0, 0.5, 0.5, 1.0], dtype=np.float32)

    for kf_idx in keyframes:
        if not pose_valid[kf_idx]:
            print(f"     Skipping keyframe {kf_idx}: no valid hand pose")
            continue

        # 1. Build point cloud from RGB-D, excluding hand/arm mask
        coords, colors = rgbd_to_points(
            rgb_frames[kf_idx], depth_frames[kf_idx],
            intrinsic, masks[kf_idx]
        )

        if len(coords) == 0:
            print(f"     Skipping keyframe {kf_idx}: empty point cloud")
            continue

        # 2. Compute jaw angle from continuous width
        w = float(ee_widths[kf_idx])
        jaw_angle = width_to_jaw_angle(w, max_width, jaw_lower, jaw_upper)
        width_norm = float(np.clip(w / max_width, 0.0, 1.0)) if max_width > 0 else 0.0

        # 3. Create gripper points at hand pose with TCP compensation
        gripper_coords, gripper_colors = create_gripper_points(
            positions[kf_idx], orientations[kf_idx], jaw_angle, gripper_offset=gripper_offset
        )

        # 4. Merge scene + gripper points
        all_coords = np.concatenate([coords, gripper_coords], axis=0)
        all_colors = np.concatenate([colors, gripper_colors], axis=0)

        # 5. Voxelize
        voxel_grid = voxelize_point_cloud(all_coords, all_colors,
                                           voxel_size=voxel_size,
                                           coord_bounds=coord_bounds)

        # 6. Build action vector:
        # [x, y, z, r00..r22, width_norm]  (13 dims)
        rot = orientations[kf_idx].flatten()  # 9 elements
        action = np.concatenate([
            positions[kf_idx],       # 3: position (= TCP target)
            rot,                      # 9: rotation matrix flattened
            [width_norm],             # 1: normalised gripper width [0, 1]
        ]).astype(np.float32)

        if prev_action is None:
            prev_action = action.copy()
            dropped_first_valid = True
            print(f"     Dropping first valid keyframe {kf_idx} to avoid zero-filled proprio")
            continue

        episode_voxel_grids.append(voxel_grid)
        episode_actions.append(action)
        episode_proprios.append(prev_action.copy())  # proprio = previous keyframe's action
        episode_keyframe_indices.append(kf_idx)
        prev_action = action.copy()  # advance for next keyframe

    if not dropped_first_valid:
        print("     WARNING: No valid keyframe found to seed previous-action proprio!")
        return None

    if len(episode_voxel_grids) == 0:
        print("     WARNING: No valid episodes generated!")
        return None

    episode_voxel_grids = np.array(episode_voxel_grids, dtype=np.float32)
    episode_actions = np.array(episode_actions, dtype=np.float32)
    episode_proprios = np.array(episode_proprios, dtype=np.float32)
    episode_keyframe_indices = np.array(episode_keyframe_indices, dtype=np.int64)

    np.savez_compressed(
        output_path,
        voxel_grids=episode_voxel_grids,
        actions=episode_actions,
        proprios=episode_proprios,
        keyframe_indices=episode_keyframe_indices,
        coord_bounds=coord_bounds,
        voxel_size=voxel_size,
        max_width=max_width,
    )
    print(f"     Saved {len(episode_voxel_grids)} episodes to {output_path}")
    print(f"     Voxel grid shape: {episode_voxel_grids.shape}")
    print(f"     Action shape: {episode_actions.shape}")
    return output_path