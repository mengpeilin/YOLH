import os
import numpy as np
import struct
import xml.etree.ElementTree as ET

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SO101_URDF_PATH = os.path.join(
    PROJECT_ROOT, "SO-ARM100", "Simulation", "SO101", "so101_new_calib.urdf"
)


_GRIPPER_MODEL_CACHE = {}


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


def _load_so101_gripper_points_in_frame(gripper_open, num_points):
    cache_key = (bool(gripper_open), int(num_points))
    if cache_key in _GRIPPER_MODEL_CACHE:
        return _GRIPPER_MODEL_CACHE[cache_key]

    if not os.path.isfile(SO101_URDF_PATH):
        raise FileNotFoundError(f"Missing URDF: {SO101_URDF_PATH}")

    urdf_dir = os.path.dirname(SO101_URDF_PATH)
    root = ET.parse(SO101_URDF_PATH).getroot()

    links = {link.get("name"): link for link in root.findall("link") if link.get("name")}
    joints = {joint.get("name"): joint for joint in root.findall("joint") if joint.get("name")}

    if "gripper_link" not in links or "moving_jaw_so101_v1_link" not in links:
        raise ValueError("URDF is missing gripper links.")
    if "gripper_frame_joint" not in joints or "gripper" not in joints:
        raise ValueError("URDF is missing required gripper joints.")

    # Fixed frame offset: gripper_link -> gripper_frame_link
    gripper_frame_joint = joints["gripper_frame_joint"]
    xyz_gf, rpy_gf = _parse_origin(gripper_frame_joint)
    t_gripper_to_frame = _make_transform(xyz_gf, rpy_gf)
    t_frame_to_gripper = _invert_transform(t_gripper_to_frame)

    # Revolute moving jaw: gripper_link -> moving_jaw
    jaw_joint = joints["gripper"]
    xyz_j, rpy_j = _parse_origin(jaw_joint)
    t_gripper_to_jaw = _make_transform(xyz_j, rpy_j)
    axis_elem = jaw_joint.find("axis")
    axis = np.fromstring(axis_elem.get("xyz", "0 0 1"), sep=" ", dtype=np.float64) \
        if axis_elem is not None else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    limit_elem = jaw_joint.find("limit")
    lower = float(limit_elem.get("lower", "0")) if limit_elem is not None else 0.0
    upper = float(limit_elem.get("upper", "0")) if limit_elem is not None else 0.0
    jaw_angle = upper if gripper_open else lower
    r_jaw = _axis_angle_rot(axis, jaw_angle)
    t_jaw_rot = np.eye(4, dtype=np.float64)
    t_jaw_rot[:3, :3] = r_jaw

    rng = np.random.default_rng(123 if gripper_open else 456)
    n_static = max(1, int(num_points * 0.6))
    n_moving = max(1, num_points - n_static)

    def load_link_points(link_name, n_samples):
        link = links[link_name]
        mesh_specs = _collect_link_visual_meshes(link)
        if not mesh_specs:
            return np.zeros((0, 3), dtype=np.float32)

        # Sample proportionally to triangle count per mesh.
        tri_counts = []
        tri_list = []
        tf_list = []
        for mesh_file, t_link_to_mesh in mesh_specs:
            mesh_path = _resolve_mesh_path(urdf_dir, mesh_file)
            tri = _read_stl_triangles(mesh_path)
            tri_counts.append(max(len(tri), 1))
            tri_list.append(tri)
            tf_list.append(t_link_to_mesh)

        total = float(sum(tri_counts))
        points_all = []
        assigned = 0
        for i, tri in enumerate(tri_list):
            count = n_samples - assigned if i == len(tri_list) - 1 else max(1, int(round(n_samples * tri_counts[i] / total)))
            assigned += count
            pts_mesh = _sample_points_from_triangles(tri, count, rng)
            pts_link = _transform_points(pts_mesh, tf_list[i])
            points_all.append(pts_link)

        return np.concatenate(points_all, axis=0) if points_all else np.zeros((0, 3), dtype=np.float32)

    # Static gripper body in gripper_link frame.
    pts_gripper = load_link_points("gripper_link", n_static).astype(np.float64)

    # Moving jaw transformed into gripper_link frame.
    pts_jaw_child = load_link_points("moving_jaw_so101_v1_link", n_moving).astype(np.float64)
    pts_jaw_in_gripper = _transform_points(pts_jaw_child, t_gripper_to_jaw @ t_jaw_rot)

    pts_link = np.concatenate([pts_gripper, pts_jaw_in_gripper], axis=0)

    # Express model points in gripper_frame coordinates so caller pose can be that frame.
    pts_frame = _transform_points(pts_link, t_frame_to_gripper).astype(np.float32)
    _GRIPPER_MODEL_CACHE[cache_key] = pts_frame
    return pts_frame


def create_gripper_points(position, orientation, gripper_open,
                          num_points=1200, gripper_length=0.06, gripper_width=0.04):
    """
    Create SO-ARM100 gripper point cloud from URDF + STL meshes at the given pose.
    The inserted gripper is rendered as pure black (color=[0,0,0]).
    
    Args:
        position: (3,) hand center position
        orientation: (3,3) rotation matrix
        gripper_open: bool
        num_points: number of sampled mesh surface points
        gripper_length: unused, kept for API compatibility
        gripper_width: unused, kept for API compatibility
    
    Returns:
        coords: (N, 3) float32
        colors: (N, 3) float32 (all zeros = black)
    """
    _ = (gripper_length, gripper_width)
    gripper_local = _load_so101_gripper_points_in_frame(bool(gripper_open), int(num_points))

    # Transform from gripper frame to camera/world frame.
    all_pts = (orientation @ gripper_local.T).T + position

    # All black color
    colors = np.zeros_like(all_pts)
    return all_pts, colors


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
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / voxel_size

    # Compute voxel indices
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
    hand_pose_path: str,
    hand_states_path: str,
    keyframes_path: str,
    output_path: str,
    voxel_size: int = 100,
    coord_bounds: list = None,
):
    """
    Build training episodes from pipeline outputs.
    
    Each episode = (voxel_grid, action) at a keyframe.
    Action = [x, y, z, r00, r01, r02, r10, r11, r12, r20, r21, r22, gripper_open]
    """
    # Load data
    raw = np.load(raw_npz_path, allow_pickle=True)
    rgb_frames = raw["rgb"]           # (N, H, W, 3)
    depth_frames = raw["depth"]       # (N, H, W)
    intrinsic = raw["intrinsic"]      # [fx, fy, cx, cy]

    masks_data = np.load(masks_npz_path, allow_pickle=True)
    masks = masks_data["masks"]       # (N, H, W) bool

    pose_data = np.load(hand_pose_path)
    positions = pose_data["positions"]       # (N, 3)
    orientations = pose_data["orientations"] # (N, 3, 3)
    pose_valid = pose_data["valid"]          # (N,) bool

    hand_open = np.load(hand_states_path)    # (N,) bool
    keyframes = np.load(keyframes_path)      # (K,) int

    if coord_bounds is not None:
        coord_bounds = np.array(coord_bounds, dtype=np.float32)

    print(f"[05] Building episodes for {len(keyframes)} keyframes")
    print(f"     Voxel size: {voxel_size}^3")

    episode_voxel_grids = []
    episode_actions = []
    episode_keyframe_indices = []

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

        # 2. Create gripper points at hand pose
        gripper_coords, gripper_colors = create_gripper_points(
            positions[kf_idx], orientations[kf_idx],
            hand_open[kf_idx]
        )

        # 3. Merge scene + gripper points
        all_coords = np.concatenate([coords, gripper_coords], axis=0)
        all_colors = np.concatenate([colors, gripper_colors], axis=0)

        # 4. Voxelize
        voxel_grid = voxelize_point_cloud(all_coords, all_colors,
                                           voxel_size=voxel_size,
                                           coord_bounds=coord_bounds)

        # 5. Build action vector:
        # [x, y, z, r00, r01, r02, r10, r11, r12, r20, r21, r22, gripper_open]
        rot = orientations[kf_idx].flatten()  # 9 elements
        action = np.concatenate([
            positions[kf_idx],              # 3: position
            rot,                             # 9: rotation matrix flattened
            [float(hand_open[kf_idx])],     # 1: gripper state
        ]).astype(np.float32)

        episode_voxel_grids.append(voxel_grid)
        episode_actions.append(action)
        episode_keyframe_indices.append(kf_idx)

    if len(episode_voxel_grids) == 0:
        print("     WARNING: No valid episodes generated!")
        return None

    episode_voxel_grids = np.array(episode_voxel_grids, dtype=np.float32)
    episode_actions = np.array(episode_actions, dtype=np.float32)
    episode_keyframe_indices = np.array(episode_keyframe_indices, dtype=np.int64)

    np.savez_compressed(
        output_path,
        voxel_grids=episode_voxel_grids,
        actions=episode_actions,
        keyframe_indices=episode_keyframe_indices,
        coord_bounds=coord_bounds,
        voxel_size=voxel_size,
    )
    print(f"     Saved {len(episode_voxel_grids)} episodes to {output_path}")
    print(f"     Voxel grid shape: {episode_voxel_grids.shape}")
    print(f"     Action shape: {episode_actions.shape}")
    return output_path