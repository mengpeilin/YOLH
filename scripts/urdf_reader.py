import os
import numpy as np
import struct
import xml.etree.ElementTree as ET

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SO101_URDF_PATH = os.path.join(
    PROJECT_ROOT, "URDF", "SO-ARM100", "Simulation", "SO101", "so101_new_calib.urdf"
)

_GRIPPER_MODEL_CACHE = {}
_URDF_DATA_CACHE = {}


def rgbd_to_points(rgb, depth, intrinsic, mask=None):
    """Convert RGB-D image to point cloud."""
    fx, fy, cx, cy = intrinsic
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    depth_m = depth.astype(np.float32) / 1000.0
    valid = depth_m > 0

    # Exclude masked pixels.
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
    """Sample surface points from visual meshes of one URDF link."""
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


def _get_urdf_data(tip_sample_points=20000, contact_offset_z=0.05, z_band=0.005):
    """Parse URDF and cache gripper kinematics and TCP contact references."""
    global _URDF_DATA_CACHE
    cache_key = (int(tip_sample_points), float(contact_offset_z), float(z_band))
    if cache_key in _URDF_DATA_CACHE:
        return _URDF_DATA_CACHE[cache_key]

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

    # Compute TCP contact references from sampled jaw surfaces.
    rng = np.random.default_rng(seed=999)
    n_tip = int(tip_sample_points)
    contact_z = float(contact_offset_z)
    z_band_v = float(z_band)

    # Static jaw (part of gripper_link) -> gripper_frame
    static_pts_link = _sample_link_points(links["gripper_link"], urdf_dir, n_tip, rng)
    static_pts_frame = _transform_points(static_pts_link, t_frame_to_gripper)

    # Moving jaw in its local frame
    moving_pts_local = _sample_link_points(links["moving_jaw_so101_v1_link"], urdf_dir, n_tip, rng)

    # Transform moving jaw to gripper_frame at closed angle
    r_closed = _axis_angle_rot(jaw_axis, jaw_lower)
    t_closed = np.eye(4, dtype=np.float64)
    t_closed[:3, :3] = r_closed
    moving_pts_frame_closed = _transform_points(
        _transform_points(moving_pts_local, t_gripper_to_jaw @ t_closed),
        t_frame_to_gripper,
    )

    def _inner_contact_point(pts, other_centroid_x, z_tip):
        z_target = z_tip + contact_z
        zmask = np.abs(pts[:, 2] - z_target) < z_band_v
        if zmask.sum() < 5:
            dists = np.abs(pts[:, 2] - z_target)
            zmask = dists < np.percentile(dists, 10)
        cands = pts[zmask]
        # Pick the inner side along x.
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

    static_contact = _inner_contact_point(static_pts_frame, moving_cx, static_z_tip)

    moving_contact_frame = _inner_contact_point(
        moving_pts_frame_closed, static_cx, moving_z_tip)
    t_local_to_frame = t_frame_to_gripper @ t_gripper_to_jaw @ t_closed
    moving_contact_local = _transform_points(
        moving_contact_frame.reshape(1, 3),
        _invert_transform(t_local_to_frame),
    ).flatten()

    _URDF_DATA_CACHE[cache_key] = {
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
    return _URDF_DATA_CACHE[cache_key]


def _compute_tcp_in_frame(jaw_angle, tip_sample_points=20000, contact_offset_z=0.05, z_band=0.005):
    """Compute TCP in gripper_frame coordinates for one jaw angle."""
    d = _get_urdf_data(
        tip_sample_points=tip_sample_points,
        contact_offset_z=contact_offset_z,
        z_band=z_band,
    )
    static_pt = d["static_contact_in_frame"]

    pt_h = np.array([*d["moving_contact_in_link"], 1.0], dtype=np.float64)
    r_jaw = _axis_angle_rot(d["jaw_axis"], jaw_angle)
    t_rot = np.eye(4, dtype=np.float64)
    t_rot[:3, :3] = r_jaw
    pt_in_frame = d["t_frame_to_gripper"] @ d["t_gripper_to_jaw"] @ t_rot @ pt_h
    moving_pt = pt_in_frame[:3]

    return ((static_pt + moving_pt) / 2.0).astype(np.float64)


def width_to_jaw_angle(width, max_width, lower=None, upper=None):
    """Map fingertip width (m) to jaw angle (rad) linearly."""
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
    """Sample gripper surface points in gripper_frame at one jaw angle."""
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


def create_gripper_points(
    position,
    orientation,
    jaw_angle,
    num_points=1200,
    gripper_offset=None,
    tip_sample_points=20000,
    contact_offset_z=0.05,
    z_band=0.005,
):
    """Create SO-ARM100 gripper point cloud at the given pose."""
    gripper_local = _load_so101_gripper_points_in_frame(jaw_angle, int(num_points))

    tcp_local = _compute_tcp_in_frame(
        jaw_angle,
        tip_sample_points=tip_sample_points,
        contact_offset_z=contact_offset_z,
        z_band=z_band,
    )

    p_gripper = position - orientation @ tcp_local

    if gripper_offset is not None:
        p_gripper = p_gripper + np.asarray(gripper_offset, dtype=np.float64)

    all_pts = (orientation @ gripper_local.T).T + p_gripper
    colors = np.zeros_like(all_pts)
    return all_pts.astype(np.float32), colors.astype(np.float32)