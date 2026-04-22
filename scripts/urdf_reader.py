import os
import numpy as np
import struct
import xml.etree.ElementTree as ET

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SO101_URDF_PATH = os.path.join(
    PROJECT_ROOT, "URDF", "SO-ARM100", "Simulation", "SO101", "so101_new_calib.urdf"
)
ROBOTIQ_2F85_URDF_PATH = os.path.join(
    PROJECT_ROOT,
    "URDF",
    "robotiq_arg85_description",
    "robots",
    "robotiq_arg85_description.URDF",
)

_GRIPPER_MODEL_CACHE = {}
_URDF_DATA_CACHE = {}
_ROBOTIQ_DATA_CACHE = {}

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
    if mesh_file.startswith("package://"):
        rel = mesh_file[len("package://"):]
        package_name, _, subpath = rel.partition("/")
        package_root = os.path.dirname(urdf_dir)

        package_candidates = []
        if package_name and os.path.basename(package_root) == package_name:
            package_candidates.append(os.path.join(package_root, subpath))
        if package_name:
            package_candidates.append(os.path.join(PROJECT_ROOT, "URDF", package_name, subpath))

        for cand in package_candidates:
            if os.path.isfile(cand):
                return cand

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


def _normalize_gripper_type(gripper_type):
    gt = str(gripper_type or "so101").strip().lower()
    aliases = {
        "so-arm100": "so101",
        "so_arm100": "so101",
        "robotiq": "robotiq_2f85",
        "robotiq2f85": "robotiq_2f85",
        "robotiq_2f_85": "robotiq_2f85",
        "2f85": "robotiq_2f85",
    }
    return aliases.get(gt, gt)


def _resolve_urdf_path(urdf_path, default_path):
    if urdf_path is None:
        path = default_path
    elif os.path.isabs(urdf_path):
        path = urdf_path
    else:
        path = os.path.join(PROJECT_ROOT, urdf_path)
    return os.path.abspath(path)


def _parse_joint_info(joint_elem):
    xyz, rpy = _parse_origin(joint_elem)
    axis_elem = joint_elem.find("axis")
    axis = (
        np.fromstring(axis_elem.get("xyz", "0 0 1"), sep=" ", dtype=np.float64)
        if axis_elem is not None
        else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )
    if axis.size != 3:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    limit_elem = joint_elem.find("limit")
    lower = float(limit_elem.get("lower", "0")) if limit_elem is not None else 0.0
    upper = float(limit_elem.get("upper", "0")) if limit_elem is not None else 0.0

    mimic_elem = joint_elem.find("mimic")
    mimic = None
    if mimic_elem is not None:
        mimic = {
            "joint": mimic_elem.get("joint"),
            "multiplier": float(mimic_elem.get("multiplier", "1")),
            "offset": float(mimic_elem.get("offset", "0")),
        }

    return {
        "name": joint_elem.get("name"),
        "type": joint_elem.get("type", "fixed"),
        "parent": joint_elem.find("parent").get("link"),
        "child": joint_elem.find("child").get("link"),
        "origin": _make_transform(xyz, rpy),
        "axis": axis,
        "lower": lower,
        "upper": upper,
        "mimic": mimic,
    }


def _resolve_joint_angle(joint_info, master_angles):
    if joint_info["type"] not in ("revolute", "continuous"):
        return 0.0

    if joint_info["name"] in master_angles:
        return float(master_angles[joint_info["name"]])

    mimic = joint_info.get("mimic")
    if mimic is not None and mimic.get("joint") in master_angles:
        return float(
            mimic["multiplier"] * master_angles[mimic["joint"]] + mimic["offset"]
        )

    return 0.0


def _compute_link_transforms(base_link, child_joints, master_angles):
    transforms = {base_link: np.eye(4, dtype=np.float64)}

    def dfs(parent_link):
        parent_t = transforms[parent_link]
        for joint_info in child_joints.get(parent_link, []):
            child_t = parent_t @ joint_info["origin"]
            if joint_info["type"] in ("revolute", "continuous"):
                angle = _resolve_joint_angle(joint_info, master_angles)
                rot = np.eye(4, dtype=np.float64)
                rot[:3, :3] = _axis_angle_rot(joint_info["axis"], angle)
                child_t = child_t @ rot
            transforms[joint_info["child"]] = child_t
            dfs(joint_info["child"])

    dfs(base_link)
    return transforms


def _pick_inner_contact_point(pts, z_tip, contact_offset_z, z_band):
    z_target = z_tip - contact_offset_z
    zmask = np.abs(pts[:, 2] - z_target) < z_band
    if zmask.sum() < 5:
        dists = np.abs(pts[:, 2] - z_target)
        zmask = dists < np.percentile(dists, 10)
    cands = pts[zmask]
    cx = cands[:, 0]
    if cands[:, 0].mean() >= 0:
        inner = cands[cx <= np.percentile(cx, 30)]
    else:
        inner = cands[cx >= np.percentile(cx, 70)]
    return inner.mean(axis=0) if len(inner) > 0 else cands.mean(axis=0)


def _get_robotiq_data(urdf_path=None, tip_sample_points=20000, contact_offset_z=0.05, z_band=0.005):
    global _ROBOTIQ_DATA_CACHE

    resolved_urdf_path = _resolve_urdf_path(urdf_path, ROBOTIQ_2F85_URDF_PATH)
    cache_key = (
        resolved_urdf_path,
        int(tip_sample_points),
        float(contact_offset_z),
        float(z_band),
    )
    if cache_key in _ROBOTIQ_DATA_CACHE:
        return _ROBOTIQ_DATA_CACHE[cache_key]

    if not os.path.isfile(resolved_urdf_path):
        raise FileNotFoundError(f"Missing URDF: {resolved_urdf_path}")

    urdf_dir = os.path.dirname(resolved_urdf_path)
    root = ET.parse(resolved_urdf_path).getroot()
    links = {lk.get("name"): lk for lk in root.findall("link") if lk.get("name")}
    joint_list = [_parse_joint_info(jt) for jt in root.findall("joint") if jt.get("name")]
    joints = {jt["name"]: jt for jt in joint_list}
    child_joints = {}
    for jt in joint_list:
        child_joints.setdefault(jt["parent"], []).append(jt)

    base_link = "robotiq_85_base_link"
    master_joint = "finger_joint"
    if base_link not in links or master_joint not in joints:
        raise ValueError("Robotiq URDF is missing robotiq_85_base_link or finger_joint")

    render_links = [
        "robotiq_85_base_link",
        "left_outer_knuckle",
        "left_outer_finger",
        "left_inner_knuckle",
        "left_inner_finger",
        "right_inner_knuckle",
        "right_inner_finger",
        "right_outer_knuckle",
        "right_outer_finger",
    ]

    rng = np.random.default_rng(seed=4242)
    render_points_local = {}
    for link_name in render_links:
        if link_name in links:
            render_points_local[link_name] = _sample_link_points(
                links[link_name], urdf_dir, 900, rng
            ).astype(np.float64)

    contact_rng = np.random.default_rng(seed=4343)
    left_contact_local = _sample_link_points(
        links["left_outer_finger"], urdf_dir, int(tip_sample_points), contact_rng
    ).astype(np.float64)
    right_contact_local = _sample_link_points(
        links["right_outer_finger"], urdf_dir, int(tip_sample_points), contact_rng
    ).astype(np.float64)

    jaw_upper = joints[master_joint]["lower"]
    jaw_lower = joints[master_joint]["upper"]

    data = {
        "urdf_path": resolved_urdf_path,
        "urdf_dir": urdf_dir,
        "links": links,
        "joints": joints,
        "child_joints": child_joints,
        "base_link": base_link,
        "master_joint": master_joint,
        "jaw_lower": jaw_lower,
        "jaw_upper": jaw_upper,
        "render_points_local": render_points_local,
        "left_contact_local": left_contact_local,
        "right_contact_local": right_contact_local,
        "contact_offset_z": float(contact_offset_z),
        "z_band": float(z_band),
    }
    _ROBOTIQ_DATA_CACHE[cache_key] = data
    return data


def get_gripper_data(gripper_type="so101", urdf_path=None, tip_sample_points=20000, contact_offset_z=0.05, z_band=0.005):
    gt = _normalize_gripper_type(gripper_type)
    if gt == "so101":
        return _get_urdf_data(
            tip_sample_points=tip_sample_points,
            contact_offset_z=contact_offset_z,
            z_band=z_band,
        )
    if gt == "robotiq_2f85":
        return _get_robotiq_data(
            urdf_path=urdf_path,
            tip_sample_points=tip_sample_points,
            contact_offset_z=contact_offset_z,
            z_band=z_band,
        )
    raise ValueError(f"Unsupported gripper_type: {gripper_type}")


def _load_robotiq_gripper_points_in_base(jaw_angle, num_points, urdf_path=None, tip_sample_points=20000, contact_offset_z=0.05, z_band=0.005):
    resolved_urdf_path = _resolve_urdf_path(urdf_path, ROBOTIQ_2F85_URDF_PATH)
    angle_q = round(float(jaw_angle), 3)
    cache_key = ("robotiq_2f85", resolved_urdf_path, angle_q, int(num_points))
    if cache_key in _GRIPPER_MODEL_CACHE:
        return _GRIPPER_MODEL_CACHE[cache_key]

    data = _get_robotiq_data(
        urdf_path=resolved_urdf_path,
        tip_sample_points=tip_sample_points,
        contact_offset_z=contact_offset_z,
        z_band=z_band,
    )
    transforms = _compute_link_transforms(
        data["base_link"],
        data["child_joints"],
        {data["master_joint"]: float(jaw_angle)},
    )

    all_pts = []
    for link_name, pts_local in data["render_points_local"].items():
        if pts_local.size == 0:
            continue
        link_t = transforms.get(link_name, np.eye(4, dtype=np.float64))
        all_pts.append(_transform_points(pts_local, link_t))

    if not all_pts:
        return np.zeros((0, 3), dtype=np.float32)

    pts = np.concatenate(all_pts, axis=0).astype(np.float32)
    rng = np.random.default_rng(int(abs(angle_q * 1000)) % (2 ** 31))
    if len(pts) > num_points:
        idx = rng.choice(len(pts), size=int(num_points), replace=False)
        pts = pts[idx]
    elif len(pts) < num_points and len(pts) > 0:
        idx = rng.choice(len(pts), size=int(num_points - len(pts)), replace=True)
        pts = np.concatenate([pts, pts[idx]], axis=0)

    _GRIPPER_MODEL_CACHE[cache_key] = pts
    return pts


def _compute_robotiq_tcp_in_base(jaw_angle, urdf_path=None, tip_sample_points=20000, contact_offset_z=0.05, z_band=0.005):
    data = _get_robotiq_data(
        urdf_path=urdf_path,
        tip_sample_points=tip_sample_points,
        contact_offset_z=contact_offset_z,
        z_band=z_band,
    )
    transforms = _compute_link_transforms(
        data["base_link"],
        data["child_joints"],
        {data["master_joint"]: float(jaw_angle)},
    )

    left_pts = _transform_points(data["left_contact_local"], transforms["left_outer_finger"])
    right_pts = _transform_points(data["right_contact_local"], transforms["right_outer_finger"])

    left_contact = _pick_inner_contact_point(
        left_pts,
        float(left_pts[:, 2].max()),
        data["contact_offset_z"],
        data["z_band"],
    )
    right_contact = _pick_inner_contact_point(
        right_pts,
        float(right_pts[:, 2].max()),
        data["contact_offset_z"],
        data["z_band"],
    )
    return ((left_contact + right_contact) / 2.0).astype(np.float64)


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
    num_points=1500,
    gripper_offset=None,
    gripper_type="so101",
    urdf_path=None,
    tcp_local=None,
    tip_sample_points=20000,
    contact_offset_z=0.05,
    z_band=0.005,
):
    """Create gripper point cloud at the given pose."""
    gt = _normalize_gripper_type(gripper_type)
    if gt == "so101":
        gripper_local = _load_so101_gripper_points_in_frame(jaw_angle, int(num_points))
        default_tcp_local = _compute_tcp_in_frame(
            jaw_angle,
            tip_sample_points=tip_sample_points,
            contact_offset_z=contact_offset_z,
            z_band=z_band,
        )
    elif gt == "robotiq_2f85":
        gripper_local = _load_robotiq_gripper_points_in_base(
            jaw_angle,
            int(num_points),
            urdf_path=urdf_path,
            tip_sample_points=tip_sample_points,
            contact_offset_z=contact_offset_z,
            z_band=z_band,
        )
        default_tcp_local = _compute_robotiq_tcp_in_base(
            jaw_angle,
            urdf_path=urdf_path,
            tip_sample_points=tip_sample_points,
            contact_offset_z=contact_offset_z,
            z_band=z_band,
        )
    else:
        raise ValueError(f"Unsupported gripper_type: {gripper_type}")

    if tcp_local is None:
        tcp_local_arr = default_tcp_local
    else:
        tcp_local_arr = np.asarray(tcp_local, dtype=np.float64)
        if tcp_local_arr.shape != (3,):
            raise ValueError(f"tcp_local must have shape (3,), got {tcp_local_arr.shape}")

    p_gripper = position - orientation @ tcp_local_arr

    if gripper_offset is not None:
        p_gripper = p_gripper + np.asarray(gripper_offset, dtype=np.float64)

    all_pts = (orientation @ gripper_local.T).T + p_gripper
    colors = np.zeros_like(all_pts)
    return all_pts.astype(np.float32), colors.astype(np.float32)