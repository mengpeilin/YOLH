import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def create_gripper_points(position, orientation, gripper_open,
                          num_points=200, gripper_length=0.06, gripper_width=0.04):
    """
    Create a simplified SO-ARM100 gripper point cloud at given pose.
    The gripper is rendered as a pure black object (color=[0,0,0]).
    
    Args:
        position: (3,) hand center position
        orientation: (3,3) rotation matrix
        gripper_open: bool
        num_points: number of points to generate
        gripper_length: length of each jaw (meters)
        gripper_width: width between jaws when open (meters)
    
    Returns:
        coords: (N, 3) float32
        colors: (N, 3) float32 (all zeros = black)
    """
    # Gripper body (wrist part) - a small rectangular block
    body_pts = np.random.uniform(-1, 1, (num_points // 4, 3)).astype(np.float32)
    body_pts[:, 0] *= 0.015  # thin in x
    body_pts[:, 1] *= 0.015  # thin in y
    body_pts[:, 2] *= 0.02   # short in z

    # Left jaw
    jaw_l = np.random.uniform(-1, 1, (num_points // 4, 3)).astype(np.float32)
    jaw_l[:, 0] *= 0.005
    jaw_l[:, 1] *= gripper_length / 2
    jaw_l[:, 2] *= 0.005

    # Right jaw
    jaw_r = jaw_l.copy()

    # Position jaws based on open/close state
    jaw_gap = gripper_width / 2 if gripper_open else 0.005
    jaw_l[:, 0] -= jaw_gap
    jaw_r[:, 0] += jaw_gap

    # Shift jaws forward (along y-axis in gripper frame)
    jaw_l[:, 1] += gripper_length / 2 + 0.02
    jaw_r[:, 1] += gripper_length / 2 + 0.02

    # Connecting bridge
    bridge = np.random.uniform(-1, 1, (num_points // 4, 3)).astype(np.float32)
    bridge[:, 0] *= jaw_gap + 0.005
    bridge[:, 1] *= 0.005
    bridge[:, 2] *= 0.005
    bridge[:, 1] += 0.02

    all_pts = np.concatenate([body_pts, jaw_l, jaw_r, bridge], axis=0)

    # Transform to world frame
    all_pts = (orientation @ all_pts.T).T + position

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