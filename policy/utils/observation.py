"""Observation preprocessing: RGB-D to point cloud, voxelisation, normalisation."""

import numpy as np

from policy.utils.constants import IMG_MEAN, IMG_STD
from scripts.urdf_reader import rgbd_to_points


def build_observation_cloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    arm_filter,
    joint_angles: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """Build an observation point cloud from an RGB-D frame.

    Steps:
        1. RGB-D back-projection
        2. Arm self-filter (remove arm body, keep gripper)
        3. Workspace crop
        4. Voxel down-sample
        5. ImageNet colour normalisation

    Returns:
        (N, 6) float32 array [x, y, z, r, g, b] or (0, 6) if empty.
    """
    voxel_size = cfg["voxel_size"]
    ws_min = cfg["workspace_min"]
    ws_max = cfg["workspace_max"]

    # 1. RGB-D → points
    coords, colors = rgbd_to_points(rgb, depth, intrinsic, mask=None)
    if len(coords) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # 2. Arm self-filter
    if cfg.get("arm_filter", {}).get("enabled", True):
        coords, colors = arm_filter.filter(coords, colors, joint_angles)
    if len(coords) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # 3. Workspace filter
    in_ws = np.all((coords >= ws_min) & (coords <= ws_max), axis=1)
    coords = coords[in_ws]
    colors = colors[in_ws]
    if len(coords) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # 4. Voxel downsample
    vi = np.floor(coords / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(vi, axis=0, return_index=True)
    coords = coords[unique_idx]
    colors = colors[unique_idx]

    # 5. ImageNet normalise
    colors = (colors - IMG_MEAN) / IMG_STD

    cloud = np.concatenate([coords, colors], axis=-1).astype(np.float32)
    return cloud
