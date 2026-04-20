import numpy as np
from policy.utils.constants import IMG_MEAN, IMG_STD
from policy.utils.transformation import rgbd_to_points
from scripts.urdf_reader import (
    create_gripper_points,
    get_gripper_data,
    width_to_jaw_angle,
)

def build_point_clouds(
    raw_npz_path: str,
    masks_npz_path: str,
    gripper_action_path: str,
    output_path: str,
    voxel_size: float = 0.005,
    workspace_min=(-0.5, -0.1, 0.0),
    workspace_max=(0.5, 0.5, 1.0),
    gripper_offset=(0.0, 0.0, 0.0),
    gripper_type: str = "so101",
    gripper_urdf_path: str = None,
    gripper_tcp_local=None,
    gripper_num_points: int = 1200,
    tip_sample_points: int = 20000,
    contact_offset_z: float = 0.05,
    z_band: float = 0.005,
):
    workspace_min = np.asarray(workspace_min, dtype=np.float32)
    workspace_max = np.asarray(workspace_max, dtype=np.float32)

    raw = np.load(raw_npz_path, allow_pickle=True)
    rgb_frames = raw["rgb"]
    depth_frames = raw["depth"]
    intrinsic = raw["intrinsic"]

    masks_data = np.load(masks_npz_path, allow_pickle=True)
    masks_key = "arm_hand_masks" if "arm_hand_masks" in masks_data else "masks"
    masks = masks_data[masks_key]

    action_data = np.load(gripper_action_path, allow_pickle=True)
    ee_pts = action_data["ee_pts"]
    ee_oris = action_data["ee_oris"]
    ee_widths = action_data["ee_widths"]
    hand_detected = action_data["hand_detected"]
    max_width = (
        float(action_data["max_width"])
        if "max_width" in action_data
        else float(ee_widths.max())
    )

    # Gripper jaw limits
    gripper = get_gripper_data(
        gripper_type=gripper_type,
        urdf_path=gripper_urdf_path,
        tip_sample_points=tip_sample_points,
        contact_offset_z=contact_offset_z,
        z_band=z_band,
    )
    jaw_lower = gripper["jaw_lower"]
    jaw_upper = gripper["jaw_upper"]

    num_frames = len(rgb_frames)
    print(f"     Building point clouds for {num_frames} frames "
          f"({int(hand_detected.sum())} with hand)")

    clouds = []
    for i in range(num_frames):
        coords, colors = rgbd_to_points(rgb_frames[i], depth_frames[i], intrinsic, masks[i])

        # workspace crop
        if len(coords) > 0:
            in_ws = np.all(
                (coords >= workspace_min) & (coords <= workspace_max), axis=1
            )
            coords = coords[in_ws]
            colors = colors[in_ws]

        # voxel downsample
        if len(coords) > 0:
            vi = np.floor(coords / voxel_size).astype(np.int64)
            _, unique_idx = np.unique(vi, axis=0, return_index=True)
            coords = coords[unique_idx]
            colors = colors[unique_idx]

        # insert gripper mesh points
        if hand_detected[i]:
            ee_pt = ee_pts[i].astype(np.float64)
            ee_ori = ee_oris[i].astype(np.float64)
            w = float(ee_widths[i])
            jaw_angle = width_to_jaw_angle(w, max_width, jaw_lower, jaw_upper)
            g_coords, g_colors = create_gripper_points(
                ee_pt, ee_ori, jaw_angle,
                num_points=gripper_num_points,
                gripper_offset=list(gripper_offset) if gripper_offset is not None else None,
                gripper_type=gripper_type,
                urdf_path=gripper_urdf_path,
                tcp_local=gripper_tcp_local,
                tip_sample_points=tip_sample_points,
                contact_offset_z=contact_offset_z,
                z_band=z_band,
            )
            if len(coords) > 0:
                coords = np.concatenate([coords, g_coords], axis=0)
                colors = np.concatenate([colors, g_colors], axis=0)
            else:
                coords = g_coords
                colors = g_colors

        if len(colors) > 0:
            colors = (colors - IMG_MEAN) / IMG_STD

        if len(coords) > 0:
            cloud = np.concatenate([coords, colors], axis=-1).astype(np.float32)
        else:
            cloud = np.zeros((0, 6), dtype=np.float32)

        clouds.append(cloud)

    # variable-length clouds require object array
    np.savez_compressed(
        output_path,
        clouds=np.array(clouds, dtype=object),
        ee_pts=ee_pts,
        ee_oris=ee_oris,
        ee_widths=ee_widths,
        hand_detected=hand_detected,
        max_width=np.float32(max_width),
        num_frames=np.int32(num_frames),
    )
    valid_count = sum(1 for c in clouds if len(c) > 0)
    print(f"     Saved {num_frames} point clouds ({valid_count} non-empty) to {output_path}")
