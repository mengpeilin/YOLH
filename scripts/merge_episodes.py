import numpy as np
from pathlib import Path


def _mat_to_rot6d(mat):
    return mat[:2, :].flatten().astype(np.float32)


def merge_episodes(
    data_dir: str,
    output_path: str,
    num_action: int = 20,
    trans_min=(-0.5, -0.1, 0.0),
    trans_max=(0.5, 0.5, 1.0),
    max_gripper_width: float = None,
    task_name: str = "default_task",
):
    trans_min = np.asarray(trans_min, dtype=np.float32)
    trans_max = np.asarray(trans_max, dtype=np.float32)

    data_dir = Path(data_dir)
    session_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("rosbag"))

    all_max_widths = []
    valid_sessions = []
    for sess_dir in session_dirs:
        pc_path = sess_dir / "episodes.npz"
        if not pc_path.exists():
            print(f"  Skipping {sess_dir.name}: no episodes.npz")
            continue
        pc = np.load(pc_path, allow_pickle=True)
        all_max_widths.append(float(pc["max_width"]))
        valid_sessions.append(sess_dir)

    if not valid_sessions:
        raise ValueError(f"No episodes.npz found in {data_dir}")

    if max_gripper_width is None:
        max_gripper_width = max(all_max_widths)
    if max_gripper_width <= 0:
        raise ValueError("max_gripper_width must be > 0. ")

    print(f"     Generating YOLH dataset from {len(valid_sessions)} sessions")
    print(f"     num_action={num_action}, max_gripper_width={max_gripper_width:.4f}m")
    print(f"     trans_min={trans_min}, trans_max={trans_max}")

    all_clouds = []
    all_actions = []
    all_actions_norm = []
    all_session_ids = []
    all_frame_indices = []

    trans_range = trans_max - trans_min

    for sess_dir in valid_sessions:
        pc = np.load(sess_dir / "episodes.npz", allow_pickle=True)
        clouds = pc["clouds"]
        ee_pts = pc["ee_pts"]
        ee_oris = pc["ee_oris"]
        ee_widths = pc["ee_widths"]
        hand_detected = pc["hand_detected"]
        num_frames = int(pc["num_frames"])

        sess_count = 0
        for frame_idx in range(num_frames - 1):
            if not hand_detected[frame_idx]:
                continue
            if len(clouds[frame_idx]) == 0:
                continue

            # Build fixed-length future chunk.
            actions = []
            for t in range(1, num_action + 1):
                fid = min(frame_idx + t, num_frames - 1)
                pos = ee_pts[fid].astype(np.float32)
                rot6d = _mat_to_rot6d(ee_oris[fid])
                width = np.float32(ee_widths[fid])
                actions.append(np.concatenate([pos, rot6d, [width]]))
            actions = np.stack(actions)

            actions_norm = actions.copy()
            actions_norm[:, :3] = (actions_norm[:, :3] - trans_min) / (trans_range + 1e-8) * 2 - 1
            actions_norm[:, -1] = actions_norm[:, -1] / (max_gripper_width + 1e-8) * 2 - 1

            all_clouds.append(clouds[frame_idx])
            all_actions.append(actions)
            all_actions_norm.append(actions_norm)
            all_session_ids.append(sess_dir.name)
            all_frame_indices.append(frame_idx)
            sess_count += 1

        print(f"  {sess_dir.name}: {sess_count} samples from {num_frames} frames")

    total = len(all_clouds)
    if total == 0:
        raise ValueError("No valid samples found!")

    all_actions = np.stack(all_actions).astype(np.float32)
    all_actions_norm = np.stack(all_actions_norm).astype(np.float32)
    all_frame_indices = np.array(all_frame_indices, dtype=np.int64)

    np.savez_compressed(
        output_path,
        clouds=np.array(all_clouds, dtype=object),
        actions=all_actions,
        actions_normalized=all_actions_norm,
        session_ids=np.array(all_session_ids, dtype=object),
        sample_frame_idx=all_frame_indices,
        max_gripper_width=np.float32(max_gripper_width),
        trans_min=trans_min,
        trans_max=trans_max,
        num_action=np.int32(num_action),
        task_name=task_name,
    )
    print(f"     Saved {total} samples to {output_path}")
    print(f"     Actions shape: {all_actions.shape}")
