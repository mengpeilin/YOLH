import os
import numpy as np
from pathlib import Path


def merge_episodes(
    data_dir: str,
    output_path: str,
    task_name: str = "default_task",
):
    """Merge all episodes.npz files from session directories."""
    data_dir = Path(data_dir)
    
    all_voxel_grids = []
    all_actions = []
    all_session_ids = []
    coord_bounds = None
    voxel_size = None

    session_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("rosbag")
    ])
    for sess_dir in session_dirs:
        ep_file = sess_dir / "episodes.npz"
        if not ep_file.exists():
            print(f"  Skipping {sess_dir.name}: no episodes.npz")
            continue

        ep = np.load(ep_file)
        vg = ep["voxel_grids"]
        ac = ep["actions"]
        
        all_voxel_grids.append(vg)
        all_actions.append(ac)
        all_session_ids.extend([sess_dir.name] * len(vg))

        if coord_bounds is None:
            coord_bounds = ep["coord_bounds"]
            voxel_size = int(ep["voxel_size"])

        print(f"  {sess_dir.name}: {len(vg)} episodes")

    if not all_voxel_grids:
        raise ValueError(f"No episodes found in {data_dir}")

    merged_voxels = np.concatenate(all_voxel_grids, axis=0)
    merged_actions = np.concatenate(all_actions, axis=0)

    np.savez_compressed(
        output_path,
        voxel_grids=merged_voxels,
        actions=merged_actions,
        coord_bounds=coord_bounds,
        voxel_size=voxel_size,
        task_name=task_name,
        session_ids=np.array(all_session_ids, dtype=object),
    )
    print(f"[merge] Saved {len(merged_voxels)} total episodes to {output_path}")
    print(f"        From {len(session_dirs)} sessions")
    return output_path