"""Dataset loader for train_dataset.npz."""

import numpy as np
import torch
import MinkowskiEngine as ME
import collections.abc as container_abcs
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation

TO_TENSOR_KEYS = [
    "input_coords_list",
    "input_feats_list",
    "action",
    "action_normalized",
]


class YolhDataset(Dataset):
    """Dataset wrapper around the merged YOLH archive."""

    def __init__(self, dataset_path: str, voxel_size: float = 0.005):
        data = np.load(dataset_path, allow_pickle=True)
        self.clouds = data["clouds"]
        self.actions = data["actions"]
        self.actions_normalized = data["actions_normalized"]
        self.voxel_size = voxel_size

        print(f"Loaded {len(self.clouds)} samples from {dataset_path}")
        print(f"  Actions shape: {self.actions.shape}")
        if "max_gripper_width" in data:
            print(f"  max_gripper_width: {float(data['max_gripper_width']):.4f}m")

    def __len__(self):
        return len(self.clouds)

    def __getitem__(self, index):
        cloud = self.clouds[index].copy()

        # Data augmentation: random translation and rotation
        if len(cloud) > 0:
            # Random translation: [-0.2m, 0.2m] along X/Y/Z
            translation = np.random.uniform(-0.2, 0.2, size=(3,))
            
            # Random rotation: [-30°, 30°] around X/Y/Z (in radians)
            angles = np.random.uniform(np.radians(-30), np.radians(30), size=(3,))
            rotation = Rotation.from_euler('xyz', angles)
            
            # Apply transformation to point coordinates
            coords_3d = cloud[:, :3]
            coords_3d = rotation.apply(coords_3d) + translation
            cloud[:, :3] = coords_3d
            
            voxel_coords = np.ascontiguousarray(
                cloud[:, :3] / self.voxel_size, dtype=np.int32
            )
        else:
            voxel_coords = np.zeros((0, 3), dtype=np.int32)

        actions = torch.from_numpy(self.actions[index]).float()
        actions_norm = torch.from_numpy(self.actions_normalized[index]).float()

        return {
            "input_coords_list": [voxel_coords],
            "input_feats_list": [cloud.astype(np.float32)],
            "action": actions,
            "action_normalized": actions_norm,
        }


def collate_fn(batch):
    """Collate function that builds ME sparse batches."""
    if type(batch[0]).__module__ == "numpy":
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            if key in TO_TENSOR_KEYS:
                ret_dict[key] = collate_fn([d[key] for d in batch])
            else:
                ret_dict[key] = [d[key] for d in batch]
        coords_batch = ret_dict["input_coords_list"]
        feats_batch = ret_dict["input_feats_list"]
        coords_batch, feats_batch = ME.utils.sparse_collate(
            coords_batch, feats_batch
        )
        ret_dict["input_coords_list"] = coords_batch
        ret_dict["input_feats_list"] = feats_batch
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [sample for b in batch for sample in b]
    raise TypeError(
        f"batch must contain tensors, dicts or lists; found {type(batch[0])}"
    )
