# Voxelizer modified from ARM for DDP training
# Source: https://github.com/stepjam/ARM
# License: https://github.com/stepjam/ARM/LICENSE

from functools import reduce
from operator import mul
import numpy as np
import torch
from torch import nn

MIN_DENOMINATOR = 1e-12
INCLUDE_PER_VOXEL_COORD = False


class VoxelGrid(nn.Module):

    def __init__(self,
                 coord_bounds,
                 voxel_size: int,
                 device,
                 batch_size,
                 feature_size,  # e.g. rgb or image features
                 max_num_coords: int,):
        super(VoxelGrid, self).__init__()
        self._device = device
        self._voxel_size = voxel_size
        self._voxel_shape = [voxel_size] * 3
        self._voxel_d = float(self._voxel_shape[-1])
        self._voxel_feature_size = 4 + feature_size
        self._voxel_shape_spec = torch.tensor(self._voxel_shape,
                                              ).unsqueeze(
            0) + 2  # +2 because we crop the edges.
        self._coord_bounds = torch.tensor(coord_bounds, dtype=torch.float,
                                          ).unsqueeze(0)
        max_dims = self._voxel_shape_spec[0]
        self._total_dims_list = torch.cat(
            [torch.tensor([batch_size], ), max_dims,
             torch.tensor([4 + feature_size], )], -1).tolist()

        self.register_buffer('_ones_max_coords', torch.ones((batch_size, max_num_coords, 1)))
        self._num_coords = max_num_coords

        shape = self._total_dims_list
        result_dim_sizes = torch.tensor(
            [reduce(mul, shape[i + 1:], 1) for i in range(len(shape) - 1)] + [1], )
        self.register_buffer('_result_dim_sizes', result_dim_sizes)
        flat_result_size = reduce(mul, shape, 1)

        self._initial_val = torch.tensor(0, dtype=torch.float)
        flat_output = torch.ones(flat_result_size, dtype=torch.float) * self._initial_val
        self.register_buffer('_flat_output', flat_output)

        self.register_buffer('_arange_to_max_coords', torch.arange(4 + feature_size))
        self._flat_zeros = torch.zeros(flat_result_size, dtype=torch.float)

        self._const_1 = torch.tensor(1.0, )
        self._batch_size = batch_size

        # Coordinate Bounds:
        bb_mins = self._coord_bounds[..., 0:3]
        self.register_buffer('_bb_mins', bb_mins)
        bb_maxs = self._coord_bounds[..., 3:6]
        bb_ranges = bb_maxs - bb_mins
        # get voxel dimensions. 'DIMS' mode
        self._dims = dims = self._voxel_shape_spec.int()
        dims_orig = self._voxel_shape_spec.int() - 2
        self.register_buffer('_dims_orig', dims_orig)

        # self._dims_m_one = (dims - 1).int()
        dims_m_one = (dims - 1).int()
        self.register_buffer('_dims_m_one', dims_m_one)

        # BS x 1 x 3
        res = bb_ranges / (dims_orig.float() + MIN_DENOMINATOR)
        self._res_minis_2 = bb_ranges / (dims.float() - 2 + MIN_DENOMINATOR)
        self.register_buffer('_res', res)

        voxel_indicy_denmominator = res + MIN_DENOMINATOR
        self.register_buffer('_voxel_indicy_denmominator', voxel_indicy_denmominator)

        self.register_buffer('_dims_m_one_zeros', torch.zeros_like(dims_m_one))

        batch_indices = torch.arange(self._batch_size, dtype=torch.int).view(self._batch_size, 1, 1)
        self.register_buffer('_tiled_batch_indices', batch_indices.repeat([1, self._num_coords, 1]))

        w = self._voxel_shape[0] + 2
        arange = torch.arange(0, w, dtype=torch.float, )
        index_grid = torch.cat([
            arange.view(w, 1, 1, 1).repeat([1, w, w, 1]),
            arange.view(1, w, 1, 1).repeat([w, 1, w, 1]),
            arange.view(1, 1, w, 1).repeat([w, w, 1, 1])], dim=-1).unsqueeze(
            0).repeat([self._batch_size, 1, 1, 1, 1])
        self.register_buffer('_index_grid', index_grid)

    def _broadcast(self, src: torch.Tensor, other: torch.Tensor, dim: int):
        if dim < 0:
            dim = other.dim() + dim
        if src.dim() == 1:
            for _ in range(0, dim):
                src = src.unsqueeze(0)
        for _ in range(src.dim(), other.dim()):
            src = src.unsqueeze(-1)
        src = src.expand_as(other)
        return src

    def _scatter_mean(self, src: torch.Tensor, index: torch.Tensor, out: torch.Tensor,
                      dim: int = -1):
        out = out.scatter_add_(dim, index, src)

        index_dim = dim
        if index_dim < 0:
            index_dim = index_dim + src.dim()
        if index.dim() <= index_dim:
            index_dim = index.dim() - 1

        ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
        out_count = torch.zeros(out.size(), dtype=out.dtype, device=out.device)
        out_count = out_count.scatter_add_(index_dim, index, ones)
        out_count.clamp_(1)
        count = self._broadcast(out_count, out, dim)
        if torch.is_floating_point(out):
            out.true_divide_(count)
        else:
            out.floor_divide_(count)
        return out

    def _scatter_nd(self, indices, updates):
        indices_shape = indices.shape
        num_index_dims = indices_shape[-1]
        flat_updates = updates.view((-1,))
        indices_scales = self._result_dim_sizes[0:num_index_dims].view(
            [1] * (len(indices_shape) - 1) + [num_index_dims])
        indices_for_flat_tiled = ((indices * indices_scales).sum(
            dim=-1, keepdims=True)).view(-1, 1).repeat(
            *[1, self._voxel_feature_size])

        implicit_indices = self._arange_to_max_coords[
                           :self._voxel_feature_size].unsqueeze(0).repeat(
            *[indices_for_flat_tiled.shape[0], 1])
        indices_for_flat = indices_for_flat_tiled + implicit_indices
        flat_indices_for_flat = indices_for_flat.view((-1,)).long()

        flat_scatter = self._scatter_mean(
            flat_updates, flat_indices_for_flat,
            out=torch.zeros_like(self._flat_output))
        return flat_scatter.view(self._total_dims_list)

    def coords_to_bounding_voxel_grid(self, coords, coord_features=None,
                                      coord_bounds=None):
        voxel_indicy_denmominator = self._voxel_indicy_denmominator
        res, bb_mins = self._res, self._bb_mins
        if coord_bounds is not None:
            bb_mins = coord_bounds[..., 0:3]
            bb_maxs = coord_bounds[..., 3:6]
            bb_ranges = bb_maxs - bb_mins
            res = bb_ranges / (self._dims_orig.float() + MIN_DENOMINATOR)
            voxel_indicy_denmominator = res + MIN_DENOMINATOR

        bb_mins_shifted = bb_mins - res  # shift back by one
        floor = torch.floor(
            (coords - bb_mins_shifted.unsqueeze(1)) / voxel_indicy_denmominator.unsqueeze(1)).int()
        voxel_indices = torch.min(floor, self._dims_m_one)
        voxel_indices = torch.max(voxel_indices, self._dims_m_one_zeros)

        # BS x NC x 3
        voxel_values = coords
        if coord_features is not None:
            voxel_values = torch.cat([voxel_values, coord_features], -1)

        _, num_coords, _ = voxel_indices.shape
        # BS x N x (num_batch_dims + 2)
        all_indices = torch.cat([
            self._tiled_batch_indices[:, :num_coords], voxel_indices], -1)

        # BS x N x 4
        voxel_values_pruned_flat = torch.cat(
            [voxel_values, self._ones_max_coords[:, :num_coords]], -1)

        # BS x x_max x y_max x z_max x 4
        scattered = self._scatter_nd(
            all_indices.view([-1, 1 + 3]),
            voxel_values_pruned_flat.view(-1, self._voxel_feature_size))

        vox = scattered[:, 1:-1, 1:-1, 1:-1]
        if INCLUDE_PER_VOXEL_COORD:
            res_expanded = res.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            res_centre = (res_expanded * self._index_grid) + res_expanded / 2.0
            coord_positions = (res_centre + bb_mins_shifted.unsqueeze(
                1).unsqueeze(1).unsqueeze(1))[:, 1:-1, 1:-1, 1:-1]
            vox = torch.cat([vox[..., :-1], coord_positions, vox[..., -1:]], -1)

        occupied = (vox[..., -1:] > 0).float()
        vox = torch.cat([
            vox[..., :-1], occupied], -1)

        return torch.cat(
           [vox[..., :-1], self._index_grid[:, :-2, :-2, :-2] / self._voxel_d,
            vox[..., -1:]], -1)
    
def get_point_cloud_from_rgbd(color_img, depth_img, intrinsic):
    """
    将 RGB-D 图像转换为点云和特征
    intrinsic: [fx, fy, cx, cy]
    """
    fx, fy, cx, cy = intrinsic
    height, width = depth_img.shape
    
    # 生成像素网格
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # 深度单位转换（通常 RealSense 为毫米，转为米）
    depth = depth_img.astype(float) / 1000.0
    valid = depth > 0 # 过滤无效深度点
    
    # 计算 3D 坐标
    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    
    coords = np.stack([x, y, z], axis=-1) # [N, 3]
    features = color_img[valid].astype(float) / 255.0 # [N, 3] (RGB)
    
    return coords, features

def visualize_voxel_grid(voxels, occ_threshold=0.5, max_points=50000, title=None):
    """
    Visualize occupied voxels with a 3D scatter plot.
    voxels: torch.Tensor with shape [1, X, Y, Z, C]
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for voxel visualization. Install with: pip install matplotlib") from exc

    vox_np = voxels[0].detach().cpu().numpy()
    occupied = vox_np[..., -1] > occ_threshold

    if not np.any(occupied):
        print("No occupied voxels to visualize for this frame.")
        return

    xs, ys, zs = np.where(occupied)
    rgb = vox_np[..., :3][occupied]

    if len(xs) > max_points:
        keep = np.random.choice(len(xs), size=max_points, replace=False)
        xs, ys, zs, rgb = xs[keep], ys[keep], zs[keep], rgb[keep]

    rgb = np.clip(rgb, 0.0, 1.0)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c=rgb, s=4, marker='s', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title or 'Voxel Occupancy')
    plt.tight_layout()
    plt.show()


def process_npz(npz_path, voxelizer, max_frames=None, visualize=False, visualize_every=1):
    data = np.load(npz_path, allow_pickle=True)
    coords_list = data['coords_list']
    features_list = data['features_list']

    if len(coords_list) != len(features_list):
        raise ValueError("coords_list and features_list length mismatch in npz")

    num_frames = len(coords_list)
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    for frame_idx in range(num_frames):
        coords = coords_list[frame_idx]
        rgb_features = features_list[frame_idx]

        if len(coords) == 0:
            continue

        # 限制点数以符合 max_num_coords
        num_points = min(len(coords), voxelizer._num_coords)
        indices = np.random.choice(len(coords), num_points, replace=False)

        coords_tensor = torch.from_numpy(coords[indices]).float().unsqueeze(0).to(voxelizer._device)
        features_tensor = torch.from_numpy(rgb_features[indices]).float().unsqueeze(0).to(voxelizer._device)

        # 如果点数不够，需要 padding 到 max_num_coords
        if num_points < voxelizer._num_coords:
            pad_size = voxelizer._num_coords - num_points
            coords_tensor = torch.nn.functional.pad(coords_tensor, (0, 0, 0, pad_size))
            features_tensor = torch.nn.functional.pad(features_tensor, (0, 0, 0, pad_size))

        with torch.no_grad():
            voxels = voxelizer.coords_to_bounding_voxel_grid(coords_tensor, features_tensor)

        print(f"Frame {frame_idx}: generated voxel grid shape {voxels.shape}")
        if visualize and (frame_idx % visualize_every == 0):
            visualize_voxel_grid(voxels, title=f"Frame {frame_idx}")

# --- 初始化示例 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
voxel_size = 32
# 设置场景边界 [min_x, min_y, min_z, max_x, max_y, max_z]
bounds = [-0.5, -0.5, 0.2, 0.5, 0.5, 1.2] 

my_voxelizer = VoxelGrid(
    coord_bounds=bounds,
    voxel_size=voxel_size,
    device=device,
    batch_size=1,
    feature_size=3, # RGB
    max_num_coords=100000 # 根据内存调整
).to(device)

process_npz(
    "/home/mercury/EECS467/Data/test_frames.npz",
    my_voxelizer,
    max_frames=10,
    visualize=True,
    visualize_every=1,
)