#!/usr/bin/env python3
"""
YOLH Training Script
====================
Trains a PerAct-based model (without language input) on pipeline-processed data.
Each task trains a separate model.

Usage (inside Docker):
    python train.py \
        --dataset data/train_dataset.npz \
        --task-name pick_cup \
        --output-dir checkpoints/pick_cup \
        --batch-size 2 \
        --epochs 100
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation

# Add peract to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "peract"))

from agents.peract_bc.perceiver_io import PerceiverVoxelEncoder
from helpers.optim.lamb import Lamb


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Dataset
# ============================================================

class YOLHDataset(Dataset):
    """Dataset for YOLH training episodes."""

    def __init__(self, dataset_path, voxel_size=100, rotation_resolution=5):
        data = np.load(dataset_path, allow_pickle=True)
        self.voxel_grids = data['voxel_grids']    # (N, V, V, V, 4)
        self.actions = data['actions']              # (N, 13): [pos(3), rot(9), width_norm(1)]
        self.coord_bounds = data['coord_bounds']    # (6,)
        self.voxel_size = int(data['voxel_size'])
        self.rotation_resolution = rotation_resolution
        # proprios: previous keyframe's EE state, raw 13-dim (same format as actions)
        # Falls back to zeros if dataset was generated without proprio support.
        if 'proprios' in data:
            self.proprios = data['proprios']          # (N, 13)
        else:
            self.proprios = np.zeros_like(self.actions)

        logger.info(f"Loaded {len(self.voxel_grids)} episodes from {dataset_path}")
        logger.info(f"  Voxel grid shape: {self.voxel_grids.shape}")
        logger.info(f"  Action shape: {self.actions.shape}")
        logger.info(f"  Coord bounds: {self.coord_bounds}")

    def __len__(self):
        return len(self.voxel_grids)

    def _action_to_gt(self, action):
        """Convert continuous action to discretized GT for training."""
        position = action[:3]
        rot_matrix = action[3:12].reshape(3, 3)
        width_norm = float(action[12])   # continuous [0, 1]

        # Discretize translation: position -> voxel index
        bb_mins = self.coord_bounds[:3]
        bb_maxs = self.coord_bounds[3:]
        bb_ranges = bb_maxs - bb_mins
        res = bb_ranges / self.voxel_size
        trans_idx = np.minimum(
            np.floor((position - bb_mins) / (res + 1e-12)).astype(np.int32),
            self.voxel_size - 1)
        trans_idx = np.maximum(trans_idx, 0)

        # Discretize rotation: rotation matrix -> discrete euler
        try:
            quat = Rotation.from_matrix(rot_matrix).as_quat()
            quat = quat / (np.linalg.norm(quat) + 1e-8)
            if quat[-1] < 0:
                quat = -quat
            euler = Rotation.from_quat(quat).as_euler('xyz', degrees=True) + 180
            euler = np.clip(euler, 0, 360)
            disc_rot = np.around(euler / self.rotation_resolution).astype(int)
            disc_rot[disc_rot == int(360 / self.rotation_resolution)] = 0
        except Exception:
            disc_rot = np.zeros(3, dtype=np.int32)

        rot_indices = disc_rot.astype(np.int32)

        return trans_idx, rot_indices, width_norm

    def __getitem__(self, idx):
        voxel_grid = self.voxel_grids[idx]    # (V, V, V, 4)
        action = self.actions[idx]             # (13,)

        trans_idx, rot_indices, width_norm = self._action_to_gt(action)

        # Convert voxel grid to channel-first: (4, V, V, V)
        # The 4 channels: R, G, B, occupancy
        voxel_tensor = torch.from_numpy(voxel_grid).float().permute(3, 0, 1, 2)

        # Expand to match PerAct input format: need (initial_dim, V, V, V)
        # initial_dim = 3(rgb) + 3(xyz_coord) + 1(occupancy) + 3(voxel_coord) = 10
        # We have 4 channels (RGB + occupancy), need to add coordinate channels
        V = self.voxel_size
        coord_bounds = self.coord_bounds
        bb_mins = coord_bounds[:3]
        bb_maxs = coord_bounds[3:]
        bb_ranges = bb_maxs - bb_mins
        res = bb_ranges / V

        # Create xyz coordinate grid
        xs = torch.linspace(bb_mins[0] + res[0]/2, bb_maxs[0] - res[0]/2, V)
        ys = torch.linspace(bb_mins[1] + res[1]/2, bb_maxs[1] - res[1]/2, V)
        zs = torch.linspace(bb_mins[2] + res[2]/2, bb_maxs[2] - res[2]/2, V)
        grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing='ij')
        xyz_coords = torch.stack([grid_x, grid_y, grid_z], dim=0)  # (3, V, V, V)

        # normalized voxel index coords
        vox_idx = torch.stack([
            torch.linspace(0, 1, V).view(V, 1, 1).expand(V, V, V),
            torch.linspace(0, 1, V).view(1, V, 1).expand(V, V, V),
            torch.linspace(0, 1, V).view(1, 1, V).expand(V, V, V),
        ], dim=0)  # (3, V, V, V)

        # Stack: rgb(3) + xyz(3) + occupancy(1) + voxel_coord(3) = 10 channels
        rgb = voxel_tensor[:3]
        occ = voxel_tensor[3:4]
        full_voxel = torch.cat([rgb, xyz_coords, occ, vox_idx], dim=0)  # (10, V, V, V)

        # --- Build proprio vector (8-dim): pos_norm(3) + quat(4) + width_norm(1) ---
        proprio_raw = self.proprios[idx]   # (13,)
        p_pos = proprio_raw[:3].astype(np.float64)
        p_rot = proprio_raw[3:12].reshape(3, 3).astype(np.float64)
        p_wn  = float(proprio_raw[12])

        # Normalize position to [-1, 1] using coord_bounds
        bb_r = (self.coord_bounds[3:] - self.coord_bounds[:3]).astype(np.float64)
        p_pos_norm = np.clip(
            2.0 * (p_pos - self.coord_bounds[:3]) / (bb_r + 1e-8) - 1.0,
            -1.0, 1.0,
        ).astype(np.float32)

        # Rotation matrix -> unit quaternion [x,y,z,w]
        try:
            p_quat = Rotation.from_matrix(p_rot).as_quat().astype(np.float32)
            if p_quat[-1] < 0:
                p_quat = -p_quat
            p_quat /= (np.linalg.norm(p_quat) + 1e-8)
        except Exception:
            p_quat = np.array([0., 0., 0., 1.], dtype=np.float32)

        proprio = np.concatenate([p_pos_norm, p_quat, [p_wn]]).astype(np.float32)  # (8,)

        return {
            'voxel_grid': full_voxel,
            'trans_action_indices': torch.from_numpy(trans_idx).long(),
            'rot_action_indices': torch.from_numpy(rot_indices).long(),
            'width_norm': torch.tensor(width_norm, dtype=torch.float32),
            'proprio': torch.from_numpy(proprio),
        }


# ============================================================
# Training
# ============================================================

def create_model(cfg):
    """Create PerceiverVoxelEncoder without language."""
    num_rotation_classes = int(360 // cfg['rotation_resolution'])

    encoder = PerceiverVoxelEncoder(
        depth=cfg.get('transformer_depth', 6),
        iterations=cfg.get('transformer_iterations', 1),
        voxel_size=cfg.get('voxel_size', 100),
        initial_dim=10,  # rgb(3) + xyz(3) + occupancy(1) + voxel_coord(3)
        low_dim_size=8,  # pos_norm(3) + quat(4) + width_norm(1)
        layer=0,
        num_rotation_classes=num_rotation_classes,
        num_grip_classes=2,
        num_collision_classes=2,
        input_axis=3,
        num_latents=cfg.get('num_latents', 2048),
        latent_dim=cfg.get('latent_dim', 512),
        cross_heads=cfg.get('cross_heads', 1),
        latent_heads=cfg.get('latent_heads', 8),
        cross_dim_head=cfg.get('cross_dim_head', 64),
        latent_dim_head=cfg.get('latent_dim_head', 64),
        activation=cfg.get('activation', 'lrelu'),
        weight_tie_layers=False,
        input_dropout=cfg.get('input_dropout', 0.1),
        attn_dropout=cfg.get('attn_dropout', 0.1),
        decoder_dropout=cfg.get('decoder_dropout', 0.0),
        voxel_patch_size=cfg.get('voxel_patch_size', 5),
        voxel_patch_stride=cfg.get('voxel_patch_stride', 5),
        no_skip_connection=False,
        no_perceiver=False,
        final_dim=cfg.get('final_dim', 64),
    )
    return encoder


def train_one_epoch(model, dataloader, optimizer, scheduler, device, cfg):
    model.train()
    total_loss_sum = 0.0
    trans_loss_sum = 0.0
    rot_loss_sum = 0.0
    grip_loss_sum = 0.0
    num_batches = 0

    ce_loss = nn.CrossEntropyLoss(reduction='mean')
    voxel_size = cfg['voxel_size']
    num_rotation_classes = int(360 // cfg['rotation_resolution'])

    for batch in dataloader:
        voxel_grid = batch['voxel_grid'].to(device)           # (B, 10, V, V, V)
        trans_gt = batch['trans_action_indices'].to(device)    # (B, 3)
        rot_gt = batch['rot_action_indices'].to(device)        # (B, 3)
        width_target = batch['width_norm'].to(device)          # (B,)
        proprio = batch['proprio'].to(device)                  # (B, 8)

        bs = voxel_grid.shape[0]
        bounds = torch.tensor(cfg['coord_bounds'], device=device).unsqueeze(0).expand(bs, -1)

        # Forward: no language, with proprioception
        q_trans, q_rot_grip, q_collision = model(
            voxel_grid, proprio, None, bounds, None)

        # Translation loss
        trans_target = (trans_gt[:, 0] * voxel_size * voxel_size +
                        trans_gt[:, 1] * voxel_size +
                        trans_gt[:, 2])
        q_trans_flat = q_trans.view(bs, -1)
        loss_trans = ce_loss(q_trans_flat, trans_target.long())

        # Rotation + grip loss
        loss_rot = torch.tensor(0.0, device=device)
        loss_grip = torch.tensor(0.0, device=device)
        if q_rot_grip is not None:
            nrc = num_rotation_classes
            loss_rot += ce_loss(q_rot_grip[:, 0*nrc:1*nrc], rot_gt[:, 0].long())
            loss_rot += ce_loss(q_rot_grip[:, 1*nrc:2*nrc], rot_gt[:, 1].long())
            loss_rot += ce_loss(q_rot_grip[:, 2*nrc:3*nrc], rot_gt[:, 2].long())
            # Continuous grip regression: softmax -> MSE against width_norm
            grip_logits = q_rot_grip[:, 3*nrc:]
            grip_pred = F.softmax(grip_logits, dim=1)[:, 1]  # P(open) in [0,1]
            loss_grip = F.mse_loss(grip_pred, width_target)

        total_loss = (loss_trans * cfg.get('trans_loss_weight', 1.0) +
                      loss_rot * cfg.get('rot_loss_weight', 1.0) +
                      loss_grip * cfg.get('grip_loss_weight', 1.0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss_sum += total_loss.item()
        trans_loss_sum += loss_trans.item()
        rot_loss_sum += loss_rot.item()
        grip_loss_sum += loss_grip.item()
        num_batches += 1

    return {
        'total_loss': total_loss_sum / max(num_batches, 1),
        'trans_loss': trans_loss_sum / max(num_batches, 1),
        'rot_loss': rot_loss_sum / max(num_batches, 1),
        'grip_loss': grip_loss_sum / max(num_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="YOLH Training")
    parser.add_argument("--dataset", required=True,
                        help="Path to train_dataset.npz")
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--optimizer", choices=['lamb', 'adam'], default='lamb')
    parser.add_argument("--voxel-size", type=int, default=100)
    parser.add_argument("--rotation-resolution", type=float, default=5.0)
    parser.add_argument("--num-latents", type=int, default=2048)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--transformer-depth", type=int, default=6)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = YOLHDataset(
        args.dataset,
        voxel_size=args.voxel_size,
        rotation_resolution=args.rotation_resolution,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # Config
    cfg = {
        'voxel_size': args.voxel_size,
        'rotation_resolution': args.rotation_resolution,
        'num_latents': args.num_latents,
        'latent_dim': args.latent_dim,
        'transformer_depth': args.transformer_depth,
        'transformer_iterations': 1,
        'cross_heads': 1,
        'latent_heads': 8,
        'cross_dim_head': 64,
        'latent_dim_head': 64,
        'activation': 'lrelu',
        'input_dropout': 0.1,
        'attn_dropout': 0.1,
        'decoder_dropout': 0.0,
        'voxel_patch_size': 5,
        'voxel_patch_stride': 5,
        'final_dim': 64,
        'trans_loss_weight': 1.0,
        'rot_loss_weight': 1.0,
        'grip_loss_weight': 1.0,
        'coord_bounds': dataset.coord_bounds.tolist(),
    }

    # Create model
    model = create_model(cfg)
    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer
    if args.optimizer == 'lamb':
        optimizer = Lamb(model.parameters(), lr=args.lr,
                         weight_decay=1e-6, betas=(0.9, 0.999), adam=False)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    # Scheduler
    total_steps = args.epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(total_steps // 10, 1), T_mult=1)

    # Training loop
    logger.info(f"Training {args.task_name} for {args.epochs} epochs")
    logger.info(f"  Dataset size: {len(dataset)}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Steps per epoch: {len(dataloader)}")

    best_loss = float('inf')
    for epoch in range(args.epochs):
        t0 = time.time()
        metrics = train_one_epoch(model, dataloader, optimizer, scheduler, device, cfg)
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch+1:04d}/{args.epochs} | "
            f"loss={metrics['total_loss']:.4f} "
            f"(trans={metrics['trans_loss']:.4f} rot={metrics['rot_loss']:.4f} "
            f"grip={metrics['grip_loss']:.4f}) | "
            f"{elapsed:.1f}s"
        )

        # Save checkpoints
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = output_dir / f"checkpoint_{epoch+1:05d}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'cfg': cfg,
                'task_name': args.task_name,
            }, ckpt_path)
            logger.info(f"  Saved checkpoint: {ckpt_path}")

        if metrics['total_loss'] < best_loss:
            best_loss = metrics['total_loss']
            best_path = output_dir / "best.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
                'cfg': cfg,
                'task_name': args.task_name,
            }, best_path)

    # Save final model
    final_path = output_dir / "final.pt"
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'cfg': cfg,
        'task_name': args.task_name,
    }, final_path)
    logger.info(f"Training complete. Best loss: {best_loss:.4f}")
    logger.info(f"Final model: {final_path}")
    logger.info(f"Best model: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
