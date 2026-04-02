# Q-Attention agent for YOLH (no language input)
# Modified from qattention_peract_bc_agent.py

import copy
import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from scipy.spatial.transform import Rotation

from helpers.network_utils import DenseBlock, SpatialSoftmax3D, Conv3DBlock
from helpers.optim.lamb import Lamb

from torch.nn.parallel import DistributedDataParallel as DDP

import transformers

NAME = 'YOLHQAttentionAgent'


class QFunction(nn.Module):
    def __init__(self, perceiver_encoder, voxelizer, bounds_offset,
                 rotation_resolution, device, training):
        super(QFunction, self).__init__()
        self._rotation_resolution = rotation_resolution
        self._voxelizer = voxelizer
        self._bounds_offset = bounds_offset
        self._qnet = perceiver_encoder.to(device)
        if training:
            self._qnet = DDP(self._qnet, device_ids=[device])

    def _argmax_3d(self, tensor_orig):
        b, c, d, h, w = tensor_orig.shape
        idxs = tensor_orig.view(b, c, -1).argmax(-1)
        indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1)
        return indices

    def choose_highest_action(self, q_trans, q_rot_grip, q_collision):
        coords = self._argmax_3d(q_trans)
        rot_and_grip_indicies = None
        ignore_collision = None
        if q_rot_grip is not None:
            q_rot = torch.stack(torch.split(
                q_rot_grip[:, :-2],
                int(360 // self._rotation_resolution),
                dim=1), dim=1)
            rot_and_grip_indicies = torch.cat(
                [q_rot[:, 0:1].argmax(-1),
                 q_rot[:, 1:2].argmax(-1),
                 q_rot[:, 2:3].argmax(-1),
                 q_rot_grip[:, -2:].argmax(-1, keepdim=True)], -1)
            ignore_collision = q_collision[:, -2:].argmax(-1, keepdim=True)
        return coords, rot_and_grip_indicies, ignore_collision

    def forward(self, voxel_grid, proprio, bounds=None,
                prev_bounds=None, prev_layer_voxel_grid=None):
        # No language inputs
        q_trans, q_rot_and_grip, q_ignore_collisions = self._qnet(
            voxel_grid, proprio,
            prev_layer_voxel_grid, bounds, prev_bounds)
        return q_trans, q_rot_and_grip, q_ignore_collisions, voxel_grid


def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)


def quaternion_to_discrete_euler(quaternion, resolution):
    euler = Rotation.from_quat(quaternion).as_euler('xyz', degrees=True) + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc


def discrete_euler_to_quaternion(discrete_euler, resolution):
    euler = (discrete_euler * resolution) - 180
    return Rotation.from_euler('xyz', euler, degrees=True).as_quat()


def point_to_voxel_index(point, voxel_size, coord_bounds):
    bb_mins = np.array(coord_bounds[0:3])
    bb_maxs = np.array(coord_bounds[3:])
    dims_m_one = np.array([voxel_size] * 3) - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([voxel_size] * 3) + 1e-12)
    voxel_indicy = np.minimum(
        np.floor((point - bb_mins) / (res + 1e-12)).astype(np.int32), dims_m_one)
    return voxel_indicy


class YOLHQAttentionAgent:
    """
    Q-attention agent for YOLH without language input.
    Simplified from QAttentionPerActBCAgent - works directly with
    pre-built voxel grids from the pipeline.
    """

    def __init__(
        self,
        perceiver_encoder: nn.Module,
        coordinate_bounds: list,
        voxel_size: int = 100,
        rotation_resolution: float = 5.0,
        lr: float = 0.0005,
        lr_scheduler: bool = False,
        training_iterations: int = 100000,
        num_warmup_steps: int = 3000,
        trans_loss_weight: float = 1.0,
        rot_loss_weight: float = 1.0,
        grip_loss_weight: float = 1.0,
        lambda_weight_l2: float = 1e-6,
        optimizer_type: str = 'lamb',
        num_rotation_classes: int = 72,
    ):
        self._perceiver_encoder = perceiver_encoder
        self._coordinate_bounds = coordinate_bounds
        self._voxel_size = voxel_size
        self._rotation_resolution = rotation_resolution
        self._lr = lr
        self._lr_scheduler = lr_scheduler
        self._training_iterations = training_iterations
        self._num_warmup_steps = num_warmup_steps
        self._trans_loss_weight = trans_loss_weight
        self._rot_loss_weight = rot_loss_weight
        self._grip_loss_weight = grip_loss_weight
        self._lambda_weight_l2 = lambda_weight_l2
        self._optimizer_type = optimizer_type
        self._num_rotation_classes = num_rotation_classes
        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

        self._q = None
        self._optimizer = None
        self._scheduler = None
        self._device = None

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device or torch.device('cpu')

        # Wrap encoder in QFunction (no voxelizer needed - input is already voxelized)
        self._q = nn.ModuleDict({
            'qnet': self._perceiver_encoder
        }).to(self._device)

        if training:
            self._q = DDP(self._q, device_ids=[self._device])

            if self._optimizer_type == 'lamb':
                self._optimizer = Lamb(
                    self._q.parameters(), lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999), adam=False)
            else:
                self._optimizer = torch.optim.Adam(
                    self._q.parameters(), lr=self._lr,
                    weight_decay=self._lambda_weight_l2)

            if self._lr_scheduler:
                self._scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                    self._optimizer,
                    num_warmup_steps=self._num_warmup_steps,
                    num_training_steps=self._training_iterations,
                    num_cycles=self._training_iterations // 10000)

        logging.info('# Q Params: %d' % sum(
            p.numel() for p in self._q.parameters() if p.requires_grad))

    def update(self, voxel_grid, action_trans, action_rot_grip, action_grip,
               proprio=None):
        """
        Training step.
        
        Args:
            voxel_grid: (B, C, V, V, V) pre-built voxel grid
            action_trans: (B, 3) ground truth voxel indices for translation
            action_rot_grip: (B, 4) [rot_x_idx, rot_y_idx, rot_z_idx, grip_idx]
            action_grip: not used separately (included in rot_grip)
            proprio: (B, D) proprioception (optional)
        """
        device = self._device
        bs = voxel_grid.shape[0]
        bounds = torch.tensor(self._coordinate_bounds, device=device).unsqueeze(0)

        # Get the underlying module from DDP
        qnet = self._q.module['qnet'] if hasattr(self._q, 'module') else self._q['qnet']

        # Forward pass
        q_trans, q_rot_grip, q_collision = qnet(
            voxel_grid, proprio, None, bounds, None)

        # Translation loss
        action_trans_one_hot = torch.zeros(
            (bs, 1, self._voxel_size, self._voxel_size, self._voxel_size),
            dtype=torch.float32, device=device)
        for b in range(bs):
            gt = action_trans[b].int()
            action_trans_one_hot[b, 0, gt[0], gt[1], gt[2]] = 1.0

        q_trans_flat = q_trans.view(bs, -1)
        action_trans_flat = action_trans_one_hot.view(bs, -1)
        trans_loss = self._cross_entropy_loss(q_trans_flat, action_trans_flat.argmax(-1))

        # Rotation and grip loss
        rot_loss = torch.tensor(0.0, device=device)
        grip_loss = torch.tensor(0.0, device=device)
        if q_rot_grip is not None:
            rot_x_one_hot = torch.zeros((bs, self._num_rotation_classes), device=device)
            rot_y_one_hot = torch.zeros((bs, self._num_rotation_classes), device=device)
            rot_z_one_hot = torch.zeros((bs, self._num_rotation_classes), device=device)
            grip_one_hot = torch.zeros((bs, 2), device=device)

            for b in range(bs):
                gt = action_rot_grip[b].int()
                rot_x_one_hot[b, gt[0]] = 1.0
                rot_y_one_hot[b, gt[1]] = 1.0
                rot_z_one_hot[b, gt[2]] = 1.0
                grip_one_hot[b, gt[3]] = 1.0

            nrc = self._num_rotation_classes
            rot_loss += self._cross_entropy_loss(q_rot_grip[:, 0*nrc:1*nrc], rot_x_one_hot.argmax(-1))
            rot_loss += self._cross_entropy_loss(q_rot_grip[:, 1*nrc:2*nrc], rot_y_one_hot.argmax(-1))
            rot_loss += self._cross_entropy_loss(q_rot_grip[:, 2*nrc:3*nrc], rot_z_one_hot.argmax(-1))
            grip_loss += self._cross_entropy_loss(q_rot_grip[:, 3*nrc:], grip_one_hot.argmax(-1))

        total_loss = (trans_loss * self._trans_loss_weight +
                      rot_loss * self._rot_loss_weight +
                      grip_loss * self._grip_loss_weight).mean()

        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        if self._lr_scheduler:
            self._scheduler.step()

        return {
            'total_loss': total_loss.item(),
            'trans_loss': trans_loss.mean().item(),
            'rot_loss': rot_loss.mean().item(),
            'grip_loss': grip_loss.mean().item(),
        }

    def save_weights(self, path):
        state = self._q.module.state_dict() if hasattr(self._q, 'module') else self._q.state_dict()
        torch.save(state, path)

    def load_weights(self, path):
        state = torch.load(path, map_location=self._device, weights_only=False)
        if hasattr(self._q, 'module'):
            self._q.module.load_state_dict(state)
        else:
            self._q.load_state_dict(state)
