# YOLH — You Only Learn from Humans

A robotics imitation learning framework that trains manipulation models directly from human demonstrations captured with an overhead RGB-D camera — no language conditioning, no simulation required.

## Overview

YOLH records a human hand performing manipulation tasks, processes the raw RGB-D video through a 6-stage pipeline, and trains a vision-based PerAct model to reproduce the behavior. The trained model outputs 6-DoF actions (3D position + rotation + gripper state) from voxelized scene observations.

**Hardware:** Intel RealSense D435i mounted above a table + [SO-ARM100](SO-ARM100/) 6-DOF robot arm

## Pipeline

```
ROS2 bag → RGB-D frames → arm mask → hand state → hand pose → keyframes → voxels → training data
```

| Step | Script | Conda Env | Output |
|------|--------|-----------|--------|
| 00 — Extract frames | `00_ros2bag_process.py` | system Python | `raw.npz` |
| 01 — Mask arm | `01_mask_generation.py` | `sam2` | `masks.npz` |
| 02 — Hand open/close | `02_hand_openclose.py` | `handstate` | `hand_states.npy` |
| 03 — Hand pose | `03_hand_pose.py` | `wilor` | `hand_pose.npz` |
| 04 — Keyframes | `04_keyframe_detection.py` | — | `keyframes.npy` |
| 05 — Voxelize | `05_voxelization.py` | — | `episodes.npz` |
| 06 — Merge dataset | `06_generate_dataset.py` | — | `train_dataset.npz` |

## Quick Start

### 1. Run the full pipeline

```bash
python run_voxelization_pipeline.py \
    --input-dir /path/to/ros2bags \
    --output-dir data/ \
    --task-name pick_cup
```

Or run steps individually from `voxelization_pipeline/`.

### 2. Train the model (Docker)

```bash
# Build image
docker build -t peract .

# Launch container
bash run_peract.sh

# Train inside container
python train.py \
    --dataset /app/data/train_dataset.npz \
    --task-name pick_cup \
    --output-dir /app/data/checkpoints/pick_cup \
    --batch-size 2 \
    --epochs 100 \
    --lr 5e-4 \
    --optimizer lamb \
    --voxel-size 100 \
    --rotation-resolution 5 \
    --num-latents 2048 \
    --transformer-depth 6 \
    --save-every 10
```

Checkpoints are saved to `data/checkpoints/{task_name}/`.

## Repository Structure

```
YOLH/
├── run_voxelization_pipeline.py   # Full pipeline orchestrator
├── train.py                       # PerAct training script
├── Dockerfile / run_peract.sh     # Docker environment
│
├── scripts/                       # Core processing modules
├── voxelization_pipeline/         # Batch pipeline wrappers (00–06)
│
├── peract/                        # PerAct model (vision-only variant)
│   └── agents/peract_bc/
│       ├── perceiver_io.py        # Vision-only PerceiverIO encoder
│       └── yolh_agent.py         # YOLH Q-attention agent
│
├── WiLoR/                         # Hand pose estimation (CVPR 2025)
├── hand_object_detector/          # Hand-object detection (CVPR 2020)
├── sam2/                          # Segment Anything Model v2
└── SO-ARM100/                     # Robot arm hardware (STL files, specs)
```

## Action Format

Actions are 13-dimensional vectors:

```
[x, y, z,                     # 3D gripper position (world coords)
 r00, r01, r02,                # 3x3 rotation matrix (row-major)
 r10, r11, r12,
 r20, r21, r22,
 gripper_open]                 # 1 = open, 0 = closed
```

## Pretrained Models

| Model | Location | Purpose |
|-------|----------|---------|
| WiLoR detector | `WiLoR/pretrained_models/detector.pt` | Hand detection |
| WiLoR reconstructor | `WiLoR/pretrained_models/wilor_final.ckpt` | Hand mesh |
| Hand-object detector | (download from Google Drive) | Interaction detection |
| SAM2 checkpoints | `sam2/checkpoints/` | Arm segmentation |

## Key Dependencies

- PyTorch 1.11.0 (CUDA 11.3)
- [WiLoR](https://github.com/rolpotamias/WiLoR) — 3D hand pose estimation
- [SAM2](https://github.com/facebookresearch/sam2) — segmentation
- [PerAct](https://github.com/peract/peract) — voxel-based transformer policy
- RLBench / PyRep / YARR — robotics simulation stack
- PyTorch3D, trimesh — 3D geometry

## Notes

- Per-task models: train a separate model for each manipulation task (e.g., `pick_cup`, `pour_water`)
- Voxel grid resolution: 100×100×100 by default
- Each pipeline step runs in its own conda environment due to conflicting dependencies
