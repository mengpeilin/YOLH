# YOLH

YOLH is a robot imitation learning project built on RGB-D demonstrations. The repository focuses on three core workflows:

- Offline data pipeline: convert ROS2 bags into trainable point-cloud/action datasets.
- Policy training: train the YOLH policy with sparse 3D encoding and temporal action decoding.
- Online deployment: run two-machine observation, remote inference, action feedback, and SO-101 control.

The repository also includes vendored third-party source trees. Day-to-day development is usually concentrated in the root scripts, scripts, yolh_pipeline, policy, interface, and dataset directories.

## Repository Layout

```text
YOLH/
├── arm_control.py              # Robot-side control client
├── inference.py                # Inference server
├── test_inference.py           # FK replay debug server
├── train.py                    # Training entrypoint
├── run_yolh_pipeline.py        # One-command offline pipeline launcher
├── configs/                    # Inference, calibration, and pipeline configs
├── dataset/                    # Training dataset wrapper
├── interface/                  # Robot interface, IK, and ZMQ transport
├── policy/                     # YOLH model and inference utilities
├── scripts/                    # Core logic for each processing stage
├── yolh_pipeline/              # Batch wrappers for stages 00-06
├── URDF/                       # SO-ARM100 / SO-101 models and hardware assets
├── lerobot/                    # Vendored LeRobot code
└── dependencies/               # Source trees for SAM2, HaMeR, MinkowskiEngine, PyTorch3D, etc.
```

## Environment Setup

Base environment for training and inference:

```bash
conda env create -f environment.yml
conda activate yolh
```

The offline pipeline also expects two additional environments by default:

- `sam2`: used by `02_mask_generation.py`
- `phantom`: used for hand bbox detection, hand state estimation, and gripper action estimation

`run_yolh_pipeline.py` lets you override these names with `--sam2-env` and `--phantom-env`.

Notes:

- `dependencies/` keeps source code for related dependencies, but not all of them are installed automatically in the base environment.
- `scripts/hand_state.py` imports HaMeR directly from `dependencies/phantom-hamer/`.
- If you plan to run the full offline pipeline end-to-end, check the README files inside the vendored dependency folders as well.

## Data Pipeline

Full pipeline entrypoint:

```bash
python run_yolh_pipeline.py \
  --input-dir /path/to/ros2bags \
  --output-dir /path/to/output \
  --task-name pick_cup \
  --config configs/pipeline.yaml
```

Each `rosbag*` generates a same-name session directory under the output path. Stage outputs are:

| Step | Script | Main Inputs | Main Outputs |
| --- | --- | --- | --- |
| 00 | `yolh_pipeline/00_ros2bag_process.py` | ROS2 bag | `raw.npz` |
| 01 | `yolh_pipeline/01_hand_bbox.py` | `raw.npz` | `hand_bboxes.npz` |
| 02 | `yolh_pipeline/02_mask_generation.py` | `raw.npz`, hand bboxes | `arm_bboxes.npz`, `masks.npz` |
| 03 | `yolh_pipeline/03_hand_state.py` | `raw.npz`, hand bboxes, mask | `hand_state.npz` |
| 04 | `yolh_pipeline/04_gripper_action.py` | `hand_state.npz` | `gripper_action.npz` |
| 05 | `yolh_pipeline/05_point_cloud.py` | `raw.npz`, `masks.npz`, `gripper_action.npz` | `episodes.npz` |
| 06 | `yolh_pipeline/06_generate_dataset.py` | multiple sessions | `train_dataset.npz` |

`train_dataset.npz` includes at least the following fields:

- `clouds`
- `actions`
- `actions_normalized`
- `trans_min`
- `trans_max`
- `max_gripper_width`

These normalization statistics are loaded again in `inference.py` to map policy outputs back to physical units.

## Training

Single GPU:

```bash
python train.py \
  --dataset /path/to/train_dataset.npz \
  --ckpt-dir /path/to/checkpoints
```

Multi-GPU:

```bash
torchrun --nproc_per_node=2 train.py \
  --dataset /path/to/train_dataset.npz \
  --ckpt-dir /path/to/checkpoints \
  --batch-size 48
```

Training artifacts:

- periodic checkpoints: `policy_epoch_*.ckpt`
- final model: `policy_last.ckpt`

Model implementation lives in `policy/yolh/`. The current training entrypoint uses:

- a sparse 3D encoder
- a Transformer readout layer
- an action diffusion decoder

## Online Inference and Control

YOLH online deployment follows a two-machine setup:

- robot side: capture RGB-D, read joint angles, execute actions
- inference side: receive observations, build point clouds, run policy, send action chunks

Start on the inference machine:

```bash
python inference.py \
  --ckpt /path/to/policy_last.ckpt \
  --config configs/inference.yaml \
  --dataset-meta /path/to/train_dataset.npz \
  --urdf URDF/SO-ARM100/Simulation/SO101/so101_new_calib.urdf
```

Start on the robot machine:

```bash
python arm_control.py \
  --config configs/inference.yaml \
  --host 192.168.1.100 \
  --serial-port /dev/ttyACM0
```

For debugging:

```bash
python test_inference.py \
  --config configs/inference.yaml \
  --dataset-meta /path/to/train_dataset.npz \
  --urdf URDF/SO-ARM100/Simulation/SO101/so101_new_calib.urdf
```

This mode does not run the policy. It replays actions from FK of current joint angles, which is useful for validating ZMQ transport, coordinate conventions, and execution flow.

## Key Configs

- `configs/pipeline.yaml`: per-stage offline pipeline parameters
- `configs/inference.yaml`: camera intrinsics, camera-to-base extrinsics, workspace, and inference parameters
- `configs/lerobot.json`: motor IDs, homing offsets, and range calibration

## Scope Boundaries

Focus here for project development:

- `scripts/` and `yolh_pipeline/`: data processing
- `policy/`: model and inference logic
- `interface/`: robot interface and communication
- `dataset/`: training data loading

These directories are mainly vendored dependencies and should generally be left unchanged unless you intentionally patch upstream code:

- `dependencies/`
- `lerobot/`

Also, `policy/README.md` records attribution and source information for part of the policy code.
