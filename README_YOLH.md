# YOLH - You Only Learn from Humans

## 项目概述

使用固定在桌上方的RealSense D435i录制人手执行特定任务的视频，通过数据处理pipeline生成voxel grid和action，训练基于PerAct（去除language输入）的模型。每个任务训练一个独立模型。

## 项目结构

```
EECS467/
├── scripts/                          # 核心处理逻辑
│   ├── ros2bag_to_npz.py            # ROS2 bag → NPZ
│   ├── mask_generation.py            # SAM2 手臂掩码生成
│   ├── hand_openclose.py            # 手部开合检测
│   ├── hand_pose.py                 # 手部位姿提取(WiLoR)
│   ├── keyframe_detection.py        # 关键帧检测
│   ├── voxelize_and_insert_gripper.py  # 体素化 + 夹爪插入
│   └── merge_episodes.py           # 合并训练数据
│
├── voxelization_pipeline/           # 批量处理编排器
│   ├── 00_ros2bag_process.py        # 批量ROS2 bag转换
│   ├── 01_mask_generation.py        # 批量掩码生成
│   ├── 02_hand_openclose.py         # 批量手部状态检测
│   ├── 03_hand_pose.py              # 批量手部位姿提取
│   ├── 04_keyframe_detection.py     # 批量关键帧检测
│   ├── 05_voxelization.py           # 批量体素化
│   └── 06_gripper_insertion.py      # 合并所有session为训练集
│
├── run_voxelization_pipeline.py     # Pipeline运行器(生成shell脚本)
├── train.py                         # 训练脚本(Docker中运行)
│
├── peract/agents/peract_bc/
│   ├── perceiver_io.py              # [新] 无语言输入的PerceiverIO
│   ├── yolh_agent.py                # [新] YOLH Q-attention agent
│   ├── perceiver_lang_io.py         # [原] 带语言的PerceiverIO
│   └── ...
│
├── Dockerfile                        # Docker镜像(PerAct训练环境)
└── run_peract.sh                    # Docker容器启动脚本
```

## 数据流

```
ROS2 bags (input_dir/)
    │
    ▼ Step 00 (系统Python, 不激活任何conda)
data/{session}/raw.npz
    │   └── rgb(N,H,W,3) + depth(N,H,W) + intrinsic(4,)
    │
    ├──▶ Step 01 (conda: sam2) ──▶ data/{session}/masks.npz
    │       └── masks(N,H,W) bool - 手臂区域为True
    │
    ├──▶ Step 02 (conda: handstate) ──▶ data/{session}/hand_states.npy
    │       └── (N,) bool - True=张开, False=闭合
    │
    └──▶ Step 03 (conda: wilor) ──▶ data/{session}/hand_pose.npz
            └── positions(N,3) + orientations(N,3,3) + valid(N,) bool
                        │
                        ▼ Step 04 (无特殊环境)
              data/{session}/keyframes.npy
                  └── (K,) int64 - 关键帧索引
                        │
                        ▼ Step 05 (无特殊环境)
              data/{session}/episodes.npz
                  └── voxel_grids(K,100,100,100,4) + actions(K,13)
                        │
                        ▼ Step 06 (无特殊环境)
              data/train_dataset.npz  ← 合并所有session
                        │
                        ▼ (Docker中)
              train.py ──▶ checkpoints/{task}/best.pt
```

## 使用方法

### 1. 运行Pipeline

#### 方式一：生成并运行完整脚本
```bash
cd ~/EECS467
python run_voxelization_pipeline.py \
    --input-dir /path/to/ros2bags \
    --output-dir data/ \
    --task-name pick_cup
```

#### 方式二：分步手动运行

**Step 00: ROS2 Bag → NPZ** (必须在系统Python下运行，不激活任何conda)
```bash
conda deactivate  # 确保没有激活conda
python voxelization_pipeline/00_ros2bag_process.py \
    --input-dir /path/to/ros2bags \
    --output-dir data/
```

**Step 01: SAM2 掩码生成** (交互式 - 需要为每条数据画bbox)
```bash
conda activate sam2
python voxelization_pipeline/01_mask_generation.py --data-dir data/
```

**Step 02: 手部开合检测**
```bash
conda activate handstate
python voxelization_pipeline/02_hand_openclose.py --data-dir data/
```

**Step 03: 手部位姿提取**
```bash
conda activate wilor
python voxelization_pipeline/03_hand_pose.py --data-dir data/
```

**Step 04: 关键帧检测**
```bash
python voxelization_pipeline/04_keyframe_detection.py --data-dir data/
```

**Step 05: 体素化 + 夹爪插入**
```bash
python voxelization_pipeline/05_voxelization.py --data-dir data/
```

**Step 06: 合并训练数据集**
```bash
python voxelization_pipeline/06_gripper_insertion.py \
    --data-dir data/ \
    --output-path data/train_dataset.npz \
    --task-name pick_cup
```

### 2. Docker中训练模型

#### 构建Docker镜像
```bash
cd ~/EECS467
docker build -t peract .
```

#### 启动Docker容器
```bash
# 确保training数据在 ~/EECS467/data/ 目录下
bash run_peract.sh
```

如果容器已存在:
```bash
docker start -ai peract_dev
```

#### 在Docker容器内训练
```bash
# 进入容器后:
cd /app

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

# 训练多个任务:
python train.py \
    --dataset /app/data/train_dataset_pour_water.npz \
    --task-name pour_water \
    --output-dir /app/data/checkpoints/pour_water \
    --batch-size 2 \
    --epochs 100
```

#### 查看训练结果
```bash
# 训练完成后,模型保存在:
# /app/data/checkpoints/{task_name}/best.pt   - 最佳模型
# /app/data/checkpoints/{task_name}/final.pt  - 最终模型
# 
# 宿主机路径: ~/EECS467/data/checkpoints/{task_name}/
```

### 3. 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--voxel-size` | 100 | 体素网格分辨率 (100³) |
| `--rotation-resolution` | 5 | 旋转离散化分辨率 (5°) |
| `--num-latents` | 2048 | Perceiver latent向量数 |
| `--transformer-depth` | 6 | Transformer层数 |
| `--batch-size` | 2 | 批次大小 (根据GPU显存调整) |
| `--lr` | 5e-4 | 学习率 |
| `--optimizer` | lamb | 优化器 (lamb/adam) |

### 4. 注意事项

1. **Step 00** 必须在系统Python下运行，不能激活任何conda环境，否则ROS2会报错
2. **Step 01** 是交互式的，需要为每条数据在第一帧上画手臂的bounding box
3. 所有步骤都是**幂等**的 - 如果中间文件已存在会自动跳过
4. 如果某步失败，修复问题后可以直接重新运行该步骤
5. **训练**必须在Docker中进行（PerAct依赖与本地环境不兼容）
6. Action格式: `[x, y, z, r00-r22, gripper_open]` (13维)
   - position(3): 手部中心3D坐标
   - rotation(9): 3×3旋转矩阵展平
   - gripper_open(1): 1=张开, 0=闭合

### 5. 模型架构

模型基于PerAct的Perceiver Transformer，主要修改：
- **移除语言输入**: 不使用CLIP编码的语言目标
- **纯视觉输入**: 10通道体素网格 (RGB + XYZ坐标 + 占据标志 + 归一化体素坐标)
- **无proprioception**: 不使用机器人本体感知数据
- **输出**: 平移Q值 + 离散化旋转 + 夹爪开合
- **每个任务独立模型**: 不同任务训练不同权重
