import os
import sys
from contextlib import contextmanager
import numpy as np
import cv2
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WILOR_ROOT = os.path.join(PROJECT_ROOT, "WiLoR")
sys.path.insert(0, WILOR_ROOT)

from ultralytics import YOLO
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import cam_crop_to_full


# MANO joint indices
# 0=wrist, 1-4=thumb(CMC,MCP,IP,tip), 5-8=index(MCP,PIP,DIP,tip),
# 9-12=middle, 13-16=ring, 17-20=pinky
WRIST_IDX = 0
INDEX_MCP_IDX = 5
RING_MCP_IDX = 13
FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips


def load_yolo_detector(detector_path: str):
    original_torch_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    try:
        torch.load = _patched_load
        return YOLO(detector_path)
    finally:
        torch.load = original_torch_load


def compute_hand_center(joints_3d):
    tips = joints_3d[FINGERTIP_INDICES]
    return tips.mean(axis=0)


def project_point_to_pixel(point_3d, intrinsic):
    """Project a 3D camera-frame point to image pixel coordinates."""
    fx, fy, cx, cy = intrinsic
    x, y, z = point_3d
    if z <= 1e-8:
        return None
    u = int(np.round((x * fx / z) + cx))
    v = int(np.round((y * fy / z) + cy))
    return u, v


def sample_depth_m(depth_frame, u, v, window=2):
    """Sample robust depth (meters) around pixel (u, v) using median of non-zero values."""
    h, w = depth_frame.shape
    if u < 0 or u >= w or v < 0 or v >= h:
        return None

    u0 = max(0, u - window)
    u1 = min(w, u + window + 1)
    v0 = max(0, v - window)
    v1 = min(h, v + window + 1)

    patch = depth_frame[v0:v1, u0:u1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None

    depth_mm = np.median(valid)
    return float(depth_mm) / 1000.0


def backproject_pixel_to_3d(u, v, depth_m, intrinsic):
    """Back-project an image pixel and metric depth to 3D camera coordinates."""
    fx, fy, cx, cy = intrinsic
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    return np.array([x, y, z], dtype=np.float32)


def compute_hand_orientation(joints_3d):
    p_wri = joints_3d[WRIST_IDX]
    p_ind = joints_3d[INDEX_MCP_IDX]
    p_ring = joints_3d[RING_MCP_IDX]

    l_iw = p_ind - p_wri
    l_rw = p_ring - p_wri

    # Z-axis: cross product of two lines
    v_z = np.cross(l_iw, l_rw)
    v_z_norm = v_z / (np.linalg.norm(v_z) + 1e-8)

    # Y-axis: average direction
    v_y = (l_iw + l_rw) / 2.0
    v_y_norm = v_y / (np.linalg.norm(v_y) + 1e-8)

    # X-axis: cross product of Y and Z
    v_x_norm = np.cross(v_y_norm, v_z_norm)

    # 3x3 rotation matrix
    rot_matrix = np.stack([v_x_norm, v_y_norm, v_z_norm], axis=-1)
    return rot_matrix


@contextmanager
def _pushd(path: str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def extract_hand_poses(
    npz_path: str,
    output_path: str,
    checkpoint_path: str,
    cfg_path: str,
    detector_path: str,
    device: str = "auto",
    conf_thres: float = 0.3,
    rescale_factor: float = 2.0,
    fast: bool = False,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device_obj = torch.device(device)

    # Ensure WiLoR internal relative assets (e.g. ./mano_data/...) resolve correctly.
    with _pushd(WILOR_ROOT):
        model, model_cfg = load_wilor(checkpoint_path=checkpoint_path, cfg_path=cfg_path)

    if fast:
        torch.set_float32_matmul_precision("high")
        model = model.half()
        model.backbone = torch.compile(model.backbone)
        model.backbone.skip_blocks = True

    detector = load_yolo_detector(detector_path)

    model = model.to(device_obj)
    detector = detector.to(device_obj)
    model.eval()

    # Load frames
    data = np.load(npz_path, allow_pickle=True)
    rgb_frames = data["rgb"]
    depth_frames = data["depth"]
    intrinsic = data["intrinsic"]  # [fx, fy, cx, cy]
    num_frames = len(rgb_frames)
    print(f"     Processing {num_frames} frames for hand pose extraction")

    positions = np.zeros((num_frames, 3), dtype=np.float32)
    orientations = np.zeros((num_frames, 3, 3), dtype=np.float32)
    valid = np.zeros(num_frames, dtype=np.bool_)

    for i in range(num_frames):
        rgb = rgb_frames[i]
        img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        detections = detector(img_bgr, conf=conf_thres, verbose=False)[0]
        bboxes = []
        is_right = []
        for det in detections:
            bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            hand_cls = det.boxes.cls.cpu().detach().squeeze().item()
            bboxes.append(bbox[:4].tolist())
            is_right.append(hand_cls)

        if len(bboxes) == 0:
            if (i + 1) % 25 == 0:
                print(f"     [{i+1}/{num_frames}] no hand detected")
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        dataset = ViTDetDataset(model_cfg, img_bgr, boxes, right,
                                rescale_factor=rescale_factor, fp16=fast)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        best_conf = -1.0
        best_joints = None

        for batch in dataloader:
            batch = recursive_to(batch, device_obj)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2 * batch["right"] - 1)
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal
            ).detach().cpu().numpy()

            batch_size = batch["img"].shape[0]
            for n in range(batch_size):
                joints = out["pred_keypoints_3d"][n].detach().cpu().numpy()
                hand_flag = batch["right"][n].cpu().numpy()
                joints[:, 0] = (2 * hand_flag - 1) * joints[:, 0]
                cam_t = pred_cam_t_full[n]

                # Transform joints to camera frame (add camera translation)
                joints_cam = joints + cam_t

                # Use detection confidence to pick best hand
                conf = float(bboxes[min(n, len(bboxes)-1)][-1]) if len(bboxes[0]) > 4 else 1.0
                if conf > best_conf:
                    best_conf = conf
                    best_joints = joints_cam

        if best_joints is not None:
            center_wilor = compute_hand_center(best_joints)

            # Depth-correct center: use projected 2D hand center and RGB-D depth.
            uv = project_point_to_pixel(center_wilor, intrinsic)
            if uv is not None:
                u, v = uv
                depth_m = sample_depth_m(depth_frames[i], u, v, window=2)
                if depth_m is not None:
                    positions[i] = backproject_pixel_to_3d(u, v, depth_m, intrinsic)
                else:
                    positions[i] = center_wilor
            else:
                positions[i] = center_wilor

            orientations[i] = compute_hand_orientation(best_joints)
            valid[i] = True

        if (i + 1) % 25 == 0:
            print(f"     [{i+1}/{num_frames}]")

    np.savez_compressed(
        output_path,
        positions=positions,
        orientations=orientations,
        valid=valid,
    )
    print(f"     Saved hand poses ({valid.sum()}/{num_frames} valid) to {output_path}")
    return output_path