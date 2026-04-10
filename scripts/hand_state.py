import sys
import os
import numpy as np
import torch
import trimesh
import open3d as o3d

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAMER_ROOT = os.path.join(PROJECT_ROOT, "dependencies", "phantom-hamer")
sys.path.insert(0, HAMER_ROOT)
from hamer.models import HAMER, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import cam_crop_to_full
from hamer.utils.geometry import perspective_projection
from hamer.configs import get_config
from pathlib import Path

def _load_hamer(checkpoint_path=None):
    if checkpoint_path is None:
        root_dir = HAMER_ROOT
        checkpoint_path = str(Path(root_dir, DEFAULT_CHECKPOINT))
    else:
        root_dir = str(Path(checkpoint_path).parent.parent.parent)

    model_cfg_path = str(Path(checkpoint_path).parent.parent / "model_config.yaml")
    model_cfg = get_config(model_cfg_path, update_cachedir=True)

    model_cfg.defrost()
    model_cfg.MANO.DATA_DIR = os.path.join(root_dir, model_cfg.MANO.DATA_DIR)
    model_cfg.MANO.MODEL_PATH = os.path.join(
        root_dir, model_cfg.MANO.MODEL_PATH.replace("./", "")
    )
    model_cfg.MANO.MEAN_PARAMS = os.path.join(
        root_dir, model_cfg.MANO.MEAN_PARAMS.replace("./", "")
    )
    model_cfg.freeze()

    if (
        model_cfg.MODEL.BACKBONE.TYPE == "vit"
        and "BBOX_SHAPE" not in model_cfg.MODEL
    ):
        model_cfg.defrost()
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()

    if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        model_cfg.freeze()

    model = HAMER.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    model.cuda()
    model.eval()
    return model, model_cfg


def _run_hamer_on_frame(model, model_cfg, img_rgb, bbox, hand_side, rescale_factor=2.0):
    bboxes = bbox.reshape(1, 4)
    is_right_val = 1 if hand_side == "right" else 0
    is_right = np.array([is_right_val])

    img_h, img_w = img_rgb.shape[:2]
    scaled_focal_length = (
        model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(img_w, img_h)
    )
    camera_center = torch.tensor([img_w, img_h], dtype=torch.float).reshape(1, 2) / 2.0

    dataset = ViTDetDataset(
        model_cfg, img_rgb, bboxes, is_right, rescale_factor=rescale_factor
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    for batch in dataloader:
        batch = recursive_to(batch, "cuda")
        with torch.no_grad():
            out = model(batch)

        kpts_3d = out["pred_keypoints_3d"][0].detach().cpu().numpy()  # (21, 3)
        verts = out["pred_vertices"][0].detach().cpu().numpy()  # (778, 3)

        if hand_side == "left":
            kpts_3d[:, 0] = -kpts_3d[:, 0]
            verts[:, 0] = -verts[:, 0]

        # Camera translation
        multiplier = 2 * batch["right"] - 1
        pred_cam = out["pred_cam"].clone()
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        W_H = batch["img_size"].float()

        T_cam_pred = cam_crop_to_full(
            pred_cam, box_center, box_size, W_H, scaled_focal_length
        )
        T_cam = T_cam_pred[0]

        kpts_3d_cam = kpts_3d + T_cam.cpu().numpy()
        verts_cam = verts + T_cam.cpu().numpy()

        kpts_2d = _project_3d_to_2d(
            kpts_3d, img_w, img_h, scaled_focal_length, camera_center, T_cam
        )

        return {
            "kpts_3d": kpts_3d_cam,
            "kpts_2d": kpts_2d,
            "verts": verts_cam,
            "T_cam_pred": T_cam,
            "scaled_focal_length": scaled_focal_length,
            "camera_center": camera_center,
            "img_w": img_w,
            "img_h": img_h,
        }

    return None


def _project_3d_to_2d(kpts_3d, img_w, img_h, scaled_focal_length, camera_center, T_cam):
    rotation = torch.eye(3).unsqueeze(0).cuda()
    kpts = torch.tensor(kpts_3d, dtype=torch.float32).cuda()
    T = T_cam.clone().cuda()
    focal = torch.tensor(
        [[scaled_focal_length, scaled_focal_length]], dtype=torch.float32
    )

    kpts_2d = perspective_projection(
        kpts.reshape(1, -1, 3),
        rotation=rotation,
        translation=T.reshape(1, -1),
        focal_length=focal,
        camera_center=camera_center.cuda(),
    ).reshape(1, -1, 2)

    return np.rint(kpts_2d[0].cpu().numpy()).astype(np.int32)

def _get_visible_points(mesh, origin):
    """Return vertices of *mesh* visible from *origin* via ray casting."""
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    pts = mesh.vertices
    vectors = pts - origin
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    directions = vectors / norms

    hit_tri = intersector.intersects_first(
        np.tile(origin, (len(pts), 1)), directions
    )
    vis_idx = np.unique(mesh.faces[hit_tri])
    return pts[vis_idx].astype(np.float32), vis_idx


def _pixels_to_3d(pixels_2d, depth_img, intrinsics):
    px = pixels_2d[:, 0].astype(int)
    py = pixels_2d[:, 1].astype(int)
    depth_mm = depth_img[py, px].astype(np.float32)
    depth_m = depth_mm / 1000.0

    X = (px - intrinsics["cx"]) / intrinsics["fx"] * depth_m
    Y = (py - intrinsics["cy"]) / intrinsics["fy"] * depth_m
    return np.stack([X, Y, depth_m], axis=1)


def _mask_to_pointcloud(mask, depth_img, intrinsics):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    pixels = np.stack([xs, ys], axis=1)
    pts_3d = _pixels_to_3d(pixels, depth_img, intrinsics)
    valid = pts_3d[:, 2] > 0
    pts_3d = pts_3d[valid]
    if len(pts_3d) == 0:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_3d)
    pcd.remove_non_finite_points()
    return pcd


def _depth_align_frame(hamer_out, depth_img, mask, intrinsics, faces):
    verts = hamer_out["verts"]
    kpts_3d = hamer_out["kpts_3d"]

    # 1. Create hand mesh
    mesh = trimesh.Trimesh(verts.copy(), faces.copy(), process=False)

    # 2. Visible vertices via ray-cast
    vis_verts, _ = _get_visible_points(mesh, origin=np.array([0.0, 0.0, 0.0]))
    if len(vis_verts) < 10:
        return kpts_3d

    # 3. Project visible verts → 2D → sample depth → 3D
    vis_2d = _project_3d_to_2d(
        (vis_verts - hamer_out["T_cam_pred"].cpu().numpy()).astype(np.float32),
        hamer_out["img_w"],
        hamer_out["img_h"],
        hamer_out["scaled_focal_length"],
        hamer_out["camera_center"],
        hamer_out["T_cam_pred"],
    )
    H, W = depth_img.shape[:2]
    valid = (
        (vis_2d[:, 0] >= 0)
        & (vis_2d[:, 0] < W)
        & (vis_2d[:, 1] >= 0)
        & (vis_2d[:, 1] < H)
    )
    vis_2d = vis_2d[valid]
    vis_verts = vis_verts[valid]
    if len(vis_2d) < 10:
        return kpts_3d

    depth_3d = _pixels_to_3d(vis_2d, depth_img, intrinsics)

    # 4. Hand + arm point cloud from mask
    mask_pcd = _mask_to_pointcloud(mask, depth_img, intrinsics)
    if mask_pcd is None or len(mask_pcd.points) < 10:
        return kpts_3d

    # 5. Initial transform (median translation)
    T_init = np.eye(4)
    translation = np.nanmedian(depth_3d - vis_verts, axis=0)
    if not np.isnan(translation).any():
        T_init[:3, 3] = translation

    # 6. ICP refinement
    hamer_pcd = o3d.geometry.PointCloud()
    hamer_pcd.points = o3d.utility.Vector3dVector(vis_verts)
    try:
        result = o3d.pipelines.registration.registration_icp(
            source=hamer_pcd,
            target=mask_pcd,
            max_correspondence_distance=0.025,
            init=T_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        T = result.transformation
    except Exception:
        T = T_init

    # 7. Apply to keypoints
    kpts_h = np.hstack([kpts_3d, np.ones((21, 1))])
    kpts_aligned = (T @ kpts_h.T).T[:, :3]
    return kpts_aligned.astype(np.float32)

def estimate_hand_states(
    npz_path: str,
    hand_bboxes_path: str,
    masks_path: str,
    output_path: str,
    hand_side: str = "right",
    hamer_checkpoint: str = None,
    rescale_factor: float = 2.0,
):
    data = np.load(npz_path, allow_pickle=True)
    rgb_frames = data["rgb"]
    depth_frames = data["depth"]
    intrinsic = data["intrinsic"]  # [fx, fy, cx, cy]
    intrinsics_dict = {
        "fx": float(intrinsic[0]),
        "fy": float(intrinsic[1]),
        "cx": float(intrinsic[2]),
        "cy": float(intrinsic[3]),
    }

    bbox_data = np.load(hand_bboxes_path, allow_pickle=True)
    bboxes = bbox_data["bboxes"]
    bbox_detected = bbox_data["hand_detected"]

    mask_data = np.load(masks_path, allow_pickle=True)
    masks = mask_data["hand_masks"] if "hand_masks" in mask_data else mask_data["masks"]

    N = len(rgb_frames)
    print(f"     Processing {N} frames for hand state estimation")

    # Load HaMeR
    model, model_cfg = _load_hamer(hamer_checkpoint)
    faces_right = model.mano.faces
    faces_left = faces_right[:, [0, 2, 1]]
    faces = faces_left if hand_side == "left" else faces_right

    # Output arrays
    kpts_3d_all = np.zeros((N, 21, 3), dtype=np.float32)
    kpts_2d_all = np.zeros((N, 21, 2), dtype=np.int32)
    hand_detected = np.zeros(N, dtype=bool)

    for i in range(N):
        if not bbox_detected[i]:
            continue

        hamer_out = _run_hamer_on_frame(
            model, model_cfg, rgb_frames[i], bboxes[i], hand_side, rescale_factor
        )
        if hamer_out is None:
            continue

        # Depth alignment via ICP
        kpts_3d_aligned = _depth_align_frame(
            hamer_out, depth_frames[i], masks[i], intrinsics_dict, faces
        )

        kpts_3d_all[i] = kpts_3d_aligned
        kpts_2d_all[i] = hamer_out["kpts_2d"]
        hand_detected[i] = True

        if (i + 1) % 25 == 0:
            print(f"     [{i + 1}/{N}] detected={hand_detected[:i+1].sum()}")

    np.savez_compressed(
        output_path,
        kpts_3d=kpts_3d_all,
        kpts_2d=kpts_2d_all,
        hand_detected=hand_detected,
    )
    print(
        f"     Saved hand states ({N} frames, {hand_detected.sum()} detected) "
        f"to {output_path}"
    )
