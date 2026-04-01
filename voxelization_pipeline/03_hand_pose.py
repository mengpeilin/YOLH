import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import torch
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.utils.renderer import cam_crop_to_full

# TODO: Later you can replace this list by loading from a file.
TARGET_FRAME_INDICES = [0, 10, 20]


def load_yolo_detector_compat(detector_path: str):
    """
    PyTorch >= 2.6 changed torch.load default to weights_only=True,
    which can break older Ultralytics checkpoints. Temporarily force
    weights_only=False only while loading the detector checkpoint.
    """
    original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    try:
        torch.load = _patched_torch_load
        return YOLO(detector_path)
    finally:
        torch.load = original_torch_load


def project_full_img(points, cam_trans, focal_length, img_res):
    camera_center = [img_res[0] / 2.0, img_res[1] / 2.0]
    k = torch.eye(3, dtype=torch.float32)
    k[0, 0] = focal_length
    k[1, 1] = focal_length
    k[0, 2] = camera_center[0]
    k[1, 2] = camera_center[1]

    points = points + cam_trans
    points = points / points[..., -1:]
    v_2d = (k @ points.T).T
    return v_2d[..., :-1]


def load_selected_color_frames_from_npz(
    npz_path: str,
    frames_key: str,
    frame_indices: List[int],
) -> Dict[int, np.ndarray]:
    """
    Read selected color frames from a NPZ file.
    Returns: {frame_idx: rgb_image}
    """
    target_set = set(frame_indices)
    if not target_set:
        return {}

    data = np.load(npz_path, allow_pickle=True)
    if frames_key in data:
        all_frames = data[frames_key]
    else:
        candidate_keys = ["rgb_frames", "color_frames", "images", "frames"]
        all_frames = None
        for key in candidate_keys:
            if key in data:
                all_frames = data[key]
                frames_key = key
                print(f"Use fallback frame key from NPZ: {frames_key}")
                break
        if all_frames is None:
            raise ValueError(
                f"Frame key '{frames_key}' not found in NPZ. "
                f"Available keys: {list(data.keys())}"
            )

    # Guard against scalar metadata keys (e.g. color_topic) being passed as frames_key.
    if not hasattr(all_frames, "shape") or all_frames.shape == ():
        raise ValueError(
            f"NPZ key '{frames_key}' is not a frame array. "
            f"Please use a key like 'rgb_frames'. Available keys: {list(data.keys())}"
        )

    frames = {}
    for frame_idx in sorted(target_set):
        if frame_idx < 0 or frame_idx >= len(all_frames):
            continue

        rgb = all_frames[frame_idx]
        if rgb is None:
            continue
        rgb = np.asarray(rgb)

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Frame {frame_idx} has invalid shape: {rgb.shape}. Expected [H, W, 3]")

        if rgb.dtype != np.uint8:
            if np.max(rgb) <= 1.0:
                rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
            else:
                rgb = rgb.clip(0, 255).astype(np.uint8)

        frames[frame_idx] = rgb

    missing = sorted(list(target_set - set(frames.keys())))
    if missing:
        print(f"Warning: some requested frames were not found in NPZ: {missing}")

    return frames


def infer_hand_joints_on_frames(
    frames: Dict[int, np.ndarray],
    checkpoint_path: str,
    cfg_path: str,
    detector_path: str,
    device_str: str,
    conf_thres: float,
    rescale_factor: float,
    fast: bool,
):
    model, model_cfg = load_wilor(checkpoint_path=checkpoint_path, cfg_path=cfg_path)

    if fast:
        torch.set_float32_matmul_precision("high")
        model = model.half()
        model.backbone = torch.compile(model.backbone)
        model.backbone.skip_blocks = True

    detector = load_yolo_detector_compat(detector_path)

    if device_str == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(device_str)

    model = model.to(device)
    detector = detector.to(device)
    model.eval()

    all_results = []

    for frame_idx in sorted(frames.keys()):
        rgb = frames[frame_idx]
        img_bgr = rgb[:, :, ::-1].copy()

        detections = detector(img_bgr, conf=conf_thres, verbose=False)[0]
        bboxes = []
        is_right = []

        for det in detections:
            bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            hand_cls = det.boxes.cls.cpu().detach().squeeze().item()
            bboxes.append(bbox[:4].tolist())
            is_right.append(hand_cls)

        frame_payload = {
            "frame_index": int(frame_idx),
            "num_hands": int(len(bboxes)),
            "hands": [],
        }

        if len(bboxes) == 0:
            all_results.append(frame_payload)
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        dataset = ViTDetDataset(
            model_cfg,
            img_bgr,
            boxes,
            right,
            rescale_factor=rescale_factor,
            fp16=fast,
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        hand_counter = 0
        for batch in dataloader:
            batch = recursive_to(batch, device)

            with torch.no_grad():
                out = model(batch)

            multiplier = (2 * batch["right"] - 1)
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]

            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(
                pred_cam,
                box_center,
                box_size,
                img_size,
                scaled_focal_length,
            ).detach().cpu().numpy()

            batch_size = batch["img"].shape[0]
            for n in range(batch_size):
                joints = out["pred_keypoints_3d"][n].detach().cpu().numpy()
                hand_flag = batch["right"][n].cpu().numpy()

                joints[:, 0] = (2 * hand_flag - 1) * joints[:, 0]
                cam_t = pred_cam_t_full[n]

                joints_2d = project_full_img(
                    torch.from_numpy(joints).float(),
                    torch.from_numpy(cam_t).float(),
                    float(scaled_focal_length),
                    img_size[n].detach().cpu().numpy(),
                ).detach().cpu().numpy()

                bbox_for_hand = bboxes[hand_counter] if hand_counter < len(bboxes) else None
                frame_payload["hands"].append(
                    {
                        "hand_index": int(hand_counter),
                        "is_right": bool(hand_flag > 0.5),
                        "bbox_xyxy": bbox_for_hand,
                        "camera_translation": cam_t.tolist(),
                        "joints_3d": joints.tolist(),
                        "joints_2d": joints_2d.tolist(),
                    }
                )
                hand_counter += 1

        all_results.append(frame_payload)

    return all_results


def build_parser():
    parser = argparse.ArgumentParser(description="Extract hand joints for selected frames from NPZ")
    parser.add_argument("--npz-path", type=str, required=True, help="Input NPZ path")
    parser.add_argument(
        "--frames-key",
        type=str,
        default="rgb_frames",
        help="NPZ key for RGB frames array",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="out_selected_frame_hand_joints.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="./pretrained_models/wilor_final.ckpt",
    )
    parser.add_argument(
        "--cfg-path",
        type=str,
        default="./pretrained_models/model_config.yaml",
    )
    parser.add_argument(
        "--detector-path",
        type=str,
        default="./pretrained_models/detector.pt",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")
    parser.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold")
    parser.add_argument("--rescale-factor", type=float, default=2.0, help="BBox rescale factor")
    parser.add_argument("--fast", action="store_true", default=False)
    return parser


def main():
    args = build_parser().parse_args()

    frame_indices = sorted(set(TARGET_FRAME_INDICES))
    if len(frame_indices) == 0:
        raise ValueError("TARGET_FRAME_INDICES is empty")

    frames = load_selected_color_frames_from_npz(
        npz_path=args.npz_path,
        frames_key=args.frames_key,
        frame_indices=frame_indices,
    )

    results = infer_hand_joints_on_frames(
        frames=frames,
        checkpoint_path=args.checkpoint_path,
        cfg_path=args.cfg_path,
        detector_path=args.detector_path,
        device_str=args.device,
        conf_thres=args.conf,
        rescale_factor=args.rescale_factor,
        fast=args.fast,
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "npz_path": args.npz_path,
        "frames_key": args.frames_key,
        "frame_indices_requested": frame_indices,
        "frame_indices_processed": [r["frame_index"] for r in results],
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print(f"Saved hand joints to: {output_path}")


if __name__ == "__main__":
    main()
