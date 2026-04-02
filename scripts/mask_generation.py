import os
import sys
import numpy as np
import cv2
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM2_ROOT = os.path.join(PROJECT_ROOT, "sam2")
sys.path.insert(0, SAM2_ROOT)

from sam2.build_sam import build_sam2_video_predictor

def get_bbox_from_user(rgb_frame, window_name="Draw BBox on hand+arm"):
    """Let user draw a bounding box on the first frame via OpenCV GUI."""
    img_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    bbox = cv2.selectROI(window_name, img_bgr, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    x, y, w, h = bbox
    if w == 0 or h == 0:
        raise ValueError("Invalid bounding box selected (zero width or height)")
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def generate_masks(
    npz_path: str,
    output_path: str,
    sam2_checkpoint: str,
    sam2_config: str,
    device: str = "auto",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Load RGB frames from NPZ
    data = np.load(npz_path, allow_pickle=True)
    rgb_frames = data["rgb"]  # (N, H, W, 3), uint8
    num_frames = len(rgb_frames)
    print(f"     Loaded {num_frames} frames from {npz_path}")

    # SAM2 video predictor expects frames as JPEG files in a directory
    # We write temp frames, run predictor, then clean up
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(num_frames):
            frame_path = os.path.join(tmpdir, f"{i:06d}.jpg")
            img_bgr = cv2.cvtColor(rgb_frames[i], cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_path, img_bgr)

        # Build predictor
        predictor = build_sam2_video_predictor(sam2_config, sam2_checkpoint, device=device)
        inference_state = predictor.init_state(video_path=tmpdir)

        # Get bounding box from user on first frame
        bbox = get_bbox_from_user(rgb_frames[0])
        print(f"     User-selected bbox: {bbox}")

        # Add box prompt on frame 0
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            box=bbox,
        )

        # Propagate through entire video
        masks = np.zeros((num_frames, rgb_frames.shape[1], rgb_frames.shape[2]), dtype=np.bool_)

        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
            masks[frame_idx] = mask

    np.savez_compressed(output_path, masks=masks)
    print(f"     Saved masks ({masks.shape}) to {output_path}")
    return output_path