import os
import sys
import numpy as np
import cv2
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM2_ROOT = os.path.join(PROJECT_ROOT, "dependencies", "sam2")
sys.path.insert(0, SAM2_ROOT)

from sam2.build_sam import build_sam2_video_predictor


def _select_user_bbox(rgb_frame, window_name="Draw BBox on hand+arm"):

    bbox = []
    cursor = [None]

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            bbox.clear()
            bbox.append((x, y))
            cursor[0] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            cursor[0] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            bbox.append((x, y))

    img_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display = img_bgr.copy()
        if len(bbox) == 1 and cursor[0] is not None:
            cv2.rectangle(display, bbox[0], cursor[0], (0, 255, 0), 2)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to cancel
            cv2.destroyAllWindows()
            raise ValueError("Bounding box selection canceled by user.")
        if len(bbox) == 2:  # Two points selected
            break

    cv2.destroyAllWindows()

    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    x_min, y_min = min(x1, x2), min(y1, y2)
    x_max, y_max = max(x1, x2), max(y1, y2)

    if x_max - x_min == 0 or y_max - y_min == 0:
        raise ValueError("Invalid bounding box (zero size)")

    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def _track_masks(rgb_frames, sam2_config, sam2_checkpoint, device, init_frame, bbox):
    num_frames = len(rgb_frames)
    masks = np.zeros((num_frames, rgb_frames.shape[1], rgb_frames.shape[2]), dtype=np.bool_)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(num_frames):
            frame_path = os.path.join(tmpdir, f"{i:06d}.jpg")
            img_bgr = cv2.cvtColor(rgb_frames[i], cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_path, img_bgr)

        predictor = build_sam2_video_predictor(sam2_config, sam2_checkpoint, device=device)
        inference_state = predictor.init_state(video_path=tmpdir)

        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=init_frame,
            obj_id=1,
            box=bbox,
        )

        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
            masks[frame_idx] = mask

    return masks


def generate_masks(
    npz_path: str,
    output_path: str,
    sam2_checkpoint: str,
    sam2_config: str,
    device: str = "auto",
    hand_bboxes_path: str = None,
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

    # Branch A: user-provided bbox for full hand+arm removal mask
    arm_hand_bbox = _select_user_bbox(rgb_frames[0], "Draw BBox on hand+arm")
    arm_hand_init_frame = 0
    print(f"     User arm+hand bbox (frame 0): {arm_hand_bbox}")

    # Branch B: auto bbox from step 01 for hand-only mask
    if hand_bboxes_path is None:
        raise ValueError("hand_bboxes_path is required to generate hand-only masks")

    if hand_bboxes_path is not None:
        bbox_data = np.load(hand_bboxes_path, allow_pickle=True)
        bboxes = bbox_data["bboxes"]
        det = bbox_data["hand_detected"]
        if not np.any(det):
            raise ValueError("No detected hand bbox found in hand_bboxes_path")
        init_frame = int(np.argmax(det))
        hand_bbox = bboxes[init_frame].astype(np.float32)
        print(f"     Auto hand bbox (frame {init_frame}): {hand_bbox}")

    arm_hand_masks = _track_masks(
        rgb_frames,
        sam2_config,
        sam2_checkpoint,
        device,
        arm_hand_init_frame,
        arm_hand_bbox,
    )
    hand_masks = _track_masks(
        rgb_frames,
        sam2_config,
        sam2_checkpoint,
        device,
        init_frame,
        hand_bbox,
    )

    np.savez_compressed(
        output_path,
        arm_hand_masks=arm_hand_masks,
        hand_masks=hand_masks,
    )
    print(
        f"     Saved arm+hand masks ({arm_hand_masks.shape}) and hand masks ({hand_masks.shape}) to {output_path}"
    )
    return output_path