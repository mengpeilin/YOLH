import numpy as np
from PIL import Image
from transformers import pipeline as hf_pipeline

def detect_hand_bboxes(
    npz_path: str,
    output_path: str,
    dino_model_id: str = "IDEA-Research/grounding-dino-tiny",
    threshold: float = 0.2,
    max_jump: float = 200.0,
    max_gap: int = 10,
):
    data = np.load(npz_path, allow_pickle=True)
    rgb_frames = data["rgb"]
    N = len(rgb_frames)
    print(f"     Processing {N} frames for hand bbox detection")

    detector = hf_pipeline(
        model=dino_model_id,
        task="zero-shot-object-detection",
        device="cuda",
        batch_size=4,
    )

    bboxes = np.zeros((N, 4), dtype=np.float32)
    scores = np.zeros(N, dtype=np.float32)
    hand_detected = np.zeros(N, dtype=bool)

    for i in range(N):
        img_pil = Image.fromarray(rgb_frames[i])
        results = detector(img_pil, candidate_labels=["a hand."], threshold=threshold)
        if results:
            best = max(results, key=lambda r: r["score"])
            box = best["box"]
            bboxes[i] = [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]
            scores[i] = best["score"]
            hand_detected[i] = True

        if (i + 1) % 25 == 0:
            print(f"     [{i + 1}/{N}] detected={hand_detected[:i+1].sum()}")

    # Post-processing
    bboxes, hand_detected = _postprocess_bboxes(
        bboxes, hand_detected, max_jump=max_jump, max_gap=max_gap
    )

    np.savez_compressed(
        output_path,
        bboxes=bboxes,
        scores=scores,
        hand_detected=hand_detected,
    )
    print(
        f"     Saved hand bboxes ({N} frames, {hand_detected.sum()} detected) "
        f"to {output_path}"
    )


def _postprocess_bboxes(bboxes, hand_detected, max_jump=200.0, max_gap=10):
    """Filter large jumps in bbox centres and linearly interpolate short gaps."""
    N = len(bboxes)

    # --- 1. Filter large centre jumps ----------------------------------------
    centres = np.zeros((N, 2), dtype=np.float32)
    centres[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    centres[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2.0

    for i in range(1, N):
        if hand_detected[i] and hand_detected[i - 1]:
            dist = np.linalg.norm(centres[i] - centres[i - 1])
            if dist > max_jump:
                hand_detected[i] = False
                bboxes[i] = 0
                scores_here = 0  # noqa: F841 (not stored back – already zero)

    # --- 2. Interpolate short gaps -------------------------------------------
    i = 0
    while i < N:
        if not hand_detected[i]:
            gap_start = i
            while i < N and not hand_detected[i]:
                i += 1
            gap_end = i
            gap_len = gap_end - gap_start

            if gap_len <= max_gap and gap_start > 0 and gap_end < N:
                start_bbox = bboxes[gap_start - 1]
                end_bbox = bboxes[gap_end]
                for j in range(gap_start, gap_end):
                    alpha = (j - gap_start + 1) / (gap_len + 1)
                    bboxes[j] = start_bbox * (1 - alpha) + end_bbox * alpha
                    hand_detected[j] = True
        else:
            i += 1

    return bboxes, hand_detected
