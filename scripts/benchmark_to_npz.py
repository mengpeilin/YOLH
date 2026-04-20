from pathlib import Path
import numpy as np
import av

def decode_rgb_frames(video_path: Path) -> np.ndarray:
    frames = []
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()

    if not frames:
        raise ValueError(f"No RGB frames decoded from {video_path}")
    return np.stack(frames).astype(np.uint8)


def decode_depth_frames(video_path: Path) -> np.ndarray:
    frames = []
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            arr = frame.to_ndarray()
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            if arr.ndim != 2:
                raise ValueError(f"Depth frame should be 2D, got shape={arr.shape} from {video_path}")
            frames.append(arr.astype(np.uint16, copy=False))
    finally:
        container.close()

    if not frames:
        raise ValueError(f"No depth frames decoded from {video_path}")
    return np.stack(frames).astype(np.uint16)


def find_video_file(video_root: Path, video_key: str, episode_index: int, preferred_ext: str | None = None) -> Path:
    base = video_root / video_key / f"episode_{episode_index:06d}"
    if preferred_ext:
        candidate = base.with_suffix(preferred_ext)
        if candidate.exists():
            return candidate

    for ext in (".mp4", ".mkv", ".avi", ".mov"):
        candidate = base.with_suffix(ext)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Video file not found for {video_key} episode {episode_index:06d}")

def parse_intrinsic_from_cfg(step_cfg: dict) -> np.ndarray:
    # Preferred format: step00_human.camera_info.intrinsic as a 3x3 matrix.
    camera_info = step_cfg.get("camera_info")
    if isinstance(camera_info, dict):
        k = camera_info.get("intrinsic")
        if k is not None:
            k_arr = np.asarray(k, dtype=np.float64)
            if k_arr.shape != (3, 3):
                raise ValueError(
                    "step00_human.camera_info.intrinsic must be a 3x3 matrix "
                    f"but got shape {k_arr.shape}"
                )
            fx = float(k_arr[0, 0])
            fy = float(k_arr[1, 1])
            cx = float(k_arr[0, 2])
            cy = float(k_arr[1, 2])
            return np.asarray([fx, fy, cx, cy], dtype=np.float64)

    # Backward compatible format: step00_human.intrinsic as [fx, fy, cx, cy].
    intrinsic = step_cfg.get("intrinsic")
    if intrinsic is not None:
        if not isinstance(intrinsic, (list, tuple)) or len(intrinsic) != 4:
            raise ValueError("step00_human.intrinsic must be [fx, fy, cx, cy]")
        return np.asarray(intrinsic, dtype=np.float64)

    raise ValueError(
        "Missing camera intrinsic config. Provide either step00_human.camera_info.intrinsic (3x3) "
        "or step00_human.intrinsic ([fx, fy, cx, cy])."
    )