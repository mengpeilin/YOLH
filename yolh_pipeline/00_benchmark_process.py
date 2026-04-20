#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolh_pipeline.config_utils import load_pipeline_config, get_step_cfg
from scripts.benchmark_to_npz import decode_rgb_frames, decode_depth_frames, find_video_file, parse_intrinsic_from_cfg

def main():
    parser = argparse.ArgumentParser(description="Convert benchmark human demo data to pipeline raw.npz format")
    parser.add_argument("--input-dir", required=True, help="Benchmark dataset root containing data/meta/videos")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    step_cfg = get_step_cfg(cfg, "step00_human")

    color_video_key = step_cfg.get(
        "color_video_key",
        "observation.images.cam_azure_kinect_front.color",
    )
    depth_video_key = step_cfg.get(
        "depth_video_key",
        "observation.images.cam_azure_kinect_front.transformed_depth",
    )
    color_ext = step_cfg.get("color_ext")
    depth_ext = step_cfg.get("depth_ext")
    max_frames = step_cfg.get("max_frames")
    chunk_filter = step_cfg.get("chunk", "*")
    intrinsic_arr = parse_intrinsic_from_cfg(step_cfg)
    fps = float(step_cfg.get("fps", 30.0))

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_chunks = sorted((input_dir / "data").glob(f"chunk-{chunk_filter}"))
    if not data_chunks:
        raise FileNotFoundError(f"No data chunks found under {(input_dir / 'data')} with chunk filter {chunk_filter}")

    all_episode_files = []
    for chunk_dir in data_chunks:
        all_episode_files.extend(sorted(chunk_dir.glob("episode_*.parquet")))

    if not all_episode_files:
        raise FileNotFoundError("No episode_*.parquet files found in selected chunk(s)")

    print(f"Found {len(all_episode_files)} episode parquet file(s)")

    for idx, parquet_path in enumerate(all_episode_files, start=1):
        episode_index = int(parquet_path.stem.split("_")[-1])

        videos_chunk_dir = input_dir / "videos" / parquet_path.parent.name
        if not videos_chunk_dir.exists():
            raise FileNotFoundError(f"Video chunk directory does not exist: {videos_chunk_dir}")

        out_session = output_dir / f"rosbag_episode_{episode_index:06d}"
        out_session.mkdir(parents=True, exist_ok=True)
        out_npz = out_session / "raw.npz"

        if out_npz.exists():
            print(f"[{idx}/{len(all_episode_files)}] episode_{episode_index:06d}: skip (raw.npz exists)")
            continue

        color_video = find_video_file(videos_chunk_dir, color_video_key, episode_index, color_ext)
        depth_video = find_video_file(videos_chunk_dir, depth_video_key, episode_index, depth_ext)

        print(f"[{idx}/{len(all_episode_files)}] episode_{episode_index:06d}")
        print(f"  color: {color_video}")
        print(f"  depth: {depth_video}")

        rgb_arr = decode_rgb_frames(color_video)
        depth_arr = decode_depth_frames(depth_video)

        ep_df = pd.read_parquet(parquet_path, columns=["timestamp", "frame_index"])
        if "timestamp" in ep_df.columns:
            ts = np.asarray(ep_df["timestamp"], dtype=np.float64)
            ts_arr = (ts * 1e9).astype(np.int64)
        else:
            n = len(ep_df)
            ts_arr = (np.arange(n, dtype=np.float64) / fps * 1e9).astype(np.int64)

        n_frames = min(len(rgb_arr), len(depth_arr), len(ts_arr))
        if max_frames is not None:
            n_frames = min(n_frames, int(max_frames))

        if n_frames <= 0:
            raise ValueError(f"No usable frames for episode_{episode_index:06d}")

        if len(rgb_arr) != len(depth_arr) or len(rgb_arr) != len(ts_arr):
            print(
                "  warning: frame count mismatch "
                f"(rgb={len(rgb_arr)}, depth={len(depth_arr)}, ts={len(ts_arr)}), using first {n_frames}"
            )

        np.savez_compressed(
            out_npz,
            rgb=rgb_arr[:n_frames],
            depth=depth_arr[:n_frames],
            intrinsic=intrinsic_arr,
            timestamps=ts_arr[:n_frames],
        )
        print(f"  saved: {out_npz} ({n_frames} frames)")


if __name__ == "__main__":
    main()
