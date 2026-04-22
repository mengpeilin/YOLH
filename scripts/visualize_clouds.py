"""
Visualize point clouds saved in episodes.npz.

Expected data format:
  - clouds: object array of per-frame point clouds
    each cloud is (Ni, 6) float32: [x, y, z, r, g, b]

Color handling:
  The pipeline stores colors with ImageNet normalization. By default this script
  denormalizes colors back to displayable RGB in [0, 1].

Keyboard controls in playback:
  - Space: pause/resume
  - Left/Right: previous/next frame
  - q or Esc: close
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_rgb(colors: np.ndarray, denorm: bool) -> np.ndarray:
    if colors.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    c = colors.astype(np.float32)
    if denorm:
        c = c * IMG_STD + IMG_MEAN

    return np.clip(c, 0.0, 1.0)


def load_clouds(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    if "clouds" not in data:
        keys = ", ".join(data.files)
        raise KeyError(f"clouds not found in {npz_path}. Available keys: {keys}")

    clouds = data["clouds"]
    if clouds.ndim != 1:
        raise ValueError(
            f"Expected clouds to be a 1D object array of frames, got shape {clouds.shape}"
        )
    return clouds


def get_axis_limits(clouds, sample_every=10):
    mins = []
    maxs = []
    step = max(1, sample_every)

    for i in range(0, len(clouds), step):
        cloud = clouds[i]
        if cloud is None or len(cloud) == 0:
            continue
        xyz = np.asarray(cloud)[:, :3]
        mins.append(xyz.min(axis=0))
        maxs.append(xyz.max(axis=0))

    if not mins:
        return np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5])

    min_xyz = np.min(np.stack(mins), axis=0)
    max_xyz = np.max(np.stack(maxs), axis=0)

    center = (min_xyz + max_xyz) * 0.5
    half = np.max(max_xyz - min_xyz) * 0.55
    half = max(half, 1e-3)
    return center - half, center + half


def draw_cloud(ax, cloud, frame_idx, total, point_size, denorm, min_xyz, max_xyz):
    ax.cla()

    if cloud is not None and len(cloud) > 0:
        arr = np.asarray(cloud)
        xyz = arr[:, :3]
        rgb = _to_rgb(arr[:, 3:6], denorm=denorm)
        ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=point_size, marker=".", depthshade=False
        )
        n_pts = xyz.shape[0]
    else:
        n_pts = 0

    ax.set_xlim(min_xyz[0], max_xyz[0])
    ax.set_ylim(min_xyz[1], max_xyz[1])
    ax.set_zlim(min_xyz[2], max_xyz[2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Cloud Frame {frame_idx} / {total - 1} | points: {n_pts}")


def show_static(clouds, frame, point_size, denorm, elev, azim):
    frame = int(np.clip(frame, 0, len(clouds) - 1))
    min_xyz, max_xyz = get_axis_limits(clouds)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    draw_cloud(ax, clouds[frame], frame, len(clouds), point_size, denorm, min_xyz, max_xyz)
    plt.tight_layout()
    plt.show()


def animate(clouds, point_size, denorm, fps, loop, elev, azim, output_video=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    total = len(clouds)
    state = {"i": 0, "paused": False}
    min_xyz, max_xyz = get_axis_limits(clouds)

    def render(i):
        draw_cloud(ax, clouds[i], i, total, point_size, denorm, min_xyz, max_xyz)
        ax.view_init(elev=elev, azim=azim)

    def step(*_):
        if state["paused"]:
            return
        render(state["i"])
        state["i"] += 1
        if state["i"] >= total:
            if loop:
                state["i"] = 0
            else:
                state["i"] = total - 1
                state["paused"] = True

    def on_key(event):
        if event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key == "right":
            state["i"] = min(state["i"] + 1, total - 1)
            render(state["i"])
            fig.canvas.draw_idle()
        elif event.key == "left":
            state["i"] = max(state["i"] - 1, 0)
            render(state["i"])
            fig.canvas.draw_idle()
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    render(0)
    interval_ms = max(1, int(1000 / max(1, fps)))
    _anim = FuncAnimation(
        fig,
        step,
        frames=total,
        interval=interval_ms,
        repeat=loop,
        cache_frame_data=False,
        save_count=total,
    )
    
    if output_video is not None:
        print(f"Saving video to {output_video}...")
        lower_path = output_video.lower()
        if lower_path.endswith((".mp4", ".mov", ".avi", ".mkv")):
            if not FFMpegWriter.isAvailable():
                raise RuntimeError(
                    "ffmpeg is not available. Install ffmpeg or save as .gif instead."
                )
            writer = FFMpegWriter(
                fps=fps,
                codec="libx264",
                extra_args=["-pix_fmt", "yuv420p"],
            )
        elif lower_path.endswith(".gif"):
            writer = PillowWriter(fps=fps)
        else:
            raise ValueError(
                "Unsupported output format. Use .mp4/.mov/.avi/.mkv for ffmpeg or .gif for Pillow."
            )
        plt.tight_layout()
        _anim.save(output_video, writer=writer)
        print(f"Video saved successfully!")
        plt.close(fig)
        return
    
    plt.tight_layout()
    plt.show()


def build_parser():
    p = argparse.ArgumentParser(description="Visualize clouds in episodes.npz")
    p.add_argument("--npz-path", required=True, help="Path to episodes.npz")
    p.add_argument("--frame", type=int, default=0, help="Frame index in static mode")
    p.add_argument("--play", action="store_true", help="Play all frames")
    p.add_argument("--fps", type=int, default=30, help="Playback FPS")
    p.add_argument("--loop", action="store_true", help="Loop playback")
    p.add_argument("--output-video", type=str, default=None, help="Save video to file (e.g., video.gif or video.mp4)")
    p.add_argument("--point-size", type=float, default=1.0, help="Scatter point size")
    p.add_argument("--elev", type=float, default=270.0, help="Camera elevation")
    p.add_argument("--azim", type=float, default=90.0, help="Camera azimuth")
    p.add_argument(
        "--no-denorm-colors",
        action="store_true",
        help="Do not denormalize RGB from ImageNet normalization",
    )
    return p


def main():
    args = build_parser().parse_args()
    clouds = load_clouds(args.npz_path)
    print(f"Loaded clouds with {len(clouds)} frames from {args.npz_path}")

    if args.play:
        animate(
            clouds=clouds,
            point_size=args.point_size,
            denorm=not args.no_denorm_colors,
            fps=args.fps,
            loop=args.loop,
            elev=args.elev,
            azim=args.azim,
            output_video=args.output_video,
        )
    else:
        show_static(
            clouds=clouds,
            frame=args.frame,
            point_size=args.point_size,
            denorm=not args.no_denorm_colors,
            elev=args.elev,
            azim=args.azim,
        )


if __name__ == "__main__":
    main()