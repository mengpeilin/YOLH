"""
Visualize voxel grids saved in episodes.npz.

Supports:
- Single frame visualization
- Animated playback across episodes
- Keyboard controls in playback window

Keyboard controls:
- Space: pause/resume
- Left/Right: previous/next frame
- q or Esc: close
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def voxel_grid_to_points(voxel_grid, occ_thresh=0.5, max_points=30000):
    """Convert one voxel grid (V,V,V,4) to xyz points and RGB colors."""
    occ = voxel_grid[..., 3] > occ_thresh
    idx = np.argwhere(occ)

    if idx.shape[0] == 0:
        return idx, np.empty((0, 3), dtype=np.float32)

    colors = voxel_grid[occ, :3]
    colors = np.clip(colors, 0.0, 1.0)

    if idx.shape[0] > max_points:
        sel = np.random.choice(idx.shape[0], size=max_points, replace=False)
        idx = idx[sel]
        colors = colors[sel]

    return idx, colors


def load_voxel_grids(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "voxel_grids" not in data:
        raise KeyError(f"voxel_grids not found in {npz_path}")

    grids = data["voxel_grids"]
    if grids.ndim == 4:
        grids = grids[None, ...]
    if grids.ndim != 5 or grids.shape[-1] != 4:
        raise ValueError(
            f"Expected shape (E,V,V,V,4) or (V,V,V,4), got {grids.shape}"
        )

    return grids


def show_static(grids, frame_idx, occ_thresh, max_points, point_size, elev, azim):
    frame_idx = int(np.clip(frame_idx, 0, len(grids) - 1))
    pts, cols = voxel_grid_to_points(grids[frame_idx], occ_thresh, max_points)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    if pts.shape[0] > 0:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=point_size, marker="s")

    v = grids.shape[1]
    ax.set_xlim(0, v)
    ax.set_ylim(0, v)
    ax.set_zlim(0, v)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"Voxel Grid Frame {frame_idx} / {len(grids)-1} | occupied: {pts.shape[0]}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def animate(
    grids,
    occ_thresh,
    max_points,
    point_size,
    fps,
    elev,
    azim,
    loop,
):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    total = len(grids)
    v = grids.shape[1]
    state = {"i": 0, "paused": False}

    def draw_frame(i):
        ax.cla()
        pts, cols = voxel_grid_to_points(grids[i], occ_thresh, max_points)
        if pts.shape[0] > 0:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=point_size, marker="s")

        ax.set_xlim(0, v)
        ax.set_ylim(0, v)
        ax.set_zlim(0, v)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Voxel Grid Frame {i} / {total-1} | occupied: {pts.shape[0]}")

    def step(*_):
        if state["paused"]:
            return
        draw_frame(state["i"])
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
            draw_frame(state["i"])
            fig.canvas.draw_idle()
        elif event.key == "left":
            state["i"] = max(state["i"] - 1, 0)
            draw_frame(state["i"])
            fig.canvas.draw_idle()
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    draw_frame(0)
    interval_ms = max(1, int(1000 / max(1, fps)))
    _anim = FuncAnimation(fig, step, interval=interval_ms)
    plt.tight_layout()
    plt.show()


def build_parser():
    p = argparse.ArgumentParser(description="Visualize voxel grids from episodes.npz")
    p.add_argument("--npz-path", required=True, help="Path to .npz with voxel_grids")
    p.add_argument("--frame", type=int, default=0, help="Frame index for static mode")
    p.add_argument("--play", action="store_true", help="Play all frames as animation")
    p.add_argument("--fps", type=int, default=5, help="Playback FPS in --play mode")
    p.add_argument("--loop", action="store_true", help="Loop playback in --play mode")
    p.add_argument("--occ-thresh", type=float, default=0.5, help="Occupancy threshold")
    p.add_argument("--max-points", type=int, default=30000, help="Max rendered occupied voxels")
    p.add_argument("--point-size", type=float, default=6.0, help="Scatter marker size")
    p.add_argument("--elev", type=float, default=25.0, help="Camera elevation")
    p.add_argument("--azim", type=float, default=45.0, help="Camera azimuth")
    return p


def main():
    args = build_parser().parse_args()
    grids = load_voxel_grids(args.npz_path)
    print(f"Loaded voxel_grids shape: {grids.shape}")

    if args.play:
        animate(
            grids=grids,
            occ_thresh=args.occ_thresh,
            max_points=args.max_points,
            point_size=args.point_size,
            fps=args.fps,
            elev=args.elev,
            azim=args.azim,
            loop=args.loop,
        )
    else:
        show_static(
            grids=grids,
            frame_idx=args.frame,
            occ_thresh=args.occ_thresh,
            max_points=args.max_points,
            point_size=args.point_size,
            elev=args.elev,
            azim=args.azim,
        )


if __name__ == "__main__":
    main()
