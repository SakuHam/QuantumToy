from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--npz",
        type=str,
        required=True,
        help="Path to debug npz file produced by worker with --save",
    )
    p.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Optional summary json for click point overlay. "
             "If omitted, tries to infer from npz filename.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output mp4 path. Default: same directory / <stem>_anim.mp4",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Output video FPS",
    )
    p.add_argument(
        "--interval-ms",
        type=int,
        default=80,
        help="Preview/update interval for matplotlib animation",
    )
    p.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start frame index",
    )
    p.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="End frame index (inclusive). Default: last frame",
    )
    p.add_argument(
        "--shared-scale",
        action="store_true",
        help="Use one common vmax across all shown fields",
    )
    p.add_argument(
        "--fixed-q",
        type=float,
        default=0.995,
        help="Quantile for display vmax",
    )
    p.add_argument(
        "--show-click",
        action="store_true",
        help="Overlay click point if summary json is available",
    )
    return p.parse_args()


def infer_summary_path(npz_path: Path) -> Path | None:
    name = npz_path.name
    if not name.endswith("_debug.npz"):
        return None
    summary_name = name.replace("_debug.npz", "_summary.json")
    return npz_path.with_name(summary_name)


def load_summary(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def choose_vmax(arr: np.ndarray, q: float) -> float:
    flat = np.asarray(arr, dtype=np.float32).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 1.0
    vmax = float(np.quantile(flat, q))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(flat)) if flat.size > 0 else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    return vmax


def safe_get_npz_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data:
        return None
    arr = data[key]
    if isinstance(arr, np.ndarray) and arr.size == 0:
        return None
    return arr


def frame_slice_bounds(n_frames: int, start_frame: int, end_frame: int | None) -> tuple[int, int]:
    start = max(0, int(start_frame))
    end = n_frames - 1 if end_frame is None else min(int(end_frame), n_frames - 1)
    if end < start:
        raise ValueError(f"Invalid frame range: start={start}, end={end}, n_frames={n_frames}")
    return start, end


def main() -> int:
    args = parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    out_path = Path(args.out) if args.out else npz_path.with_name(npz_path.stem + "_anim.mp4")

    summary_path = Path(args.summary_json) if args.summary_json else infer_summary_path(npz_path)
    summary = load_summary(summary_path)

    data = np.load(npz_path)

    frames_psi = safe_get_npz_array(data, "frames_psi")
    base_rho = safe_get_npz_array(data, "base_rho")
    rho_wl = safe_get_npz_array(data, "rho_wl")
    times_arr = safe_get_npz_array(data, "times_arr")

    if frames_psi is None:
        raise RuntimeError("frames_psi missing in npz")
    if base_rho is None:
        raise RuntimeError("base_rho missing in npz")
    if times_arr is None:
        raise RuntimeError("times_arr missing in npz")

    if frames_psi.ndim != 3:
        raise RuntimeError(f"frames_psi expected 3D, got shape={frames_psi.shape}")
    if base_rho.ndim != 3:
        raise RuntimeError(f"base_rho expected 3D, got shape={base_rho.shape}")
    if rho_wl is not None and rho_wl.ndim != 3:
        raise RuntimeError(f"rho_wl expected 3D, got shape={rho_wl.shape}")

    n_frames = frames_psi.shape[0]
    if base_rho.shape[0] != n_frames or times_arr.shape[0] != n_frames:
        raise RuntimeError(
            f"Inconsistent frame counts: "
            f"frames_psi={frames_psi.shape[0]}, base_rho={base_rho.shape[0]}, times_arr={times_arr.shape[0]}"
        )
    if rho_wl is not None and rho_wl.shape[0] != n_frames:
        raise RuntimeError(
            f"Inconsistent frame counts: rho_wl={rho_wl.shape[0]}, expected={n_frames}"
        )

    start_i, end_i = frame_slice_bounds(n_frames, args.start_frame, args.end_frame)
    sel = slice(start_i, end_i + 1)

    frames_psi = frames_psi[sel]
    base_rho = base_rho[sel]
    times_arr = times_arr[sel]
    if rho_wl is not None:
        rho_wl = rho_wl[sel]

    n_frames_sel = frames_psi.shape[0]
    ny, nx = frames_psi.shape[1], frames_psi.shape[2]

    click_x = None
    click_y = None
    click_ix = None
    click_iy = None

    if summary is not None:
        click_x = summary.get("click_x")
        click_y = summary.get("click_y")

    # visible geometry from worker constants
    visible_lx = 40.0
    visible_ly = 20.0
    x_extent = (-visible_lx / 2.0, visible_lx / 2.0)
    y_extent = (-visible_ly / 2.0, visible_ly / 2.0)
    extent = [x_extent[0], x_extent[1], y_extent[0], y_extent[1]]

    if click_x is not None and click_y is not None:
        # convert to pixel only if needed later
        x_coords = np.linspace(x_extent[0], x_extent[1], nx, endpoint=False)
        y_coords = np.linspace(y_extent[0], y_extent[1], ny, endpoint=False)
        click_ix = int(np.argmin(np.abs(x_coords - float(click_x))))
        click_iy = int(np.argmin(np.abs(y_coords - float(click_y))))

    if args.shared_scale:
        arrays = [frames_psi, base_rho]
        if rho_wl is not None:
            arrays.append(rho_wl)
        combined = np.concatenate([a.ravel() for a in arrays]).astype(np.float32)
        common_vmax = choose_vmax(combined, args.fixed_q)
        vmax_psi = common_vmax
        vmax_base = common_vmax
        vmax_wl = common_vmax
    else:
        vmax_psi = choose_vmax(frames_psi, args.fixed_q)
        vmax_base = choose_vmax(base_rho, args.fixed_q)
        vmax_wl = choose_vmax(rho_wl, args.fixed_q) if rho_wl is not None else 1.0

    ncols = 3
    fig, axes = plt.subplots(1, ncols, figsize=(15, 5), constrained_layout=True)

    ax0, ax1, ax2 = axes

    im0 = ax0.imshow(
        frames_psi[0],
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=vmax_psi,
    )
    ax0.set_title("Forward density")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")

    im1 = ax1.imshow(
        base_rho[0],
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=vmax_base,
    )
    ax1.set_title("Posthoc / TRF base_rho")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    if rho_wl is not None:
        im2 = ax2.imshow(
            rho_wl[0],
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=vmax_wl,
        )
        ax2.set_title("Posthoc / WL rho_wl")
    else:
        blank = np.zeros_like(frames_psi[0], dtype=np.float32)
        im2 = ax2.imshow(
            blank,
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        ax2.set_title("rho_wl not available")
        ax2.text(
            0.5,
            0.5,
            "rho_wl\nnot available",
            transform=ax2.transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    click_artists = []
    if args.show_click and click_x is not None and click_y is not None:
        for ax in axes:
            artist = ax.plot(
                [float(click_x)],
                [float(click_y)],
                marker="x",
                markersize=8,
                markeredgewidth=2,
                linestyle="None",
            )[0]
            click_artists.append(artist)

    title = fig.suptitle(f"time={times_arr[0]:.3f}")

    def update(frame_idx: int):
        im0.set_data(frames_psi[frame_idx])
        im1.set_data(base_rho[frame_idx])

        if rho_wl is not None:
            im2.set_data(rho_wl[frame_idx])

        title.set_text(f"time={times_arr[frame_idx]:.3f}")
        artists = [im0, im1, im2, title]
        artists.extend(click_artists)
        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames_sel,
        interval=args.interval_ms,
        blit=False,
        repeat=False,
    )

    writer = FFMpegWriter(fps=args.fps)
    anim.save(str(out_path), writer=writer)
    plt.close(fig)

    print(f"Saved animation to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())