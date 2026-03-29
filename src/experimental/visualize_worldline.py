from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider


RENDER_MODES = (
    "forward_density",
    "base_rho",
    "rho_wl",
)


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("npz_path", help="Path to worker debug npz")
    p.add_argument(
        "--summary-json",
        default=None,
        help="Optional summary json path. If omitted, inferred from npz filename.",
    )
    p.add_argument("--save-mp4", action="store_true", help="Save animation to mp4")
    p.add_argument("--output-mp4", default=None, help="Override mp4 output path")
    p.add_argument("--fps", type=int, default=25, help="MP4 fps")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None)

    p.add_argument(
        "--split-view",
        action="store_true",
        help="Show two panels side by side instead of the default three-panel view.",
    )
    p.add_argument(
        "--left-mode",
        choices=RENDER_MODES,
        default="forward_density",
        help="Left panel mode in split view",
    )
    p.add_argument(
        "--right-mode",
        choices=RENDER_MODES,
        default="base_rho",
        help="Right panel mode in split view",
    )

    p.add_argument(
        "--single-mode",
        choices=RENDER_MODES,
        default=None,
        help="Render only one mode fullscreen",
    )

    p.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Gamma display exponent",
    )
    p.add_argument(
        "--display-q",
        type=float,
        default=0.995,
        help="Quantile for fixed display reference",
    )
    p.add_argument(
        "--use-fixed-display-scale",
        action="store_true",
        help="Use fixed display reference instead of per-frame normalization",
    )

    p.add_argument("--show-barrier", action="store_true")
    p.add_argument("--show-screen", action="store_true")
    p.add_argument("--show-click", action="store_true")
    p.add_argument("--show-panel-titles", action="store_true")

    return p.parse_args()


# ============================================================
# Constants / geometry
# ============================================================

VISIBLE_LX = 40.0
VISIBLE_LY = 20.0

BARRIER_CENTER_X = 0.0
SCREEN_CENTER_X = 10.0


# ============================================================
# Helpers
# ============================================================

def infer_summary_path(npz_path: Path) -> Path | None:
    name = npz_path.name
    if name.endswith("_debug.npz"):
        return npz_path.with_name(name.replace("_debug.npz", "_summary.json"))
    return None


def load_summary(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_finite_scalar(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def compute_click_frame_idx(times: np.ndarray, t_det) -> int | None:
    if not is_finite_scalar(t_det):
        return None

    t_det = float(t_det)

    if len(times) == 0:
        return None

    if t_det <= float(times[0]):
        return 0
    if t_det > float(times[-1]):
        return None

    return int(np.searchsorted(times, t_det, side="left"))


def gamma_display(
    arr: np.ndarray,
    vref: float,
    gamma: float = 0.5,
    use_fixed_scale: bool = True,
) -> np.ndarray:
    if use_fixed_scale:
        disp = np.clip(arr / (vref + 1e-30), 0.0, 1.0)
        return disp ** gamma

    m = float(np.max(arr))
    if m <= 0:
        return np.zeros_like(arr)
    return (arr / m) ** gamma


def choose_vref(arr: np.ndarray | None, q: float) -> float:
    if arr is None:
        return 1.0
    vals = np.asarray(arr, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    ref = float(np.quantile(vals, q))
    if ref <= 0.0:
        ref = float(np.max(vals))
    if ref <= 0.0:
        ref = 1.0
    return ref


def get_extent():
    return (
        -VISIBLE_LX / 2.0,
        +VISIBLE_LX / 2.0,
        -VISIBLE_LY / 2.0,
        +VISIBLE_LY / 2.0,
    )


def load_npz_bundle(npz_path: Path):
    data = np.load(npz_path)

    required = ["frames_psi", "base_rho", "times_arr"]
    for key in required:
        if key not in data:
            raise RuntimeError(f"Missing required array in npz: {key}")

    frames_psi = data["frames_psi"]
    base_rho = data["base_rho"]
    times_arr = data["times_arr"]

    rho_wl = None
    if "rho_wl" in data and data["rho_wl"].size > 0:
        rho_wl = data["rho_wl"]

    return {
        "frames_psi": frames_psi,
        "base_rho": base_rho,
        "rho_wl": rho_wl,
        "times_arr": times_arr,
    }


def slice_frames(arr: np.ndarray | None, i0: int, i1: int):
    if arr is None:
        return None
    return arr[i0:i1]


def mode_title(mode: str) -> str:
    if mode == "forward_density":
        return "Forward density"
    if mode == "base_rho":
        return "Posthoc / TRF base_rho"
    if mode == "rho_wl":
        return "Posthoc / WL rho_wl"
    return mode


def get_mode_array(mode: str, frames_psi: np.ndarray, base_rho: np.ndarray, rho_wl: np.ndarray | None):
    if mode == "forward_density":
        return frames_psi
    if mode == "base_rho":
        return base_rho
    if mode == "rho_wl":
        return rho_wl
    raise ValueError(f"Unsupported render mode: {mode}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    summary_path = Path(args.summary_json) if args.summary_json else infer_summary_path(npz_path)
    summary = load_summary(summary_path)

    bundle = load_npz_bundle(npz_path)
    frames_psi = bundle["frames_psi"]
    base_rho = bundle["base_rho"]
    rho_wl = bundle["rho_wl"]
    times = bundle["times_arr"]

    if frames_psi.ndim != 3:
        raise RuntimeError(f"frames_psi must be 3D, got {frames_psi.shape}")
    if base_rho.ndim != 3:
        raise RuntimeError(f"base_rho must be 3D, got {base_rho.shape}")
    if rho_wl is not None and rho_wl.ndim != 3:
        raise RuntimeError(f"rho_wl must be 3D, got {rho_wl.shape}")

    Nt = len(times)
    if Nt == 0:
        raise RuntimeError("No frames in times array")

    i0 = max(0, int(args.start_frame))
    i1 = Nt if args.end_frame is None else min(Nt, int(args.end_frame) + 1)
    if i1 <= i0:
        raise RuntimeError(f"Invalid frame range: start={i0}, end={i1}")

    frames_psi = slice_frames(frames_psi, i0, i1)
    base_rho = slice_frames(base_rho, i0, i1)
    rho_wl = slice_frames(rho_wl, i0, i1)
    times = times[i0:i1]

    Nt = len(times)

    vref_fwd = choose_vref(frames_psi, args.display_q)
    vref_base = choose_vref(base_rho, args.display_q)
    vref_wl = choose_vref(rho_wl, args.display_q) if rho_wl is not None else 1.0

    extent = get_extent()

    click_x = None
    click_y = None
    click_t = None
    click_frame_idx = None

    if summary is not None:
        click_x = summary.get("click_x")
        click_y = summary.get("click_y")
        click_t = summary.get("click_time")
        click_frame_idx = compute_click_frame_idx(times, click_t)

    # --------------------------------------------------------
    # Figure / panels
    # --------------------------------------------------------
    split_view = bool(args.split_view)

    if args.single_mode is not None:
        fig = plt.figure(figsize=(10.5, 6.5))
        ax = fig.add_axes([0.06, 0.10, 0.90, 0.85])  # lähes full screen
        axes = np.array([ax])
        panel_modes = [args.single_mode]

    elif split_view:
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))
        plt.subplots_adjust(left=0.05, right=0.98, bottom=0.16, top=0.90, wspace=0.12)
        panel_modes = [args.left_mode, args.right_mode]

    else:
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6))
        plt.subplots_adjust(left=0.05, right=0.98, bottom=0.16, top=0.90, wspace=0.12)
        panel_modes = ["forward_density", "base_rho", "rho_wl"]
        
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    panel_states = []

    for ax, mode in zip(axes, panel_modes):
        arr = get_mode_array(mode, frames_psi, base_rho, rho_wl)

        if arr is None:
            frame0 = np.zeros_like(frames_psi[0])
            vref = 1.0
        else:
            frame0 = arr[0]
            vref = (
                vref_fwd if mode == "forward_density"
                else vref_base if mode == "base_rho"
                else vref_wl
            )

        img0 = gamma_display(
            frame0,
            vref=vref,
            gamma=args.gamma,
            use_fixed_scale=args.use_fixed_display_scale,
        )

        im = ax.imshow(
            img0,
            extent=extent,
            origin="lower",
            cmap="magma",
            interpolation="nearest",
            zorder=1,
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        if args.show_panel_titles:
            ax.set_title(mode_title(mode))
        else:
            ax.set_title("")

        if args.show_barrier:
            ax.axvline(BARRIER_CENTER_X, color="white", linestyle="--", alpha=0.65, linewidth=1.1, zorder=5)

        if args.show_screen:
            ax.axvline(SCREEN_CENTER_X, color="cyan", linestyle="--", alpha=0.50, linewidth=1.0, zorder=5)

        if arr is None:
            ax.text(
                0.5,
                0.5,
                f"{mode}\nnot available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="white",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.4, edgecolor="none"),
                zorder=10,
            )

        click_marker, = ax.plot(
            [],
            [],
            marker="x",
            markersize=8,
            linestyle="None",
            color="deepskyblue",
            alpha=0.9,
            zorder=9,
        )

        panel_states.append(
            {
                "ax": ax,
                "mode": mode,
                "array": arr,
                "vref": vref,
                "im": im,
                "click_marker": click_marker,
            }
        )

    main_title = fig.suptitle(f"time={times[0]:.3f}", fontsize=12)

    # --------------------------------------------------------
    # Slider
    # --------------------------------------------------------
    ax_frame = fig.add_axes([0.12, 0.06, 0.76, 0.04])
    slider = Slider(
        ax=ax_frame,
        label="frame",
        valmin=0,
        valmax=Nt - 1,
        valinit=0,
        valstep=1,
    )

    def update_click_marker(marker, i: int):
        visible = (
            args.show_click
            and click_x is not None
            and click_y is not None
            and click_frame_idx is not None
            and i >= click_frame_idx
        )

        if visible:
            marker.set_data([float(click_x)], [float(click_y)])
        else:
            marker.set_data([], [])

    def redraw(i: int):
        artists = []

        for panel in panel_states:
            arr = panel["array"]
            if arr is None:
                img = np.zeros_like(frames_psi[0])
            else:
                img = gamma_display(
                    arr[i],
                    vref=panel["vref"],
                    gamma=args.gamma,
                    use_fixed_scale=args.use_fixed_display_scale,
                )

            panel["im"].set_data(img)
            update_click_marker(panel["click_marker"], i)

            artists.append(panel["im"])
            artists.append(panel["click_marker"])

        main_title.set_text(f"time={times[i]:.3f}")
        artists.append(main_title)
        return artists

    def on_slider(_val):
        i = int(slider.val)
        redraw(i)
        fig.canvas.draw_idle()

    slider.on_changed(on_slider)

    def update(i: int):
        if int(slider.val) != i:
            slider.eventson = False
            slider.set_val(i)
            slider.eventson = True
        return redraw(i)

    ani = FuncAnimation(
        fig,
        update,
        frames=Nt,
        interval=40,
        blit=False,
    )

    if args.save_mp4:
        output_mp4 = args.output_mp4
        if output_mp4 is None:
            output_mp4 = str(npz_path.with_suffix(".mp4"))
        print(f"[SAVE] animation -> {output_mp4}")
        ani.save(output_mp4, writer="ffmpeg", fps=args.fps, dpi=150)

    plt.show()


if __name__ == "__main__":
    main()