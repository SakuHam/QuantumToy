from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

from config import AppConfig
from core.grid import build_grid
from file.run_io import load_run_bundle, apply_cfg_dict


# ============================================================
# Helpers
# ============================================================

def load_clicks(path: Path):
    """
    Load click_y from JSONL or JSON.
    Expected:
      {"click_y": [...]}  OR  lines with {"click_y": value}
    """
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return np.array(data["click_y"], dtype=float)

    # JSONL
    ys = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "click_y" in row:
                ys.append(row["click_y"])
    return np.array(ys, dtype=float)


def estimate_branch_centers(click_y: np.ndarray, n_bins: int = 200):
    """
    Estimate branch centers from histogram peaks.
    """
    hist, edges = np.histogram(click_y, bins=n_bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # simple peak detection
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
            peaks.append(i)

    return centers[peaks]


def assign_clicks_to_branch_centers(click_y, branch_centers):
    d = np.abs(click_y[:, None] - branch_centers[None, :])
    return np.argmin(d, axis=1)


# ============================================================
# Main visualization
# ============================================================

def plot_phase_branch_mapping(
    psi,
    x,
    y,
    click_y,
    branch_centers,
    screen_x=None,
):
    rho = np.abs(psi) ** 2
    phase = np.angle(psi)

    click_branch_idx = assign_clicks_to_branch_centers(click_y, branch_centers)

    extent = (x[0], x[-1], y[0], y[-1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # -------------------------------
    # Phase map
    # -------------------------------
    ax = axes[0]
    im = ax.imshow(
        phase,
        origin="lower",
        extent=extent,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    ax.set_title("Phase map + branch centers")
    for yc in branch_centers:
        ax.axhline(yc, color="white", linestyle="--", alpha=0.6)

    if screen_x is not None:
        ax.axvline(screen_x, color="cyan", linestyle=":")

    fig.colorbar(im, ax=ax)

    # -------------------------------
    # Density + clicks
    # -------------------------------
    ax = axes[1]
    im = ax.imshow(
        rho / rho.max(),
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0,
        vmax=1,
    )

    if screen_x is not None:
        x_clicks = np.full_like(click_y, screen_x)
    else:
        x_clicks = np.full_like(click_y, x[-1])

    sc = ax.scatter(
        x_clicks,
        click_y,
        c=click_branch_idx,
        cmap="tab10",
        s=20,
        edgecolors="black",
        linewidths=0.3,
    )

    ax.set_title("Density + clicks (colored by branch)")
    for yc in branch_centers:
        ax.axhline(yc, color="white", linestyle="--", alpha=0.4)

    fig.colorbar(sc, ax=ax)

    plt.show()


# ============================================================
# Entry point
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--clicks", required=True)
    ap.add_argument("--t-index", type=int, default=-1)
    args = ap.parse_args()

    click_path = Path(args.clicks)

    # --------------------------------------------------------
    # Load run bundle
    # --------------------------------------------------------
#    bundle = load_run_bundle(run_path)
    npz = np.load(args.npz, allow_pickle=True)
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))

    print("[DEBUG] npz keys:", npz.files)

    if "state_vis_frames" in npz:
        psi_all = npz["state_vis_frames"]
        print("[INFO] Using state_vis_frames as psi")
    elif "psi" in npz:
        psi_all = npz["psi"]
    elif "psi_frames" in npz:
        psi_all = npz["psi_frames"]
    else:
        raise RuntimeError("No complex field found (psi/state_vis_frames)")

    psi = psi_all[args.t_index]

    # Use visualization grid directly from NPZ
    if "x_vis_1d" in npz and "y_vis_1d" in npz:
        x = npz["x_vis_1d"]
        y = npz["y_vis_1d"]
        print("[INFO] Using x_vis_1d / y_vis_1d from npz")
    else:
        raise RuntimeError("x_vis_1d / y_vis_1d not found in npz")

    # Detector screen x:
    # first try meta/config, otherwise leave None
    screen_x = None
    try:
        cfg = AppConfig()
        apply_cfg_dict(cfg, meta["config"])
        screen_x = getattr(cfg, "screen_x", None)
    except Exception as e:
        print(f"[WARN] Could not recover screen_x from meta config: {e}")
        

    #grid = build_grid(cfg)

    # --------------------------------------------------------
    # Load clicks
    # --------------------------------------------------------
    if "y_click" in npz:
        click_y = np.asarray(npz["y_click"], dtype=float)
        print("[INFO] Using y_click from npz")
    else:
        click_y = load_clicks(click_path)

    print(f"[INFO] Loaded {len(click_y)} clicks")

    click_x = None
    if "x_click" in npz:
        click_x = np.asarray(npz["x_click"], dtype=float)
        print("[INFO] Using x_click from npz")
        
    # --------------------------------------------------------
    # Estimate branches
    # --------------------------------------------------------
    branch_centers = estimate_branch_centers(click_y)

    print("[INFO] Branch centers:", branch_centers)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    plot_phase_branch_mapping(
        psi=psi,
        x=x,
        y=y,
#        click_x=click_x,
        click_y=click_y,
        branch_centers=branch_centers,
        screen_x=screen_x,
    )


if __name__ == "__main__":
    main()