from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def load_flux_summary(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_clicks_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize(arr: np.ndarray) -> np.ndarray:
    s = float(np.sum(arr))
    if s <= 0.0:
        raise ValueError("Cannot normalize zero array")
    return arr / s


def estimate_lambda_from_k(k0x: float, k0y: float) -> float:
    k = float(np.hypot(k0x, k0y))
    if k <= 0.0:
        raise ValueError("k magnitude must be > 0")
    return 2.0 * np.pi / k


def find_fringe_peaks(
    y: np.ndarray,
    py: np.ndarray,
    smooth_sigma_bins: float = 1.0,
    prominence: float = 0.002,
    distance_bins: int = 3,
):
    py_smooth = gaussian_filter1d(py, smooth_sigma_bins)
    peaks, props = find_peaks(py_smooth, prominence=prominence, distance=distance_bins)
    return py_smooth, peaks, props


def compute_spacing(y: np.ndarray, peaks: np.ndarray):
    if len(peaks) < 2:
        return None, None
    yp = y[peaks]
    spacings = np.diff(yp)
    return float(np.mean(spacings)), spacings


def nearest_to_zero(values: np.ndarray) -> int:
    return int(np.argmin(np.abs(values)))


def local_visibility(py_smooth: np.ndarray, peaks: np.ndarray):
    """
    Estimate average fringe visibility using neighboring peak-valley pairs:
        V = (Imax - Imin) / (Imax + Imin)
    """
    if len(peaks) < 2:
        return None, []

    vis = []
    for i in range(len(peaks) - 1):
        p1 = peaks[i]
        p2 = peaks[i + 1]
        if p2 <= p1 + 1:
            continue
        valley_idx_rel = np.argmin(py_smooth[p1:p2 + 1])
        valley = p1 + valley_idx_rel
        i_max = 0.5 * (py_smooth[p1] + py_smooth[p2])
        i_min = py_smooth[valley]
        denom = i_max + i_min
        if denom > 0:
            vis.append(float((i_max - i_min) / denom))
    if not vis:
        return None, []
    return float(np.mean(vis)), vis


def symmetry_error(y: np.ndarray, py: np.ndarray):
    """
    Compare P(y) and P(-y) on the common overlap grid.
    """
    py_interp = np.interp(-y, y, py, left=np.nan, right=np.nan)
    mask = np.isfinite(py_interp)
    if not np.any(mask):
        return None
    num = np.sum(np.abs(py[mask] - py_interp[mask]))
    den = np.sum(np.abs(py[mask])) + 1e-12
    return float(num / den)


def main():
    parser = argparse.ArgumentParser(description="Validate double-slit detector distribution.")
    parser.add_argument("--summary", required=True, help="Path to *_flux_summary.json")
    parser.add_argument("--clicks-jsonl", required=True, help="Path to *_pseudo_clicks.jsonl")

    parser.add_argument("--k0x", type=float, required=True)
    parser.add_argument("--k0y", type=float, default=0.0)
    parser.add_argument("--screen-x", type=float, required=True)
    parser.add_argument("--barrier-x", type=float, required=True)
    parser.add_argument("--slit-separation", type=float, required=True)

    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--smooth-sigma-bins", type=float, default=1.0)
    parser.add_argument("--prominence", type=float, default=0.002)
    parser.add_argument("--distance-bins", type=int, default=3)

    args = parser.parse_args()

    summary = load_flux_summary(args.summary)
    clicks = load_clicks_jsonl(args.clicks_jsonl)

    y = np.asarray(summary["y_coords"], dtype=float)
    flux_y = normalize(np.asarray(summary["flux_y_accum"], dtype=float))
    y_click = np.asarray([float(c["y"]) for c in clicks], dtype=float)

    # Click histogram on same y support
    click_hist, edges = np.histogram(
        y_click,
        bins=len(y),
        range=(float(np.min(y)), float(np.max(y))),
        density=False,
    )
    click_hist = normalize(click_hist.astype(float))

    # Smooth both for peak analysis
    flux_smooth, flux_peaks, _ = find_fringe_peaks(
        y=y,
        py=flux_y,
        smooth_sigma_bins=args.smooth_sigma_bins,
        prominence=args.prominence,
        distance_bins=args.distance_bins,
    )
    click_smooth, click_peaks, _ = find_fringe_peaks(
        y=y,
        py=click_hist,
        smooth_sigma_bins=args.smooth_sigma_bins,
        prominence=args.prominence,
        distance_bins=args.distance_bins,
    )

    flux_spacing_mean, flux_spacings = compute_spacing(y, flux_peaks)
    click_spacing_mean, click_spacings = compute_spacing(y, click_peaks)

    # Theory
    lam = estimate_lambda_from_k(args.k0x, args.k0y)
    L = float(args.screen_x - args.barrier_x)
    d = float(args.slit_separation)
    theory_spacing = lam * L / d

    # Visibility
    flux_visibility_mean, flux_visibility_all = local_visibility(flux_smooth, flux_peaks)
    click_visibility_mean, click_visibility_all = local_visibility(click_smooth, click_peaks)

    # Symmetry
    flux_sym_err = symmetry_error(y, flux_y)
    click_sym_err = symmetry_error(y, click_hist)

    print("=== Detector / Double-slit validation ===")
    print(f"lambda_theory = {lam:.6g}")
    print(f"L = {L:.6g}")
    print(f"d = {d:.6g}")
    print(f"fringe_spacing_theory = {theory_spacing:.6g}")
    print()

    print(f"num_flux_peaks = {len(flux_peaks)}")
    print(f"num_click_peaks = {len(click_peaks)}")
    print()

    if flux_spacing_mean is not None:
        rel_err_flux = abs(flux_spacing_mean - theory_spacing) / (abs(theory_spacing) + 1e-12)
        print(f"flux_spacing_mean = {flux_spacing_mean:.6g}")
        print(f"flux_spacing_rel_err = {rel_err_flux:.3%}")
    else:
        print("flux_spacing_mean = not enough peaks")

    if click_spacing_mean is not None:
        rel_err_click = abs(click_spacing_mean - theory_spacing) / (abs(theory_spacing) + 1e-12)
        print(f"click_spacing_mean = {click_spacing_mean:.6g}")
        print(f"click_spacing_rel_err = {rel_err_click:.3%}")
    else:
        print("click_spacing_mean = not enough peaks")

    print()

    if flux_visibility_mean is not None:
        print(f"flux_visibility_mean = {flux_visibility_mean:.6g}")
    else:
        print("flux_visibility_mean = not enough peaks")

    if click_visibility_mean is not None:
        print(f"click_visibility_mean = {click_visibility_mean:.6g}")
    else:
        print("click_visibility_mean = not enough peaks")

    print()

    if flux_sym_err is not None:
        print(f"flux_symmetry_error = {flux_sym_err:.6g}")
    if click_sym_err is not None:
        print(f"click_symmetry_error = {click_sym_err:.6g}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(y, flux_y, label="Flux P(y)", alpha=0.45)
    plt.plot(y, flux_smooth, label="Flux smooth", linewidth=2)
    if len(flux_peaks) > 0:
        plt.scatter(y[flux_peaks], flux_smooth[flux_peaks], marker="o", label="Flux peaks")

    plt.plot(y, click_hist, label="Click hist", alpha=0.35)
    plt.plot(y, click_smooth, label="Click smooth", linewidth=2)
    if len(click_peaks) > 0:
        plt.scatter(y[click_peaks], click_smooth[click_peaks], marker="x", label="Click peaks")

    plt.xlabel("y")
    plt.ylabel("Probability density / normalized mass")
    plt.title("Double-slit validation: flux vs pseudo-click distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()