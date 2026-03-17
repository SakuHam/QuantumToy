from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from config import AppConfig
from core.grid import build_grid
from file.run_io import load_run_bundle, apply_cfg_dict


def load_summary_latest_per_case(path: Path):
    """
    Load JSONL summary and keep only the latest row for each case_name.
    Useful when summary.jsonl contains repeated entries due to resume/rerun.
    """
    latest_by_case: dict[str, dict] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            case_name = row.get("case_name")
            if case_name is None:
                continue
            latest_by_case[case_name] = row

    return list(latest_by_case.values())


def build_cfg_from_meta(meta: dict):
    cfg = AppConfig()
    return apply_cfg_dict(cfg, meta["config"])


def density_from_state_vis(state_vis: np.ndarray) -> np.ndarray:
    if state_vis.ndim == 2:
        return np.abs(state_vis) ** 2
    if state_vis.ndim == 3:
        return np.sum(np.abs(state_vis) ** 2, axis=0)
    raise ValueError(f"Unsupported state_vis ndim={state_vis.ndim}")


def branchiness_metrics(click_y: np.ndarray) -> dict[str, float]:
    ys = np.sort(np.asarray(click_y, dtype=float))
    if ys.size < 2:
        return {
            "n": float(ys.size),
            "range": 0.0,
            "mean_gap": 0.0,
            "median_gap": 0.0,
            "p90_gap": 0.0,
            "max_gap": 0.0,
        }

    gaps = np.diff(ys)
    return {
        "n": float(ys.size),
        "range": float(ys[-1] - ys[0]),
        "mean_gap": float(np.mean(gaps)),
        "median_gap": float(np.median(gaps)),
        "p90_gap": float(np.percentile(gaps, 90)),
        "max_gap": float(np.max(gaps)),
    }


def compute_frame_idx_from_time(times: np.ndarray, t_ref: float | None) -> int:
    if t_ref is None or not np.isfinite(float(t_ref)):
        return len(times) - 1

    idx = int(np.searchsorted(times, float(t_ref), side="left"))
    idx = max(0, min(idx, len(times) - 1))
    return idx


def extract_screen_profile_from_reference(
    npz_path: Path,
    meta_path: Path | None,
    *,
    selection_mode: str,
    click_times: np.ndarray | None = None,
):
    """
    Extract a Born-like screen profile rho(y) from a reference bundle.

    selection_mode:
      - "mean_click_time" : use mean click time from sweep
      - "bundle_t_det"    : use t_det stored in bundle
      - "screen_mass_max" : use frame with maximal mass on screen line
    """
    bundle = load_run_bundle(str(npz_path), str(meta_path) if meta_path is not None else None)
    meta = bundle["meta"]
    cfg = build_cfg_from_meta(meta)

    grid = build_grid(
        visible_lx=cfg.VISIBLE_LX,
        visible_ly=cfg.VISIBLE_LY,
        n_visible_x=cfg.N_VISIBLE_X,
        n_visible_y=cfg.N_VISIBLE_Y,
        pad_factor=cfg.PAD_FACTOR,
    )

    times = np.asarray(bundle["times"], dtype=float)
    state_vis_frames = bundle["state_vis_frames"]
    t_det_bundle = bundle.get("t_det", None)

    ix_screen = int(np.argmin(np.abs(grid.x_vis_1d - float(cfg.screen_center_x))))
    y_vals = np.asarray(grid.y_vis_1d, dtype=float)
    dy = float(grid.dy)

    if selection_mode == "mean_click_time":
        if click_times is None or click_times.size == 0:
            raise ValueError("mean_click_time mode requires click_times from sweep")
        t_ref = float(np.mean(click_times))
        i_ref = compute_frame_idx_from_time(times, t_ref)
        selection_detail = f"mean click time from sweep ({t_ref:.6f})"

    elif selection_mode == "bundle_t_det":
        t_ref = None if t_det_bundle is None else float(t_det_bundle)
        i_ref = compute_frame_idx_from_time(times, t_ref)
        selection_detail = f"bundle t_det ({t_ref})"

    elif selection_mode == "screen_mass_max":
        screen_mass = np.empty(len(times), dtype=float)
        for i in range(len(times)):
            rho_i = density_from_state_vis(state_vis_frames[i])
            rho_screen_i = np.asarray(rho_i[:, ix_screen], dtype=float)
            screen_mass[i] = float(np.sum(rho_screen_i) * dy)

        i_ref = int(np.argmax(screen_mass))
        t_ref = float(times[i_ref])
        selection_detail = "max screen mass"

    else:
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")

    state_ref = state_vis_frames[i_ref]
    rho_ref = density_from_state_vis(state_ref)
    rho_screen = np.asarray(rho_ref[:, ix_screen], dtype=float)

    area = float(np.sum(rho_screen) * dy)
    if area > 0.0:
        rho_screen_pdf = rho_screen / area
    else:
        rho_screen_pdf = rho_screen.copy()

    info = {
        "i_ref": i_ref,
        "t_ref": float(times[i_ref]),
        "screen_x": float(grid.x_vis_1d[ix_screen]),
        "dy": dy,
        "selection_mode": selection_mode,
        "selection_detail": selection_detail,
    }

    return y_vals, rho_screen_pdf, info


def find_emergent_branch_centers(
    click_y: np.ndarray,
    *,
    bins: int = 80,
    smooth_sigma_bins: float = 1.0,
    prominence: float = 0.03,
):
    """
    Estimate branch centers from empirical click_y distribution:
      histogram -> smooth -> detect peaks
    """
    hist, edges = np.histogram(click_y, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=smooth_sigma_bins)

    if np.max(hist_smooth) <= 0.0:
        return {
            "hist": hist,
            "hist_smooth": hist_smooth,
            "centers": centers,
            "peak_indices": np.array([], dtype=int),
            "peak_y": np.array([], dtype=float),
            "peak_val": np.array([], dtype=float),
        }

    prom_abs = prominence * float(np.max(hist_smooth))
    peak_indices, _props = find_peaks(hist_smooth, prominence=prom_abs)

    return {
        "hist": hist,
        "hist_smooth": hist_smooth,
        "centers": centers,
        "peak_indices": peak_indices,
        "peak_y": centers[peak_indices],
        "peak_val": hist_smooth[peak_indices],
    }


def find_born_peaks(
    y_ref: np.ndarray,
    rho_screen_pdf: np.ndarray,
    *,
    smooth_sigma_samples: float = 1.0,
    prominence: float = 0.03,
):
    """
    Detect local maxima in Born-like screen profile rho(y).
    Uses a smoothed version for stable peak detection, but raw profile
    can be plotted separately.
    """
    rho_smooth = gaussian_filter1d(np.asarray(rho_screen_pdf, dtype=float), sigma=smooth_sigma_samples)

    if np.max(rho_smooth) <= 0.0:
        return {
            "rho_smooth": rho_smooth,
            "peak_indices": np.array([], dtype=int),
            "peak_y": np.array([], dtype=float),
            "peak_val": np.array([], dtype=float),
        }

    prom_abs = prominence * float(np.max(rho_smooth))
    peak_indices, _props = find_peaks(rho_smooth, prominence=prom_abs)

    return {
        "rho_smooth": rho_smooth,
        "peak_indices": peak_indices,
        "peak_y": y_ref[peak_indices],
        "peak_val": rho_smooth[peak_indices],
    }


def nearest_peak_matches(source_y: np.ndarray, target_y: np.ndarray):
    """
    For each source peak, find nearest target peak.
    Returns tuples:
      (source_y, nearest_target_y, abs_distance)
    """
    source_y = np.asarray(source_y, dtype=float)
    target_y = np.asarray(target_y, dtype=float)

    matches = []
    if source_y.size == 0 or target_y.size == 0:
        return matches

    for y in source_y:
        j = int(np.argmin(np.abs(target_y - y)))
        matches.append((float(y), float(target_y[j]), float(abs(target_y[j] - y))))
    return matches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_jsonl", help="Sweep summary jsonl")
    parser.add_argument("reference_npz", nargs="?", default=None, help="Optional reference run npz")
    parser.add_argument("reference_meta", nargs="?", default=None, help="Optional reference run meta json")
    parser.add_argument(
        "--reference-frame-mode",
        choices=("screen_mass_max", "mean_click_time", "bundle_t_det"),
        default="screen_mass_max",
        help="How to choose the reference frame for Born/Emergent overlay",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_jsonl)
    if not summary_path.exists():
        print(f"File not found: {summary_path}")
        return 1

    reference_npz = Path(args.reference_npz) if args.reference_npz is not None else None
    reference_meta = Path(args.reference_meta) if args.reference_meta is not None else None

    rows = load_summary_latest_per_case(summary_path)

    successful_rows = [
        r for r in rows
        if int(r.get("returncode", 999999)) == 0
    ]
    clicked_rows = [
        r for r in successful_rows
        if bool(r.get("clicked", False))
    ]

    if not clicked_rows:
        print("No clicks found.")
        return 1

    click_x = np.array([r["click_x"] for r in clicked_rows], dtype=float)
    click_y = np.array([r["click_y"] for r in clicked_rows], dtype=float)
    click_t = np.array([r["click_time"] for r in clicked_rows], dtype=float)

    metrics = branchiness_metrics(click_y)

    print()
    print("=======================================")
    print("Detector sweep summary")
    print("=======================================")
    print(f"rows in jsonl           : {len(rows)} (latest per case)")
    print(f"successful runs         : {len(successful_rows)}")
    print(f"clicked runs            : {len(clicked_rows)}")
    print()

    print("click_y statistics")
    print("------------------")
    print(f"mean                    : {click_y.mean():.4f}")
    print(f"std                     : {click_y.std():.4f}")
    print(f"min                     : {click_y.min():.4f}")
    print(f"max                     : {click_y.max():.4f}")
    print()

    print("click_time statistics")
    print("---------------------")
    print(f"mean                    : {click_t.mean():.4f}")
    print(f"std                     : {click_t.std():.4f}")
    print(f"min                     : {click_t.min():.4f}")
    print(f"max                     : {click_t.max():.4f}")
    print()

    print("branchiness diagnostics")
    print("-----------------------")
    print(f"range(click_y)          : {metrics['range']:.4f}")
    print(f"mean nearest gap        : {metrics['mean_gap']:.4f}")
    print(f"median nearest gap      : {metrics['median_gap']:.4f}")
    print(f"90th percentile gap     : {metrics['p90_gap']:.4f}")
    print(f"max nearest gap         : {metrics['max_gap']:.4f}")
    print()

    # -------------------------------------------------
    # Histogram: click_y
    # -------------------------------------------------
    plt.figure()
    plt.hist(click_y, bins=30)
    plt.title("click_y distribution")
    plt.xlabel("y")
    plt.ylabel("count")

    # -------------------------------------------------
    # Histogram: click_time
    # -------------------------------------------------
    plt.figure()
    plt.hist(click_t, bins=30)
    plt.title("click_time distribution")
    plt.xlabel("t")
    plt.ylabel("count")

    # -------------------------------------------------
    # Scatter: click_y vs click_time
    # -------------------------------------------------
    plt.figure()
    plt.scatter(click_y, click_t)
    plt.title("click_y vs click_time")
    plt.xlabel("click_y")
    plt.ylabel("click_time")

    # -------------------------------------------------
    # Scatter: click_x vs click_y
    # -------------------------------------------------
    plt.figure()
    plt.scatter(click_x, click_y)
    plt.title("screen hit positions")
    plt.xlabel("click_x")
    plt.ylabel("click_y")

    # -------------------------------------------------
    # Branch-selection diagnostics
    # -------------------------------------------------
    rng = np.random.default_rng(12345)
    x_jitter = rng.normal(loc=0.0, scale=0.03, size=click_y.shape[0])
    sorted_y = np.sort(click_y)
    ranks = np.arange(sorted_y.size)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(x_jitter, click_y, alpha=0.8)
    axes[0].set_title("branch-selection diagnostic: strip plot")
    axes[0].set_xlabel("jittered x (no meaning)")
    axes[0].set_ylabel("click_y")
    axes[0].axvline(0.0, linestyle="--", alpha=0.4)

    axes[1].plot(ranks, sorted_y, marker="o", linewidth=1.0, markersize=3)
    axes[1].set_title("branch-selection diagnostic: sorted click_y")
    axes[1].set_xlabel("rank")
    axes[1].set_ylabel("sorted click_y")
    plt.tight_layout()

    # -------------------------------------------------
    # Born vs Emergent overlay + automatic peak comparison
    # -------------------------------------------------
    if reference_npz is not None:
        if not reference_npz.exists():
            print(f"[WARN] reference npz not found: {reference_npz}")
        else:
            y_ref, rho_screen_pdf, ref_info = extract_screen_profile_from_reference(
                reference_npz,
                reference_meta if (reference_meta is not None and reference_meta.exists()) else None,
                selection_mode=args.reference_frame_mode,
                click_times=click_t,
            )

            # Raw Born profile without smoothing
            plt.figure(figsize=(10, 6))
            plt.plot(
                y_ref,
                rho_screen_pdf,
                linewidth=1.5,
                label=(
                    f"Born raw screen rho(y) @ x≈{ref_info['screen_x']:.3f}, "
                    f"t≈{ref_info['t_ref']:.3f}"
                ),
            )
            plt.title("Born raw screen profile (no smoothing)")
            plt.xlabel("y")
            plt.ylabel("density")
            plt.legend()

            # Overlay histogram vs raw Born-like screen rho(y)
            fig = plt.figure(figsize=(10, 6))
            plt.hist(
                click_y,
                bins=30,
                density=True,
                alpha=0.45,
                label="Emergent click_y histogram",
            )
            plt.plot(
                y_ref,
                rho_screen_pdf,
                linewidth=1.5,
                label=(
                    f"Born raw screen rho(y) @ x≈{ref_info['screen_x']:.3f}, "
                    f"t≈{ref_info['t_ref']:.3f}"
                ),
            )
            plt.title("Born raw vs Emergent overlay")
            plt.xlabel("y")
            plt.ylabel("density")
            plt.legend()

            # CDF comparison from raw Born profile
            y_sorted = np.sort(click_y)
            ecdf = np.arange(1, len(y_sorted) + 1) / len(y_sorted)

            born_cdf = np.cumsum(rho_screen_pdf) * ref_info["dy"]
            born_cdf = np.clip(born_cdf, 0.0, 1.0)

            plt.figure(figsize=(10, 6))
            plt.step(y_sorted, ecdf, where="post", label="Emergent empirical CDF")
            plt.plot(y_ref, born_cdf, linewidth=1.5, label="Born raw screen CDF")
            plt.title("Born raw vs Emergent cumulative comparison")
            plt.xlabel("y")
            plt.ylabel("CDF")
            plt.legend()

            # Automatic peak/branch comparison still uses smoothed born
            emergent_peaks = find_emergent_branch_centers(
                click_y,
                bins=80,
                smooth_sigma_bins=1.0,
                prominence=0.03,
            )

            born_peaks = find_born_peaks(
                y_ref,
                rho_screen_pdf,
                smooth_sigma_samples=1.0,
                prominence=0.03,
            )

            matches_e_to_b = nearest_peak_matches(
                emergent_peaks["peak_y"],
                born_peaks["peak_y"],
            )
            matches_b_to_e = nearest_peak_matches(
                born_peaks["peak_y"],
                emergent_peaks["peak_y"],
            )

            print("reference overlay info")
            print("----------------------")
            print(f"reference frame mode    : {ref_info['selection_mode']}")
            print(f"reference detail        : {ref_info['selection_detail']}")
            print(f"reference frame index   : {ref_info['i_ref']}")
            print(f"reference time          : {ref_info['t_ref']:.6f}")
            print(f"reference screen_x      : {ref_info['screen_x']:.6f}")
            print()

            print("automatic branch / peak comparison")
            print("----------------------------------")
            print(f"emergent branch centers : {np.array2string(emergent_peaks['peak_y'], precision=3)}")
            print(f"born rho peaks          : {np.array2string(born_peaks['peak_y'], precision=3)}")
            print()

            if matches_e_to_b:
                print("nearest Born peak for each Emergent branch:")
                for y_e, y_b, d in matches_e_to_b:
                    print(f"  emergent {y_e:+.3f} -> born {y_b:+.3f}   |Δ|={d:.3f}")
                print()

            if matches_b_to_e:
                print("nearest Emergent branch for each Born peak:")
                for y_b, y_e, d in matches_b_to_e:
                    print(f"  born     {y_b:+.3f} -> emergent {y_e:+.3f}   |Δ|={d:.3f}")
                print()

            # Raw vs smoothed Born directly
            plt.figure(figsize=(10, 6))
            plt.plot(y_ref, rho_screen_pdf, linewidth=1.2, label="Born raw screen rho(y)")
            plt.plot(y_ref, born_peaks["rho_smooth"], linewidth=2.0, label="Born smoothed screen rho(y)")
            plt.title("Born raw vs smoothed screen profile")
            plt.xlabel("y")
            plt.ylabel("density")
            plt.legend()

            # Comparison with detected peaks
            fig = plt.figure(figsize=(10, 6))
            plt.hist(
                click_y,
                bins=80,
                density=True,
                alpha=0.25,
                label="Emergent click_y histogram",
            )
            plt.plot(
                emergent_peaks["centers"],
                emergent_peaks["hist_smooth"],
                linewidth=2.0,
                label="Emergent smoothed histogram",
            )
            plt.plot(
                y_ref,
                rho_screen_pdf,
                linewidth=1.0,
                alpha=0.8,
                label="Born raw screen rho(y)",
            )
            plt.plot(
                y_ref,
                born_peaks["rho_smooth"],
                linewidth=2.0,
                label="Born-like screen rho(y), smoothed",
            )

            for k, y0 in enumerate(emergent_peaks["peak_y"]):
                plt.axvline(
                    y0,
                    linestyle="--",
                    alpha=0.55,
                    label="Emergent branch center" if k == 0 else None,
                )

            for k, y0 in enumerate(born_peaks["peak_y"]):
                plt.axvline(
                    y0,
                    linestyle=":",
                    alpha=0.75,
                    label="Born rho peak" if k == 0 else None,
                )

            plt.title("Automatic branch center vs Born peak comparison")
            plt.xlabel("y")
            plt.ylabel("density")
            plt.legend()

    else:
        print("[INFO] No reference NPZ provided, skipping Born overlay.")

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())