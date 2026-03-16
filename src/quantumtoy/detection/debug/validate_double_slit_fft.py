from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def load_flux_summary(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def quadratic_peak_interp(xm1: float, x0: float, xp1: float, ym1: float, y0: float, yp1: float):
    """
    Parabolic interpolation around a discrete peak using three points.
    Returns refined x_peak, y_peak.
    """
    denom = (ym1 - 2.0 * y0 + yp1)
    if abs(denom) < 1e-30:
        return x0, y0

    dx = x0 - xm1
    offset = 0.5 * (ym1 - yp1) / denom
    x_peak = x0 + offset * dx
    y_peak = y0 - 0.25 * (ym1 - yp1) * offset
    return float(x_peak), float(y_peak)


def dominant_fft_spacing(
    y: np.ndarray,
    py: np.ndarray,
    smooth_sigma_bins: float = 1.0,
    min_freq: float = 0.02,
    max_freq: float | None = None,
):
    """
    Estimate dominant fringe spacing from FFT of the flux distribution.

    Steps:
      - smooth lightly
      - subtract smooth DC baseline (global mean)
      - FFT
      - choose strongest positive frequency inside [min_freq, max_freq]
      - spacing = 1 / f_peak
    """
    if y.ndim != 1 or py.ndim != 1 or len(y) != len(py):
        raise ValueError("y and py must be 1D arrays of same length")

    dy = float(np.mean(np.diff(y)))
    if not np.allclose(np.diff(y), dy, rtol=1e-6, atol=1e-9):
        raise ValueError("y grid must be approximately uniform for FFT")

    py_smooth = gaussian_filter1d(py, smooth_sigma_bins)

    # remove DC / baseline
    signal = py_smooth - float(np.mean(py_smooth))

    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=dy)
    spec = np.abs(np.fft.rfft(signal)) ** 2

    # ignore zero frequency
    valid = freqs > float(min_freq)
    if max_freq is not None:
        valid &= freqs < float(max_freq)

    if not np.any(valid):
        raise RuntimeError("No valid FFT frequency range left after masking")

    spec_masked = spec.copy()
    spec_masked[~valid] = 0.0

    i_peak = int(np.argmax(spec_masked))
    f_peak = float(freqs[i_peak])
    p_peak = float(spec[i_peak])

    # refine with quadratic interpolation if interior point
    if 1 <= i_peak < len(freqs) - 1:
        f_refined, p_refined = quadratic_peak_interp(
            freqs[i_peak - 1], freqs[i_peak], freqs[i_peak + 1],
            spec[i_peak - 1], spec[i_peak], spec[i_peak + 1],
        )
    else:
        f_refined, p_refined = f_peak, p_peak

    spacing = 1.0 / f_refined if f_refined > 0 else np.inf

    return {
        "dy": dy,
        "py_smooth": py_smooth,
        "signal": signal,
        "freqs": freqs,
        "spec": spec,
        "i_peak": i_peak,
        "f_peak_raw": f_peak,
        "f_peak_refined": f_refined,
        "power_peak_raw": p_peak,
        "power_peak_refined": p_refined,
        "spacing_fft": spacing,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate double-slit spacing using FFT of flux distribution.")
    parser.add_argument("--summary", required=True, help="Path to *_flux_summary.json")
    parser.add_argument("--k0x", type=float, required=True)
    parser.add_argument("--k0y", type=float, default=0.0)
    parser.add_argument("--screen-x", type=float, required=True)
    parser.add_argument("--barrier-x", type=float, required=True)
    parser.add_argument("--slit-separation", type=float, required=True)

    parser.add_argument("--smooth-sigma-bins", type=float, default=1.0)
    parser.add_argument("--min-freq", type=float, default=0.02)
    parser.add_argument("--max-freq", type=float, default=None)

    args = parser.parse_args()

    summary = load_flux_summary(args.summary)

    y = np.asarray(summary["y_coords"], dtype=float)
    flux_y = np.asarray(summary["flux_y_accum"], dtype=float)
    flux_y = normalize(flux_y)

    lam = estimate_lambda_from_k(args.k0x, args.k0y)
    L = float(args.screen_x - args.barrier_x)
    d = float(args.slit_separation)
    spacing_theory = lam * L / d

    fft_res = dominant_fft_spacing(
        y=y,
        py=flux_y,
        smooth_sigma_bins=args.smooth_sigma_bins,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
    )

    spacing_fft = float(fft_res["spacing_fft"])
    rel_err = abs(spacing_fft - spacing_theory) / (abs(spacing_theory) + 1e-12)

    print("=== Double-slit FFT validation ===")
    print(f"lambda_theory = {lam:.6g}")
    print(f"L = {L:.6g}")
    print(f"d = {d:.6g}")
    print(f"fringe_spacing_theory = {spacing_theory:.6g}")
    print()
    print(f"dy_grid = {fft_res['dy']:.6g}")
    print(f"fft_peak_frequency_raw = {fft_res['f_peak_raw']:.6g}")
    print(f"fft_peak_frequency_refined = {fft_res['f_peak_refined']:.6g}")
    print(f"fringe_spacing_fft = {spacing_fft:.6g}")
    print(f"fft_spacing_rel_err = {rel_err:.3%}")

    # --------------------------------------------------------
    # plots
    # --------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(y, flux_y, label="Flux P(y)", alpha=0.5)
    ax1.plot(y, fft_res["py_smooth"], label="Flux smooth", linewidth=2)
    ax1.set_xlabel("y")
    ax1.set_ylabel("Normalized flux")
    ax1.set_title("Detector-screen flux distribution")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(fft_res["freqs"], fft_res["spec"], label="FFT power")
    ax2.axvline(fft_res["f_peak_refined"], linestyle="--", label=f"Peak f={fft_res['f_peak_refined']:.4f}")
    ax2.set_xlabel("Spatial frequency [1 / y]")
    ax2.set_ylabel("Power")
    ax2.set_title("FFT spectrum of screen distribution")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()