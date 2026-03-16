from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parents[2]
TREND_ROOT = ROOT / "trend_runs"


def load_flux_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize(arr: np.ndarray) -> np.ndarray:
    s = float(np.sum(arr))
    if s <= 0.0:
        raise ValueError("Cannot normalize zero array")
    return arr / s


def quadratic_peak_interp(xm1, x0, xp1, ym1, y0, yp1):
    denom = (ym1 - 2.0 * y0 + yp1)
    if abs(denom) < 1e-30:
        return float(x0), float(y0)
    dx = x0 - xm1
    offset = 0.5 * (ym1 - yp1) / denom
    x_peak = x0 + offset * dx
    y_peak = y0 - 0.25 * (ym1 - yp1) * offset
    return float(x_peak), float(y_peak)


def dominant_fft_spacing(
    y: np.ndarray,
    py: np.ndarray,
    smooth_sigma_bins: float = 1.0,
    min_freq: float = 0.15,
    max_freq: float | None = None,
) -> float:
    dy = float(np.mean(np.diff(y)))
    py_smooth_fast = gaussian_filter1d(py, smooth_sigma_bins)
    py_smooth_slow = gaussian_filter1d(py, 8.0 * smooth_sigma_bins)
    signal = py_smooth_fast - py_smooth_slow

    freqs = np.fft.rfftfreq(len(signal), d=dy)
    spec = np.abs(np.fft.rfft(signal)) ** 2

    valid = freqs > min_freq
    if max_freq is not None:
        valid &= freqs < max_freq

    spec_masked = spec.copy()
    spec_masked[~valid] = 0.0

    i_peak = int(np.argmax(spec_masked))
    f_peak = float(freqs[i_peak])

    if 1 <= i_peak < len(freqs) - 1:
        f_try, _ = quadratic_peak_interp(
            freqs[i_peak - 1], freqs[i_peak], freqs[i_peak + 1],
            spec[i_peak - 1], spec[i_peak], spec[i_peak + 1],
        )
        if freqs[i_peak - 1] <= f_try <= freqs[i_peak + 1]:
            f_peak = f_try

    if f_peak <= 0.0:
        raise RuntimeError("Invalid FFT peak frequency")

    return 1.0 / f_peak


def estimate_slit_separation_from_summary_or_case(summary: dict, case_name: str) -> float | None:
    # If later you store d in summary, use it here. For now infer slit_offset cases from name.
    m = re.match(r"slit_offset_(\d+)p(\d+)", case_name)
    if m:
        off = float(f"{m.group(1)}.{m.group(2)}")
        return 2.0 * off
    return None


def parse_case_value(case_name: str, prefix: str) -> float:
    m = re.match(rf"{re.escape(prefix)}_(\d+)p(\d+)", case_name)
    if not m:
        raise ValueError(f"Cannot parse value from case name: {case_name}")
    return float(f"{m.group(1)}.{m.group(2)}")


def collect_cases():
    rows = []

    for case_dir in sorted(TREND_ROOT.iterdir()):
        if not case_dir.is_dir():
            continue

        case_name = case_dir.name
        summary_path = case_dir / f"{case_name}_flux_summary.json"
        if not summary_path.exists():
            print(f"[WARN] missing summary: {summary_path}")
            continue

        summary = load_flux_summary(summary_path)
        y = np.asarray(summary["y_coords"], dtype=float)
        flux_y = normalize(np.asarray(summary["flux_y_accum"], dtype=float))

        spacing_fft = dominant_fft_spacing(y, flux_y)

        row = {
            "case": case_name,
            "spacing_fft": float(spacing_fft),
        }

        if case_name.startswith("screen_x_"):
            row["group"] = "screen_x"
            row["x"] = parse_case_value(case_name, "screen_x")

        elif case_name.startswith("k0x_"):
            row["group"] = "k0x"
            row["x"] = parse_case_value(case_name, "k0x")

        elif case_name.startswith("slit_offset_"):
            row["group"] = "slit_offset"
            row["x"] = parse_case_value(case_name, "slit_offset")
            row["estimated_d"] = estimate_slit_separation_from_summary_or_case(summary, case_name)

        else:
            row["group"] = "other"
            row["x"] = np.nan

        rows.append(row)

    return rows


def monotonic_direction_ok(xs: list[float], ys: list[float], expected: str) -> bool:
    # expected: "increasing" or "decreasing"
    pairs = sorted(zip(xs, ys), key=lambda t: t[0])
    ys_sorted = [p[1] for p in pairs]

    diffs = np.diff(ys_sorted)
    if expected == "increasing":
        return bool(np.all(diffs > 0))
    if expected == "decreasing":
        return bool(np.all(diffs < 0))
    raise ValueError(expected)


def print_group(rows, group, expected):
    group_rows = [r for r in rows if r["group"] == group]
    group_rows = sorted(group_rows, key=lambda r: r["x"])

    if not group_rows:
        print(f"\n[{group}] no rows")
        return

    print(f"\n=== {group} ===")
    for r in group_rows:
        extra = ""
        if "estimated_d" in r:
            extra = f", estimated_d≈{r['estimated_d']:.3f}"
        print(
            f"{r['case']:<20} "
            f"x={r['x']:<6.3f} "
            f"spacing_fft={r['spacing_fft']:.6f}"
            f"{extra}"
        )

    xs = [r["x"] for r in group_rows]
    ys = [r["spacing_fft"] for r in group_rows]
    ok = monotonic_direction_ok(xs, ys, expected)

    print(f"expected trend: {expected}")
    print(f"trend ok      : {ok}")

    # crude relative slope summary
    if len(xs) >= 2:
        ratio = ys[-1] / ys[0]
        print(f"end/start spacing ratio: {ratio:.6f}")


def main():
    if not TREND_ROOT.exists():
        raise RuntimeError(f"trend_runs not found: {TREND_ROOT}")

    rows = collect_cases()
    if not rows:
        raise RuntimeError("No trend cases found")

    print_group(rows, "screen_x", expected="increasing")
    print_group(rows, "k0x", expected="decreasing")
    print_group(rows, "slit_offset", expected="decreasing")


if __name__ == "__main__":
    main()