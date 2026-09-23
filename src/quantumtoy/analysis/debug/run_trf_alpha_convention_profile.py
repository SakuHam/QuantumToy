"""Profile clock-convention uncertainty around the locked TRF alpha calibration."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.record_environment import RecordExperiment
from analysis.record_sensitivity import (
    build_reference_runs,
    profile_clock_convention_widths,
    threshold_grid,
)
from analysis.trf_physical_scale import (
    LOCKED_REFERENCE_ALPHA,
    REFERENCE_SPATIAL_SIGMA_T,
)


G_VALUES = (0.5, 1.0, 2.0)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "minimum": np.min(values, axis=0),
        "quantile_05": np.quantile(values, 0.05, axis=0),
        "median": np.median(values, axis=0),
        "quantile_95": np.quantile(values, 0.95, axis=0),
        "maximum": np.max(values, axis=0),
    }


def _plot(profile, output):
    rows = profile.rows
    latencies = np.asarray([row.latencies for row in rows])
    locked = np.asarray([row.locked_alpha_widths for row in rows])
    calibrated = np.asarray([
        row.calibration_profiled_widths for row in rows])
    alphas = np.asarray([row.calibration_profiled_alpha for row in rows])
    baseline_index = next(
        index for index, row in enumerate(rows)
        if (row.information_deficit == 0.1 and row.required_copies == 3
            and row.coherence_tolerance == 0.05 and row.hold_time == 0.5))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    for values in latencies:
        axes[0, 0].plot(G_VALUES, values, color="#94a3b8", alpha=0.22)
    axes[0, 0].plot(
        G_VALUES, latencies[baseline_index], color="#dc2626", marker="o",
        linewidth=2.5, label="locked baseline convention")
    axes[0, 0].set(
        xlabel="environment coupling g", ylabel="tau_stab",
        title="81 shared threshold conventions")
    axes[0, 0].grid(alpha=0.22)
    axes[0, 0].legend()

    axes[0, 1].boxplot(
        [locked[:, index] for index in range(len(G_VALUES))],
        tick_labels=[f"g={g:g}" for g in G_VALUES], showfliers=False)
    axes[0, 1].scatter(
        np.arange(1, 4), locked[baseline_index], color="#dc2626", zorder=3,
        label="baseline")
    axes[0, 1].set(
        ylabel="predicted sigma_T",
        title="Numerical alpha fixed across conventions")
    axes[0, 1].grid(axis="y", alpha=0.22)
    axes[0, 1].legend()

    axes[1, 0].boxplot(
        [calibrated[:, index] for index in range(len(G_VALUES))],
        tick_labels=[f"g={g:g}" for g in G_VALUES], showfliers=False)
    axes[1, 0].scatter(
        np.arange(1, 4), calibrated[baseline_index], color="#16a34a", zorder=3,
        label="baseline")
    axes[1, 0].set(
        ylabel="predicted sigma_T",
        title="Convention profiled through the g=1 calibration")
    axes[1, 0].grid(axis="y", alpha=0.22)
    axes[1, 0].legend()

    axes[1, 1].hist(alphas, bins=12, color="#2563eb", alpha=0.82,
                    edgecolor="white")
    axes[1, 1].axvline(
        LOCKED_REFERENCE_ALPHA, color="#dc2626", linestyle="--",
        linewidth=2, label=f"reference alpha={LOCKED_REFERENCE_ALPHA:.5f}")
    axes[1, 1].set(
        xlabel="alpha calibrated at g=1", ylabel="number of conventions",
        title="Alpha absorbs the absolute clock convention")
    axes[1, 1].grid(axis="y", alpha=0.22)
    axes[1, 1].legend()

    fig.suptitle(
        "TRF clock-convention profile with a held-out width prediction",
        fontsize=15, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument(
        "--json-output", type=Path,
        default=Path("trf_alpha_convention_profile.json"))
    parser.add_argument(
        "--plot-output", type=Path,
        default=Path("trf_alpha_convention_profile.png"))
    args = parser.parse_args()
    if args.duration <= 0 or args.dt <= 0:
        parser.error("duration and dt must be positive")

    experiment = RecordExperiment()
    runs = build_reference_runs(
        experiment, G_VALUES, duration=args.duration, dt=args.dt)
    threshold_rows = threshold_grid(
        runs,
        information_deficits=[0.05, 0.1, 0.2],
        required_copies=[2, 3, 4],
        coherence_tolerances=[0.01, 0.05, 0.1],
        hold_times=[0.25, 0.5, 1.0],
    )
    profile = profile_clock_convention_widths(
        threshold_rows, g_values=G_VALUES,
        locked_alpha=LOCKED_REFERENCE_ALPHA,
        calibration_g=1.0,
        calibration_sigma_t=REFERENCE_SPATIAL_SIGMA_T)
    if not profile.rows:
        raise RuntimeError("No fully resolved shared clock convention")

    latencies = np.asarray([row.latencies for row in profile.rows])
    locked = np.asarray([row.locked_alpha_widths for row in profile.rows])
    calibrated = np.asarray([
        row.calibration_profiled_widths for row in profile.rows])
    profiled_alpha = np.asarray([
        row.calibration_profiled_alpha for row in profile.rows])
    baseline = next(
        row for row in profile.rows
        if (row.information_deficit == experiment.information_deficit
            and row.required_copies == experiment.required_copies
            and row.coherence_tolerance == experiment.coherence_tolerance
            and row.hold_time == experiment.hold_time))
    heldout_indices = [index for index, g in enumerate(G_VALUES) if g != 1.0]
    relative_spans = (
        np.ptp(calibrated[:, heldout_indices], axis=0)
        / np.median(calibrated[:, heldout_indices], axis=0))

    payload = {
        "status": "synthetic_discrete_convention_profile",
        "policy": {
            "locked_reference_alpha": LOCKED_REFERENCE_ALPHA,
            "calibration_g": 1.0,
            "calibration_sigma_t": REFERENCE_SPATIAL_SIGMA_T,
            "threshold_conventions_are_shared_across_g": True,
            "heldout_g": [0.5, 2.0],
        },
        "interpretation": {
            "strict_locked_alpha": (
                "Keeps the numerical alpha fixed while changing the operational "
                "clock convention; this is a conservative envelope but no longer "
                "preserves the g=1 calibration."),
            "calibration_profiled_convention": (
                "Treats the shared convention as a discrete nuisance, calibrates "
                "alpha using only g=1, and predicts g=0.5 and g=2 without refitting."),
            "not_a_likelihood_profile": (
                "This fast run profiles deterministic clock conventions. It does "
                "not use empirical response counts or assign prior weights to conventions."),
        },
        "resolved_conventions": len(profile.rows),
        "unresolved_conventions": profile.unresolved_conventions,
        "g_values": G_VALUES,
        "baseline": asdict(baseline),
        "latency_summary": _summary(latencies),
        "strict_locked_alpha_width_summary": _summary(locked),
        "calibration_profiled_width_summary": _summary(calibrated),
        "calibration_profiled_alpha_summary": _summary(profiled_alpha[:, None]),
        "heldout_relative_full_spans": relative_spans,
        "rows": [asdict(row) for row in profile.rows],
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, default=_json_value, indent=2) + "\n")
    _plot(profile, args.plot_output)

    print(f"locked alpha={LOCKED_REFERENCE_ALPHA:.10f}")
    print(f"resolved conventions={len(profile.rows)}; "
          f"unresolved={profile.unresolved_conventions}")
    for label, values in (
            ("strict locked alpha", locked),
            ("calibration-profiled", calibrated)):
        print(label)
        for index, g in enumerate(G_VALUES):
            print(f"  g={g:g}: min={np.min(values[:, index]):.8g} "
                  f"median={np.median(values[:, index]):.8g} "
                  f"max={np.max(values[:, index]):.8g}")
    print("profiled alpha range="
          f"[{np.min(profiled_alpha):.8g}, {np.max(profiled_alpha):.8g}]")
    print("held-out relative full spans="
          f"{dict(zip((0.5, 2.0), relative_spans))}")
    print(f"results={args.json_output}; plot={args.plot_output}")


if __name__ == "__main__":
    main()
