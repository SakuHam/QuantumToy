"""Design and verify a complementary detector for the double-slit geometry."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, PowerNorm


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.spatial_effect_measurement import (
    double_slit_effect_experiment,
    fit_spatial_effect_joint_response,
    fit_spatial_effect_multi_response,
    run_spatial_effect_measurement,
    select_double_slit_detector_setting,
    spatial_effect_convergence,
    spatial_effect_evolution,
    spatial_effect_potential,
)


TRUE_SIGMA_T = 0.2
TRUE_LAMBDA = 1.0
DETECTOR_X_GRID = (0.0, 0.5, 1.0, 1.5, 2.0)
DETECTOR_WIDTH_GRID = (0.2, 0.35, 0.5)
TIME_OFFSET_GRID = (-0.4, -0.2, 0.0, 0.2)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _diagnostics(profile):
    return {
        "best_sigma_t": profile.best_sigma_t,
        "best_lambda_strength": profile.best_lambda_strength,
        "fisher_condition_number": profile.fisher_condition_number,
        "local_parameter_correlation": profile.local_parameter_correlation,
        "local_standard_errors": profile.local_standard_errors,
    }


def _bin_centers(y, count):
    return np.asarray([
        np.mean(y[rows]) for rows in np.array_split(np.arange(y.size), count)
    ])


def plot_design(
        experiment, second, design, first_run, second_run, evolution,
        single_profile, combined_profile, output_path):
    plt.style.use("dark_background")
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    density = np.tensordot(
        evolution.delay_weights, evolution.densities, axes=(0, 0))
    dx = experiment.lx / experiment.nx
    dy = experiment.ly / experiment.ny
    extent = (
        evolution.x[0] - dx / 2, evolution.x[-1] + dx / 2,
        evolution.y[0] - dy / 2, evolution.y[-1] + dy / 2,
    )
    image = axes[0, 0].imshow(
        density, origin="lower", extent=extent, aspect="equal",
        cmap="magma", interpolation="bilinear",
        norm=PowerNorm(gamma=0.5, vmin=0, vmax=float(np.max(density))))
    potential = spatial_effect_potential(experiment)
    axes[0, 0].contour(
        potential, levels=[0.1 * float(np.max(potential))],
        origin="lower", extent=extent, colors="white", linewidths=1.2)
    axes[0, 0].axvline(
        experiment.detector_x, color="#f472b6", linestyle="--",
        label="original detector")
    axes[0, 0].axvline(
        second.detector_x, color="#22d3ee", linestyle="--",
        label="designed detector")
    axes[0, 0].set(
        title="Double-slit geometry and detector planes", xlabel="x", ylabel="y")
    axes[0, 0].legend(framealpha=0.75)
    figure.colorbar(image, ax=axes[0, 0], fraction=0.046).set_label(
        "mixed probability density")

    eligible = design.improves_both_standard_errors
    scatter = axes[0, 1].scatter(
        design.candidate_detector_x[eligible],
        design.candidate_reference_time[eligible],
        s=75 + 180 * design.candidate_detector_width[eligible],
        c=design.combined_determinant[eligible], cmap="viridis",
        norm=LogNorm(), edgecolors="white", linewidths=0.4)
    axes[0, 1].scatter(
        design.candidate_detector_x[~eligible],
        design.candidate_reference_time[~eligible],
        s=40, facecolors="none", edgecolors="#64748b", alpha=0.5)
    axes[0, 1].scatter(
        [second.detector_x], [second.reference_time], marker="*", s=260,
        color="#fbbf24", edgecolors="black", linewidths=0.8,
        label="selected")
    axes[0, 1].set(
        title="Constrained Fisher design scan",
        xlabel="detector x", ylabel="reference time")
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend(framealpha=0.75)
    figure.colorbar(scatter, ax=axes[0, 1], fraction=0.046).set_label(
        "combined Fisher determinant")

    centers = _bin_centers(evolution.y, experiment.y_bins)
    axes[1, 0].plot(
        centers, first_run.conditional_y_distribution, marker="o",
        color="#f472b6", label=(
            f"original: x={experiment.detector_x:g}, "
            f"t={experiment.reference_time:g}"))
    axes[1, 0].plot(
        centers, second_run.conditional_y_distribution, marker="o",
        color="#22d3ee", label=(
            f"designed: x={second.detector_x:g}, "
            f"t={second.reference_time:.3f}"))
    axes[1, 0].set(
        title="Complementary conditional click laws",
        xlabel="y-bin center", ylabel="p(y bin | click)")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(framealpha=0.75)

    axes[1, 1].plot(
        single_profile.candidate_sigmas,
        single_profile.profiled_expected_deviance,
        color="#f472b6", linewidth=2, label="original detector")
    axes[1, 1].plot(
        combined_profile.candidate_sigmas,
        combined_profile.profiled_expected_deviance,
        color="#22d3ee", linewidth=2, label="50/50 detector pair")
    axes[1, 1].axvline(TRUE_SIGMA_T, color="#a3e635", linestyle=":")
    axes[1, 1].axhline(2.30, color="white", linestyle="--", alpha=0.7)
    axes[1, 1].set(
        title="Width profile after profiling response strength",
        xlabel=r"$\sigma_T$", ylabel=r"minimum $2N\,KL$", ylim=(0, 25))
    axes[1, 1].grid(alpha=0.2)
    axes[1, 1].legend(framealpha=0.75)
    before = single_profile.local_standard_errors
    after = combined_profile.local_standard_errors
    axes[1, 1].text(
        0.98, 0.95,
        f"SE sigma: {before[0]:.4f} -> {after[0]:.4f}\n"
        f"SE lambda: {before[1]:.4f} -> {after[1]:.4f}\n"
        f"corr: {single_profile.local_parameter_correlation:.3f} -> "
        f"{combined_profile.local_parameter_correlation:.3f}",
        transform=axes[1, 1].transAxes, ha="right", va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#111827",
              "edgecolor": "#64748b", "alpha": 0.9})

    figure.suptitle(
        "Double-slit detector design under a fixed shot budget",
        fontsize=16, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Design a complementary complete double-slit detector")
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--profile-count", type=int, default=51)
    parser.add_argument(
        "--plot-output", type=Path,
        default=Path("spatial_effect_double_slit_detector_design.png"))
    parser.add_argument(
        "--json-output", type=Path,
        default=Path("spatial_effect_double_slit_detector_design.json"))
    args = parser.parse_args()
    if args.profile_count < 5:
        parser.error("--profile-count must be at least 5")

    experiment = double_slit_effect_experiment()
    design = select_double_slit_detector_setting(
        experiment, TRUE_SIGMA_T, TRUE_LAMBDA,
        DETECTOR_X_GRID, DETECTOR_WIDTH_GRID, TIME_OFFSET_GRID)
    second = replace(
        experiment,
        detector_x=design.best_detector_x,
        detector_width=design.best_detector_width,
        reference_time=design.best_reference_time)
    sigmas = np.unique(np.append(
        np.linspace(0.1, 0.35, args.profile_count), TRUE_SIGMA_T))
    lambdas = np.unique(np.append(
        np.linspace(0.3, 2.0, args.profile_count), TRUE_LAMBDA))
    single = fit_spatial_effect_joint_response(
        experiment, TRUE_SIGMA_T, TRUE_LAMBDA, sigmas, lambdas,
        shots=args.shots)
    combined = fit_spatial_effect_multi_response(
        [experiment, second], TRUE_SIGMA_T, TRUE_LAMBDA, sigmas, lambdas,
        shots=args.shots, shot_fractions=[0.5, 0.5],
        setting_names=["original", "designed"])
    first_run = run_spatial_effect_measurement(
        experiment, TRUE_SIGMA_T, lambda_strength=TRUE_LAMBDA)
    second_run = run_spatial_effect_measurement(
        second, TRUE_SIGMA_T, lambda_strength=TRUE_LAMBDA)
    evolution = spatial_effect_evolution(
        second, TRUE_SIGMA_T, lambda_strength=TRUE_LAMBDA)
    convergence = spatial_effect_convergence(
        second, TRUE_SIGMA_T, lambda_strength=TRUE_LAMBDA)
    plot_design(
        experiment, second, design, first_run, second_run, evolution,
        single, combined, args.plot_output)

    print("Double-slit detector design")
    print(
        f"  selected (x, width, time) = ({second.detector_x:.6g}, "
        f"{second.detector_width:.6g}, {second.reference_time:.6g})")
    print(
        f"  click mass original -> designed = "
        f"{first_run.click_probability:.6g} -> "
        f"{second_run.click_probability:.6g}")
    print(
        f"  condition original -> pair = "
        f"{single.fisher_condition_number:.6g} -> "
        f"{combined.fisher_condition_number:.6g}")
    print(
        f"  correlation original -> pair = "
        f"{single.local_parameter_correlation:.6g} -> "
        f"{combined.local_parameter_correlation:.6g}")
    print(
        f"  standard errors original -> pair = "
        f"{single.local_standard_errors} -> {combined.local_standard_errors}")
    print(f"  convergence TV = {convergence}")
    print(f"  plot = {args.plot_output}")

    payload = {
        "status": "synthetic_double_slit_detector_design_not_empirical_measurement",
        "objective": (
            "maximize the combined per-shot Fisher determinant among settings "
            "that improve both marginal standard errors at a 50/50 shot split"),
        "candidate_grid": {
            "detector_x": DETECTOR_X_GRID,
            "detector_width": DETECTOR_WIDTH_GRID,
            "classical_arrival_time_offset": TIME_OFFSET_GRID,
        },
        "original_experiment": asdict(experiment),
        "designed_experiment": asdict(second),
        "design": asdict(design),
        "original_run": asdict(first_run),
        "designed_run": asdict(second_run),
        "original_profile": _diagnostics(single),
        "combined_profile": _diagnostics(combined),
        "convergence_total_variation": convergence,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, default=_json_value, indent=2) + "\n",
        encoding="utf-8")
    print(f"  json = {args.json_output}")


if __name__ == "__main__":
    main()
