"""Run the locked double-slit effect-instrument robustness study."""

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
from matplotlib.colors import PowerNorm


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.spatial_effect_measurement import (
    double_slit_effect_experiment,
    fit_spatial_effect_joint_response,
    fit_spatial_effect_multi_response,
    run_spatial_effect_measurement,
    spatial_effect_convergence,
    spatial_effect_evolution,
    spatial_effect_potential,
)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _bin_centers(y, count):
    return np.asarray([
        np.mean(y[rows]) for rows in np.array_split(np.arange(y.size), count)
    ])


def _profile(experiment, second, sigmas, lambdas, shots):
    single = fit_spatial_effect_joint_response(
        experiment, 0.2, 1.0, sigmas, lambdas, shots=shots)
    combined = fit_spatial_effect_multi_response(
        [experiment, second], 0.2, 1.0, sigmas, lambdas,
        shots=shots, shot_fractions=[0.5, 0.5],
        setting_names=[f"detector_x={experiment.detector_x:g}",
                       f"detector_x={second.detector_x:g}"])
    return single, combined


def _diagnostics(profile):
    return {
        "best_sigma_t": profile.best_sigma_t,
        "best_lambda_strength": profile.best_lambda_strength,
        "fisher_condition_number": profile.fisher_condition_number,
        "local_parameter_correlation": profile.local_parameter_correlation,
        "local_standard_errors": profile.local_standard_errors,
    }


def plot_study(experiment, free_run, slit_run, evolution,
               free_profiles, slit_profiles, output_path):
    plt.style.use("dark_background")
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    mixed_density = np.tensordot(
        evolution.delay_weights, evolution.densities, axes=(0, 0))
    dx = experiment.lx / experiment.nx
    dy = experiment.ly / experiment.ny
    extent = (
        evolution.x[0] - dx / 2, evolution.x[-1] + dx / 2,
        evolution.y[0] - dy / 2, evolution.y[-1] + dy / 2,
    )
    density_image = axes[0, 0].imshow(
        mixed_density, origin="lower", extent=extent, aspect="equal",
        cmap="magma", interpolation="bilinear",
        norm=PowerNorm(gamma=0.5, vmin=0, vmax=float(np.max(mixed_density))))
    potential = spatial_effect_potential(experiment)
    axes[0, 0].contour(
        potential, levels=[0.1 * float(np.max(potential))],
        origin="lower", extent=extent, colors="white", linewidths=1.2)
    axes[0, 0].axvline(
        experiment.detector_x, color="#67e8f9", linestyle="--",
        linewidth=1.5)
    axes[0, 0].set(
        title="Temporally mixed double-slit density",
        xlabel="x", ylabel="y")
    colorbar = figure.colorbar(density_image, ax=axes[0, 0], fraction=0.046)
    colorbar.set_label("mixed probability density")

    centers = _bin_centers(evolution.y, experiment.y_bins)
    axes[0, 1].plot(
        centers, free_run.conditional_y_distribution,
        marker="o", color="#f472b6", label="matched free propagation")
    axes[0, 1].plot(
        centers, slit_run.conditional_y_distribution,
        marker="o", color="#22d3ee", label="double slit")
    axes[0, 1].set(
        title="Conditional click pattern",
        xlabel="y-bin center", ylabel="p(y bin | click)")
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend(framealpha=0.75)
    axes[0, 1].text(
        0.98, 0.95,
        f"click mass\nfree {free_run.click_probability:.4f}\n"
        f"slit {slit_run.click_probability:.4f}",
        transform=axes[0, 1].transAxes, ha="right", va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#111827",
              "edgecolor": "#64748b", "alpha": 0.9})

    slit_single, slit_two = slit_profiles
    axes[1, 0].plot(
        slit_single.candidate_sigmas,
        slit_single.profiled_expected_deviance,
        color="#fbbf24", linewidth=2, label="one detector")
    axes[1, 0].plot(
        slit_two.candidate_sigmas,
        slit_two.profiled_expected_deviance,
        color="#22d3ee", linewidth=2, label="fixed detectors x=1.5 and x=0")
    axes[1, 0].axvline(0.2, color="#a3e635", linestyle=":")
    for level, color in ((2.30, "white"), (5.99, "#fb7185")):
        axes[1, 0].axhline(level, color=color, linestyle="--", alpha=0.7)
    axes[1, 0].set(
        title="Double-slit width profile after profiling lambda",
        xlabel=r"$\sigma_T$", ylabel=r"minimum $2N\,KL$", ylim=(0, 25))
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(framealpha=0.75)

    free_single, free_two = free_profiles
    profiles = [free_single, free_two, slit_single, slit_two]
    names = ["free\n1 det.", "free\n2 det.", "slit\n1 det.", "slit\n2 det."]
    positions = np.arange(len(profiles))
    width = 0.34
    axes[1, 1].bar(
        positions - width / 2,
        [profile.local_standard_errors[0] for profile in profiles],
        width, color="#a3e635", label=r"SE($\sigma_T$)")
    axes[1, 1].bar(
        positions + width / 2,
        [profile.local_standard_errors[1] for profile in profiles],
        width, color="#c084fc", label=r"SE($\lambda$)")
    axes[1, 1].set(
        title="Matched Fisher comparison, 100,000 total shots",
        ylabel="Asymptotic standard error", xticks=positions,
        xticklabels=names)
    axes[1, 1].grid(axis="y", alpha=0.2)
    axes[1, 1].legend(framealpha=0.75)
    metric_lines = [
        f"{name.replace(chr(10), ' ')}: cond={profile.fisher_condition_number:.1f}, "
        f"corr={profile.local_parameter_correlation:.3f}"
        for name, profile in zip(names, profiles)
    ]
    axes[1, 1].text(
        0.98, 0.96, "\n".join(metric_lines),
        transform=axes[1, 1].transAxes, ha="right", va="top", fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#111827",
              "edgecolor": "#64748b", "alpha": 0.9})

    figure.suptitle(
        "Complete double-slit effect instrument: geometry and identifiability",
        fontsize=16, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Run the locked complete double-slit effect study")
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--profile-count", type=int, default=51)
    parser.add_argument(
        "--plot-output", type=Path,
        default=Path("spatial_effect_double_slit_study.png"))
    parser.add_argument(
        "--json-output", type=Path,
        default=Path("spatial_effect_double_slit_study.json"))
    args = parser.parse_args()
    if args.profile_count < 5:
        parser.error("--profile-count must be at least 5")

    slit = double_slit_effect_experiment()
    matched_free = replace(slit, potential_mode="free")
    slit_second = replace(slit, detector_x=0.0)
    free_second = replace(matched_free, detector_x=0.0)
    sigmas = np.unique(np.append(np.linspace(0.1, 0.35, args.profile_count), 0.2))
    lambdas = np.unique(np.append(np.linspace(0.3, 2.0, args.profile_count), 1.0))

    slit_run = run_spatial_effect_measurement(slit, 0.2, lambda_strength=1.0)
    free_run = run_spatial_effect_measurement(
        matched_free, 0.2, lambda_strength=1.0)
    evolution = spatial_effect_evolution(slit, 0.2, lambda_strength=1.0)
    free_profiles = _profile(
        matched_free, free_second, sigmas, lambdas, args.shots)
    slit_profiles = _profile(
        slit, slit_second, sigmas, lambdas, args.shots)
    convergence = spatial_effect_convergence(slit, 0.2, lambda_strength=1.0)
    plot_study(
        slit, free_run, slit_run, evolution,
        free_profiles, slit_profiles, args.plot_output)

    free_single, free_two = free_profiles
    slit_single, slit_two = slit_profiles
    distribution_tv = 0.5 * float(np.sum(np.abs(
        slit_run.probabilities - free_run.probabilities)))
    print("Complete double-slit effect study")
    print(f"  injected (sigma, lambda) = (0.2, 1)")
    print(
        f"  recovered slit minimum   = "
        f"({slit_single.best_sigma_t:.6g}, "
        f"{slit_single.best_lambda_strength:.6g})")
    print(f"  TV(slit, matched free)   = {distribution_tv:.6g}")
    print(
        f"  click mass free -> slit  = "
        f"{free_run.click_probability:.6g} -> {slit_run.click_probability:.6g}")
    print(
        f"  slit condition 1 -> 2 det. = "
        f"{slit_single.fisher_condition_number:.3f} -> "
        f"{slit_two.fisher_condition_number:.3f}")
    print(
        f"  slit correlation 1 -> 2 det. = "
        f"{slit_single.local_parameter_correlation:.6f} -> "
        f"{slit_two.local_parameter_correlation:.6f}")
    print(f"  convergence TV           = {convergence}")
    print(f"  plot                     = {args.plot_output}")

    payload = {
        "status": "synthetic_double_slit_effect_study_not_empirical_measurement",
        "double_slit_experiment": asdict(slit),
        "matched_free_experiment": asdict(matched_free),
        "distribution_total_variation": distribution_tv,
        "double_slit_run": asdict(slit_run),
        "matched_free_run": asdict(free_run),
        "free_single": _diagnostics(free_single),
        "free_two_setting": _diagnostics(free_two),
        "double_slit_single": _diagnostics(slit_single),
        "double_slit_two_setting": _diagnostics(slit_two),
        "double_slit_single_profile": asdict(slit_single),
        "double_slit_two_setting_profile": asdict(slit_two),
        "convergence_total_variation": convergence,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, default=_json_value, indent=2) + "\n",
        encoding="utf-8")
    print(f"  json                     = {args.json_output}")


if __name__ == "__main__":
    main()
