"""Profile and plot joint spatial-effect width/coupling identifiability."""

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

from analysis.spatial_effect_measurement import (
    SpatialEffectExperiment,
    fit_spatial_effect_joint_response,
)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _candidate_axis(lower, upper, count, injected):
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("Profile bounds must be finite and increasing")
    if count < 3:
        raise ValueError("Profile axes require at least three points")
    return np.unique(np.append(np.linspace(lower, upper, count), injected))


def plot_joint_profile(profile, output_path):
    plt.style.use("dark_background")
    figure = plt.figure(figsize=(13.5, 7.4), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.45, 1))
    surface_ax = figure.add_subplot(grid[:, 0])
    ridge_ax = figure.add_subplot(grid[0, 1])
    profile_ax = figure.add_subplot(grid[1, 1])

    transformed = np.log10(1.0 + profile.expected_deviance)
    surface = surface_ax.pcolormesh(
        profile.candidate_sigmas, profile.candidate_lambdas, transformed,
        shading="auto", cmap="viridis")
    colorbar = figure.colorbar(surface, ax=surface_ax, fraction=0.05)
    colorbar.set_label(r"$\log_{10}(1 + 2N\,KL)$")
    maximum = float(np.max(profile.expected_deviance))
    contour_levels = [level for level in (2.30, 5.99, 9.21)
                      if level < maximum]
    if contour_levels:
        contours = surface_ax.contour(
            profile.candidate_sigmas, profile.candidate_lambdas,
            profile.expected_deviance, levels=contour_levels,
            colors=["#f8fafc", "#fbbf24", "#fb7185"][:len(contour_levels)],
            linewidths=1.2)
        surface_ax.clabel(contours, inline=True, fontsize=8, fmt="%.2f")
    surface_ax.plot(
        profile.candidate_sigmas, profile.profiled_lambda_strength,
        color="#e2e8f0", linewidth=1.4, linestyle="--",
        label="profile ridge")
    surface_ax.scatter(
        [profile.true_sigma_t], [profile.true_lambda_strength], marker="*",
        s=180, color="#a3e635", edgecolor="black", linewidth=0.8,
        label="injected")
    surface_ax.scatter(
        [profile.best_sigma_t], [profile.best_lambda_strength], marker="x",
        s=90, color="#22d3ee", linewidth=2.2, label="grid minimum")
    surface_ax.set(
        title="Joint expected-likelihood surface",
        xlabel=r"Temporal width $\sigma_T$",
        ylabel=r"Response strength $\lambda$")
    surface_ax.legend(loc="upper right", framealpha=0.8)

    ridge_ax.plot(
        profile.candidate_sigmas, profile.profiled_lambda_strength,
        color="#38bdf8", linewidth=2)
    ridge_ax.axhline(
        profile.true_lambda_strength, color="#a3e635", linestyle=":",
        label="injected lambda")
    ridge_ax.axvline(
        profile.true_sigma_t, color="#a3e635", linestyle=":")
    ridge_ax.set(
        title="Best coupling at each width",
        xlabel=r"$\sigma_T$", ylabel=r"profiled $\hat\lambda(\sigma_T)$")
    ridge_ax.grid(alpha=0.2)
    ridge_ax.legend(framealpha=0.75)

    profile_ax.plot(
        profile.candidate_sigmas, profile.profiled_expected_deviance,
        color="#f472b6", linewidth=2)
    for level, color in ((2.30, "#f8fafc"), (5.99, "#fbbf24")):
        profile_ax.axhline(level, color=color, linestyle="--", alpha=0.8)
    profile_ax.axvline(
        profile.true_sigma_t, color="#a3e635", linestyle=":")
    upper = min(25.0, max(7.0, 1.05 * float(np.percentile(
        profile.profiled_expected_deviance, 70))))
    profile_ax.set(
        title="Likelihood after profiling lambda",
        xlabel=r"$\sigma_T$", ylabel=r"minimum $2N\,KL$",
        ylim=(0, upper))
    profile_ax.grid(alpha=0.2)
    profile_ax.text(
        0.98, 0.95,
        "Fisher condition = "
        f"{profile.fisher_condition_number:.1f}\n"
        "local parameter corr. = "
        f"{profile.local_parameter_correlation:.3f}",
        transform=profile_ax.transAxes, ha="right", va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#111827",
              "edgecolor": "#64748b", "alpha": 0.9})

    figure.suptitle(
        "Spatial effect instrument: joint width–coupling identifiability",
        fontsize=16, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Joint sigma_T/lambda profile for the complete spatial POVM")
    parser.add_argument("--true-sigma", type=float, default=0.6)
    parser.add_argument("--true-lambda", type=float, default=1.0)
    parser.add_argument("--sigma-min", type=float, default=0.25)
    parser.add_argument("--sigma-max", type=float, default=1.2)
    parser.add_argument("--sigma-count", type=int, default=81)
    parser.add_argument("--lambda-min", type=float, default=0.05)
    parser.add_argument("--lambda-max", type=float, default=2.5)
    parser.add_argument("--lambda-count", type=int, default=81)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--delay-step", type=float, default=0.1)
    parser.add_argument("--horizon-sigmas", type=float, default=4.0)
    parser.add_argument(
        "--plot-output", type=Path,
        default=Path("spatial_effect_joint_profile.png"))
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    try:
        sigmas = _candidate_axis(
            args.sigma_min, args.sigma_max, args.sigma_count, args.true_sigma)
        lambdas = _candidate_axis(
            args.lambda_min, args.lambda_max, args.lambda_count,
            args.true_lambda)
    except ValueError as error:
        parser.error(str(error))
    experiment = SpatialEffectExperiment(
        nx=args.nx, ny=args.ny, delay_step=args.delay_step,
        horizon_sigmas=args.horizon_sigmas)
    profile = fit_spatial_effect_joint_response(
        experiment, args.true_sigma, args.true_lambda, sigmas, lambdas,
        shots=args.shots)
    plot_joint_profile(profile, args.plot_output)

    print("Joint spatial effect profile")
    print(
        f"  injected (sigma_T, lambda) = "
        f"({profile.true_sigma_t:.6g}, {profile.true_lambda_strength:.6g})")
    print(
        f"  recovered grid minimum     = "
        f"({profile.best_sigma_t:.6g}, {profile.best_lambda_strength:.6g})")
    print(f"  Fisher eigenvalues         = {profile.fisher_eigenvalues}")
    print(f"  Fisher condition number    = {profile.fisher_condition_number:.6g}")
    print(f"  local standard errors      = {profile.local_standard_errors}")
    print(f"  local parameter correlation= {profile.local_parameter_correlation:.6g}")
    print(f"  plot                        = {args.plot_output}")

    if args.json_output:
        payload = {
            "status": "synthetic_joint_profile_not_empirical_measurement",
            "experiment": asdict(experiment),
            "profile": asdict(profile),
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, default=_json_value, indent=2) + "\n",
            encoding="utf-8")
        print(f"  json                        = {args.json_output}")


if __name__ == "__main__":
    main()
