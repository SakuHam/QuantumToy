"""Design and evaluate a second independent spatial detector setting."""

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


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.spatial_effect_measurement import (
    SpatialEffectExperiment,
    fit_spatial_effect_joint_response,
    fit_spatial_effect_multi_response,
    select_complementary_detector_setting,
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


def _draw_surface(axis, sigmas, lambdas, deviance, ridge, truth, title,
                  color_limits):
    transformed = np.log10(1.0 + deviance)
    surface = axis.pcolormesh(
        sigmas, lambdas, transformed, shading="auto", cmap="viridis",
        vmin=color_limits[0], vmax=color_limits[1])
    levels = [level for level in (2.30, 5.99, 9.21)
              if level < float(np.max(deviance))]
    if levels:
        axis.contour(
            sigmas, lambdas, deviance, levels=levels,
            colors=["#f8fafc", "#fbbf24", "#fb7185"][:len(levels)],
            linewidths=1.0)
    axis.plot(sigmas, ridge, color="#e2e8f0", linestyle="--", linewidth=1.3)
    axis.scatter(
        [truth[0]], [truth[1]], marker="*", s=140,
        color="#a3e635", edgecolor="black", linewidth=0.7)
    axis.set(title=title, xlabel=r"$\sigma_T$", ylabel=r"$\lambda$")
    return surface


def plot_comparison(single, combined, second_detector_x, output_path):
    plt.style.use("dark_background")
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)
    color_max = max(
        float(np.max(np.log10(1 + single.expected_deviance))),
        float(np.max(np.log10(1 + combined.expected_deviance))))
    truth = (single.true_sigma_t, single.true_lambda_strength)
    first_surface = _draw_surface(
        axes[0, 0], single.candidate_sigmas, single.candidate_lambdas,
        single.expected_deviance, single.profiled_lambda_strength, truth,
        r"One setting: detector $x=1.5$", (0, color_max))
    _draw_surface(
        axes[0, 1], combined.candidate_sigmas, combined.candidate_lambdas,
        combined.expected_deviance, combined.profiled_lambda_strength, truth,
        rf"Two settings: detector $x=1.5$ and $x={second_detector_x:g}$",
        (0, color_max))
    colorbar = figure.colorbar(
        first_surface, ax=axes[0, :], orientation="vertical", fraction=0.025)
    colorbar.set_label(r"$\log_{10}(1 + 2N\,KL)$")

    axes[1, 0].plot(
        single.candidate_sigmas, single.profiled_expected_deviance,
        color="#f472b6", linewidth=2, label="one setting")
    axes[1, 0].plot(
        combined.candidate_sigmas, combined.profiled_expected_deviance,
        color="#22d3ee", linewidth=2, label="two settings")
    for level, color in ((2.30, "#f8fafc"), (5.99, "#fbbf24")):
        axes[1, 0].axhline(level, color=color, linestyle="--", alpha=0.7)
    axes[1, 0].axvline(
        single.true_sigma_t, color="#a3e635", linestyle=":")
    axes[1, 0].set(
        title="Width likelihood after profiling lambda",
        xlabel=r"$\sigma_T$", ylabel=r"minimum $2N\,KL$", ylim=(0, 25))
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(framealpha=0.75)

    positions = np.arange(2)
    width = 0.35
    axes[1, 1].bar(
        positions - width / 2, single.local_standard_errors, width,
        color="#f472b6", label="one setting")
    axes[1, 1].bar(
        positions + width / 2, combined.local_standard_errors, width,
        color="#22d3ee", label="two settings")
    axes[1, 1].set(
        title="Local uncertainty with the same total shot budget",
        ylabel="Asymptotic standard error", xticks=positions,
        xticklabels=[r"$\sigma_T$", r"$\lambda$"])
    axes[1, 1].grid(axis="y", alpha=0.2)
    axes[1, 1].legend(framealpha=0.75)
    axes[1, 1].text(
        0.98, 0.95,
        "one setting\n"
        f"  condition {single.fisher_condition_number:.1f}\n"
        f"  corr. {single.local_parameter_correlation:.3f}\n\n"
        "two settings\n"
        f"  condition {combined.fisher_condition_number:.1f}\n"
        f"  corr. {combined.local_parameter_correlation:.3f}",
        transform=axes[1, 1].transAxes, ha="right", va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#111827",
              "edgecolor": "#64748b", "alpha": 0.9})

    figure.suptitle(
        "Independent detector setting breaks the width–coupling degeneracy",
        fontsize=16, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Design a second detector setting and combine both laws")
    parser.add_argument("--true-sigma", type=float, default=0.6)
    parser.add_argument("--true-lambda", type=float, default=1.0)
    parser.add_argument(
        "--candidate-detector-x", type=float, nargs="+",
        default=[-0.5, 0.0, 0.5, 1.0, 2.5])
    parser.add_argument("--sigma-min", type=float, default=0.25)
    parser.add_argument("--sigma-max", type=float, default=1.2)
    parser.add_argument("--sigma-count", type=int, default=81)
    parser.add_argument("--lambda-min", type=float, default=0.05)
    parser.add_argument("--lambda-max", type=float, default=2.5)
    parser.add_argument("--lambda-count", type=int, default=81)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--first-setting-fraction", type=float, default=0.5)
    parser.add_argument(
        "--plot-output", type=Path,
        default=Path("spatial_effect_two_setting_profile.png"))
    parser.add_argument(
        "--json-output", type=Path,
        default=Path("spatial_effect_two_setting_profile.json"))
    args = parser.parse_args()

    try:
        sigmas = _candidate_axis(
            args.sigma_min, args.sigma_max, args.sigma_count, args.true_sigma)
        lambdas = _candidate_axis(
            args.lambda_min, args.lambda_max, args.lambda_count,
            args.true_lambda)
    except ValueError as error:
        parser.error(str(error))
    first = SpatialEffectExperiment()
    design = select_complementary_detector_setting(
        first, args.true_sigma, args.true_lambda, args.candidate_detector_x,
        first_setting_fraction=args.first_setting_fraction)
    second = replace(first, detector_x=design.best_detector_x)
    single = fit_spatial_effect_joint_response(
        first, args.true_sigma, args.true_lambda, sigmas, lambdas,
        shots=args.shots)
    combined = fit_spatial_effect_multi_response(
        [first, second], args.true_sigma, args.true_lambda, sigmas, lambdas,
        shots=args.shots,
        shot_fractions=[args.first_setting_fraction,
                        1.0 - args.first_setting_fraction],
        setting_names=[f"detector_x={first.detector_x:g}",
                       f"detector_x={second.detector_x:g}"])
    plot_comparison(single, combined, second.detector_x, args.plot_output)

    print("Two-setting spatial effect profile")
    print(f"  selected second detector x = {second.detector_x:.6g}")
    print(f"  shot fractions             = {combined.shot_fractions}")
    print(
        f"  recovered (sigma, lambda)  = "
        f"({combined.best_sigma_t:.6g}, {combined.best_lambda_strength:.6g})")
    print(
        f"  Fisher condition           = "
        f"{single.fisher_condition_number:.3f} -> "
        f"{combined.fisher_condition_number:.3f}")
    print(
        f"  parameter correlation      = "
        f"{single.local_parameter_correlation:.6f} -> "
        f"{combined.local_parameter_correlation:.6f}")
    print(
        f"  standard errors            = "
        f"{single.local_standard_errors} -> {combined.local_standard_errors}")
    print(f"  plot                       = {args.plot_output}")

    payload = {
        "status": "synthetic_two_setting_design_not_empirical_measurement",
        "first_experiment": asdict(first),
        "second_experiment": asdict(second),
        "design": asdict(design),
        "single_setting_profile": asdict(single),
        "two_setting_profile": asdict(combined),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, default=_json_value, indent=2) + "\n",
        encoding="utf-8")
    print(f"  json                       = {args.json_output}")


if __name__ == "__main__":
    main()
