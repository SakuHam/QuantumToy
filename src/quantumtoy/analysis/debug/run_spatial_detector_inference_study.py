"""Run calibrated, robust double-slit design and Monte Carlo inference."""

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
from scipy.stats import binom


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.spatial_detector_inference import (
    DetectorResponse,
    bootstrap_spatial_detection_threshold,
    build_spatial_detector_inference_library,
    detector_response_matrix,
    fit_detector_calibration,
    fit_spatial_detector_counts,
    monte_carlo_spatial_detector_recovery,
    realistic_spatial_fisher_information,
    select_robust_double_slit_detector_setting,
    simulate_realistic_counts,
)
from analysis.spatial_effect_measurement import double_slit_effect_experiment


TRUE_RESPONSE = DetectorResponse(
    efficiency=0.88,
    dark_probability=0.012,
    blur_sigma_bins=0.45,
    timing_jitter=0.05,
)
TRUE_SIGMA_T = 0.2
TRUE_LAMBDA = 1.0
DESIGN_SIGMAS = (0.12, 0.2, 0.28)
DESIGN_LAMBDAS = (0.5, 1.0, 1.5)
DETECTOR_X_GRID = (0.0, 0.5, 1.0)
DETECTOR_WIDTH_GRID = (0.35, 0.5)
TIME_OFFSET_GRID = (-0.4, -0.2, 0.0)
JITTER_GRID = (0.025, 0.05, 0.075)
FIT_SIGMAS = np.linspace(0.12, 0.28, 7)
FIT_LAMBDAS = np.array([
    0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16,
    0.25, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0,
    1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5,
])


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _wilson_interval(proportion, repetitions, z=1.959963984540054):
    successes = int(round(float(proportion) * repetitions))
    center = successes / repetitions
    denominator = 1 + z ** 2 / repetitions
    midpoint = (center + z ** 2 / (2 * repetitions)) / denominator
    half_width = z * np.sqrt(
        center * (1 - center) / repetitions
        + z ** 2 / (4 * repetitions ** 2)) / denominator
    return np.array([midpoint - half_width, midpoint + half_width])


def _quantile_order_statistic_interval(values, probability=0.95):
    values = np.sort(np.asarray(values, dtype=float))
    lower_rank = int(binom.ppf(0.025, values.size, probability))
    upper_rank = int(binom.ppf(0.975, values.size, probability))
    return np.array([
        values[max(0, lower_rank - 1)],
        values[min(values.size - 1, upper_rank)],
    ])


def _calibration_experiments(experiment):
    return tuple(replace(
        experiment, potential_mode="free", detector_x=0,
        detector_width=0.3, reference_time=time)
        for time in (0.55, 0.7, 0.85, 1.0, 1.15))


def _selected_experiment(experiment, design):
    return replace(
        experiment,
        detector_x=design.best_detector_x,
        detector_width=design.best_detector_width,
        reference_time=design.best_reference_time)


def _high_resolution_verification(response, design):
    first = double_slit_effect_experiment()
    second = _selected_experiment(first, design)
    gains, ratios = [], []
    for sigma_t in DESIGN_SIGMAS:
        for lambda_strength in DESIGN_LAMBDAS:
            first_fisher = realistic_spatial_fisher_information(
                first, sigma_t, lambda_strength, response)
            second_fisher = realistic_spatial_fisher_information(
                second, sigma_t, lambda_strength, response)
            combined = 0.5 * (first_fisher + second_fisher)
            gains.append(float(
                np.linalg.slogdet(combined)[1]
                - np.linalg.slogdet(first_fisher)[1]))
            ratios.append(np.sqrt(
                np.diag(np.linalg.inv(combined))
                / np.diag(np.linalg.inv(first_fisher))))
    ratios = np.asarray(ratios)
    return {
        "minimum_log_determinant_gain": float(np.min(gains)),
        "worst_sigma_standard_error_ratio": float(np.max(ratios[:, 0])),
        "worst_lambda_standard_error_ratio": float(np.max(ratios[:, 1])),
    }


def plot_study(
        calibration_fit, design, library, example_fit, recoveries,
        output_path):
    plt.style.use("dark_background")
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    matrix = detector_response_matrix(16, calibration_fit.response)
    image = axes[0, 0].imshow(matrix, origin="lower", aspect="auto", cmap="magma")
    axes[0, 0].set(
        title="Calibrated complete detector channel",
        xlabel="ideal outcome (last = no click)",
        ylabel="observed outcome (last = no click)")
    figure.colorbar(image, ax=axes[0, 0], fraction=0.046).set_label(
        "conditional probability")

    selected = design.best_index
    scatter = axes[0, 1].scatter(
        design.candidate_detector_x,
        design.candidate_reference_time,
        s=90 + 220 * design.candidate_detector_width,
        c=design.worst_log_determinant_gain, cmap="viridis",
        edgecolors="white", linewidths=0.4)
    axes[0, 1].scatter(
        [design.best_detector_x], [design.best_reference_time],
        marker="*", s=280, color="#fbbf24", edgecolors="black",
        label="minimax selection")
    axes[0, 1].set(
        title="Worst-case Fisher gain over parameter region",
        xlabel="detector x", ylabel="reference time")
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend(framealpha=0.75)
    figure.colorbar(scatter, ax=axes[0, 1], fraction=0.046).set_label(
        "minimum log determinant gain")
    axes[0, 1].text(
        0.98, 0.04,
        f"worst SE ratios\nsigma {design.worst_sigma_error_ratio[selected]:.3f}\n"
        f"lambda {design.worst_lambda_error_ratio[selected]:.3f}",
        transform=axes[0, 1].transAxes, ha="right", va="bottom",
        bbox={"boxstyle": "round", "facecolor": "#111827", "alpha": 0.9})

    delta = 2 * (
        example_fit.profile_negative_log_likelihood
        - example_fit.negative_log_likelihood)
    profile_image = axes[1, 0].pcolormesh(
        library.candidate_sigmas, library.candidate_lambdas, delta,
        shading="nearest", cmap="viridis", vmin=0,
        vmax=min(20, float(np.max(delta))))
    axes[1, 0].contour(
        library.candidate_sigmas, library.candidate_lambdas, delta,
        levels=[2.30, 5.991], colors=["white", "#fb7185"])
    axes[1, 0].scatter(
        [TRUE_SIGMA_T], [TRUE_LAMBDA], marker="x", s=90,
        color="#fbbf24", label="injection")
    axes[1, 0].set(
        title="One profiled synthetic test",
        xlabel=r"$\sigma_T$", ylabel=r"$\lambda$")
    axes[1, 0].legend(framealpha=0.75)
    figure.colorbar(profile_image, ax=axes[1, 0], fraction=0.046).set_label(
        r"$2\Delta$ negative log likelihood")

    names = [f"lambda={row.true_lambda_strength:g}" for row in recoveries]
    positions = np.arange(len(recoveries))
    width = 0.34
    axes[1, 1].bar(
        positions - width / 2,
        [row.detection_rate for row in recoveries], width,
        color="#22d3ee", label="detection rate")
    axes[1, 1].bar(
        positions + width / 2,
        [row.joint_coverage for row in recoveries], width,
        color="#a3e635", label="95% joint coverage")
    axes[1, 1].axhline(0.95, color="white", linestyle="--", alpha=0.6)
    axes[1, 1].set(
        title="Repeated calibration and test",
        ylabel="fraction", ylim=(0, 1.08), xticks=positions,
        xticklabels=names)
    axes[1, 1].grid(axis="y", alpha=0.2)
    axes[1, 1].legend(framealpha=0.75)

    figure.suptitle(
        "Calibrated robust double-slit detector model",
        fontsize=16, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Run calibrated robust spatial-detector inference")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--calibration-shots", type=int, default=50_000)
    parser.add_argument("--test-shots", type=int, default=100_000)
    parser.add_argument("--design-grid-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--plot-output", type=Path,
        default=Path("spatial_detector_inference_study.png"))
    parser.add_argument(
        "--json-output", type=Path,
        default=Path("spatial_detector_inference_study.json"))
    args = parser.parse_args()
    if (args.repetitions <= 0 or args.bootstrap_repetitions <= 1
            or args.calibration_shots <= 0
            or args.test_shots <= 0 or args.test_shots % 2
            or args.workers <= 0):
        parser.error("Use positive repetitions and shots; test-shots must be even")
    if args.design_grid_size < 24:
        parser.error("--design-grid-size must be at least 24")

    rng = np.random.default_rng(args.seed)
    experiment = double_slit_effect_experiment(
        nx=args.design_grid_size, ny=args.design_grid_size)
    calibration_experiments = _calibration_experiments(experiment)
    pilot_counts = simulate_realistic_counts(
        calibration_experiments, TRUE_SIGMA_T, 0, TRUE_RESPONSE,
        args.calibration_shots, rng=rng)
    calibration_fit = fit_detector_calibration(
        calibration_experiments, pilot_counts, TRUE_SIGMA_T, 0, JITTER_GRID)
    print("Pilot calibration complete; running robust detector design", flush=True)
    design = select_robust_double_slit_detector_setting(
        experiment, calibration_fit.response,
        DESIGN_SIGMAS, DESIGN_LAMBDAS,
        DETECTOR_X_GRID, DETECTOR_WIDTH_GRID, TIME_OFFSET_GRID)
    second = _selected_experiment(experiment, design)
    high_resolution = _high_resolution_verification(
        calibration_fit.response, design)
    print("Robust design verified; building inference library", flush=True)

    library = build_spatial_detector_inference_library(
        calibration_experiments, (experiment, second),
        calibration_sigma_t=TRUE_SIGMA_T, calibration_lambda_strength=0,
        candidate_sigmas=FIT_SIGMAS,
        candidate_lambdas=FIT_LAMBDAS,
        candidate_timing_jitters=JITTER_GRID)
    example_calibration_counts = simulate_realistic_counts(
        calibration_experiments, TRUE_SIGMA_T, 0, TRUE_RESPONSE,
        args.calibration_shots, rng=rng)
    example_test_counts = simulate_realistic_counts(
        (experiment, second), TRUE_SIGMA_T, TRUE_LAMBDA, TRUE_RESPONSE,
        [args.test_shots // 2, args.test_shots // 2], rng=rng)
    example_fit = fit_spatial_detector_counts(
        library, example_calibration_counts, example_test_counts)
    print(
        f"Inference library complete; bootstrapping with {args.workers} workers",
        flush=True)
    detection_threshold = bootstrap_spatial_detection_threshold(
        library, TRUE_RESPONSE,
        calibration_shots=args.calibration_shots,
        test_shots=[args.test_shots // 2, args.test_shots // 2],
        repetitions=args.bootstrap_repetitions, seed=args.seed + 1,
        workers=args.workers)
    print("Bootstrap complete; running independent evaluation", flush=True)
    recoveries = monte_carlo_spatial_detector_recovery(
        library, TRUE_RESPONSE,
        [(TRUE_SIGMA_T, 0), (TRUE_SIGMA_T, TRUE_LAMBDA)],
        calibration_shots=args.calibration_shots,
        test_shots=[args.test_shots // 2, args.test_shots // 2],
        repetitions=args.repetitions, seed=args.seed + 2,
        detection_threshold=detection_threshold.threshold,
        workers=args.workers)
    validation_intervals = {
        "bootstrap_threshold_order_statistic_95_interval": (
            _quantile_order_statistic_interval(
                detection_threshold.likelihood_ratios)),
        "per_scenario": [{
            "true_lambda_strength": row.true_lambda_strength,
            "detection_rate_wilson_95_interval": _wilson_interval(
                row.detection_rate, row.repetitions),
            "joint_coverage_wilson_95_interval": _wilson_interval(
                row.joint_coverage, row.repetitions),
        } for row in recoveries],
    }
    plot_study(
        calibration_fit, design, library, example_fit, recoveries,
        args.plot_output)

    print("Calibrated robust double-slit detector study")
    print(f"  pilot calibration = {calibration_fit.response}")
    print(
        f"  robust detector (x, width, time) = "
        f"({second.detector_x:.6g}, {second.detector_width:.6g}, "
        f"{second.reference_time:.6g})")
    print(
        f"  coarse worst SE ratios (sigma, lambda) = "
        f"({design.worst_sigma_error_ratio[design.best_index]:.6g}, "
        f"{design.worst_lambda_error_ratio[design.best_index]:.6g})")
    print(f"  high-resolution verification = {high_resolution}")
    print(
        f"  bootstrap LR threshold ({detection_threshold.confidence_level:.0%}) = "
        f"{detection_threshold.threshold:.6g}")
    print(
        f"  bootstrap threshold 95% order-stat interval = "
        f"{validation_intervals['bootstrap_threshold_order_statistic_95_interval']}")
    print(
        f"  example fit (sigma, lambda) = "
        f"({example_fit.best_sigma_t:.6g}, "
        f"{example_fit.best_lambda_strength:.6g})")
    for row in recoveries:
        print(
            f"  lambda={row.true_lambda_strength:g}: detection="
            f"{row.detection_rate:.3f}, coverage={row.joint_coverage:.3f}, "
            f"bias=({row.sigma_bias:.4g}, {row.lambda_bias:.4g}), "
            f"RMSE=({row.sigma_rmse:.4g}, {row.lambda_rmse:.4g})")
    print(f"  plot = {args.plot_output}")

    payload = {
        "status": "synthetic_calibrated_design_not_empirical_evidence",
        "settings": vars(args) | {
            "plot_output": str(args.plot_output),
            "json_output": str(args.json_output),
        },
        "true_detector_response": asdict(TRUE_RESPONSE),
        "pilot_calibration_fit": asdict(calibration_fit),
        "design_parameter_region": {
            "sigma_t": DESIGN_SIGMAS,
            "lambda_strength": DESIGN_LAMBDAS,
        },
        "robust_design": asdict(design),
        "selected_experiment": asdict(second),
        "high_resolution_verification": high_resolution,
        "bootstrap_detection_threshold": asdict(detection_threshold),
        "validation_intervals": validation_intervals,
        "example_fit": asdict(example_fit),
        "monte_carlo_recovery": [asdict(row) for row in recoveries],
        "interpretation": (
            "lambda=0 makes sigma_t unidentifiable; its null bias is not a "
            "physical width estimate. The bootstrap threshold is validated "
            "only if an independent null ensemble has the declared rejection "
            "rate within its sampling uncertainty."),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, default=_json_value, indent=2) + "\n",
        encoding="utf-8")
    print(f"  json = {args.json_output}")


if __name__ == "__main__":
    main()
