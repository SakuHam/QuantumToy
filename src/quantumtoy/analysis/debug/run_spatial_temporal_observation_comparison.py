"""Compare integrated, time-resolved, and coherent temporal observations."""

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

from analysis.spatial_detector_inference import DetectorResponse
from analysis.spatial_effect_measurement import double_slit_effect_experiment
from analysis.spatial_temporal_observation import (
    fixed_temporal_delays,
    temporal_observation_fisher_information,
    temporal_observation_probabilities,
)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _diagnostics(first, second, response, strategy, delay_horizon, phase, shots,
                 category_blur_sigma_bins=0.0):
    information = 0.5 * (
        temporal_observation_fisher_information(
            first, 0.2, 1.0, response, strategy=strategy,
            delay_horizon=delay_horizon, phase=phase,
            category_blur_sigma_bins=category_blur_sigma_bins)
        + temporal_observation_fisher_information(
            second, 0.2, 1.0, response, strategy=strategy,
            delay_horizon=delay_horizon, phase=phase,
            category_blur_sigma_bins=category_blur_sigma_bins))
    covariance = np.linalg.inv(shots * information)
    errors = np.sqrt(np.diag(covariance))
    correlation = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
    return {
        "strategy": strategy,
        "phase": phase,
        "category_blur_sigma_bins": category_blur_sigma_bins,
        "fisher_information_per_shot": information,
        "log_determinant_per_shot": float(np.linalg.slogdet(information)[1]),
        "standard_errors": errors,
        "parameter_correlation": float(correlation),
    }


def _plot(rows, phases, phase_logdets, output):
    baseline = rows[0]
    names = [row["label"] for row in rows]
    sigma_ratios = [row["standard_errors"][0] / baseline["standard_errors"][0]
                    for row in rows]
    lambda_ratios = [row["standard_errors"][1] / baseline["standard_errors"][1]
                     for row in rows]
    gains = [row["log_determinant_per_shot"] - baseline["log_determinant_per_shot"]
             for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    x = np.arange(len(rows))
    width = 0.36
    axes[0].bar(x - width / 2, sigma_ratios, width, label="sigma_T")
    axes[0].bar(x + width / 2, lambda_ratios, width, label="lambda")
    axes[0].axhline(1, color="gray", linestyle="--")
    axes[0].set(ylabel="standard-error ratio", title="Local precision (lower is better)",
                xticks=x, xticklabels=names)
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].legend()
    axes[1].bar(x, gains, color=plt.cm.viridis(np.linspace(0.15, 0.85, len(rows))))
    axes[1].axhline(0, color="gray", linestyle="--")
    axes[1].set(ylabel="log-det gain over current", title="Joint information gain",
                xticks=x, xticklabels=names)
    axes[1].tick_params(axis="x", rotation=18)
    axes[2].plot(phases, phase_logdets, marker="o")
    best = int(np.argmax(phase_logdets))
    axes[2].scatter([phases[best]], [phase_logdets[best]], marker="*", s=180,
                    color="#f59e0b", edgecolor="black", label="selected phase")
    axes[2].set(xlabel="phase offset (rad)", ylabel="combined log det",
                title="Coherent phase scan")
    axes[2].legend()
    fig.suptitle("Temporal observation comparison — synthetic local Fisher benchmark")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=48)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--phase-points", type=int, default=12)
    parser.add_argument("--json-output", type=Path,
                        default=Path("spatial_temporal_observation_comparison.json"))
    parser.add_argument("--plot-output", type=Path,
                        default=Path("spatial_temporal_observation_comparison.png"))
    args = parser.parse_args()
    if args.grid_size < 24 or args.shots <= 0 or args.phase_points < 3:
        parser.error("grid-size >= 24, positive shots, and phase-points >= 3 are required")

    first = double_slit_effect_experiment(nx=args.grid_size, ny=args.grid_size)
    second = replace(first, detector_x=0, detector_width=0.5, reference_time=13 / 30)
    response = DetectorResponse(0.88, 0.012, 0.45, 0.05)
    sharper = replace(response, blur_sigma_bins=0, timing_jitter=0)
    delay_horizon = 4 * 0.2
    mode_count = fixed_temporal_delays(first, delay_horizon).size + 1
    phases = np.linspace(0, 2 * np.pi / mode_count, args.phase_points, endpoint=False)
    coherent_scan = [
        _diagnostics(first, second, response, "coherent", delay_horizon,
                     float(phase), args.shots)
        for phase in phases]
    phase_logdets = np.array([row["log_determinant_per_shot"] for row in coherent_scan])
    best_index = int(np.argmax(phase_logdets))
    rows = [
        _diagnostics(first, second, response, "integrated", delay_horizon, 0, args.shots)
        | {"label": "current y only"},
        _diagnostics(first, second, sharper, "integrated", delay_horizon, 0, args.shots)
        | {"label": "sharp y only"},
        _diagnostics(first, second, response, "resolved", delay_horizon, 0,
                     args.shots, response.timing_jitter / first.delay_step)
        | {"label": "time tagged"},
        _diagnostics(first, second, response, "resolved", delay_horizon, 0, args.shots)
        | {"label": "ideal time tag"},
        coherent_scan[best_index] | {"label": "coherent ports"},
    ]
    baseline = rows[0]
    for row in rows:
        row["standard_error_ratios"] = (
            row["standard_errors"] / baseline["standard_errors"])
        row["log_determinant_gain"] = (
            row["log_determinant_per_shot"]
            - baseline["log_determinant_per_shot"])

    # Algebraic checks: resolving or coherently rotating the temporal register
    # must preserve the current spatial marginal at each detector setting.
    marginal_errors = {}
    for name, experiment in (("original", first), ("selected", second)):
        integrated = temporal_observation_probabilities(
            experiment, 0.2, 1, response, strategy="integrated",
            delay_horizon=delay_horizon)
        errors = {}
        for strategy, phase in (("resolved", 0), ("coherent", phases[best_index])):
            complete = temporal_observation_probabilities(
                experiment, 0.2, 1, response, strategy=strategy,
                delay_horizon=delay_horizon, phase=float(phase))
            y_bins = experiment.y_bins
            marginalized = np.concatenate([
                complete[:-1].reshape(-1, y_bins).sum(axis=0), complete[-1:]])
            errors[strategy] = float(np.max(np.abs(marginalized - integrated)))
        marginal_errors[name] = errors

    payload = {
        "status": "synthetic_information_comparison_not_trf_prediction",
        "settings": vars(args) | {
            "json_output": str(args.json_output), "plot_output": str(args.plot_output),
            "true_sigma_t": 0.2, "true_lambda_strength": 1.0,
            "delay_horizon": delay_horizon, "temporal_mode_count": mode_count,
            "shot_allocation": [args.shots // 2, args.shots - args.shots // 2],
        },
        "detector_response": asdict(response),
        "interpretation": (
            "Resolved-time and coherent-port rows assume access to a temporal "
            "register that is absent from the current detector. The coherent "
            "purification and phase scan are design benchmarks, not predictions "
            "of TRF-IT. Detector response is treated as known in this local Fisher study."),
        "marginal_consistency_max_abs_error": marginal_errors,
        "phase_scan": {"phases": phases, "log_determinants": phase_logdets,
                       "best_index": best_index},
        "results": rows,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, default=_json_value, indent=2) + "\n")
    _plot(rows, phases, phase_logdets, args.plot_output)
    for row in rows:
        print(f"{row['label']:<18} SE=({row['standard_errors'][0]:.6g}, "
              f"{row['standard_errors'][1]:.6g}) ratios="
              f"({row['standard_error_ratios'][0]:.4f}, "
              f"{row['standard_error_ratios'][1]:.4f}) "
              f"logdet_gain={row['log_determinant_gain']:+.4f}")
    print(f"best coherent phase={phases[best_index]:.6g}")
    print(f"marginal errors={marginal_errors}")
    print(f"results={args.json_output}; plot={args.plot_output}")


if __name__ == "__main__":
    main()
