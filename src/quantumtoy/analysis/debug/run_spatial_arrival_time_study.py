"""Validate a finite-resolution joint p(y,t) arrival-time instrument."""

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

from analysis.spatial_arrival_time import (
    ArrivalTimeResponse,
    joint_spatial_arrival_fisher_information,
    joint_spatial_arrival_probabilities,
    marginalize_joint_arrival,
)
from analysis.spatial_detector_inference import DetectorResponse
from analysis.spatial_effect_measurement import double_slit_effect_experiment


TRUE_SIGMA = 0.2
TRUE_LAMBDA = 1.0


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _combined_fisher(experiments, response, timing, horizon, names):
    return sum(
        joint_spatial_arrival_fisher_information(
            experiment, TRUE_SIGMA, TRUE_LAMBDA, response, timing,
            delay_horizon=horizon, parameter_names=names)
        for experiment in experiments
    ) / len(experiments)


def _marginal_spatial_fisher(experiments, response, timing, horizon):
    steps = (1e-4 * TRUE_SIGMA, 1e-4)
    derivatives_by_setting = []
    probabilities_by_setting = []
    for experiment in experiments:
        def spatial(sigma_t, lambda_strength):
            complete = joint_spatial_arrival_probabilities(
                experiment, sigma_t, lambda_strength, response, timing,
                delay_horizon=horizon)
            return marginalize_joint_arrival(
                complete, experiment.y_bins, timing.time_bins)[0]
        center = spatial(TRUE_SIGMA, TRUE_LAMBDA)
        d_sigma = (
            spatial(TRUE_SIGMA + steps[0], TRUE_LAMBDA)
            - spatial(TRUE_SIGMA - steps[0], TRUE_LAMBDA)
        ) / (2 * steps[0])
        d_lambda = (
            spatial(TRUE_SIGMA, TRUE_LAMBDA + steps[1])
            - spatial(TRUE_SIGMA, TRUE_LAMBDA - steps[1])
        ) / (2 * steps[1])
        probabilities_by_setting.append(center)
        derivatives_by_setting.append(np.stack([d_sigma, d_lambda], axis=1))
    information = sum(
        (derivative.T / np.maximum(probability, 1e-300)) @ derivative
        for probability, derivative in zip(
            probabilities_by_setting, derivatives_by_setting)
    ) / len(experiments)
    return 0.5 * (information + information.T)


def _precision(information, shots, parameter_count=2, prior_information=None):
    total = shots * information.copy()
    if prior_information is not None:
        total += prior_information
    covariance = np.linalg.inv(total)
    errors = np.sqrt(np.diag(covariance)[:parameter_count])
    correlation = covariance[0, 1] / np.sqrt(
        covariance[0, 0] * covariance[1, 1])
    return errors, float(correlation), covariance


def _recovery(experiments, response, timing, horizon, shots, repetitions, seed):
    sigmas = np.linspace(0.14, 0.26, 7)
    lambdas = np.linspace(0.7, 1.3, 7)
    source_widths = np.array([0.02, 0.03, 0.04])
    clock_offsets = np.array([0.0, 0.01, 0.02])
    shape = (sigmas.size, lambdas.size, source_widths.size, clock_offsets.size)
    laws = np.empty(shape + (len(experiments),), dtype=object)
    for index in np.ndindex(shape):
        sigma_index, lambda_index, source_index, clock_index = index
        candidate_timing = replace(
            timing, source_pulse_sigma=float(source_widths[source_index]),
            clock_offset=float(clock_offsets[clock_index]))
        for setting_index, experiment in enumerate(experiments):
            laws[index + (setting_index,)] = joint_spatial_arrival_probabilities(
                experiment, float(sigmas[sigma_index]),
                float(lambdas[lambda_index]), response, candidate_timing,
                delay_horizon=horizon)

    per_setting = np.full(len(experiments), shots // len(experiments), dtype=int)
    per_setting[:shots % len(experiments)] += 1
    true_laws = [joint_spatial_arrival_probabilities(
        experiment, TRUE_SIGMA, TRUE_LAMBDA, response, timing,
        delay_horizon=horizon) for experiment in experiments]
    rng = np.random.default_rng(seed)
    fixed_estimates, calibrated_estimates = [], []
    source_observations, offset_observations = [], []
    source_se, offset_se = 0.005, 0.003
    for _ in range(repetitions):
        counts = [rng.multinomial(int(count), law)
                  for count, law in zip(per_setting, true_laws)]
        data_nll = np.zeros(shape)
        for index in np.ndindex(shape):
            value = 0.0
            for setting_index, observed in enumerate(counts):
                probability = laws[index + (setting_index,)]
                supported = observed > 0
                value -= float(np.sum(
                    observed[supported] * np.log(probability[supported])))
            data_nll[index] = value
        fixed = np.unravel_index(
            int(np.argmin(data_nll[:, :, 1, 1])), (sigmas.size, lambdas.size))
        fixed_estimates.append([sigmas[fixed[0]], lambdas[fixed[1]]])

        source_observed = rng.normal(timing.source_pulse_sigma, source_se)
        offset_observed = rng.normal(timing.clock_offset, offset_se)
        source_observations.append(source_observed)
        offset_observations.append(offset_observed)
        penalty = (
            0.5 * ((source_widths[:, None] - source_observed) / source_se) ** 2
            + 0.5 * ((clock_offsets[None, :] - offset_observed) / offset_se) ** 2
        )
        calibrated_nll = data_nll + penalty[None, None, :, :]
        selected = np.unravel_index(int(np.argmin(calibrated_nll)), shape)
        calibrated_estimates.append([
            sigmas[selected[0]], lambdas[selected[1]],
            source_widths[selected[2]], clock_offsets[selected[3]],
        ])

    fixed_estimates = np.asarray(fixed_estimates)
    calibrated_estimates = np.asarray(calibrated_estimates)

    def summary(estimates, truth):
        difference = estimates - np.asarray(truth)
        return {
            "mean": np.mean(estimates, axis=0),
            "bias": np.mean(difference, axis=0),
            "rmse": np.sqrt(np.mean(difference ** 2, axis=0)),
            "exact_grid_recovery_rate": np.mean(
                np.all(np.isclose(estimates, truth), axis=1)),
        }

    return {
        "candidate_sigmas": sigmas,
        "candidate_lambdas": lambdas,
        "candidate_source_pulse_sigmas": source_widths,
        "candidate_clock_offsets": clock_offsets,
        "calibration_standard_errors": [source_se, offset_se],
        "fixed_timing_estimates": fixed_estimates,
        "calibrated_timing_estimates": calibrated_estimates,
        "source_calibration_observations": np.asarray(source_observations),
        "clock_calibration_observations": np.asarray(offset_observations),
        "fixed_timing_summary": summary(
            fixed_estimates, [TRUE_SIGMA, TRUE_LAMBDA]),
        "calibrated_timing_summary": summary(
            calibrated_estimates,
            [TRUE_SIGMA, TRUE_LAMBDA,
             timing.source_pulse_sigma, timing.clock_offset]),
    }


def _plot(experiment, response, timing, joint, scale_scan, jitter_scan,
          precision_rows, recovery, output):
    joint_click = joint[:-1].reshape(timing.time_bins, experiment.y_bins).T
    conditional = joint_click / max(float(np.sum(joint_click)), 1e-300)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    image = axes[0, 0].imshow(
        conditional, origin="lower", aspect="auto",
        extent=(timing.window_start, timing.window_stop,
                -0.5, experiment.y_bins - 0.5), cmap="magma")
    axes[0, 0].set(
        title="Recorded joint click law p(y,t | click)",
        xlabel="arrival residual t", ylabel="y bin")
    fig.colorbar(image, ax=axes[0, 0], label="conditional probability density per bin")

    axes[0, 1].plot(
        scale_scan[:, 0], scale_scan[:, 1], marker="o", label="SE(sigma_T)")
    axes[0, 1].plot(
        scale_scan[:, 0], scale_scan[:, 2], marker="s", label="SE(lambda)")
    axes[0, 1].set(
        title="Dependence on latent-to-clock coupling",
        xlabel="delay scale kappa_t", ylabel="local standard error")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].plot(
        jitter_scan[:, 0], jitter_scan[:, 1], marker="o", label="SE(sigma_T)")
    axes[1, 0].plot(
        jitter_scan[:, 0], jitter_scan[:, 2], marker="s", label="SE(lambda)")
    axes[1, 0].set(
        title="Dependence on timestamp jitter",
        xlabel="detector timestamp sigma", ylabel="local standard error")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    estimates = recovery["calibrated_timing_estimates"]
    axes[1, 1].scatter(
        estimates[:, 0], estimates[:, 1], alpha=0.65,
        color="#2563eb", edgecolor="white", linewidth=0.4)
    axes[1, 1].scatter(
        [TRUE_SIGMA], [TRUE_LAMBDA], marker="*", s=240,
        color="#f59e0b", edgecolor="black", label="injected")
    axes[1, 1].set(
        title="Small calibrated-nuisance recovery",
        xlabel="fitted sigma_T", ylabel="fitted lambda")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()
    fig.suptitle(
        "Phenomenological joint spatial/arrival-time instrument",
        fontsize=15, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=48)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--repetitions", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument(
        "--json-output", type=Path,
        default=Path("spatial_arrival_time_study.json"))
    parser.add_argument(
        "--plot-output", type=Path,
        default=Path("spatial_arrival_time_study.png"))
    args = parser.parse_args()
    if args.grid_size < 24 or args.shots < 100 or args.repetitions < 1:
        parser.error("grid-size >= 24, shots >= 100, and repetitions >= 1 required")

    first = double_slit_effect_experiment(nx=args.grid_size, ny=args.grid_size)
    second = replace(first, detector_x=0, detector_width=0.5, reference_time=13 / 30)
    experiments = (first, second)
    response = DetectorResponse(0.88, 0.012, 0.45, 0.05)
    timing = ArrivalTimeResponse(
        source_pulse_sigma=0.03, clock_offset=0.01, delay_scale=1.0,
        window_start=-0.2, window_stop=1.0, time_bins=24)
    horizon = 4 * TRUE_SIGMA

    marginal_information = _marginal_spatial_fisher(
        experiments, response, timing, horizon)
    joint_information = _combined_fisher(
        experiments, response, timing, horizon,
        ("sigma_t", "lambda_strength"))
    nuisance_names = (
        "sigma_t", "lambda_strength", "source_pulse_sigma", "clock_offset")
    nuisance_information = _combined_fisher(
        experiments, response, timing, horizon, nuisance_names)
    marginal_errors, marginal_corr, _ = _precision(
        marginal_information, args.shots)
    joint_errors, joint_corr, _ = _precision(joint_information, args.shots)
    nuisance_errors, nuisance_corr, _ = _precision(
        nuisance_information, args.shots)
    prior = np.zeros_like(nuisance_information)
    prior[2, 2] = 1 / 0.005 ** 2
    prior[3, 3] = 1 / 0.003 ** 2
    calibrated_errors, calibrated_corr, calibrated_covariance = _precision(
        nuisance_information, args.shots, prior_information=prior)
    precision_rows = [
        {"label": "spatial marginal only", "standard_errors": marginal_errors,
         "correlation": marginal_corr},
        {"label": "joint p(y,t), timing fixed", "standard_errors": joint_errors,
         "correlation": joint_corr},
        {"label": "joint p(y,t), timing fitted", "standard_errors": nuisance_errors,
         "correlation": nuisance_corr},
        {"label": "joint p(y,t), timing calibrated",
         "standard_errors": calibrated_errors, "correlation": calibrated_corr},
    ]

    scale_rows = []
    for scale in (0.0, 0.1, 0.25, 0.5, 1.0):
        candidate = replace(timing, delay_scale=scale)
        information = _combined_fisher(
            experiments, response, candidate, horizon,
            ("sigma_t", "lambda_strength"))
        errors, correlation, _ = _precision(information, args.shots)
        scale_rows.append([scale, errors[0], errors[1], correlation])
    scale_scan = np.asarray(scale_rows)

    jitter_rows = []
    for jitter in (0.025, 0.05, 0.1, 0.2, 0.4):
        candidate = replace(response, timing_jitter=jitter)
        information = _combined_fisher(
            experiments, candidate, timing, horizon,
            ("sigma_t", "lambda_strength"))
        errors, correlation, _ = _precision(information, args.shots)
        jitter_rows.append([jitter, errors[0], errors[1], correlation])
    jitter_scan = np.asarray(jitter_rows)

    # Including both timing widths exposes their exact quadrature-sum
    # degeneracy and documents why a separate timing calibration is required.
    width_information = _combined_fisher(
        experiments, response, timing, horizon,
        ("sigma_t", "lambda_strength", "source_pulse_sigma",
         "detector_jitter", "clock_offset"))
    width_eigenvalues = np.linalg.eigvalsh(width_information)
    recovery = _recovery(
        experiments, response, timing, horizon, args.shots,
        args.repetitions, args.seed)
    example_joint = joint_spatial_arrival_probabilities(
        first, TRUE_SIGMA, TRUE_LAMBDA, response, timing,
        delay_horizon=horizon)
    spatial_marginal, temporal_marginal = marginalize_joint_arrival(
        example_joint, first.y_bins, timing.time_bins)

    payload = {
        "status": "synthetic_phenomenological_instrument_not_trf_prediction",
        "settings": vars(args) | {
            "json_output": str(args.json_output),
            "plot_output": str(args.plot_output),
            "true_sigma_t": TRUE_SIGMA,
            "true_lambda_strength": TRUE_LAMBDA,
            "delay_horizon": horizon,
            "shot_allocation": [args.shots // 2,
                                args.shots - args.shots // 2],
        },
        "detector_response": asdict(response),
        "arrival_time_response": asdict(timing),
        "observation_definition": (
            "The stored timestamp is the residual after subtracting the standard "
            "arrival time. Its conditional mean is delay_scale times the latent "
            "delay plus clock_offset. Source width and detector jitter are Gaussian; "
            "finite-gate loss, dark records, and no-click are retained."),
        "scope": (
            "The latent-delay-to-arrival-time map and delay_scale are measurement "
            "hypotheses. They are not derived from TRF-IT v0.2."),
        "precision_comparison": precision_rows,
        "calibrated_full_covariance": calibrated_covariance,
        "delay_scale_scan": scale_scan,
        "detector_jitter_scan": jitter_scan,
        "separate_timing_width_fisher_eigenvalues": width_eigenvalues,
        "example_first_setting": {
            "joint_probabilities": example_joint,
            "spatial_marginal": spatial_marginal,
            "temporal_marginal": temporal_marginal,
            "click_probability": float(1 - example_joint[-1]),
        },
        "recovery": recovery,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, default=_json_value, indent=2) + "\n")
    _plot(first, response, timing, example_joint, scale_scan, jitter_scan,
          precision_rows, recovery, args.plot_output)

    for row in precision_rows:
        errors = row["standard_errors"]
        print(f"{row['label']:<36} SE=({errors[0]:.6g}, {errors[1]:.6g}) "
              f"corr={row['correlation']:+.4f}")
    print("delay scale scan [scale, SE sigma, SE lambda, corr]:")
    print(scale_scan)
    print("jitter scan [jitter, SE sigma, SE lambda, corr]:")
    print(jitter_scan)
    print(f"timing-width Fisher eigenvalues={width_eigenvalues}")
    print(f"recovery={recovery['calibrated_timing_summary']}")
    print(f"results={args.json_output}; plot={args.plot_output}")


if __name__ == "__main__":
    main()
