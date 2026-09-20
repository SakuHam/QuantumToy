"""Experimental design and multinomial recovery for the minimal TRF candidate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from analysis.trf_candidate import design_distributions


PARAMETER_NAMES = (
    "lambda_strength",
    "g_scale",
    "probe_strength",
    "ordinary_dephasing_rate",
)
DESIGN_PARAMETER_NAMES = PARAMETER_NAMES + ("tau_scale",)


def _validate_observations(observations):
    observations = tuple(tuple(map(float, row)) for row in observations)
    if not observations or any(len(row) != 3 or row[0] <= 0 or row[1] < 0
                               or row[2] <= 0 for row in observations):
        raise ValueError("observations must be (g>0, time>=0, tau_stab>0)")
    return observations


def _finite_difference_derivatives(experiment, observations, parameters, alpha):
    parameters = np.asarray(parameters, dtype=float)
    steps = np.maximum(np.abs(parameters) * 1e-4, [1e-6, 1e-6, 1e-6, 1e-6])
    derivatives = []
    for index, step in enumerate(steps):
        high, low = parameters.copy(), parameters.copy()
        high[index] += step
        low[index] -= step
        upper = design_distributions(experiment, observations, high, alpha)
        lower = design_distributions(experiment, observations, low, alpha)
        derivatives.append([(a - b) / (2 * step)
                            for a, b in zip(upper, lower)])
    return derivatives


def per_shot_fisher_matrices(experiment, observations, *, lambda_strength,
                             alpha=1.0, g_scale=1.0,
                             ordinary_dephasing_rate=0.02):
    """Return one expected multinomial Fisher matrix per declared setting."""
    observations = _validate_observations(observations)
    parameters = np.array([
        lambda_strength,
        g_scale,
        experiment.probe_strength,
        ordinary_dephasing_rate,
    ], dtype=float)
    if parameters[0] <= 0 or parameters[3] <= 0:
        raise ValueError("Design linearization requires positive interior strengths")
    probabilities = design_distributions(experiment, observations, parameters, alpha)
    derivatives = _finite_difference_derivatives(
        experiment, observations, parameters, alpha)
    matrices = []
    for setting, probability in enumerate(probabilities):
        supported = probability > 1e-15
        matrix = np.zeros((len(parameters), len(parameters)))
        for i in range(len(parameters)):
            for j in range(len(parameters)):
                matrix[i, j] = np.sum(
                    derivatives[i][setting][supported]
                    * derivatives[j][setting][supported]
                    / probability[supported])
        matrices.append((matrix + matrix.T) / 2)
    return np.asarray(matrices)


def calibration_prior_information(*, g_scale_sigma, probe_strength_sigma,
                                  ordinary_dephasing_sigma):
    """Gaussian calibration information; lambda intentionally has no prior."""
    sigmas = [None, g_scale_sigma, probe_strength_sigma,
              ordinary_dephasing_sigma]
    information = np.zeros((4, 4))
    for index, sigma in enumerate(sigmas):
        if sigma is None:
            continue
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Calibration standard errors must be finite and positive")
        information[index, index] = 1 / float(sigma) ** 2
    return information


def covariance_from_information(information):
    information = np.asarray(information, dtype=float)
    if (information.ndim != 2 or information.shape[0] != information.shape[1]
            or information.shape[0] == 0 or not np.all(np.isfinite(information))):
        raise ValueError("Expected a finite nonempty square information matrix")
    eigenvalues = np.linalg.eigvalsh((information + information.T) / 2)
    if eigenvalues[0] <= max(eigenvalues[-1] * 1e-12, 1e-14):
        return None
    return np.linalg.inv(information)


def per_shot_design_fisher_matrices(experiment, observations, *,
                                    lambda_strength, alpha=1.0,
                                    g_scale=1.0,
                                    ordinary_dephasing_rate=0.02,
                                    tau_scale=1.0):
    """Per-setting Fisher matrices including the clock-scale nuisance."""
    observations = _validate_observations(observations)
    if not np.isfinite(tau_scale) or tau_scale <= 0:
        raise ValueError("tau_scale must be finite and positive")
    scaled = tuple((g, time, tau * tau_scale) for g, time, tau in observations)
    physical = np.array([
        lambda_strength, g_scale, experiment.probe_strength,
        ordinary_dephasing_rate], dtype=float)
    probabilities = design_distributions(experiment, scaled, physical, alpha)
    derivatives = _finite_difference_derivatives(
        experiment, scaled, physical, alpha)
    step = max(abs(tau_scale) * 1e-4, 1e-6)
    upper_observations = tuple((g, time, tau * (tau_scale + step))
                               for g, time, tau in observations)
    lower_observations = tuple((g, time, tau * (tau_scale - step))
                               for g, time, tau in observations)
    upper = design_distributions(experiment, upper_observations, physical, alpha)
    lower = design_distributions(experiment, lower_observations, physical, alpha)
    derivatives.append([(a - b) / (2 * step) for a, b in zip(upper, lower)])

    matrices = []
    for setting, probability in enumerate(probabilities):
        supported = probability > 1e-15
        matrix = np.zeros((len(DESIGN_PARAMETER_NAMES), len(DESIGN_PARAMETER_NAMES)))
        for i in range(len(DESIGN_PARAMETER_NAMES)):
            for j in range(len(DESIGN_PARAMETER_NAMES)):
                matrix[i, j] = np.sum(
                    derivatives[i][setting][supported]
                    * derivatives[j][setting][supported]
                    / probability[supported])
        matrices.append((matrix + matrix.T) / 2)
    return np.asarray(matrices)


def design_prior_information(*, g_scale_sigma, probe_strength_sigma,
                             ordinary_dephasing_sigma, tau_scale_sigma=None):
    """Calibration information for the five-parameter robust design."""
    physical = calibration_prior_information(
        g_scale_sigma=g_scale_sigma,
        probe_strength_sigma=probe_strength_sigma,
        ordinary_dephasing_sigma=ordinary_dephasing_sigma,
    )
    information = np.zeros((5, 5))
    information[:4, :4] = physical
    if tau_scale_sigma is not None:
        if not np.isfinite(tau_scale_sigma) or tau_scale_sigma <= 0:
            raise ValueError("tau_scale_sigma must be finite and positive")
        information[4, 4] = 1 / tau_scale_sigma ** 2
    return information


@dataclass(frozen=True)
class DesignAllocation:
    g: float
    time: float
    tau_stab: float
    shots: int


@dataclass(frozen=True)
class DesignResult:
    allocations: tuple[DesignAllocation, ...]
    information: np.ndarray
    covariance: np.ndarray
    lambda_standard_error: float
    worst_case_lambda_standard_error: float
    tau_scale_standard_errors: dict[float, float]
    lambda_noise_correlation: float
    lambda_clock_correlation: float


def optimize_lambda_design(experiment, candidate_observations, *, total_shots,
                           shot_block=1_000, lambda_strength=0.2, alpha=1.0,
                           ordinary_dephasing_rate=0.02,
                           g_scale_sigma=0.01, probe_strength_sigma=0.01,
                           ordinary_dephasing_sigma=0.005,
                           tau_scales=(1.0,), tau_scale_sigma=None):
    """Greedy minimax allocation for the profiled variance of lambda.

    The same shot allocation is evaluated for every declared tau_scale. Each
    block minimizes the worst local lambda variance, preventing a design that
    is optimal only for one arbitrary stabilization convention.
    """
    observations = _validate_observations(candidate_observations)
    if (not isinstance(total_shots, (int, np.integer)) or total_shots <= 0
            or not isinstance(shot_block, (int, np.integer)) or shot_block <= 0
            or total_shots % shot_block):
        raise ValueError("total_shots must be a positive multiple of shot_block")
    tau_scales = tuple(float(scale) for scale in tau_scales)
    if not tau_scales or any(not np.isfinite(scale) or scale <= 0
                             for scale in tau_scales):
        raise ValueError("tau_scales must be finite and positive")
    matrices_by_scale = []
    for scale in tau_scales:
        matrices_by_scale.append(per_shot_design_fisher_matrices(
            experiment, observations,
            lambda_strength=lambda_strength,
            alpha=alpha,
            ordinary_dephasing_rate=ordinary_dephasing_rate,
            tau_scale=scale,
        ))
    prior = design_prior_information(
        g_scale_sigma=g_scale_sigma,
        probe_strength_sigma=probe_strength_sigma,
        ordinary_dephasing_sigma=ordinary_dephasing_sigma,
        tau_scale_sigma=tau_scale_sigma,
    )
    information_by_scale = [prior.copy() for _ in tau_scales]
    counts = np.zeros(len(observations), dtype=int)
    for _ in range(total_shots // shot_block):
        best = None
        fallback = None
        for index in range(len(observations)):
            proposed = [
                information + shot_block * matrices[index]
                for information, matrices in zip(
                    information_by_scale, matrices_by_scale)
            ]
            covariances = [covariance_from_information(value) for value in proposed]
            score = (np.inf if any(value is None for value in covariances)
                     else max(value[0, 0] for value in covariances))
            candidate = (score, index, proposed)
            if best is None or candidate[0] < best[0] - 1e-18:
                best = candidate
            if not np.isfinite(score):
                ranks, log_determinants = [], []
                for value in proposed:
                    eigenvalues = np.linalg.eigvalsh((value + value.T) / 2)
                    tolerance = max(eigenvalues[-1] * 1e-12, 1e-14)
                    positive = eigenvalues[eigenvalues > tolerance]
                    ranks.append(len(positive))
                    log_determinants.append(float(np.sum(np.log(positive))))
                fallback_score = (min(ranks), min(log_determinants))
                if fallback is None or fallback_score > fallback[0]:
                    fallback = (fallback_score, index, proposed)
        if not np.isfinite(best[0]):
            if fallback is None:
                raise ValueError("Candidate settings contain no parameter information")
            _, index, information_by_scale = fallback
        else:
            _, index, information_by_scale = best
        counts[index] += shot_block
    covariances = [covariance_from_information(value)
                   for value in information_by_scale]
    if any(value is None for value in covariances):
        raise ValueError("Calibration priors and candidate settings do not identify lambda")
    central_index = int(np.argmin(np.abs(np.asarray(tau_scales) - 1)))
    information = information_by_scale[central_index]
    covariance = covariances[central_index]
    scale_errors = {scale: float(np.sqrt(value[0, 0]))
                    for scale, value in zip(tau_scales, covariances)}
    allocations = tuple(
        DesignAllocation(*observations[index], int(shots))
        for index, shots in enumerate(counts) if shots
    )
    denominator = np.sqrt(covariance[0, 0] * covariance[3, 3])
    correlation = float(covariance[0, 3] / denominator)
    clock_denominator = np.sqrt(covariance[0, 0] * covariance[4, 4])
    clock_correlation = float(covariance[0, 4] / clock_denominator)
    return DesignResult(
        allocations,
        information,
        covariance,
        float(np.sqrt(covariance[0, 0])),
        max(scale_errors.values()),
        scale_errors,
        correlation,
        clock_correlation,
    )


def distributions_for_allocations(experiment, allocations, parameters, *,
                                  alpha=1.0, tau_scale=1.0):
    if not np.isfinite(tau_scale) or tau_scale <= 0:
        raise ValueError("tau_scale must be finite and positive")
    observations = [(row.g, row.time, row.tau_stab * tau_scale)
                    for row in allocations]
    return design_distributions(experiment, observations, parameters, alpha)


def simulate_multinomial_counts(experiment, allocations, parameters, *,
                                alpha=1.0, tau_scale=1.0, rng=None):
    """Draw complete-record counts independently at every design setting."""
    rng = np.random.default_rng() if rng is None else rng
    probabilities = distributions_for_allocations(
        experiment, allocations, parameters, alpha=alpha, tau_scale=tau_scale)
    return tuple(
        rng.multinomial(row.shots, probability.ravel()).reshape(probability.shape)
        for row, probability in zip(allocations, probabilities)
    )


def _prior_penalty(parameters, calibration_means, calibration_sigmas):
    penalty = 0.0
    for index in range(1, 4):
        sigma = calibration_sigmas[index]
        if sigma is not None:
            penalty += 0.5 * ((parameters[index] - calibration_means[index]) / sigma) ** 2
    return float(penalty)


def _negative_log_posterior(parameters, experiment, allocations, counts, alpha,
                            tau_scale, calibration_means, calibration_sigmas):
    try:
        probabilities = distributions_for_allocations(
            experiment, allocations, parameters, alpha=alpha, tau_scale=tau_scale)
    except ValueError:
        return np.inf
    value = _prior_penalty(parameters, calibration_means, calibration_sigmas)
    for observed, probability in zip(counts, probabilities):
        supported = observed > 0
        if np.any(probability[supported] <= 0):
            return np.inf
        value -= float(np.sum(observed[supported] * np.log(probability[supported])))
    return value


@dataclass(frozen=True)
class FitResult:
    parameters: np.ndarray
    negative_log_posterior: float
    null_negative_log_posterior: float
    likelihood_ratio: float
    detected: bool
    tau_scale: float
    success: bool


def fit_candidate_counts(experiment, allocations, counts, *, alpha=1.0,
                         tau_scales=(1.0,), ordinary_dephasing_mean=0.02,
                         g_scale_sigma=0.01, probe_strength_sigma=0.01,
                         ordinary_dephasing_sigma=0.005,
                         detection_threshold=2.7055):
    """Profile calibrated nuisances and a discrete clock-convention scale.

    detection_threshold is the common one-sided 95% likelihood-ratio threshold
    for a nonnegative signal strength. Coverage is checked by Monte Carlo.
    """
    allocations = tuple(allocations)
    counts = tuple(np.asarray(value, dtype=int) for value in counts)
    if len(allocations) == 0 or len(counts) != len(allocations):
        raise ValueError("Counts must match a nonempty allocation")
    calibration_means = np.array([
        0.0, 1.0, experiment.probe_strength, ordinary_dephasing_mean], dtype=float)
    calibration_sigmas = (None, g_scale_sigma, probe_strength_sigma,
                          ordinary_dephasing_sigma)
    bounds = [(0, 5), (0.5, 1.5), (1e-5, 1 - 1e-5), (0, 1)]
    full_fits, null_fits = [], []
    for tau_scale in tau_scales:
        tau_scale = float(tau_scale)
        initial = calibration_means.copy()
        initial[0] = 0.1
        full = minimize(
            _negative_log_posterior,
            initial,
            args=(experiment, allocations, counts, alpha, tau_scale,
                  calibration_means, calibration_sigmas),
            method="L-BFGS-B",
            bounds=bounds,
        )

        def null_objective(nuisances):
            parameters = np.r_[0.0, nuisances]
            return _negative_log_posterior(
                parameters, experiment, allocations, counts, alpha, tau_scale,
                calibration_means, calibration_sigmas)

        null = minimize(
            null_objective,
            calibration_means[1:],
            method="L-BFGS-B",
            bounds=bounds[1:],
        )
        full_fits.append((float(full.fun), tau_scale, full))
        null_fits.append((float(null.fun), tau_scale, null))
    full_value, tau_scale, full = min(full_fits, key=lambda row: row[0])
    null_value, _, null = min(null_fits, key=lambda row: row[0])
    statistic = max(0.0, 2 * (null_value - full_value))
    return FitResult(
        np.asarray(full.x), full_value, null_value, statistic,
        statistic >= detection_threshold, tau_scale,
        bool(full.success and null.success),
    )


@dataclass(frozen=True)
class RecoverySummary:
    true_lambda: float
    repetitions: int
    successful_fits: int
    detection_rate: float
    mean_lambda: float
    bias: float
    rmse: float
    quantiles: np.ndarray
    selected_tau_scale_counts: dict[float, int]


def monte_carlo_recovery(experiment, allocations, *, true_lambda_values,
                         repetitions=100, seed=0, alpha=1.0,
                         true_g_scale=1.0, true_ordinary_dephasing_rate=0.02,
                         tau_scales=(1.0,), g_scale_sigma=0.01,
                         probe_strength_sigma=0.01,
                         ordinary_dephasing_sigma=0.005):
    """Measure false positives and recovery using full multinomial records."""
    if not isinstance(repetitions, (int, np.integer)) or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    rng = np.random.default_rng(seed)
    summaries = []
    for true_lambda in true_lambda_values:
        estimates, detections, selected, successes = [], [], [], 0
        parameters = np.array([
            float(true_lambda), float(true_g_scale), experiment.probe_strength,
            float(true_ordinary_dephasing_rate)])
        for _ in range(repetitions):
            counts = simulate_multinomial_counts(
                experiment, allocations, parameters, alpha=alpha, rng=rng)
            fit = fit_candidate_counts(
                experiment, allocations, counts, alpha=alpha,
                tau_scales=tau_scales,
                ordinary_dephasing_mean=true_ordinary_dephasing_rate,
                g_scale_sigma=g_scale_sigma,
                probe_strength_sigma=probe_strength_sigma,
                ordinary_dephasing_sigma=ordinary_dephasing_sigma,
            )
            successes += int(fit.success)
            estimates.append(fit.parameters[0])
            detections.append(fit.detected)
            selected.append(fit.tau_scale)
        estimates = np.asarray(estimates)
        scale_counts = {float(scale): selected.count(scale)
                        for scale in sorted(set(selected))}
        summaries.append(RecoverySummary(
            float(true_lambda), repetitions, successes,
            float(np.mean(detections)), float(np.mean(estimates)),
            float(np.mean(estimates) - true_lambda),
            float(np.sqrt(np.mean((estimates - true_lambda) ** 2))),
            np.quantile(estimates, [0.05, 0.5, 0.95]),
            scale_counts,
        ))
    return summaries
