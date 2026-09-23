"""Realistic detector response, robust design, and synthetic inference.

The ideal complete spatial POVM is retained as the physical input law.  A
column-stochastic classical channel then models finite efficiency, dark
records, and y-bin blur.  Symmetric readout-time jitter is averaged before the
classical channel.  Calibration and double-slit test records share the same
detector nuisance parameters but have separate likelihood contributions.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import multiprocessing as mp

import numpy as np
from scipy.optimize import minimize

from analysis.spatial_effect_measurement import (
    SpatialEffectExperiment,
    run_spatial_effect_measurement,
)


_REPLICATE_CONTEXT = None
_RESPONSE_BOUNDS = ((0.2, 1.0), (0.0, 0.2), (0.0, 2.5))


def _process_pool_context():
    """Use direct fork where available; Python 3.14 forkserver needs IPC sockets."""
    return mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None


@dataclass(frozen=True)
class OptimizerDiagnostic:
    success: bool
    status: int
    message: str
    iterations: int
    function_evaluations: int
    negative_log_likelihood: float


@dataclass(frozen=True)
class ProfileDiagnostics:
    """Optimizer state and boundary flags, without changing fit selection."""

    best: OptimizerDiagnostic
    null: OptimizerDiagnostic
    null_response: DetectorResponse
    calibration: tuple[OptimizerDiagnostic, ...]
    profile_fits: int
    failed_profile_fits: int
    nonfinite_profile_fits: int
    failed_profile_status_counts: dict[str, int]
    best_boundaries: dict[str, str]
    null_response_boundaries: dict[str, str]
    sigma_identified_at_best: bool


@dataclass(frozen=True)
class DetectorResponse:
    """Calibrated classical response following the ideal spatial POVM."""

    efficiency: float = 0.9
    dark_probability: float = 0.01
    blur_sigma_bins: float = 0.5
    timing_jitter: float = 0.025

    def __post_init__(self):
        values = (
            self.efficiency, self.dark_probability,
            self.blur_sigma_bins, self.timing_jitter,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Detector response parameters must be finite")
        if not 0 <= self.efficiency <= 1:
            raise ValueError("efficiency must lie in [0, 1]")
        if not 0 <= self.dark_probability < 1:
            raise ValueError("dark_probability must lie in [0, 1)")
        if self.blur_sigma_bins < 0 or self.timing_jitter < 0:
            raise ValueError("blur and timing jitter must be nonnegative")


@dataclass(frozen=True)
class RobustDetectorDesign:
    """Minimax detector design over a declared physical parameter grid."""

    parameter_sigmas: np.ndarray
    parameter_lambdas: np.ndarray
    candidate_detector_x: np.ndarray
    candidate_detector_width: np.ndarray
    candidate_reference_time: np.ndarray
    worst_log_determinant_gain: np.ndarray
    mean_log_determinant_gain: np.ndarray
    worst_sigma_error_ratio: np.ndarray
    worst_lambda_error_ratio: np.ndarray
    best_index: int
    best_detector_x: float
    best_detector_width: float
    best_reference_time: float
    first_setting_fraction: float


@dataclass(frozen=True)
class DetectorCalibrationFit:
    response: DetectorResponse
    negative_log_likelihood: float
    response_standard_errors: np.ndarray
    success: bool


@dataclass(frozen=True)
class SpatialDetectorInferenceLibrary:
    """Precomputed ideal laws used repeatedly by profile fits."""

    calibration_experiments: tuple[SpatialEffectExperiment, ...]
    test_experiments: tuple[SpatialEffectExperiment, ...]
    calibration_sigma_t: float
    calibration_lambda_strength: float
    candidate_sigmas: np.ndarray
    candidate_lambdas: np.ndarray
    candidate_timing_jitters: np.ndarray
    calibration_ideal_probabilities: np.ndarray
    test_ideal_probabilities: np.ndarray


@dataclass(frozen=True)
class SpatialDetectorCountFit:
    best_sigma_t: float
    best_lambda_strength: float
    response: DetectorResponse
    negative_log_likelihood: float
    null_negative_log_likelihood: float
    likelihood_ratio: float
    detected: bool
    profile_negative_log_likelihood: np.ndarray
    joint_confidence_region: np.ndarray
    success: bool
    diagnostics: ProfileDiagnostics


@dataclass(frozen=True)
class SpatialDetectorReplicate:
    """Compact, seed-addressable record of a complete profile fit."""

    seed: int
    best_sigma_t: float
    best_lambda_strength: float
    response: DetectorResponse
    negative_log_likelihood: float
    null_negative_log_likelihood: float
    likelihood_ratio: float
    detected: bool
    success: bool
    diagnostics: ProfileDiagnostics


@dataclass(frozen=True)
class SpatialDetectorRecovery:
    true_sigma_t: float
    true_lambda_strength: float
    repetitions: int
    successful_fits: int
    detection_rate: float
    mean_sigma_t: float
    mean_lambda_strength: float
    sigma_bias: float
    lambda_bias: float
    sigma_rmse: float
    lambda_rmse: float
    joint_coverage: float
    sigma_quantiles: np.ndarray
    lambda_quantiles: np.ndarray
    likelihood_ratio_quantiles: np.ndarray
    replicate_records: tuple[SpatialDetectorReplicate, ...]


@dataclass(frozen=True)
class SpatialDetectionThreshold:
    """Parametric-bootstrap threshold for the nonregular lambda=0 null."""

    confidence_level: float
    repetitions: int
    threshold: float
    likelihood_ratios: np.ndarray
    replicate_records: tuple[SpatialDetectorReplicate, ...]


def detector_blur_matrix(y_bins, blur_sigma_bins):
    """Return a column-stochastic finite-resolution y-bin response."""
    if not isinstance(y_bins, (int, np.integer)) or y_bins < 1:
        raise ValueError("y_bins must be a positive integer")
    blur_sigma_bins = float(blur_sigma_bins)
    if not np.isfinite(blur_sigma_bins) or blur_sigma_bins < 0:
        raise ValueError("blur_sigma_bins must be finite and nonnegative")
    if blur_sigma_bins == 0:
        return np.eye(y_bins)
    indices = np.arange(y_bins, dtype=float)
    matrix = np.exp(
        -0.5 * ((indices[:, None] - indices[None, :])
                / blur_sigma_bins) ** 2)
    return matrix / np.sum(matrix, axis=0, keepdims=True)


def detector_response_matrix(y_bins, response):
    """Return the complete observed-outcome channel for one detector.

    An ideal click is registered with probability ``efficiency`` and blurred
    in y.  If it is missed, or if the ideal result is no-click, a uniform dark
    click occurs with probability ``dark_probability``.  Every column sums to
    one, including the observed no-click row.
    """
    if not isinstance(response, DetectorResponse):
        raise TypeError("response must be a DetectorResponse")
    blur = detector_blur_matrix(y_bins, response.blur_sigma_bins)
    background = np.full(y_bins, 1.0 / y_bins)
    matrix = np.zeros((y_bins + 1, y_bins + 1), dtype=float)
    for ideal_bin in range(y_bins):
        matrix[:-1, ideal_bin] = (
            response.efficiency * blur[:, ideal_bin]
            + (1.0 - response.efficiency)
            * response.dark_probability * background
        )
        matrix[-1, ideal_bin] = (
            (1.0 - response.efficiency)
            * (1.0 - response.dark_probability)
        )
    matrix[:-1, -1] = response.dark_probability * background
    matrix[-1, -1] = 1.0 - response.dark_probability
    return matrix


def apply_detector_response(ideal_probabilities, response):
    """Apply the classical detector channel to one complete ideal law."""
    ideal = np.asarray(ideal_probabilities, dtype=float)
    if (ideal.ndim != 1 or ideal.size < 2 or np.any(~np.isfinite(ideal))
            or np.min(ideal) < -1e-12
            or not np.isclose(np.sum(ideal), 1.0, atol=1e-12)):
        raise ValueError("ideal_probabilities must be one complete outcome law")
    observed = detector_response_matrix(ideal.size - 1, response) @ ideal
    if np.min(observed) < -1e-12 or not np.isclose(
            np.sum(observed), 1.0, atol=1e-12):
        raise RuntimeError("Detector response produced an invalid outcome law")
    return np.maximum(observed, 0.0)


def _jitter_nodes(timing_jitter):
    if timing_jitter == 0:
        return np.array([0.0]), np.array([1.0])
    # Three-point Gauss-Hermite-equivalent rule with the requested variance.
    spread = np.sqrt(3.0) * timing_jitter
    return np.array([-spread, 0.0, spread]), np.array([1 / 6, 2 / 3, 1 / 6])


def ideal_probabilities_with_timing_jitter(
        experiment, sigma_t, lambda_strength, timing_jitter):
    """Average ideal complete laws over symmetric readout-time jitter."""
    timing_jitter = float(timing_jitter)
    if not np.isfinite(timing_jitter) or timing_jitter < 0:
        raise ValueError("timing_jitter must be finite and nonnegative")
    offsets, weights = _jitter_nodes(timing_jitter)
    if experiment.reference_time + offsets[0] < 0:
        raise ValueError("timing jitter extends before time zero")
    laws = [run_spatial_effect_measurement(
        replace(experiment, reference_time=experiment.reference_time + offset),
        sigma_t, lambda_strength=lambda_strength).probabilities
        for offset in offsets]
    ideal = np.tensordot(weights, np.asarray(laws), axes=(0, 0))
    if not np.isclose(np.sum(ideal), 1.0, atol=1e-12):
        raise RuntimeError("Timing mixture produced an incomplete law")
    return ideal


def realistic_spatial_probabilities(
        experiment, sigma_t, lambda_strength, response):
    """Return the normalized law including timing and detector response."""
    ideal = ideal_probabilities_with_timing_jitter(
        experiment, sigma_t, lambda_strength, response.timing_jitter)
    return apply_detector_response(ideal, response)


def realistic_spatial_fisher_information(
        experiment, sigma_t, lambda_strength, response, *, relative_step=1e-3):
    """Per-shot Fisher information for ``(sigma_t, lambda)`` after response."""
    sigma_t = float(sigma_t)
    lambda_strength = float(lambda_strength)
    if sigma_t <= 0 or lambda_strength < 0:
        raise ValueError("sigma_t must be positive and lambda nonnegative")
    sigma_step = min(relative_step * sigma_t, 0.25 * sigma_t)
    lambda_step = relative_step * max(1.0, lambda_strength)
    sigma_minus = realistic_spatial_probabilities(
        experiment, sigma_t - sigma_step, lambda_strength, response)
    sigma_plus = realistic_spatial_probabilities(
        experiment, sigma_t + sigma_step, lambda_strength, response)
    derivative_sigma = (sigma_plus - sigma_minus) / (2 * sigma_step)
    if lambda_strength > lambda_step:
        lambda_minus = realistic_spatial_probabilities(
            experiment, sigma_t, lambda_strength - lambda_step, response)
        lambda_plus = realistic_spatial_probabilities(
            experiment, sigma_t, lambda_strength + lambda_step, response)
        derivative_lambda = (lambda_plus - lambda_minus) / (2 * lambda_step)
    else:
        central = realistic_spatial_probabilities(
            experiment, sigma_t, lambda_strength, response)
        lambda_plus = realistic_spatial_probabilities(
            experiment, sigma_t, lambda_strength + lambda_step, response)
        derivative_lambda = (lambda_plus - central) / lambda_step
    probabilities = realistic_spatial_probabilities(
        experiment, sigma_t, lambda_strength, response)
    derivatives = np.stack([derivative_sigma, derivative_lambda], axis=1)
    fisher = ((derivatives.T / np.maximum(probabilities, 1e-300))
              @ derivatives)
    return 0.5 * (fisher + fisher.T)


def _covariance(information):
    eigenvalues = np.linalg.eigvalsh(information)
    threshold = max(1e-14, 1e-12 * eigenvalues[-1])
    if eigenvalues[0] <= threshold:
        return None
    return np.linalg.inv(information)


def select_robust_double_slit_detector_setting(
        experiment, response, parameter_sigmas, parameter_lambdas,
        candidate_detector_x, candidate_detector_width, candidate_time_offset,
        *, first_setting_fraction=0.5):
    """Choose the setting maximizing the worst log-determinant gain.

    Every Cartesian point in the declared ``sigma`` and ``lambda`` region is
    evaluated.  Gains are relative to spending the full shot budget on the
    original detector, while the candidate comparison uses the declared fixed
    split of that same budget.
    """
    if experiment.potential_mode != "double_slit":
        raise ValueError("robust design requires double_slit mode")
    sigmas = np.asarray(parameter_sigmas, dtype=float)
    lambdas = np.asarray(parameter_lambdas, dtype=float)
    positions = np.asarray(candidate_detector_x, dtype=float)
    widths = np.asarray(candidate_detector_width, dtype=float)
    offsets = np.asarray(candidate_time_offset, dtype=float)
    if sigmas.ndim != 1 or sigmas.size == 0 or np.any(sigmas <= 0):
        raise ValueError("parameter_sigmas must contain positive values")
    if lambdas.ndim != 1 or lambdas.size == 0 or np.any(lambdas <= 0):
        raise ValueError("robust design requires positive parameter_lambdas")
    if positions.ndim != 1 or positions.size == 0:
        raise ValueError("candidate_detector_x must be nonempty")
    if widths.ndim != 1 or widths.size == 0 or np.any(widths <= 0):
        raise ValueError("candidate_detector_width must be positive")
    if offsets.ndim != 1 or offsets.size == 0:
        raise ValueError("candidate_time_offset must be nonempty")
    if not all(np.all(np.isfinite(values))
               for values in (sigmas, lambdas, positions, widths, offsets)):
        raise ValueError("design grids must be finite")
    fraction = float(first_setting_fraction)
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("first_setting_fraction must lie in (0, 1)")
    velocity = experiment.hbar * experiment.packet_kx / experiment.mass
    if velocity <= 0:
        raise ValueError("robust design requires positive packet velocity")

    points = [(sigma_t, lambda_strength)
              for sigma_t in sigmas for lambda_strength in lambdas]
    first_fishers = [realistic_spatial_fisher_information(
        experiment, sigma_t, lambda_strength, response)
        for sigma_t, lambda_strength in points]
    rows = []
    worst_gains, mean_gains = [], []
    worst_sigma_ratios, worst_lambda_ratios = [], []
    for detector_x in positions:
        arrival = (detector_x - experiment.packet_x) / velocity
        for width in widths:
            for offset in offsets:
                reference_time = arrival + offset
                if reference_time <= np.sqrt(3) * response.timing_jitter:
                    continue
                second = replace(
                    experiment, detector_x=float(detector_x),
                    detector_width=float(width),
                    reference_time=float(reference_time))
                gains, error_ratios = [], []
                for point, first_fisher in zip(points, first_fishers):
                    second_fisher = realistic_spatial_fisher_information(
                        second, *point, response)
                    combined = fraction * first_fisher + (1 - fraction) * second_fisher
                    first_covariance = _covariance(first_fisher)
                    combined_covariance = _covariance(combined)
                    first_sign, first_logdet = np.linalg.slogdet(first_fisher)
                    combined_sign, combined_logdet = np.linalg.slogdet(combined)
                    if (first_sign <= 0 or combined_sign <= 0
                            or first_covariance is None
                            or combined_covariance is None):
                        gains.append(-np.inf)
                        error_ratios.append([np.inf, np.inf])
                    else:
                        gains.append(float(combined_logdet - first_logdet))
                        error_ratios.append(np.sqrt(
                            np.diag(combined_covariance)
                            / np.diag(first_covariance)))
                rows.append((detector_x, width, reference_time))
                worst_gains.append(float(np.min(gains)))
                mean_gains.append(float(np.mean(gains)))
                error_ratios = np.asarray(error_ratios)
                worst_sigma_ratios.append(float(np.max(error_ratios[:, 0])))
                worst_lambda_ratios.append(float(np.max(error_ratios[:, 1])))
    if not rows or not np.any(np.isfinite(worst_gains)):
        raise ValueError("candidate settings contain no robust information")
    rows = np.asarray(rows, dtype=float)
    worst_gains = np.asarray(worst_gains)
    best_index = int(np.argmax(worst_gains))
    best = rows[best_index]
    return RobustDetectorDesign(
        parameter_sigmas=sigmas,
        parameter_lambdas=lambdas,
        candidate_detector_x=rows[:, 0],
        candidate_detector_width=rows[:, 1],
        candidate_reference_time=rows[:, 2],
        worst_log_determinant_gain=worst_gains,
        mean_log_determinant_gain=np.asarray(mean_gains),
        worst_sigma_error_ratio=np.asarray(worst_sigma_ratios),
        worst_lambda_error_ratio=np.asarray(worst_lambda_ratios),
        best_index=best_index,
        best_detector_x=float(best[0]),
        best_detector_width=float(best[1]),
        best_reference_time=float(best[2]),
        first_setting_fraction=fraction,
    )


def simulate_realistic_counts(
        experiments, sigma_t, lambda_strength, response, shots, *, rng=None):
    """Draw independent complete multinomial records for detector settings."""
    experiments = tuple(experiments)
    if not experiments:
        raise ValueError("experiments must be nonempty")
    if isinstance(shots, (int, np.integer)):
        shot_counts = np.full(len(experiments), int(shots), dtype=int)
    else:
        shot_counts = np.asarray(shots, dtype=int)
    if shot_counts.shape != (len(experiments),) or np.any(shot_counts <= 0):
        raise ValueError("shots must provide a positive count per experiment")
    rng = np.random.default_rng() if rng is None else rng
    return tuple(rng.multinomial(
        int(count), realistic_spatial_probabilities(
            experiment, sigma_t, lambda_strength, response))
        for experiment, count in zip(experiments, shot_counts))


def _response_from_vector(values, timing_jitter):
    return DetectorResponse(
        efficiency=float(values[0]), dark_probability=float(values[1]),
        blur_sigma_bins=float(values[2]), timing_jitter=float(timing_jitter))


def _response_nll(values, ideal_probabilities, counts, timing_jitter):
    try:
        response = _response_from_vector(values, timing_jitter)
    except ValueError:
        return np.inf
    total = 0.0
    for ideal, observed in zip(ideal_probabilities, counts):
        probabilities = apply_detector_response(ideal, response)
        supported = observed > 0
        if np.any(probabilities[supported] <= 0):
            return np.inf
        total -= float(np.sum(observed[supported]
                              * np.log(probabilities[supported])))
    return total


def _fit_response(ideal_probabilities, counts, timing_jitter, initial):
    return minimize(
        _response_nll, np.asarray(initial, dtype=float),
        args=(ideal_probabilities, counts, timing_jitter),
        method="L-BFGS-B", bounds=_RESPONSE_BOUNDS,
    )


def _optimizer_diagnostic(result):
    return OptimizerDiagnostic(
        bool(result.success), int(result.status), str(result.message),
        int(result.nit), int(result.nfev), float(result.fun))


def _boundary(value, lower, upper):
    if lower == upper:
        return "fixed"
    if np.isclose(value, lower, rtol=0, atol=1e-8):
        return "lower"
    if np.isclose(value, upper, rtol=0, atol=1e-8):
        return "upper"
    return "interior"


def _response_boundaries(response, jitters):
    values = (response.efficiency, response.dark_probability, response.blur_sigma_bins)
    flags = {name: _boundary(value, *bounds) for name, value, bounds in zip(
        ("efficiency", "dark_probability", "blur_sigma_bins"), values, _RESPONSE_BOUNDS)}
    flags["timing_jitter"] = _boundary(response.timing_jitter, min(jitters), max(jitters))
    return flags


def _replicate_record(fit, seed):
    return SpatialDetectorReplicate(
        int(seed), fit.best_sigma_t, fit.best_lambda_strength, fit.response,
        fit.negative_log_likelihood, fit.null_negative_log_likelihood,
        fit.likelihood_ratio, fit.detected, fit.success, fit.diagnostics)


def _response_fisher(ideal_probabilities, counts, response):
    center = np.array([
        response.efficiency, response.dark_probability,
        response.blur_sigma_bins])
    steps = np.array([1e-4, 1e-5, 1e-4])
    information = np.zeros((3, 3))
    for ideal, observed in zip(ideal_probabilities, counts):
        probability = apply_detector_response(ideal, response)
        derivatives = []
        for index, step in enumerate(steps):
            lower, upper = center.copy(), center.copy()
            lower[index] = max(lower[index] - step, [0.2, 0.0, 0.0][index])
            upper[index] = min(upper[index] + step, [1.0, 0.2, 2.5][index])
            denominator = upper[index] - lower[index]
            derivatives.append((
                apply_detector_response(
                    ideal, _response_from_vector(upper, response.timing_jitter))
                - apply_detector_response(
                    ideal, _response_from_vector(lower, response.timing_jitter))
            ) / denominator)
        derivatives = np.asarray(derivatives).T
        information += int(np.sum(observed)) * (
            (derivatives.T / np.maximum(probability, 1e-300)) @ derivatives)
    covariance = np.linalg.pinv(information, hermitian=True)
    return np.sqrt(np.maximum(np.diag(covariance), 0.0))


def fit_detector_calibration(
        experiments, counts, known_sigma_t, known_lambda_strength,
        candidate_timing_jitters, *, initial=(0.9, 0.01, 0.5)):
    """Fit detector nuisances from independent known-physics controls."""
    experiments = tuple(experiments)
    counts = tuple(np.asarray(value, dtype=int) for value in counts)
    jitters = np.asarray(candidate_timing_jitters, dtype=float)
    if not experiments or len(counts) != len(experiments):
        raise ValueError("calibration counts must match nonempty experiments")
    if (jitters.ndim != 1 or jitters.size == 0 or np.any(jitters < 0)
            or np.any(~np.isfinite(jitters))):
        raise ValueError("candidate_timing_jitters must be finite and nonnegative")
    fits = []
    for jitter in jitters:
        ideal = np.asarray([ideal_probabilities_with_timing_jitter(
            experiment, known_sigma_t, known_lambda_strength, float(jitter))
            for experiment in experiments])
        result = _fit_response(ideal, counts, float(jitter), initial)
        fits.append((float(result.fun), float(jitter), result, ideal))
    value, jitter, result, ideal = min(fits, key=lambda row: row[0])
    response = _response_from_vector(result.x, jitter)
    errors = _response_fisher(ideal, counts, response)
    return DetectorCalibrationFit(
        response, value, errors, bool(result.success))


def build_spatial_detector_inference_library(
        calibration_experiments, test_experiments, *,
        calibration_sigma_t, calibration_lambda_strength,
        candidate_sigmas, candidate_lambdas, candidate_timing_jitters):
    """Precompute every ideal law needed by repeated profile fits."""
    calibration_experiments = tuple(calibration_experiments)
    test_experiments = tuple(test_experiments)
    sigmas = np.asarray(candidate_sigmas, dtype=float)
    lambdas = np.asarray(candidate_lambdas, dtype=float)
    jitters = np.asarray(candidate_timing_jitters, dtype=float)
    if not calibration_experiments or not test_experiments:
        raise ValueError("calibration and test experiments must be nonempty")
    if sigmas.ndim != 1 or sigmas.size < 2 or np.any(sigmas <= 0):
        raise ValueError("candidate_sigmas must contain positive values")
    if (lambdas.ndim != 1 or lambdas.size < 2 or np.any(lambdas < 0)
            or not np.any(lambdas == 0)):
        raise ValueError("candidate_lambdas must include zero")
    if jitters.ndim != 1 or jitters.size == 0 or np.any(jitters < 0):
        raise ValueError("candidate_timing_jitters must be nonnegative")
    calibration = np.asarray([[
        ideal_probabilities_with_timing_jitter(
            experiment, calibration_sigma_t, calibration_lambda_strength, jitter)
        for experiment in calibration_experiments]
        for jitter in jitters])
    test = np.asarray([[[[
        ideal_probabilities_with_timing_jitter(
            experiment, sigma_t, lambda_strength, jitter)
        for experiment in test_experiments]
        for sigma_t in sigmas]
        for lambda_strength in lambdas]
        for jitter in jitters])
    return SpatialDetectorInferenceLibrary(
        calibration_experiments, test_experiments,
        float(calibration_sigma_t), float(calibration_lambda_strength),
        sigmas, lambdas, jitters, calibration, test)


def fit_spatial_detector_counts(
        library, calibration_counts, test_counts, *,
        initial=(0.9, 0.01, 0.5), detection_threshold=2.7055,
        joint_confidence_threshold=5.991):
    """Profile shared detector nuisances over calibration and test records."""
    calibration_counts = tuple(np.asarray(value, dtype=int)
                               for value in calibration_counts)
    test_counts = tuple(np.asarray(value, dtype=int) for value in test_counts)
    if len(calibration_counts) != len(library.calibration_experiments):
        raise ValueError("calibration_counts do not match the library")
    if len(test_counts) != len(library.test_experiments):
        raise ValueError("test_counts do not match the library")
    shape = (library.candidate_lambdas.size, library.candidate_sigmas.size)
    surface = np.full(shape, np.inf)
    fitted = {}
    successes = {}
    optimizer_diagnostics = {}
    calibration_diagnostics = []
    failed_status_counts = {}
    profile_fits = failed_fits = nonfinite_fits = 0
    for jitter_index, jitter in enumerate(library.candidate_timing_jitters):
        calibration_ideal = library.calibration_ideal_probabilities[jitter_index]
        calibration_fit = _fit_response(
            calibration_ideal, calibration_counts, float(jitter), initial)
        calibration_diagnostics.append(_optimizer_diagnostic(calibration_fit))
        start = calibration_fit.x
        for lambda_index in range(shape[0]):
            for sigma_index in range(shape[1]):
                ideal = np.concatenate([
                    calibration_ideal,
                    library.test_ideal_probabilities[
                        jitter_index, lambda_index, sigma_index],
                ])
                counts = calibration_counts + test_counts
                result = _fit_response(ideal, counts, float(jitter), start)
                profile_fits += 1
                nonfinite_fits += int(not np.isfinite(result.fun))
                if not result.success:
                    failed_fits += 1
                    status = f"{result.status}: {result.message}"
                    failed_status_counts[status] = failed_status_counts.get(status, 0) + 1
                key = (lambda_index, sigma_index)
                if float(result.fun) < surface[key]:
                    surface[key] = float(result.fun)
                    fitted[key] = (result.x.copy(), float(jitter))
                    successes[key] = bool(result.success)
                    optimizer_diagnostics[key] = _optimizer_diagnostic(result)
    if not np.any(np.isfinite(surface)):
        raise RuntimeError("No finite profile fit; cannot construct a likelihood ratio")
    best_index = np.unravel_index(int(np.argmin(surface)), surface.shape)
    best_value = float(surface[best_index])
    null_rows = np.flatnonzero(library.candidate_lambdas == 0)
    null_offset = np.unravel_index(int(np.argmin(surface[null_rows])), surface[null_rows].shape)
    null_index = (int(null_rows[null_offset[0]]), int(null_offset[1]))
    null_value = float(surface[null_index])
    if not np.isfinite(null_value):
        raise RuntimeError("No finite null fit; cannot construct a likelihood ratio")
    statistic = max(0.0, 2 * (null_value - best_value))
    values, jitter = fitted[best_index]
    response = _response_from_vector(values, jitter)
    null_values, null_jitter = fitted[null_index]
    null_response = _response_from_vector(null_values, null_jitter)
    boundaries = _response_boundaries(response, library.candidate_timing_jitters)
    boundaries["sigma_t"] = _boundary(
        library.candidate_sigmas[best_index[1]],
        min(library.candidate_sigmas), max(library.candidate_sigmas))
    boundaries["lambda_strength"] = _boundary(
        library.candidate_lambdas[best_index[0]],
        min(library.candidate_lambdas), max(library.candidate_lambdas))
    return SpatialDetectorCountFit(
        best_sigma_t=float(library.candidate_sigmas[best_index[1]]),
        best_lambda_strength=float(library.candidate_lambdas[best_index[0]]),
        response=response,
        negative_log_likelihood=best_value,
        null_negative_log_likelihood=null_value,
        likelihood_ratio=statistic,
        detected=bool(statistic > detection_threshold),
        profile_negative_log_likelihood=surface,
        joint_confidence_region=(2 * (surface - best_value)
                                 <= joint_confidence_threshold),
        success=bool(successes[best_index]),
        diagnostics=ProfileDiagnostics(
            best=optimizer_diagnostics[best_index], null=optimizer_diagnostics[null_index],
            null_response=null_response, calibration=tuple(calibration_diagnostics),
            profile_fits=profile_fits, failed_profile_fits=failed_fits,
            nonfinite_profile_fits=nonfinite_fits,
            failed_profile_status_counts=failed_status_counts,
            best_boundaries=boundaries,
            null_response_boundaries=_response_boundaries(
                null_response, library.candidate_timing_jitters),
            sigma_identified_at_best=bool(library.candidate_lambdas[best_index[0]] > 0)),
    )


def _shot_counts(shots, count):
    if isinstance(shots, (int, np.integer)):
        values = np.full(count, int(shots), dtype=int)
    else:
        values = np.asarray(shots, dtype=int)
    if values.shape != (count,) or np.any(values <= 0):
        raise ValueError("shots must provide a positive count per experiment")
    return values


def compare_spatial_detector_grids(coarse, refined, calibration_counts, test_counts):
    """Fit identical records on nested grids and expose numerical discrepancies.

    No detection threshold is calibrated here. Shared physical and jitter
    settings are required so only sigma/lambda grid density changes.
    """
    if (coarse.calibration_experiments != refined.calibration_experiments
            or coarse.test_experiments != refined.test_experiments
            or coarse.calibration_sigma_t != refined.calibration_sigma_t
            or coarse.calibration_lambda_strength != refined.calibration_lambda_strength
            or not np.array_equal(coarse.candidate_timing_jitters,
                                  refined.candidate_timing_jitters)):
        raise ValueError("Paired grid comparison requires identical physics and jitter settings")
    indices = []
    for old, new in ((coarse.candidate_lambdas, refined.candidate_lambdas),
                     (coarse.candidate_sigmas, refined.candidate_sigmas)):
        if not np.all(np.isfinite(new)) or np.any(np.diff(new) <= 0):
            raise ValueError("Refined grid must be finite and strictly increasing")
        matched = []
        for value in old:
            found = np.flatnonzero(new == value)
            if found.size != 1:
                raise ValueError("Refined grid must contain every coarse grid point exactly")
            matched.append(int(found[0]))
        indices.append(np.asarray(matched))
    common_laws = refined.test_ideal_probabilities[:, indices[0]][:, :, indices[1]]
    if (not np.allclose(common_laws, coarse.test_ideal_probabilities, rtol=0, atol=1e-12)
            or not np.allclose(refined.calibration_ideal_probabilities,
                               coarse.calibration_ideal_probabilities, rtol=0, atol=1e-12)):
        raise ValueError("Common grid points must have identical complete probability laws")
    # Pass the very same count arrays to both fits; never redraw inside a fit.
    first = fit_spatial_detector_counts(
        coarse, calibration_counts, test_counts, detection_threshold=np.inf)
    second = fit_spatial_detector_counts(
        refined, calibration_counts, test_counts, detection_threshold=np.inf)
    common_surface = second.profile_negative_log_likelihood[np.ix_(*indices)]
    return first, second, {
        "delta_likelihood_ratio": second.likelihood_ratio - first.likelihood_ratio,
        "delta_best_nll": second.negative_log_likelihood - first.negative_log_likelihood,
        "delta_null_nll": second.null_negative_log_likelihood - first.null_negative_log_likelihood,
        "max_abs_common_profile_nll_difference": float(np.max(np.abs(
            common_surface - first.profile_negative_log_likelihood))),
        "nested_minimum_violation": bool(
            second.negative_log_likelihood > first.negative_log_likelihood + 1e-7),
    }


def _library_replicate(
        library, true_response, calibration_shots, test_shots,
        sigma_index, lambda_index, detection_threshold, seed):
    rng = np.random.default_rng(seed)
    jitter_matches = np.flatnonzero(np.isclose(
        library.candidate_timing_jitters, true_response.timing_jitter,
        rtol=0, atol=1e-12))
    if jitter_matches.size != 1:
        raise ValueError("true response timing jitter must occur on the library grid")
    jitter_index = int(jitter_matches[0])
    calibration_ideal = library.calibration_ideal_probabilities[jitter_index]
    test_ideal = library.test_ideal_probabilities[
        jitter_index, lambda_index, sigma_index]
    calibration_shot_counts = _shot_counts(
        calibration_shots, len(library.calibration_experiments))
    test_shot_counts = _shot_counts(test_shots, len(library.test_experiments))
    calibration_counts = tuple(rng.multinomial(
        int(shots), apply_detector_response(ideal, true_response))
        for ideal, shots in zip(calibration_ideal, calibration_shot_counts))
    test_counts = tuple(rng.multinomial(
        int(shots), apply_detector_response(ideal, true_response))
        for ideal, shots in zip(test_ideal, test_shot_counts))
    return fit_spatial_detector_counts(
        library, calibration_counts, test_counts,
        detection_threshold=detection_threshold)


def _initialize_replicate_worker(*context):
    global _REPLICATE_CONTEXT
    _REPLICATE_CONTEXT = context


def _replicate_worker(seed):
    return _library_replicate(*_REPLICATE_CONTEXT, seed)


def _run_replicates(
        library, true_response, calibration_shots, test_shots,
        sigma_index, lambda_index, detection_threshold, repetitions,
        seed, workers):
    if not isinstance(workers, (int, np.integer)) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    seeds = _replicate_seeds(seed, repetitions)
    context = (
        library, true_response, calibration_shots, test_shots,
        sigma_index, lambda_index, detection_threshold,
    )
    if workers == 1:
        return [_library_replicate(*context, value) for value in seeds]
    with ProcessPoolExecutor(
            max_workers=workers, initializer=_initialize_replicate_worker,
            initargs=context, mp_context=_process_pool_context()) as executor:
        return list(executor.map(_replicate_worker, seeds, chunksize=1))


def _replicate_seeds(seed, repetitions):
    return [int(value.generate_state(1, dtype=np.uint64)[0])
            for value in np.random.SeedSequence(seed).spawn(repetitions)]


def monte_carlo_spatial_detector_recovery(
        library, true_response, scenarios, *, calibration_shots,
        test_shots, repetitions=100, seed=0, detection_threshold=2.7055,
        workers=1):
    """Check bias, detection rate, and joint profile-region coverage."""
    if not isinstance(repetitions, (int, np.integer)) or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    summaries = []
    for scenario_index, (true_sigma_t, true_lambda_strength) in enumerate(scenarios):
        sigma_matches = np.flatnonzero(np.isclose(
            library.candidate_sigmas, true_sigma_t, rtol=0, atol=1e-12))
        lambda_matches = np.flatnonzero(np.isclose(
            library.candidate_lambdas, true_lambda_strength, rtol=0, atol=1e-12))
        if sigma_matches.size != 1 or lambda_matches.size != 1:
            raise ValueError("each true scenario must occur exactly on the fit grid")
        sigma_index = int(sigma_matches[0])
        lambda_index = int(lambda_matches[0])
        fits = _run_replicates(
            library, true_response, calibration_shots, test_shots,
            sigma_index, lambda_index, detection_threshold, repetitions,
            seed + scenario_index, workers)
        estimates, detections, coverages, likelihood_ratios, successes = (
            [], [], [], [], 0)
        for fit in fits:
            estimates.append((fit.best_sigma_t, fit.best_lambda_strength))
            detections.append(fit.detected)
            coverages.append(bool(
                fit.joint_confidence_region[lambda_index, sigma_index]))
            likelihood_ratios.append(fit.likelihood_ratio)
            successes += int(fit.success)
        estimates = np.asarray(estimates)
        sigma_errors = estimates[:, 0] - true_sigma_t
        lambda_errors = estimates[:, 1] - true_lambda_strength
        summaries.append(SpatialDetectorRecovery(
            true_sigma_t=float(true_sigma_t),
            true_lambda_strength=float(true_lambda_strength),
            repetitions=repetitions,
            successful_fits=successes,
            detection_rate=float(np.mean(detections)),
            mean_sigma_t=float(np.mean(estimates[:, 0])),
            mean_lambda_strength=float(np.mean(estimates[:, 1])),
            sigma_bias=float(np.mean(sigma_errors)),
            lambda_bias=float(np.mean(lambda_errors)),
            sigma_rmse=float(np.sqrt(np.mean(sigma_errors ** 2))),
            lambda_rmse=float(np.sqrt(np.mean(lambda_errors ** 2))),
            joint_coverage=float(np.mean(coverages)),
            sigma_quantiles=np.quantile(estimates[:, 0], [0.05, 0.5, 0.95]),
            lambda_quantiles=np.quantile(
                estimates[:, 1], [0.05, 0.5, 0.95]),
            likelihood_ratio_quantiles=np.quantile(
                likelihood_ratios, [0.05, 0.5, 0.95]),
            replicate_records=tuple(_replicate_record(fit, replicate_seed)
                for fit, replicate_seed in zip(
                    fits, _replicate_seeds(seed + scenario_index, repetitions))),
        ))
    return tuple(summaries)


def bootstrap_spatial_detection_threshold(
        library, true_response, *, calibration_shots, test_shots,
        repetitions=100, confidence_level=0.95, seed=0, workers=1):
    """Calibrate the likelihood-ratio threshold under the lambda=0 null.

    ``sigma_t`` is not identified under this null, so a regular chi-square
    approximation is not assumed.  Every bootstrap replicate redraws both the
    independent controls and the double-slit records and repeats the complete
    nuisance profile fit.
    """
    if not isinstance(repetitions, (int, np.integer)) or repetitions <= 1:
        raise ValueError("bootstrap repetitions must be an integer greater than one")
    confidence_level = float(confidence_level)
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0, 1)")
    # The null law is exactly independent of sigma_t; a grid value simply
    # supplies the otherwise irrelevant function argument.
    null_rows = np.flatnonzero(library.candidate_lambdas == 0)
    if null_rows.size != 1:
        raise ValueError("library must contain exactly one lambda=0 grid point")
    fits = _run_replicates(
        library, true_response, calibration_shots, test_shots,
        sigma_index=0, lambda_index=int(null_rows[0]),
        detection_threshold=np.inf, repetitions=repetitions,
        seed=seed, workers=workers)
    statistics = np.asarray([fit.likelihood_ratio for fit in fits])
    try:
        threshold = float(np.quantile(
            statistics, confidence_level, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility.
        threshold = float(np.quantile(
            statistics, confidence_level, interpolation="higher"))
    return SpatialDetectionThreshold(
        confidence_level, repetitions, threshold, statistics,
        tuple(_replicate_record(fit, replicate_seed)
              for fit, replicate_seed in zip(fits, _replicate_seeds(seed, repetitions))))
