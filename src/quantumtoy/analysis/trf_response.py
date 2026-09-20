"""Temporal-response measurement of sigma_T and held-out alpha tests."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize
from scipy.special import erf, erfinv

from analysis.trf_candidate import joint_with_dephasing


HALF_RISE_FACTOR = float(np.sqrt(2) * erfinv(0.5))


def excess_dephasing(time, lambda_strength, sigma_t):
    """D(t)=-log(C_theta/C_Q) for the half-Gaussian candidate."""
    time = np.asarray(time, dtype=float)
    if (np.any(~np.isfinite(time)) or np.any(time < 0)
            or not np.isfinite(lambda_strength) or lambda_strength < 0
            or not np.isfinite(sigma_t) or sigma_t <= 0):
        raise ValueError("Invalid temporal-response parameters")
    return lambda_strength * erf(time / (np.sqrt(2) * sigma_t))


def half_rise_time(sigma_t):
    if not np.isfinite(sigma_t) or sigma_t <= 0:
        raise ValueError("sigma_t must be finite and positive")
    return HALF_RISE_FACTOR * sigma_t


@dataclass(frozen=True)
class ResponseSetting:
    g: float
    time: float
    detector_phase: float
    shots: int

    def __post_init__(self):
        if (not np.isfinite(self.g) or self.g <= 0
                or not np.isfinite(self.time) or self.time < 0
                or not np.isfinite(self.detector_phase)
                or not isinstance(self.shots, (int, np.integer)) or self.shots <= 0):
            raise ValueError("Invalid temporal-response setting")


def response_probabilities(experiment, settings, *, lambda_strength,
                           sigma_by_g, ordinary_dephasing_rate):
    """Complete p(a,Y,d) for phase scans at declared times and couplings."""
    if not np.isfinite(ordinary_dephasing_rate) or ordinary_dephasing_rate < 0:
        raise ValueError("ordinary_dephasing_rate must be finite and nonnegative")
    probabilities = []
    for setting in settings:
        try:
            sigma_t = float(sigma_by_g[setting.g])
        except (KeyError, TypeError):
            raise ValueError(f"Missing sigma_T for g={setting.g}") from None
        factor = np.exp(
            -ordinary_dephasing_rate * setting.time
            -excess_dephasing(setting.time, lambda_strength, sigma_t)
        )
        protocol = replace(experiment, detector_phase=setting.detector_phase)
        # Z readout erases the X-basis path record in these separately prepared
        # response runs, retaining sensitivity after record formation.
        probabilities.append(joint_with_dephasing(
            protocol, setting.g, setting.time, float(factor),
            fragment_readout_basis="Z"))
    return tuple(probabilities)


def simulate_response_counts(experiment, settings, *, lambda_strength,
                             sigma_by_g, ordinary_dephasing_rate, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    probabilities = response_probabilities(
        experiment, settings,
        lambda_strength=lambda_strength,
        sigma_by_g=sigma_by_g,
        ordinary_dephasing_rate=ordinary_dephasing_rate,
    )
    return tuple(
        rng.multinomial(setting.shots, probability.ravel()).reshape(probability.shape)
        for setting, probability in zip(settings, probabilities)
    )


@dataclass(frozen=True)
class ResponseFit:
    mode: str
    g_values: tuple[float, ...]
    lambda_strength: float
    sigma_by_g: dict[float, float]
    ordinary_dephasing_rate: float
    parameters: np.ndarray
    covariance: np.ndarray | None
    standard_errors: np.ndarray | None
    negative_log_posterior: float
    success: bool

    def alpha_by_g(self, tau_by_g):
        return {g: sigma / tau_by_g[g] for g, sigma in self.sigma_by_g.items()}


def _decode_parameters(parameters, mode, g_values, tau_by_g):
    lambda_strength = float(parameters[0])
    if mode == "universal_alpha":
        alpha = float(parameters[1])
        sigma_by_g = {g: alpha * tau_by_g[g] for g in g_values}
        gamma = float(parameters[2])
    elif mode == "free_widths":
        sigma_by_g = {g: float(parameters[1 + i])
                      for i, g in enumerate(g_values)}
        gamma = float(parameters[-1])
    else:
        raise ValueError("mode must be 'universal_alpha' or 'free_widths'")
    return lambda_strength, sigma_by_g, gamma


def _response_nlp(parameters, experiment, settings, counts, mode, g_values,
                  tau_by_g, gamma_mean, gamma_sigma):
    lambda_strength, sigma_by_g, gamma = _decode_parameters(
        parameters, mode, g_values, tau_by_g)
    try:
        probabilities = response_probabilities(
            experiment, settings,
            lambda_strength=lambda_strength,
            sigma_by_g=sigma_by_g,
            ordinary_dephasing_rate=gamma,
        )
    except ValueError:
        return np.inf
    value = 0.5 * ((gamma - gamma_mean) / gamma_sigma) ** 2
    for observed, probability in zip(counts, probabilities):
        supported = observed > 0
        if np.any(probability[supported] <= 0):
            return np.inf
        value -= float(np.sum(observed[supported] * np.log(probability[supported])))
    return value


def _response_fisher(parameters, experiment, settings, mode, g_values,
                     tau_by_g, gamma_mean, gamma_sigma):
    base_lambda, base_sigma, base_gamma = _decode_parameters(
        parameters, mode, g_values, tau_by_g)
    base = response_probabilities(
        experiment, settings,
        lambda_strength=base_lambda,
        sigma_by_g=base_sigma,
        ordinary_dephasing_rate=base_gamma,
    )
    steps = np.maximum(np.abs(parameters) * 1e-4, 1e-6)
    derivatives = []
    for index, step in enumerate(steps):
        upper, lower = parameters.copy(), parameters.copy()
        upper[index] += step
        lower[index] -= step
        u_lambda, u_sigma, u_gamma = _decode_parameters(
            upper, mode, g_values, tau_by_g)
        l_lambda, l_sigma, l_gamma = _decode_parameters(
            lower, mode, g_values, tau_by_g)
        high = response_probabilities(
            experiment, settings, lambda_strength=u_lambda,
            sigma_by_g=u_sigma, ordinary_dephasing_rate=u_gamma)
        low = response_probabilities(
            experiment, settings, lambda_strength=l_lambda,
            sigma_by_g=l_sigma, ordinary_dephasing_rate=l_gamma)
        derivatives.append([(a - b) / (2 * step) for a, b in zip(high, low)])
    fisher = np.zeros((len(parameters), len(parameters)))
    for setting_index, (setting, probability) in enumerate(zip(settings, base)):
        supported = probability > 1e-15
        for i in range(len(parameters)):
            for j in range(len(parameters)):
                fisher[i, j] += setting.shots * np.sum(
                    derivatives[i][setting_index][supported]
                    * derivatives[j][setting_index][supported]
                    / probability[supported])
    fisher[-1, -1] += 1 / gamma_sigma ** 2
    eigenvalues = np.linalg.eigvalsh((fisher + fisher.T) / 2)
    if eigenvalues[0] <= max(eigenvalues[-1] * 1e-12, 1e-14):
        return None
    return np.linalg.inv(fisher)


def fit_response(experiment, settings, counts, *, tau_by_g,
                 mode="universal_alpha", gamma_mean=0.02,
                 gamma_sigma=0.002, initial_lambda=0.1,
                 initial_alpha=1.0):
    """Fit amplitude and temporal width from complete phase-scan records."""
    settings = tuple(settings)
    counts = tuple(np.asarray(value, dtype=int) for value in counts)
    if not settings or len(settings) != len(counts):
        raise ValueError("Counts must match nonempty response settings")
    g_values = tuple(sorted({setting.g for setting in settings}))
    if any(g not in tau_by_g or not np.isfinite(tau_by_g[g]) or tau_by_g[g] <= 0
           for g in g_values):
        raise ValueError("Every coupling needs a finite positive tau_stab")
    if not np.isfinite(gamma_sigma) or gamma_sigma <= 0:
        raise ValueError("gamma_sigma must be finite and positive")
    if mode == "universal_alpha":
        initial = np.array([initial_lambda, initial_alpha, gamma_mean], dtype=float)
        bounds = [(0, 5), (0.05, 10), (0, 1)]
    elif mode == "free_widths":
        initial = np.array(
            [initial_lambda]
            + [initial_alpha * tau_by_g[g] for g in g_values]
            + [gamma_mean], dtype=float)
        bounds = [(0, 5)] + [(0.01, 100)] * len(g_values) + [(0, 1)]
    else:
        raise ValueError("mode must be 'universal_alpha' or 'free_widths'")
    result = minimize(
        _response_nlp,
        initial,
        args=(experiment, settings, counts, mode, g_values, tau_by_g,
              gamma_mean, gamma_sigma),
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 2000},
    )
    lambda_strength, sigma_by_g, gamma = _decode_parameters(
        result.x, mode, g_values, tau_by_g)
    covariance = _response_fisher(
        result.x, experiment, settings, mode, g_values, tau_by_g,
        gamma_mean, gamma_sigma)
    errors = None if covariance is None else np.sqrt(np.maximum(np.diag(covariance), 0))
    return ResponseFit(
        mode, g_values, lambda_strength, sigma_by_g, gamma,
        np.asarray(result.x), covariance, errors, float(result.fun), bool(result.success))


@dataclass(frozen=True)
class HeldoutWidthFit:
    g: float
    sigma_t: float
    standard_error: float | None
    negative_log_likelihood: float
    predicted_negative_log_likelihood: float
    prediction_likelihood_ratio: float
    success: bool


def _fixed_width_nll(sigma_t, experiment, settings, counts,
                     lambda_strength, ordinary_dephasing_rate):
    sigma_by_g = {setting.g: float(sigma_t) for setting in settings}
    probabilities = response_probabilities(
        experiment, settings,
        lambda_strength=lambda_strength,
        sigma_by_g=sigma_by_g,
        ordinary_dephasing_rate=ordinary_dephasing_rate,
    )
    value = 0.0
    for observed, probability in zip(counts, probabilities):
        supported = observed > 0
        if np.any(probability[supported] <= 0):
            return np.inf
        value -= float(np.sum(observed[supported] * np.log(probability[supported])))
    return value


def fit_heldout_width(experiment, settings, counts, *, lambda_strength,
                      ordinary_dephasing_rate, predicted_sigma_t):
    """Fit held-out sigma_T while keeping calibration parameters fixed."""
    settings = tuple(settings)
    counts = tuple(np.asarray(value, dtype=int) for value in counts)
    g_values = {setting.g for setting in settings}
    if len(g_values) != 1 or len(settings) != len(counts):
        raise ValueError("Held-out width fit requires one coupling and matching counts")
    if not np.isfinite(predicted_sigma_t) or predicted_sigma_t <= 0:
        raise ValueError("predicted_sigma_t must be finite and positive")
    result = minimize(
        lambda value: _fixed_width_nll(
            float(value[0]), experiment, settings, counts,
            lambda_strength, ordinary_dephasing_rate),
        [predicted_sigma_t],
        method="L-BFGS-B",
        bounds=[(0.01, 100)],
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 2000},
    )
    sigma_t = float(result.x[0])
    step = max(sigma_t * 1e-4, 1e-6)
    sigma_map = {next(iter(g_values)): sigma_t}
    base = response_probabilities(
        experiment, settings, lambda_strength=lambda_strength,
        sigma_by_g=sigma_map,
        ordinary_dephasing_rate=ordinary_dephasing_rate)
    upper = response_probabilities(
        experiment, settings, lambda_strength=lambda_strength,
        sigma_by_g={next(iter(g_values)): sigma_t + step},
        ordinary_dephasing_rate=ordinary_dephasing_rate)
    lower = response_probabilities(
        experiment, settings, lambda_strength=lambda_strength,
        sigma_by_g={next(iter(g_values)): sigma_t - step},
        ordinary_dephasing_rate=ordinary_dephasing_rate)
    information = 0.0
    for setting, probability, high, low in zip(settings, base, upper, lower):
        derivative = (high - low) / (2 * step)
        supported = probability > 1e-15
        information += setting.shots * np.sum(
            derivative[supported] ** 2 / probability[supported])
    standard_error = None if information <= 0 else float(1 / np.sqrt(information))
    predicted_nll = _fixed_width_nll(
        predicted_sigma_t, experiment, settings, counts,
        lambda_strength, ordinary_dephasing_rate)
    statistic = max(0.0, 2 * (predicted_nll - float(result.fun)))
    return HeldoutWidthFit(
        float(next(iter(g_values))), sigma_t, standard_error,
        float(result.fun), predicted_nll, statistic, bool(result.success))
