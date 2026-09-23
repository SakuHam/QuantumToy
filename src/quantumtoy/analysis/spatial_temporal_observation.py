"""Idealized temporal-register measurements for the double-slit benchmark.

The existing detector marginalizes an unresolved temporal mixture.  This
module constructs two explicitly stronger observation models from a coherent
purification of that same mixture: direct temporal-mode readout and a Fourier
(phase-port) readout.  They are information benchmarks, not TRF predictions.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np

from analysis.spatial_detector_inference import (
    DetectorResponse,
    _jitter_nodes,
    detector_blur_matrix,
    detector_response_matrix,
)
from analysis.spatial_effect_measurement import (
    _propagate_sample_times,
    _terminal_effect_diagonals,
    initial_spatial_state,
)


_STRATEGIES = {"integrated", "resolved", "coherent"}


def fixed_temporal_delays(experiment, delay_horizon):
    """Return a parameter-independent delay grid for nested local fits."""
    delay_horizon = float(delay_horizon)
    if not np.isfinite(delay_horizon) or delay_horizon <= 0:
        raise ValueError("delay_horizon must be finite and positive")
    count = max(1, int(np.ceil(delay_horizon / experiment.delay_step)))
    return np.arange(count + 1, dtype=float) * experiment.delay_step


def fixed_half_gaussian_weights(sigma_t, delays):
    """Trapezoidal half-Gaussian weights on one fixed outcome grid."""
    sigma_t = float(sigma_t)
    delays = np.asarray(delays, dtype=float)
    if (not np.isfinite(sigma_t) or sigma_t <= 0 or delays.ndim != 1
            or delays.size < 2 or np.any(~np.isfinite(delays))
            or delays[0] != 0 or np.any(np.diff(delays) <= 0)):
        raise ValueError("sigma and an ordered nonnegative delay grid are required")
    weights = np.exp(-0.5 * (delays / sigma_t) ** 2)
    weights[[0, -1]] *= 0.5
    return weights / np.sum(weights)


@lru_cache(maxsize=64)
def _temporal_mode_states(experiment, delay_count):
    delays = np.arange(delay_count + 1, dtype=float) * experiment.delay_step
    state = initial_spatial_state(experiment)
    temporal = _propagate_sample_times(
        experiment, state, experiment.reference_time + delays)
    # The response-null reference and the zero-delay temporal component have
    # equal spatial states but distinct purification labels.
    return np.concatenate([temporal[:1], temporal], axis=0)


def _mode_weights(sigma_t, lambda_strength, delays):
    lambda_strength = float(lambda_strength)
    if not np.isfinite(lambda_strength) or lambda_strength < 0:
        raise ValueError("lambda_strength must be finite and nonnegative")
    temporal = fixed_half_gaussian_weights(sigma_t, delays)
    response_fraction = -np.expm1(-lambda_strength)
    return np.concatenate([
        np.array([1.0 - response_fraction]),
        response_fraction * temporal,
    ])


def _ideal_category_laws(
        experiment, sigma_t, lambda_strength, delays, strategy, phase):
    labels, diagonals = _terminal_effect_diagonals(experiment)
    states = _temporal_mode_states(experiment, delays.size - 1)
    weights = _mode_weights(sigma_t, lambda_strength, delays)
    if strategy == "integrated":
        probabilities = np.sum(
            weights[:, None] * np.asarray([
                diagonals @ np.abs(state) ** 2 for state in states
            ]), axis=0, keepdims=True)
    elif strategy == "resolved":
        mode_probabilities = weights[:, None] * np.asarray([
            diagonals @ np.abs(state) ** 2 for state in states
        ])
        # An arrival-time record cannot distinguish the response-null branch
        # from the temporal component at exactly zero delay: both have the
        # same spatial state and timestamp.  Merge them before readout.
        probabilities = np.concatenate([
            mode_probabilities[:1] + mode_probabilities[1:2],
            mode_probabilities[2:],
        ], axis=0)
    else:
        mode_count = states.shape[0]
        indices = np.arange(mode_count)
        transform = np.exp(-1j * (
            2 * np.pi * indices[:, None] * indices[None, :] / mode_count
            + float(phase) * indices[None, :])) / np.sqrt(mode_count)
        ports = transform @ (np.sqrt(weights)[:, None] * states)
        probabilities = np.asarray([
            diagonals @ np.abs(state) ** 2 for state in ports
        ])
    probabilities = np.real_if_close(probabilities).real
    if (np.min(probabilities) < -1e-12
            or not np.isclose(np.sum(probabilities), 1.0, atol=2e-11)):
        raise RuntimeError("Temporal observation produced an invalid ideal law")
    return labels, np.maximum(probabilities, 0.0)


def temporal_observation_probabilities(
        experiment, sigma_t, lambda_strength, response, *, strategy,
        delay_horizon, phase=0.0, category_blur_sigma_bins=0.0):
    """Return one complete law for an integrated, resolved, or phase readout.

    The temporal register is read jointly with the spatial detector.  Its label
    is retained for click outcomes; all no-click outcomes are coarsened into
    one category.  Summing temporal/phase click labels recovers the integrated
    spatial law exactly.  Symmetric timing jitter remains an unresolved
    classical mixture common to all three strategies.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"strategy must be one of {sorted(_STRATEGIES)}")
    if not isinstance(response, DetectorResponse):
        raise TypeError("response must be a DetectorResponse")
    category_blur_sigma_bins = float(category_blur_sigma_bins)
    if (not np.isfinite(category_blur_sigma_bins)
            or category_blur_sigma_bins < 0):
        raise ValueError("category_blur_sigma_bins must be finite and nonnegative")
    delays = fixed_temporal_delays(experiment, delay_horizon)
    offsets, jitter_weights = _jitter_nodes(response.timing_jitter)
    if experiment.reference_time + offsets[0] < 0:
        raise ValueError("timing jitter extends before time zero")
    observed = []
    channel = detector_response_matrix(experiment.y_bins, response)
    for offset, jitter_weight in zip(offsets, jitter_weights):
        shifted = replace(
            experiment, reference_time=experiment.reference_time + float(offset))
        _, ideal = _ideal_category_laws(
            shifted, sigma_t, lambda_strength, delays, strategy, phase)
        if category_blur_sigma_bins:
            ideal = detector_blur_matrix(
                ideal.shape[0], category_blur_sigma_bins) @ ideal
        categories = np.einsum("ij,cj->ci", channel, ideal)
        click = categories[:, :-1].reshape(-1)
        complete = np.concatenate([click, [np.sum(categories[:, -1])]])
        observed.append(float(jitter_weight) * complete)
    probabilities = np.sum(observed, axis=0)
    if (np.min(probabilities) < -1e-12
            or not np.isclose(np.sum(probabilities), 1.0, atol=2e-11)):
        raise RuntimeError("Temporal detector channel produced an invalid law")
    return np.maximum(probabilities, 0.0)


def temporal_observation_fisher_information(
        experiment, sigma_t, lambda_strength, response, *, strategy,
        delay_horizon, phase=0.0, category_blur_sigma_bins=0.0,
        relative_step=1e-3):
    """Per-shot local Fisher information for ``(sigma_t, lambda)``."""
    sigma_t = float(sigma_t)
    lambda_strength = float(lambda_strength)
    if sigma_t <= 0 or lambda_strength < 0:
        raise ValueError("sigma_t must be positive and lambda nonnegative")
    sigma_step = relative_step * sigma_t
    lambda_step = relative_step * max(1.0, lambda_strength)
    sigma_minus = temporal_observation_probabilities(
        experiment, sigma_t - sigma_step, lambda_strength, response,
        strategy=strategy, delay_horizon=delay_horizon, phase=phase,
        category_blur_sigma_bins=category_blur_sigma_bins)
    sigma_plus = temporal_observation_probabilities(
        experiment, sigma_t + sigma_step, lambda_strength, response,
        strategy=strategy, delay_horizon=delay_horizon, phase=phase,
        category_blur_sigma_bins=category_blur_sigma_bins)
    derivative_sigma = (sigma_plus - sigma_minus) / (2 * sigma_step)
    if lambda_strength > lambda_step:
        lambda_minus = temporal_observation_probabilities(
            experiment, sigma_t, lambda_strength - lambda_step, response,
            strategy=strategy, delay_horizon=delay_horizon, phase=phase,
            category_blur_sigma_bins=category_blur_sigma_bins)
        lambda_plus = temporal_observation_probabilities(
            experiment, sigma_t, lambda_strength + lambda_step, response,
            strategy=strategy, delay_horizon=delay_horizon, phase=phase,
            category_blur_sigma_bins=category_blur_sigma_bins)
        derivative_lambda = (lambda_plus - lambda_minus) / (2 * lambda_step)
    else:
        center = temporal_observation_probabilities(
            experiment, sigma_t, lambda_strength, response,
            strategy=strategy, delay_horizon=delay_horizon, phase=phase,
            category_blur_sigma_bins=category_blur_sigma_bins)
        lambda_plus = temporal_observation_probabilities(
            experiment, sigma_t, lambda_strength + lambda_step, response,
            strategy=strategy, delay_horizon=delay_horizon, phase=phase,
            category_blur_sigma_bins=category_blur_sigma_bins)
        derivative_lambda = (lambda_plus - center) / lambda_step
    probabilities = temporal_observation_probabilities(
        experiment, sigma_t, lambda_strength, response,
        strategy=strategy, delay_horizon=delay_horizon, phase=phase,
        category_blur_sigma_bins=category_blur_sigma_bins)
    derivatives = np.stack([derivative_sigma, derivative_lambda], axis=1)
    information = ((derivatives.T / np.maximum(probabilities, 1e-300))
                   @ derivatives)
    return 0.5 * (information + information.T)
