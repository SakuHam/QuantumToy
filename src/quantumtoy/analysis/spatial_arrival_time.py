"""Phenomenological joint spatial/arrival-time detector for the TRF benchmark.

The latent nonnegative delay used by the spatial effect model is mapped to a
measured timestamp residual.  Source-pulse uncertainty and detector jitter
are convolved with that delay, finite time gates and bins are retained, and
all events outside the genuine-click channel end in either a dark record or
the explicit no-click outcome.

This is an instrument hypothesis.  TRF-IT v0.2 does not derive the map from
its temporal width to an arrival-time shift, so the module must not be read as
a prediction that the latent delay is experimentally accessible.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from scipy.special import ndtr

from analysis.spatial_detector_inference import (
    DetectorResponse,
    detector_blur_matrix,
)
from analysis.spatial_effect_measurement import (
    _propagate_sample_times,
    _terminal_effect_diagonals,
    initial_spatial_state,
)
from analysis.spatial_temporal_observation import (
    fixed_half_gaussian_weights,
    fixed_temporal_delays,
)


@dataclass(frozen=True)
class ArrivalTimeResponse:
    """Timing part of a pulsed or heralded detector instrument.

    Times are residuals after subtracting the standard arrival time for the
    detector setting.  ``source_pulse_sigma`` is the launch-time uncertainty;
    detector timestamp jitter is ``DetectorResponse.timing_jitter``.  The
    finite acquisition window is divided into equal bins.  A genuine arrival
    outside the window is unregistered and therefore remains in the complete
    dark/no-click channel.
    """

    source_pulse_sigma: float = 0.03
    clock_offset: float = 0.0
    delay_scale: float = 1.0
    window_start: float = -0.2
    window_stop: float = 1.0
    time_bins: int = 24

    def __post_init__(self):
        values = (
            self.source_pulse_sigma, self.clock_offset,
            self.delay_scale, self.window_start, self.window_stop,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Arrival-time response parameters must be finite")
        if self.source_pulse_sigma < 0:
            raise ValueError("source_pulse_sigma must be nonnegative")
        if self.delay_scale < 0:
            raise ValueError("delay_scale must be nonnegative")
        if self.window_stop <= self.window_start:
            raise ValueError("window_stop must exceed window_start")
        if not isinstance(self.time_bins, (int, np.integer)) or self.time_bins < 1:
            raise ValueError("time_bins must be a positive integer")

    @property
    def bin_edges(self):
        return np.linspace(self.window_start, self.window_stop,
                           self.time_bins + 1)

    @property
    def bin_centers(self):
        edges = self.bin_edges
        return 0.5 * (edges[:-1] + edges[1:])


def arrival_time_bin_probabilities(delay, timing, detector_jitter):
    """Probability of every finite timestamp bin for one latent delay."""
    if not isinstance(timing, ArrivalTimeResponse):
        raise TypeError("timing must be an ArrivalTimeResponse")
    delay = float(delay)
    detector_jitter = float(detector_jitter)
    if not np.isfinite(delay) or delay < 0:
        raise ValueError("delay must be finite and nonnegative")
    if not np.isfinite(detector_jitter) or detector_jitter < 0:
        raise ValueError("detector_jitter must be finite and nonnegative")
    sigma = float(np.hypot(timing.source_pulse_sigma, detector_jitter))
    mean = timing.delay_scale * delay + timing.clock_offset
    edges = timing.bin_edges
    if sigma == 0:
        probabilities = np.zeros(timing.time_bins)
        index = int(np.searchsorted(edges, mean, side="right") - 1)
        if mean == edges[-1]:
            index = timing.time_bins - 1
        if 0 <= index < timing.time_bins:
            probabilities[index] = 1.0
        return probabilities
    cdf = ndtr((edges - mean) / sigma)
    probabilities = np.diff(cdf)
    return np.maximum(probabilities, 0.0)


@lru_cache(maxsize=64)
def _fixed_component_spatial_probabilities(experiment, delay_count):
    delays = np.arange(delay_count + 1, dtype=float) * experiment.delay_step
    _, diagonals = _terminal_effect_diagonals(experiment)
    state = initial_spatial_state(experiment)
    states = _propagate_sample_times(
        experiment, state, experiment.reference_time + delays)
    probabilities = np.asarray([
        diagonals @ np.abs(component) ** 2 for component in states
    ])
    if (np.min(probabilities) < -1e-12
            or not np.allclose(np.sum(probabilities, axis=1), 1, atol=2e-11)):
        raise RuntimeError("Propagation produced an invalid spatial law")
    return np.maximum(probabilities, 0.0)


def joint_spatial_arrival_probabilities(
        experiment, sigma_t, lambda_strength, response, timing, *,
        delay_horizon):
    """Return the complete joint law ``p(time bin, y bin) + p(no-click)``.

    ``sigma_t`` and all timing fields use the simulation's time unit.  The
    measured mean residual is ``delay_scale * latent_delay + clock_offset``.
    ``delay_scale`` is a declared response coupling, not a TRF prediction. The
    output is time-major when its click portion is reshaped to
    ``(time_bins, experiment.y_bins)``.  Dark timestamps are uniform over the
    declared acquisition window.  No outcome is conditioned away.
    """
    sigma_t = float(sigma_t)
    lambda_strength = float(lambda_strength)
    if not isinstance(response, DetectorResponse):
        raise TypeError("response must be a DetectorResponse")
    if not isinstance(timing, ArrivalTimeResponse):
        raise TypeError("timing must be an ArrivalTimeResponse")
    if not np.isfinite(sigma_t) or sigma_t <= 0:
        raise ValueError("sigma_t must be finite and positive")
    if not np.isfinite(lambda_strength) or lambda_strength < 0:
        raise ValueError("lambda_strength must be finite and nonnegative")

    delays = fixed_temporal_delays(experiment, delay_horizon)
    temporal_weights = fixed_half_gaussian_weights(sigma_t, delays)
    response_fraction = -np.expm1(-lambda_strength)
    # The reference branch and response branch at zero delay are operationally
    # indistinguishable.  Summing both contributions implements that merging.
    component_delays = np.concatenate([[0.0], delays])
    component_weights = np.concatenate([
        [1.0 - response_fraction], response_fraction * temporal_weights,
    ])
    temporal_spatial = _fixed_component_spatial_probabilities(
        experiment, delays.size - 1)
    spatial = np.concatenate([temporal_spatial[:1], temporal_spatial], axis=0)
    blur = detector_blur_matrix(experiment.y_bins, response.blur_sigma_bins)
    dark_time = np.diff(timing.bin_edges)
    dark_time = dark_time / np.sum(dark_time)
    dark_joint = dark_time[:, None] / experiment.y_bins

    joint = np.zeros((timing.time_bins, experiment.y_bins), dtype=float)
    no_click = 0.0
    for weight, delay, ideal in zip(
            component_weights, component_delays, spatial):
        timestamp = arrival_time_bin_probabilities(
            delay, timing, response.timing_jitter)
        blurred_click = blur @ ideal[:-1]
        genuine = (response.efficiency
                   * timestamp[:, None] * blurred_click[None, :])
        remaining = max(0.0, 1.0 - float(np.sum(genuine)))
        observed = genuine + remaining * response.dark_probability * dark_joint
        joint += weight * observed
        no_click += weight * remaining * (1.0 - response.dark_probability)

    probabilities = np.concatenate([joint.reshape(-1), [no_click]])
    if (np.min(probabilities) < -1e-12
            or not np.isclose(np.sum(probabilities), 1, atol=2e-11)):
        raise RuntimeError("Joint spatial/arrival detector law is incomplete")
    return np.maximum(probabilities, 0.0)


def marginalize_joint_arrival(probabilities, y_bins, time_bins):
    """Return complete spatial and temporal marginals from a joint law."""
    probabilities = np.asarray(probabilities, dtype=float)
    expected = int(y_bins) * int(time_bins) + 1
    if probabilities.shape != (expected,):
        raise ValueError("probability vector does not match y/time bin counts")
    joint = probabilities[:-1].reshape(int(time_bins), int(y_bins))
    no_click = probabilities[-1:]
    spatial = np.concatenate([np.sum(joint, axis=0), no_click])
    temporal = np.concatenate([np.sum(joint, axis=1), no_click])
    return spatial, temporal


def joint_spatial_arrival_fisher_information(
        experiment, sigma_t, lambda_strength, response, timing, *,
        delay_horizon, parameter_names=("sigma_t", "lambda_strength"),
        relative_step=1e-3):
    """Per-shot Fisher matrix, optionally including timing nuisances.

    Supported parameters are ``sigma_t``, ``lambda_strength``,
    ``source_pulse_sigma``, ``detector_jitter``, ``clock_offset``, and
    ``delay_scale``.
    Separately fitting both widths from this law alone is expected to be
    singular because they enter through their quadrature sum; independent
    timing calibration is therefore physically necessary.
    """
    allowed = {
        "sigma_t", "lambda_strength", "source_pulse_sigma",
        "detector_jitter", "clock_offset", "delay_scale",
    }
    parameter_names = tuple(parameter_names)
    if (not parameter_names or len(set(parameter_names)) != len(parameter_names)
            or not set(parameter_names) <= allowed):
        raise ValueError(f"parameter_names must be unique members of {sorted(allowed)}")
    if not np.isfinite(relative_step) or relative_step <= 0:
        raise ValueError("relative_step must be finite and positive")

    def law(values):
        local_sigma = values.get("sigma_t", sigma_t)
        local_lambda = values.get("lambda_strength", lambda_strength)
        local_response = replace(
            response, timing_jitter=values.get(
                "detector_jitter", response.timing_jitter))
        local_timing = replace(
            timing,
            source_pulse_sigma=values.get(
                "source_pulse_sigma", timing.source_pulse_sigma),
            clock_offset=values.get("clock_offset", timing.clock_offset),
            delay_scale=values.get("delay_scale", timing.delay_scale),
        )
        return joint_spatial_arrival_probabilities(
            experiment, local_sigma, local_lambda, local_response,
            local_timing, delay_horizon=delay_horizon)

    center_values = {
        "sigma_t": float(sigma_t),
        "lambda_strength": float(lambda_strength),
        "source_pulse_sigma": timing.source_pulse_sigma,
        "detector_jitter": response.timing_jitter,
        "clock_offset": timing.clock_offset,
        "delay_scale": timing.delay_scale,
    }
    probabilities = law({})
    derivatives = []
    for name in parameter_names:
        value = center_values[name]
        if name == "clock_offset":
            scale = max(
                timing.source_pulse_sigma, response.timing_jitter,
                (timing.window_stop - timing.window_start) / timing.time_bins)
        elif name == "lambda_strength":
            scale = max(1.0, value)
        else:
            scale = max(experiment.delay_step, value)
        step = relative_step * scale
        lower_allowed = name != "clock_offset"
        if lower_allowed and value <= step:
            upper = law({name: value + step})
            derivatives.append((upper - probabilities) / step)
        else:
            lower = law({name: value - step})
            upper = law({name: value + step})
            derivatives.append((upper - lower) / (2 * step))
    derivatives = np.asarray(derivatives).T
    information = ((derivatives.T / np.maximum(probabilities, 1e-300))
                   @ derivatives)
    return 0.5 * (information + information.T)
