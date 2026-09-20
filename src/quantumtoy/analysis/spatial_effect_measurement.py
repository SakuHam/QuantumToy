"""Complete spatial POVM reference for a temporally thick detector effect.

This module deliberately stays separate from the nonlinear exploratory
``ThickFrontMeasurementGuidedTheory``.  It provides a small, exactly unitary
periodic lattice on which every detector outcome is represented by a full
operator.  A temporal width mixes Heisenberg-picture effects arithmetically,
without normalizing any field by a candidate-dependent maximum.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class SpatialEffectExperiment:
    """Declared compact geometry and preparation used by the POVM reference."""

    lx: float = 8.0
    ly: float = 6.0
    nx: int = 16
    ny: int = 16
    mass: float = 1.0
    hbar: float = 1.0
    detector_x: float = 1.5
    detector_width: float = 0.6
    y_bins: int = 4
    packet_x: float = -1.5
    packet_y: float = -0.4
    packet_sigma_x: float = 0.7
    packet_sigma_y: float = 0.8
    packet_kx: float = 2.2
    packet_ky: float = 0.4
    delay_step: float = 0.1
    horizon_sigmas: float = 4.0

    def __post_init__(self):
        positive = (
            self.lx, self.ly, self.mass, self.hbar, self.detector_width,
            self.packet_sigma_x, self.packet_sigma_y, self.delay_step,
            self.horizon_sigmas,
        )
        if not np.all(np.isfinite(positive)) or min(positive) <= 0:
            raise ValueError("Spatial effect scales must be finite and positive")
        if not isinstance(self.nx, (int, np.integer)) or self.nx < 4:
            raise ValueError("nx must be an integer >= 4")
        if not isinstance(self.ny, (int, np.integer)) or self.ny < 4:
            raise ValueError("ny must be an integer >= 4")
        if (not isinstance(self.y_bins, (int, np.integer))
                or not 1 <= self.y_bins <= self.ny):
            raise ValueError("y_bins must be an integer between 1 and ny")
        finite = (
            self.detector_x, self.packet_x, self.packet_y,
            self.packet_kx, self.packet_ky,
        )
        if not np.all(np.isfinite(finite)):
            raise ValueError("Spatial effect locations and momenta must be finite")


@dataclass(frozen=True)
class SpatialEffectInstrument:
    """One complete POVM represented at the preparation time."""

    labels: tuple[str, ...]
    effects: np.ndarray
    delays: np.ndarray
    delay_weights: np.ndarray
    sigma_t: float
    lambda_strength: float


@dataclass(frozen=True)
class SpatialEffectRun:
    sigma_t: float
    lambda_strength: float
    labels: tuple[str, ...]
    probabilities: np.ndarray
    click_probability: float
    conditional_y_distribution: np.ndarray


@dataclass(frozen=True)
class SpatialEffectEvolution:
    """Forward-picture components used to visualize the temporal mixture."""

    sigma_t: float
    lambda_strength: float
    x: np.ndarray
    y: np.ndarray
    detector_gate: np.ndarray
    labels: tuple[str, ...]
    delays: np.ndarray
    delay_weights: np.ndarray
    densities: np.ndarray
    component_probabilities: np.ndarray
    cumulative_probabilities: np.ndarray


@dataclass(frozen=True)
class SpatialEffectProfile:
    true_sigma_t: float
    lambda_strength: float
    candidate_sigmas: np.ndarray
    kl_divergence: np.ndarray
    expected_deviance: np.ndarray
    best_sigma_t: float
    target_probabilities: np.ndarray
    null_probabilities: np.ndarray
    signal_total_variation: float
    lambda_zero_max_error: float


def _validate_sigma_lambda(sigma_t, lambda_strength):
    sigma_t = float(sigma_t)
    lambda_strength = float(lambda_strength)
    if not np.isfinite(sigma_t) or sigma_t <= 0:
        raise ValueError("sigma_t must be finite and positive")
    if not np.isfinite(lambda_strength) or lambda_strength < 0:
        raise ValueError("lambda_strength must be finite and nonnegative")
    return sigma_t, lambda_strength


def _coordinates(experiment):
    dx = experiment.lx / experiment.nx
    dy = experiment.ly / experiment.ny
    x = (np.arange(experiment.nx) - experiment.nx / 2) * dx
    y = (np.arange(experiment.ny) - experiment.ny / 2) * dy
    return np.meshgrid(x, y)


def initial_spatial_state(experiment):
    """Return a normalized complex packet in the lattice orthonormal basis."""
    x, y = _coordinates(experiment)
    envelope = np.exp(
        -0.25 * ((x - experiment.packet_x) / experiment.packet_sigma_x) ** 2
        -0.25 * ((y - experiment.packet_y) / experiment.packet_sigma_y) ** 2
    )
    phase = np.exp(1j * (experiment.packet_kx * x + experiment.packet_ky * y))
    state = (envelope * phase).reshape(-1).astype(complex)
    norm = np.linalg.norm(state)
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("Initial spatial state has zero norm")
    return state / norm


def terminal_detector_effects(experiment):
    """Return y-bin click effects and the complementary no-click effect.

    The Gaussian x gate lies in [0, 1].  The y masks form a partition, so the
    click effects plus ``no_click`` sum to the identity exactly up to floating
    point roundoff.
    """
    labels, diagonals = _terminal_effect_diagonals(experiment)
    effects = np.array([np.diag(diagonal) for diagonal in diagonals], complex)
    return labels, effects


def _terminal_effect_diagonals(experiment):
    x, _ = _coordinates(experiment)
    gate = np.exp(
        -0.5 * ((x - experiment.detector_x) / experiment.detector_width) ** 2
    ).reshape(-1)
    diagonals = []
    for rows in np.array_split(np.arange(experiment.ny), experiment.y_bins):
        mask = np.zeros((experiment.ny, experiment.nx), dtype=float)
        mask[rows, :] = 1.0
        diagonals.append(mask.reshape(-1) * gate)
    diagonals.append(1.0 - gate)
    labels = tuple(
        [f"y_bin_{index}" for index in range(experiment.y_bins)]
        + ["no_click"]
    )
    return labels, np.asarray(diagonals)


def temporal_delays_and_weights(sigma_t, delay_step, horizon_sigmas):
    """Discretize a normalized half-Gaussian prior over nonnegative delays."""
    values = (float(sigma_t), float(delay_step), float(horizon_sigmas))
    if not np.all(np.isfinite(values)) or min(values) <= 0:
        raise ValueError("Temporal scales must be finite and positive")
    count = max(1, int(np.ceil(horizon_sigmas * sigma_t / delay_step)))
    delays = np.arange(count + 1, dtype=float) * delay_step
    weights = np.exp(-0.5 * (delays / sigma_t) ** 2)
    # Trapezoidal quadrature on [0, horizon].  The common delay_step cancels.
    weights[[0, -1]] *= 0.5
    weights /= np.sum(weights)
    return delays, weights


def _angular_frequencies(experiment):
    dx = experiment.lx / experiment.nx
    dy = experiment.ly / experiment.ny
    kx = 2 * np.pi * np.fft.fftfreq(experiment.nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(experiment.ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    return (experiment.hbar / (2 * experiment.mass)
            * (kx_grid ** 2 + ky_grid ** 2))


def _propagate_states(experiment, states, delay):
    """Apply the unitary free-particle propagator to one or more states."""
    states = np.asarray(states, dtype=complex)
    leading = states.shape[:-1]
    expected = experiment.nx * experiment.ny
    if states.shape[-1] != expected:
        raise ValueError(f"State dimension must be {expected}")
    spatial = states.reshape(leading + (experiment.ny, experiment.nx))
    spectrum = np.fft.fft2(spatial, axes=(-2, -1), norm="ortho")
    phase = np.exp(-1j * _angular_frequencies(experiment) * float(delay))
    propagated = np.fft.ifft2(
        spectrum * phase, axes=(-2, -1), norm="ortho")
    return propagated.reshape(leading + (expected,))


def _unitary_matrix(experiment, delay):
    dimension = experiment.nx * experiment.ny
    # Each row passed to _propagate_states is one input basis ket.  Transpose
    # the returned rows so the resulting matrix acts on column vectors.
    return _propagate_states(experiment, np.eye(dimension), delay).T


def build_spatial_effect_instrument(
        experiment, sigma_t, *, lambda_strength=1.0, component_phases=None):
    """Build the full temporally mixed POVM at the preparation time.

    ``lambda_strength`` is an explicit candidate coupling, mapped to the
    convex response fraction ``1 - exp(-lambda_strength)``.  This mapping is a
    declared model choice, not a value derived from TRF-IT.  At zero the
    returned effects are exactly the unsmeared detector POVM.

    ``component_phases`` exists to test representation invariance.  A global
    phase can be attached to every temporal propagator, but cancels from each
    operator ``U^dagger Pi U``.
    """
    sigma_t, lambda_strength = _validate_sigma_lambda(
        sigma_t, lambda_strength)
    labels, terminal = terminal_detector_effects(experiment)
    delays, weights = temporal_delays_and_weights(
        sigma_t, experiment.delay_step, experiment.horizon_sigmas)
    if lambda_strength == 0:
        return SpatialEffectInstrument(
            labels, terminal.copy(), delays, weights, sigma_t, lambda_strength)

    if component_phases is None:
        phases = np.zeros(delays.size)
    else:
        phases = np.asarray(component_phases, dtype=float)
        if phases.shape != delays.shape or np.any(~np.isfinite(phases)):
            raise ValueError(
                "component_phases must be finite with one value per delay")

    mixed = np.zeros_like(terminal)
    for delay, weight, phase in zip(delays, weights, phases):
        unitary = _unitary_matrix(experiment, delay) * np.exp(1j * phase)
        for index, effect in enumerate(terminal):
            mixed[index] += weight * (unitary.conj().T @ effect @ unitary)
    response_fraction = -np.expm1(-lambda_strength)
    effects = ((1.0 - response_fraction) * terminal
               + response_fraction * mixed)
    # Remove roundoff-level anti-Hermitian parts before eigensystem checks.
    effects = 0.5 * (effects + effects.conj().transpose(0, 2, 1))
    return SpatialEffectInstrument(
        labels, effects, delays, weights, sigma_t, lambda_strength)


def run_spatial_effect_measurement(
        experiment, sigma_t, *, lambda_strength=1.0):
    """Evaluate the complete outcome law without click postselection."""
    sigma_t, lambda_strength = _validate_sigma_lambda(
        sigma_t, lambda_strength)
    labels, diagonals = _terminal_effect_diagonals(experiment)
    state = initial_spatial_state(experiment)
    null_probabilities = diagonals @ np.abs(state) ** 2
    if lambda_strength == 0:
        probabilities = null_probabilities.copy()
    else:
        delays, weights = temporal_delays_and_weights(
            sigma_t, experiment.delay_step, experiment.horizon_sigmas)
        component_probabilities = []
        for delay in delays:
            propagated = _propagate_states(experiment, state, delay)
            component_probabilities.append(diagonals @ np.abs(propagated) ** 2)
        mixed = weights @ np.asarray(component_probabilities)
        response_fraction = -np.expm1(-lambda_strength)
        probabilities = ((1.0 - response_fraction) * null_probabilities
                         + response_fraction * mixed)
    probabilities = np.real_if_close(probabilities).real
    if (np.min(probabilities) < -1e-12
            or abs(float(np.sum(probabilities)) - 1.0) > 1e-12):
        raise RuntimeError("Complete spatial POVM produced an invalid outcome law")
    # Remove only negative floating-point dust; completeness is established
    # above rather than imposed by renormalizing the outcomes.
    probabilities = np.maximum(probabilities, 0.0)
    click_probability = float(np.sum(probabilities[:-1]))
    conditional = (probabilities[:-1] / click_probability
                   if click_probability > 0
                   else np.zeros(experiment.y_bins))
    return SpatialEffectRun(
        sigma_t, lambda_strength, labels, probabilities,
        click_probability, conditional)


def spatial_effect_evolution(
        experiment, sigma_t, *, lambda_strength=1.0):
    """Return every delay component and its cumulative complete outcome law.

    The frame index labels unresolved alternatives in the temporal mixture; it
    is not a stochastic trajectory of one particle.  The last cumulative law
    equals :func:`run_spatial_effect_measurement` for the same parameters.
    """
    sigma_t, lambda_strength = _validate_sigma_lambda(
        sigma_t, lambda_strength)
    x_grid, y_grid = _coordinates(experiment)
    labels, diagonals = _terminal_effect_diagonals(experiment)
    state = initial_spatial_state(experiment)
    null_probabilities = diagonals @ np.abs(state) ** 2
    delays, weights = temporal_delays_and_weights(
        sigma_t, experiment.delay_step, experiment.horizon_sigmas)

    densities = []
    components = []
    for delay in delays:
        propagated = _propagate_states(experiment, state, delay)
        density = np.abs(propagated.reshape(
            experiment.ny, experiment.nx)) ** 2
        densities.append(density)
        components.append(diagonals @ density.reshape(-1))
    densities = np.asarray(densities)
    components = np.asarray(components)

    weighted = np.cumsum(weights[:, None] * components, axis=0)
    cumulative_weight = np.cumsum(weights)
    temporal_laws = weighted / cumulative_weight[:, None]
    response_fraction = -np.expm1(-lambda_strength)
    cumulative = ((1.0 - response_fraction) * null_probabilities[None, :]
                  + response_fraction * temporal_laws)
    if (np.min(cumulative) < -1e-12
            or np.max(np.abs(np.sum(cumulative, axis=1) - 1.0)) > 1e-12):
        raise RuntimeError("Spatial effect frames produced an invalid outcome law")

    detector_gate = np.exp(
        -0.5 * ((x_grid[0] - experiment.detector_x)
                / experiment.detector_width) ** 2)
    return SpatialEffectEvolution(
        sigma_t=sigma_t,
        lambda_strength=lambda_strength,
        x=x_grid[0].copy(),
        y=y_grid[:, 0].copy(),
        detector_gate=detector_gate,
        labels=labels,
        delays=delays,
        delay_weights=weights,
        densities=densities,
        component_probabilities=components,
        cumulative_probabilities=np.maximum(cumulative, 0.0),
    )


def _kl_divergence(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    supported = observed > 0
    value = np.sum(observed[supported] * np.log(
        observed[supported] / np.maximum(predicted[supported], 1e-300)))
    return max(0.0, float(value))


def fit_spatial_effect_response(
        experiment, true_sigma_t, candidate_sigmas, *, lambda_strength=1.0,
        shots=100_000):
    """Profile a synthetic complete detector law over candidate widths."""
    candidates = np.asarray(candidate_sigmas, dtype=float)
    if (candidates.ndim != 1 or candidates.size < 2
            or np.any(~np.isfinite(candidates)) or np.any(candidates <= 0)):
        raise ValueError("candidate_sigmas must contain at least two positive values")
    if not isinstance(shots, (int, np.integer)) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    _, lambda_strength = _validate_sigma_lambda(true_sigma_t, lambda_strength)
    target = run_spatial_effect_measurement(
        experiment, true_sigma_t, lambda_strength=lambda_strength)
    profiles = [run_spatial_effect_measurement(
        experiment, sigma, lambda_strength=lambda_strength)
        for sigma in candidates]
    kl = np.array([
        _kl_divergence(target.probabilities, profile.probabilities)
        for profile in profiles
    ])
    null = run_spatial_effect_measurement(
        experiment, true_sigma_t, lambda_strength=0)
    other_null = run_spatial_effect_measurement(
        experiment, candidates[-1], lambda_strength=0)
    return SpatialEffectProfile(
        true_sigma_t=float(true_sigma_t),
        lambda_strength=lambda_strength,
        candidate_sigmas=candidates,
        kl_divergence=kl,
        expected_deviance=2 * shots * kl,
        best_sigma_t=float(candidates[int(np.argmin(kl))]),
        target_probabilities=target.probabilities,
        null_probabilities=null.probabilities,
        signal_total_variation=0.5 * float(np.sum(np.abs(
            target.probabilities - null.probabilities))),
        lambda_zero_max_error=float(np.max(np.abs(
            null.probabilities - other_null.probabilities))),
    )


def _refined_size(size, divisor=1):
    refined = max(size + 2, int(np.ceil(1.25 * size)))
    remainder = refined % divisor
    return refined if remainder == 0 else refined + divisor - remainder


def spatial_effect_convergence(
        experiment, sigma_t, *, lambda_strength=1.0):
    """Return complete-law TV changes under quadrature and grid refinement."""
    reference = run_spatial_effect_measurement(
        experiment, sigma_t, lambda_strength=lambda_strength)
    variants = {
        "delay_step_half": replace(
            experiment, delay_step=experiment.delay_step / 2),
        "grid_5_over_4": replace(
            experiment,
            nx=_refined_size(experiment.nx),
            ny=_refined_size(experiment.ny, experiment.y_bins)),
        "horizon_plus_one_sigma": replace(
            experiment, horizon_sigmas=experiment.horizon_sigmas + 1),
    }
    return {
        name: 0.5 * float(np.sum(np.abs(
            reference.probabilities - run_spatial_effect_measurement(
                variant, sigma_t,
                lambda_strength=lambda_strength).probabilities)))
        for name, variant in variants.items()
    }
