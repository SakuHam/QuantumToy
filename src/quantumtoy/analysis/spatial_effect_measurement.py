"""Complete spatial POVM reference for a temporally thick detector effect.

This module deliberately stays separate from the nonlinear exploratory
``ThickFrontMeasurementGuidedTheory``.  It provides a small, exactly unitary
periodic lattice on which every detector outcome is represented by a full
operator.  A temporal width mixes Heisenberg-picture effects arithmetically,
without normalizing any field by a candidate-dependent maximum.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

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
    reference_time: float = 0.0
    potential_mode: str = "free"
    propagation_step: float = 0.01
    barrier_x: float = -0.5
    barrier_width: float = 0.16
    barrier_height: float = 35.0
    slit_offset: float = 0.8
    slit_half_height: float = 0.3
    slit_edge_smooth: float = 0.08

    def __post_init__(self):
        positive = (
            self.lx, self.ly, self.mass, self.hbar, self.detector_width,
            self.packet_sigma_x, self.packet_sigma_y, self.delay_step,
            self.horizon_sigmas, self.propagation_step, self.barrier_width,
            self.barrier_height, self.slit_offset, self.slit_half_height,
            self.slit_edge_smooth,
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
            self.packet_kx, self.packet_ky, self.reference_time,
            self.barrier_x,
        )
        if not np.all(np.isfinite(finite)):
            raise ValueError("Spatial effect locations and momenta must be finite")
        if self.reference_time < 0:
            raise ValueError("reference_time must be nonnegative")
        if self.potential_mode not in {"free", "double_slit"}:
            raise ValueError("potential_mode must be 'free' or 'double_slit'")


def double_slit_effect_experiment(*, nx=80, ny=80):
    """Return the locked geometry used by the double-slit robustness study."""
    return SpatialEffectExperiment(
        nx=nx,
        ny=ny,
        lx=10.0,
        ly=8.0,
        y_bins=min(16, ny),
        detector_x=1.5,
        detector_width=0.3,
        packet_x=-2.5,
        packet_y=0.0,
        packet_sigma_x=0.45,
        packet_sigma_y=0.65,
        packet_kx=3.0,
        packet_ky=0.0,
        delay_step=0.025,
        horizon_sigmas=4.0,
        reference_time=0.9,
        potential_mode="double_slit",
        propagation_step=0.005,
        barrier_x=-0.5,
        barrier_width=0.16,
        barrier_height=35.0,
        slit_offset=0.8,
        slit_half_height=0.3,
        slit_edge_smooth=0.08,
    )


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


@dataclass(frozen=True)
class SpatialEffectJointProfile:
    """Two-parameter expected likelihood surface and local identifiability."""

    true_sigma_t: float
    true_lambda_strength: float
    candidate_sigmas: np.ndarray
    candidate_lambdas: np.ndarray
    kl_divergence: np.ndarray
    expected_deviance: np.ndarray
    best_sigma_t: float
    best_lambda_strength: float
    profiled_lambda_strength: np.ndarray
    profiled_expected_deviance: np.ndarray
    fisher_information: np.ndarray
    fisher_eigenvalues: np.ndarray
    fisher_condition_number: float
    fisher_score_correlation: float
    local_covariance: np.ndarray
    local_standard_errors: np.ndarray
    local_parameter_correlation: float


@dataclass(frozen=True)
class SpatialEffectSecondSettingDesign:
    """Fisher design scan for a complementary detector location."""

    candidate_detector_x: np.ndarray
    combined_determinant: np.ndarray
    combined_condition_number: np.ndarray
    combined_parameter_correlation: np.ndarray
    best_detector_x: float
    first_setting_fraction: float


@dataclass(frozen=True)
class SpatialEffectDetectorDesign:
    """Fisher design scan over a detector plane, gate width, and readout time."""

    candidate_detector_x: np.ndarray
    candidate_detector_width: np.ndarray
    candidate_reference_time: np.ndarray
    combined_determinant: np.ndarray
    combined_condition_number: np.ndarray
    combined_parameter_correlation: np.ndarray
    combined_standard_errors: np.ndarray
    improves_both_standard_errors: np.ndarray
    best_index: int
    best_detector_x: float
    best_detector_width: float
    best_reference_time: float
    first_setting_fraction: float


@dataclass(frozen=True)
class SpatialEffectMultiProfile:
    """Joint profile from independent settings sharing sigma and lambda."""

    setting_names: tuple[str, ...]
    total_shots: int
    shot_fractions: np.ndarray
    true_sigma_t: float
    true_lambda_strength: float
    candidate_sigmas: np.ndarray
    candidate_lambdas: np.ndarray
    per_setting_kl_divergence: np.ndarray
    combined_kl_divergence: np.ndarray
    expected_deviance: np.ndarray
    best_sigma_t: float
    best_lambda_strength: float
    profiled_lambda_strength: np.ndarray
    profiled_expected_deviance: np.ndarray
    per_setting_fisher_information: np.ndarray
    fisher_information: np.ndarray
    fisher_eigenvalues: np.ndarray
    fisher_condition_number: float
    local_standard_errors: np.ndarray
    local_parameter_correlation: float


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


@lru_cache(maxsize=32)
def _cached_spatial_effect_potential(experiment):
    x, y = _coordinates(experiment)
    if experiment.potential_mode == "free":
        return np.zeros_like(x)

    def smooth_window(center):
        lower = center - experiment.slit_half_height
        upper = center + experiment.slit_half_height
        return 0.5 * (
            np.tanh((y - lower) / experiment.slit_edge_smooth)
            - np.tanh((y - upper) / experiment.slit_edge_smooth))

    openings = np.clip(
        smooth_window(experiment.slit_offset)
        + smooth_window(-experiment.slit_offset), 0.0, 1.0)
    transverse_wall = np.exp(
        -0.5 * ((x - experiment.barrier_x) / experiment.barrier_width) ** 2)
    return experiment.barrier_height * transverse_wall * (1.0 - openings)


def spatial_effect_potential(experiment):
    """Return a copy of the declared real potential on the spatial lattice."""
    return _cached_spatial_effect_potential(experiment).copy()


def _split_step(experiment, states, step):
    leading = states.shape[:-1]
    spatial = states.reshape(leading + (experiment.ny, experiment.nx))
    potential_phase = np.exp(
        -0.5j * _cached_spatial_effect_potential(experiment)
        * step / experiment.hbar)
    spatial = spatial * potential_phase
    spectrum = np.fft.fft2(spatial, axes=(-2, -1), norm="ortho")
    kinetic_phase = np.exp(-1j * _angular_frequencies(experiment) * step)
    spatial = np.fft.ifft2(
        spectrum * kinetic_phase, axes=(-2, -1), norm="ortho")
    spatial = spatial * potential_phase
    return spatial.reshape(states.shape)


def _propagate_states(experiment, states, delay):
    """Apply the declared unitary propagator to one or more states."""
    states = np.asarray(states, dtype=complex)
    leading = states.shape[:-1]
    expected = experiment.nx * experiment.ny
    if states.shape[-1] != expected:
        raise ValueError(f"State dimension must be {expected}")
    delay = float(delay)
    if not np.isfinite(delay) or delay < 0:
        raise ValueError("Propagation delay must be finite and nonnegative")
    if experiment.potential_mode == "double_slit":
        propagated = states.copy()
        full_steps = int(np.floor(delay / experiment.propagation_step + 1e-12))
        for _ in range(full_steps):
            propagated = _split_step(
                experiment, propagated, experiment.propagation_step)
        remainder = delay - full_steps * experiment.propagation_step
        if remainder > 1e-14:
            propagated = _split_step(experiment, propagated, remainder)
        return propagated
    spatial = states.reshape(leading + (experiment.ny, experiment.nx))
    spectrum = np.fft.fft2(spatial, axes=(-2, -1), norm="ortho")
    phase = np.exp(-1j * _angular_frequencies(experiment) * delay)
    propagated = np.fft.ifft2(
        spectrum * phase, axes=(-2, -1), norm="ortho")
    return propagated.reshape(leading + (expected,))


def _propagate_sample_times(experiment, state, times):
    """Propagate sequentially to increasing absolute sample times."""
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or times.size == 0 or np.any(~np.isfinite(times))
            or np.any(times < 0) or np.any(np.diff(times) < 0)):
        raise ValueError("Sample times must be finite, nonnegative and ordered")
    current = np.asarray(state, dtype=complex)
    previous = 0.0
    samples = []
    for time in times:
        current = _propagate_states(experiment, current, float(time - previous))
        samples.append(current.copy())
        previous = float(time)
    return np.asarray(samples)


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
    reference_unitary = _unitary_matrix(experiment, experiment.reference_time)
    reference = np.asarray([
        reference_unitary.conj().T @ effect @ reference_unitary
        for effect in terminal
    ])
    if lambda_strength == 0:
        return SpatialEffectInstrument(
            labels, reference, delays, weights, sigma_t, lambda_strength)

    if component_phases is None:
        phases = np.zeros(delays.size)
    else:
        phases = np.asarray(component_phases, dtype=float)
        if phases.shape != delays.shape or np.any(~np.isfinite(phases)):
            raise ValueError(
                "component_phases must be finite with one value per delay")

    mixed = np.zeros_like(terminal)
    for delay, weight, phase in zip(delays, weights, phases):
        unitary = _unitary_matrix(
            experiment, experiment.reference_time + delay) * np.exp(1j * phase)
        for index, effect in enumerate(terminal):
            mixed[index] += weight * (unitary.conj().T @ effect @ unitary)
    response_fraction = -np.expm1(-lambda_strength)
    effects = ((1.0 - response_fraction) * reference
               + response_fraction * mixed)
    # Remove roundoff-level anti-Hermitian parts before eigensystem checks.
    effects = 0.5 * (effects + effects.conj().transpose(0, 2, 1))
    return SpatialEffectInstrument(
        labels, effects, delays, weights, sigma_t, lambda_strength)


@lru_cache(maxsize=64)
def _component_probability_curve(experiment, delay_count):
    labels, diagonals = _terminal_effect_diagonals(experiment)
    state = initial_spatial_state(experiment)
    delays = np.arange(delay_count + 1, dtype=float) * experiment.delay_step
    times = experiment.reference_time + delays
    states = _propagate_sample_times(experiment, state, times)
    components = np.asarray([
        diagonals @ np.abs(component) ** 2 for component in states
    ])
    return labels, components


def _null_and_temporal_probabilities(experiment, sigma_t):
    sigma_t, _ = _validate_sigma_lambda(sigma_t, 0)
    delays, weights = temporal_delays_and_weights(
        sigma_t, experiment.delay_step, experiment.horizon_sigmas)
    labels, components = _component_probability_curve(
        experiment, delays.size - 1)
    null_probabilities = components[0]
    temporal_probabilities = weights @ components
    return labels, null_probabilities, temporal_probabilities


def _coupled_probabilities(null_probabilities, temporal_probabilities,
                           lambda_strength):
    _, lambda_strength = _validate_sigma_lambda(1, lambda_strength)
    response_fraction = -np.expm1(-lambda_strength)
    probabilities = ((1.0 - response_fraction) * null_probabilities
                     + response_fraction * temporal_probabilities)
    probabilities = np.real_if_close(probabilities).real
    if (np.min(probabilities) < -1e-12
            or abs(float(np.sum(probabilities)) - 1.0) > 1e-12):
        raise RuntimeError("Complete spatial POVM produced an invalid outcome law")
    # Remove only negative floating-point dust; completeness is established
    # above rather than imposed by renormalizing the outcomes.
    return np.maximum(probabilities, 0.0)


def run_spatial_effect_measurement(
        experiment, sigma_t, *, lambda_strength=1.0):
    """Evaluate the complete outcome law without click postselection."""
    sigma_t, lambda_strength = _validate_sigma_lambda(
        sigma_t, lambda_strength)
    labels, null_probabilities, temporal_probabilities = (
        _null_and_temporal_probabilities(experiment, sigma_t))
    probabilities = _coupled_probabilities(
        null_probabilities, temporal_probabilities, lambda_strength)
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
    delays, weights = temporal_delays_and_weights(
        sigma_t, experiment.delay_step, experiment.horizon_sigmas)
    states = _propagate_sample_times(
        experiment, state, experiment.reference_time + delays)
    densities = np.abs(states.reshape(
        (-1, experiment.ny, experiment.nx))) ** 2
    components = np.asarray([
        diagonals @ density.reshape(-1) for density in densities
    ])
    null_probabilities = components[0]

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


def spatial_effect_fisher_information(
        experiment, sigma_t, lambda_strength, *, shots=1,
        relative_step=1e-3):
    """Return the multinomial Fisher matrix for ``(sigma_t, lambda)``.

    The derivatives are evaluated symmetrically inside the positive parameter
    domain.  At ``lambda=0`` the sigma derivative vanishes, correctly making
    the local information singular.
    """
    sigma_t, lambda_strength = _validate_sigma_lambda(
        sigma_t, lambda_strength)
    if not isinstance(shots, (int, np.integer)) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    relative_step = float(relative_step)
    if not np.isfinite(relative_step) or relative_step <= 0:
        raise ValueError("relative_step must be finite and positive")

    sigma_step = min(relative_step * sigma_t, 0.25 * sigma_t)
    lambda_step = relative_step * max(1.0, lambda_strength)
    sigma_minus = run_spatial_effect_measurement(
        experiment, sigma_t - sigma_step,
        lambda_strength=lambda_strength).probabilities
    sigma_plus = run_spatial_effect_measurement(
        experiment, sigma_t + sigma_step,
        lambda_strength=lambda_strength).probabilities
    derivative_sigma = (sigma_plus - sigma_minus) / (2 * sigma_step)

    if lambda_strength > lambda_step:
        lambda_minus = run_spatial_effect_measurement(
            experiment, sigma_t,
            lambda_strength=lambda_strength - lambda_step).probabilities
        lambda_plus = run_spatial_effect_measurement(
            experiment, sigma_t,
            lambda_strength=lambda_strength + lambda_step).probabilities
        derivative_lambda = (lambda_plus - lambda_minus) / (2 * lambda_step)
    else:
        central = run_spatial_effect_measurement(
            experiment, sigma_t,
            lambda_strength=lambda_strength).probabilities
        lambda_plus = run_spatial_effect_measurement(
            experiment, sigma_t,
            lambda_strength=lambda_strength + lambda_step).probabilities
        derivative_lambda = (lambda_plus - central) / lambda_step

    probabilities = run_spatial_effect_measurement(
        experiment, sigma_t,
        lambda_strength=lambda_strength).probabilities
    derivatives = np.stack([derivative_sigma, derivative_lambda], axis=1)
    fisher = shots * (derivatives.T / np.maximum(probabilities, 1e-300)) @ derivatives
    return 0.5 * (fisher + fisher.T)


def _fisher_diagnostics(fisher):
    fisher = np.asarray(fisher, dtype=float)
    eigenvalues = np.linalg.eigvalsh(fisher)
    threshold = 1e-12 * max(1.0, eigenvalues[-1])
    positive = eigenvalues[eigenvalues > threshold]
    condition = (float(eigenvalues[-1] / positive[0])
                 if positive.size == 2 else float("inf"))
    score_denominator = np.sqrt(max(0.0, fisher[0, 0] * fisher[1, 1]))
    score_correlation = (float(fisher[0, 1] / score_denominator)
                         if score_denominator > 0 else float("nan"))
    covariance = np.linalg.pinv(fisher, hermitian=True)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    covariance_denominator = standard_errors[0] * standard_errors[1]
    parameter_correlation = (
        float(covariance[0, 1] / covariance_denominator)
        if covariance_denominator > 0 else float("nan"))
    return (
        eigenvalues, condition, score_correlation, covariance,
        standard_errors, parameter_correlation,
    )


def fit_spatial_effect_joint_response(
        experiment, true_sigma_t, true_lambda_strength, candidate_sigmas,
        candidate_lambdas, *, shots=100_000):
    """Profile the complete law jointly over temporal width and coupling."""
    true_sigma_t, true_lambda_strength = _validate_sigma_lambda(
        true_sigma_t, true_lambda_strength)
    sigmas = np.asarray(candidate_sigmas, dtype=float)
    lambdas = np.asarray(candidate_lambdas, dtype=float)
    if (sigmas.ndim != 1 or sigmas.size < 2
            or np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0)):
        raise ValueError("candidate_sigmas must contain at least two positive values")
    if (lambdas.ndim != 1 or lambdas.size < 2
            or np.any(~np.isfinite(lambdas)) or np.any(lambdas < 0)):
        raise ValueError(
            "candidate_lambdas must contain at least two nonnegative values")
    if not isinstance(shots, (int, np.integer)) or shots <= 0:
        raise ValueError("shots must be a positive integer")

    target = run_spatial_effect_measurement(
        experiment, true_sigma_t,
        lambda_strength=true_lambda_strength).probabilities
    kl = np.empty((lambdas.size, sigmas.size), dtype=float)
    supported = target > 0
    response_fractions = -np.expm1(-lambdas)
    for sigma_index, sigma_t in enumerate(sigmas):
        _, null, temporal = _null_and_temporal_probabilities(
            experiment, sigma_t)
        predictions = (
            (1.0 - response_fractions[:, None]) * null[None, :]
            + response_fractions[:, None] * temporal[None, :])
        values = np.sum(
            target[None, supported] * np.log(
                target[None, supported]
                / np.maximum(predictions[:, supported], 1e-300)),
            axis=1)
        kl[:, sigma_index] = np.maximum(values, 0.0)

    expected_deviance = 2 * shots * kl
    best_lambda_index, best_sigma_index = np.unravel_index(
        int(np.argmin(expected_deviance)), expected_deviance.shape)
    profiled_lambda_indices = np.argmin(expected_deviance, axis=0)
    fisher = spatial_effect_fisher_information(
        experiment, true_sigma_t, true_lambda_strength, shots=shots)
    (eigenvalues, condition, score_correlation, covariance,
     standard_errors, parameter_correlation) = _fisher_diagnostics(fisher)
    return SpatialEffectJointProfile(
        true_sigma_t=true_sigma_t,
        true_lambda_strength=true_lambda_strength,
        candidate_sigmas=sigmas,
        candidate_lambdas=lambdas,
        kl_divergence=kl,
        expected_deviance=expected_deviance,
        best_sigma_t=float(sigmas[best_sigma_index]),
        best_lambda_strength=float(lambdas[best_lambda_index]),
        profiled_lambda_strength=lambdas[profiled_lambda_indices],
        profiled_expected_deviance=expected_deviance[
            profiled_lambda_indices, np.arange(sigmas.size)],
        fisher_information=fisher,
        fisher_eigenvalues=eigenvalues,
        fisher_condition_number=condition,
        fisher_score_correlation=score_correlation,
        local_covariance=covariance,
        local_standard_errors=standard_errors,
        local_parameter_correlation=parameter_correlation,
    )


def select_complementary_detector_setting(
        experiment, sigma_t, lambda_strength, candidate_detector_x, *,
        first_setting_fraction=0.5):
    """Select a second detector location by combined Fisher determinant.

    The scan changes only ``detector_x``.  Its objective is the determinant of
    the per-shot Fisher matrix for the declared allocation between the original
    and candidate settings.
    """
    sigma_t, lambda_strength = _validate_sigma_lambda(
        sigma_t, lambda_strength)
    candidates = np.asarray(candidate_detector_x, dtype=float)
    if (candidates.ndim != 1 or candidates.size < 2
            or np.any(~np.isfinite(candidates))):
        raise ValueError(
            "candidate_detector_x must contain at least two finite values")
    first_setting_fraction = float(first_setting_fraction)
    if (not np.isfinite(first_setting_fraction)
            or not 0 < first_setting_fraction < 1):
        raise ValueError("first_setting_fraction must lie strictly between 0 and 1")

    first_fisher = spatial_effect_fisher_information(
        experiment, sigma_t, lambda_strength)
    determinants = []
    conditions = []
    correlations = []
    for detector_x in candidates:
        second = replace(experiment, detector_x=float(detector_x))
        second_fisher = spatial_effect_fisher_information(
            second, sigma_t, lambda_strength)
        combined = (first_setting_fraction * first_fisher
                    + (1.0 - first_setting_fraction) * second_fisher)
        (_, condition, _, _, _, correlation) = _fisher_diagnostics(combined)
        determinants.append(float(np.linalg.det(combined)))
        conditions.append(condition)
        correlations.append(correlation)
    determinants = np.asarray(determinants)
    best_index = int(np.argmax(determinants))
    return SpatialEffectSecondSettingDesign(
        candidate_detector_x=candidates,
        combined_determinant=determinants,
        combined_condition_number=np.asarray(conditions),
        combined_parameter_correlation=np.asarray(correlations),
        best_detector_x=float(candidates[best_index]),
        first_setting_fraction=first_setting_fraction,
    )


def select_double_slit_detector_setting(
        experiment, sigma_t, lambda_strength, candidate_detector_x,
        candidate_detector_width, candidate_time_offset, *,
        first_setting_fraction=0.5):
    """Select a complementary double-slit detector by constrained D-optimality.

    Candidate readout times are specified relative to the packet's classical
    arrival time at each detector plane.  The same total shot budget is split
    between the original and candidate settings.  A candidate is eligible only
    if both marginal standard errors are smaller than for the original setting
    alone; the eligible candidate with the largest Fisher determinant wins.
    """
    if experiment.potential_mode != "double_slit":
        raise ValueError("double-slit detector design requires double_slit mode")
    sigma_t, lambda_strength = _validate_sigma_lambda(
        sigma_t, lambda_strength)
    positions = np.asarray(candidate_detector_x, dtype=float)
    widths = np.asarray(candidate_detector_width, dtype=float)
    offsets = np.asarray(candidate_time_offset, dtype=float)
    if (positions.ndim != 1 or positions.size == 0
            or np.any(~np.isfinite(positions))):
        raise ValueError("candidate_detector_x must contain finite values")
    if (widths.ndim != 1 or widths.size == 0
            or np.any(~np.isfinite(widths)) or np.any(widths <= 0)):
        raise ValueError(
            "candidate_detector_width must contain positive finite values")
    if (offsets.ndim != 1 or offsets.size == 0
            or np.any(~np.isfinite(offsets))):
        raise ValueError("candidate_time_offset must contain finite values")
    first_setting_fraction = float(first_setting_fraction)
    if (not np.isfinite(first_setting_fraction)
            or not 0 < first_setting_fraction < 1):
        raise ValueError("first_setting_fraction must lie strictly between 0 and 1")
    velocity = experiment.hbar * experiment.packet_kx / experiment.mass
    if velocity <= 0:
        raise ValueError("double-slit detector design requires packet_kx > 0")

    first_fisher = spatial_effect_fisher_information(
        experiment, sigma_t, lambda_strength)
    first_errors = _fisher_diagnostics(first_fisher)[4]
    candidate_rows = []
    determinants = []
    conditions = []
    correlations = []
    standard_errors = []
    improves_both = []
    for detector_x in positions:
        arrival_time = (detector_x - experiment.packet_x) / velocity
        for detector_width in widths:
            for offset in offsets:
                reference_time = arrival_time + offset
                if reference_time <= 0:
                    continue
                second = replace(
                    experiment,
                    detector_x=float(detector_x),
                    detector_width=float(detector_width),
                    reference_time=float(reference_time),
                )
                second_fisher = spatial_effect_fisher_information(
                    second, sigma_t, lambda_strength)
                combined = (
                    first_setting_fraction * first_fisher
                    + (1.0 - first_setting_fraction) * second_fisher
                )
                (_, condition, _, _, errors,
                 correlation) = _fisher_diagnostics(combined)
                candidate_rows.append(
                    (detector_x, detector_width, reference_time))
                determinants.append(float(np.linalg.det(combined)))
                conditions.append(condition)
                correlations.append(correlation)
                standard_errors.append(errors)
                improves_both.append(bool(np.all(errors < first_errors)))

    if not candidate_rows:
        raise ValueError("candidate grid contains no positive reference times")
    determinants = np.asarray(determinants)
    improves_both = np.asarray(improves_both, dtype=bool)
    if not np.any(improves_both):
        raise ValueError(
            "no candidate improves both standard errors at the declared allocation")
    eligible_objective = np.where(improves_both, determinants, -np.inf)
    best_index = int(np.argmax(eligible_objective))
    candidate_rows = np.asarray(candidate_rows, dtype=float)
    best = candidate_rows[best_index]
    return SpatialEffectDetectorDesign(
        candidate_detector_x=candidate_rows[:, 0],
        candidate_detector_width=candidate_rows[:, 1],
        candidate_reference_time=candidate_rows[:, 2],
        combined_determinant=determinants,
        combined_condition_number=np.asarray(conditions),
        combined_parameter_correlation=np.asarray(correlations),
        combined_standard_errors=np.asarray(standard_errors),
        improves_both_standard_errors=improves_both,
        best_index=best_index,
        best_detector_x=float(best[0]),
        best_detector_width=float(best[1]),
        best_reference_time=float(best[2]),
        first_setting_fraction=first_setting_fraction,
    )


def fit_spatial_effect_multi_response(
        experiments, true_sigma_t, true_lambda_strength, candidate_sigmas,
        candidate_lambdas, *, shots=100_000, shot_fractions=None,
        setting_names=None):
    """Combine independent complete laws with shared sigma and lambda."""
    experiments = tuple(experiments)
    if len(experiments) < 2:
        raise ValueError("At least two independent measurement settings are required")
    true_sigma_t, true_lambda_strength = _validate_sigma_lambda(
        true_sigma_t, true_lambda_strength)
    sigmas = np.asarray(candidate_sigmas, dtype=float)
    lambdas = np.asarray(candidate_lambdas, dtype=float)
    if (sigmas.ndim != 1 or sigmas.size < 2
            or np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0)):
        raise ValueError("candidate_sigmas must contain at least two positive values")
    if (lambdas.ndim != 1 or lambdas.size < 2
            or np.any(~np.isfinite(lambdas)) or np.any(lambdas < 0)):
        raise ValueError(
            "candidate_lambdas must contain at least two nonnegative values")
    if not isinstance(shots, (int, np.integer)) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    if shot_fractions is None:
        fractions = np.full(len(experiments), 1.0 / len(experiments))
    else:
        fractions = np.asarray(shot_fractions, dtype=float)
    if (fractions.shape != (len(experiments),)
            or np.any(~np.isfinite(fractions)) or np.any(fractions <= 0)
            or not np.isclose(np.sum(fractions), 1.0, atol=1e-12)):
        raise ValueError("shot_fractions must be positive and sum to one")
    if setting_names is None:
        names = tuple(f"setting_{index + 1}" for index in range(len(experiments)))
    else:
        names = tuple(str(name) for name in setting_names)
        if len(names) != len(experiments) or any(not name for name in names):
            raise ValueError("setting_names must name every measurement setting")

    response_fractions = -np.expm1(-lambdas)
    per_setting_kl = []
    per_setting_fisher = []
    for experiment in experiments:
        target = run_spatial_effect_measurement(
            experiment, true_sigma_t,
            lambda_strength=true_lambda_strength).probabilities
        supported = target > 0
        kl = np.empty((lambdas.size, sigmas.size), dtype=float)
        for sigma_index, sigma_t in enumerate(sigmas):
            _, null, temporal = _null_and_temporal_probabilities(
                experiment, sigma_t)
            predictions = (
                (1.0 - response_fractions[:, None]) * null[None, :]
                + response_fractions[:, None] * temporal[None, :])
            values = np.sum(
                target[None, supported] * np.log(
                    target[None, supported]
                    / np.maximum(predictions[:, supported], 1e-300)),
                axis=1)
            kl[:, sigma_index] = np.maximum(values, 0.0)
        per_setting_kl.append(kl)
        per_setting_fisher.append(spatial_effect_fisher_information(
            experiment, true_sigma_t, true_lambda_strength))
    per_setting_kl = np.asarray(per_setting_kl)
    per_setting_fisher = np.asarray(per_setting_fisher)
    combined_kl = np.tensordot(fractions, per_setting_kl, axes=(0, 0))
    expected_deviance = 2 * shots * combined_kl
    best_lambda_index, best_sigma_index = np.unravel_index(
        int(np.argmin(expected_deviance)), expected_deviance.shape)
    profiled_lambda_indices = np.argmin(expected_deviance, axis=0)
    fisher = shots * np.tensordot(
        fractions, per_setting_fisher, axes=(0, 0))
    (eigenvalues, condition, _, _, standard_errors,
     parameter_correlation) = _fisher_diagnostics(fisher)
    return SpatialEffectMultiProfile(
        setting_names=names,
        total_shots=int(shots),
        shot_fractions=fractions,
        true_sigma_t=true_sigma_t,
        true_lambda_strength=true_lambda_strength,
        candidate_sigmas=sigmas,
        candidate_lambdas=lambdas,
        per_setting_kl_divergence=per_setting_kl,
        combined_kl_divergence=combined_kl,
        expected_deviance=expected_deviance,
        best_sigma_t=float(sigmas[best_sigma_index]),
        best_lambda_strength=float(lambdas[best_lambda_index]),
        profiled_lambda_strength=lambdas[profiled_lambda_indices],
        profiled_expected_deviance=expected_deviance[
            profiled_lambda_indices, np.arange(sigmas.size)],
        per_setting_fisher_information=per_setting_fisher,
        fisher_information=fisher,
        fisher_eigenvalues=eigenvalues,
        fisher_condition_number=condition,
        local_standard_errors=standard_errors,
        local_parameter_correlation=parameter_correlation,
    )


def _refined_size(size, divisor=1):
    refined = max(size + 2, int(np.ceil(1.25 * size)))
    remainder = refined % divisor
    return refined if remainder == 0 else refined + divisor - remainder


def spatial_effect_convergence(
        experiment, sigma_t, *, lambda_strength=1.0):
    """Return complete-law TV changes under temporal and grid refinement."""
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
    if experiment.potential_mode == "double_slit":
        variants.update({
            "propagation_step_half": replace(
                experiment, propagation_step=experiment.propagation_step / 2),
            "slit_edge_half_sensitivity": replace(
                experiment, slit_edge_smooth=experiment.slit_edge_smooth / 2),
            "x_box_5_over_4": replace(
                experiment, lx=1.25 * experiment.lx,
                nx=_refined_size(experiment.nx)),
        })
    return {
        name: 0.5 * float(np.sum(np.abs(
            reference.probabilities - run_spatial_effect_measurement(
                variant, sigma_t,
                lambda_strength=lambda_strength).probabilities)))
        for name, variant in variants.items()
    }
