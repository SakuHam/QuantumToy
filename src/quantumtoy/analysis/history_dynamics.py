"""Causal single-selection instrument, absorbing detector and record erasure.

This is an explicit candidate model, not a derived universal evolution law.
Unitary split-step propagation is interrupted at most once by a complete
upper/lower-half-plane projective instrument. First detection is an absorbing
quantum jump, not a terminal snapshot. All paths, detector misses and erased
records are retained. Branch kets are unnormalized; their outer-product sum
is the surviving ensemble density operator.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from analysis.spatial_effect_measurement import (
    _coordinates, _propagate_states, initial_spatial_state,
)
from analysis.spatial_detector_inference import DetectorResponse, detector_blur_matrix
from analysis.temporal_history_profile import (
    TemporalHistoryEnvelope, record_survival, selection_time_weights,
)


@dataclass(frozen=True)
class HistoryDynamicsConfig:
    duration: float = 2.0
    steps: int = 100
    absorption_rate: float = 4.0
    pointer_y: float = 0.0

    def __post_init__(self):
        if (not np.all(np.isfinite([self.duration, self.absorption_rate, self.pointer_y]))
                or self.duration <= 0 or self.absorption_rate < 0):
            raise ValueError("Invalid duration, detector rate or pointer boundary")
        if not isinstance(self.steps, (int, np.integer)) or self.steps < 1:
            raise ValueError("steps must be a positive integer")


@dataclass
class HistoryDynamicsBasis:
    """Instrument for each possible selection interval, plus the quantum null.

    Case i<steps has one projective selection at that interval's midpoint.
    Case steps has no selection. Click entries are unconditioned probabilities
    per preparation, with detection time represented by interval midpoint.
    """
    experiment: object
    config: HistoryDynamicsConfig
    edges: np.ndarray
    click_times: np.ndarray
    clicks: np.ndarray  # [case, time, y-bin]
    final_kets: np.ndarray  # [case, history branch, lattice site]


@dataclass
class HistoryDynamicsResult:
    weights: np.ndarray
    density: np.ndarray
    ideal_joint: np.ndarray
    surviving_probability: float
    detected_probability: float
    accessible_selection: float
    erased_selection: float
    selected_probability: float
    # Selection clock probabilities include paths absorbed before their
    # scheduled selection; selected_probability instead counts actual jumps.


@dataclass
class HistoryDetectorLaw:
    joint: np.ndarray
    erased_record: float
    no_record: float
    before_erasure_joint: np.ndarray

    @property
    def probabilities(self):
        return np.r_[self.joint.ravel(), self.erased_record, self.no_record]


def detector_step_diagonals(experiment, config, dt):
    """K_no=exp(-ΓD dt/2); sum_j K_j†K_j=1-exp(-ΓD dt).

    Click jumps go to orthogonal absorbing record states. A detector miss is
    physical survival, not a renormalization of the remaining wavefunction.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    x, _ = _coordinates(experiment)
    gate = np.exp(-0.5 * ((x - experiment.detector_x) / experiment.detector_width)**2)
    rate_dt = config.absorption_rate * gate.ravel() * dt
    no_click = np.exp(-0.5 * rate_dt)
    absorption = -np.expm1(-rate_dt)
    clicks = np.zeros((experiment.y_bins, experiment.ny, experiment.nx))
    for j, rows in enumerate(np.array_split(np.arange(experiment.ny), experiment.y_bins)):
        clicks[j, rows] = absorption.reshape(experiment.ny, experiment.nx)[rows]
    return no_click, clicks.reshape(experiment.y_bins, -1)


def build_history_dynamics_basis(experiment, config=HistoryDynamicsConfig()):
    """Enumerate exact Kraus branches on a finite temporal/spatial grid.

    Each step applies U(dt/2), optional path projection, U(dt/2), absorbing
    detector. The quantum channel is CP and trace preserving including the
    absorbing outcomes at every resolution. Midpoint event quadrature and
    operator splitting still require time/grid convergence checks.
    """
    n = config.steps
    edges = np.linspace(0, config.duration, n+1)
    dt = config.duration/n
    no_click, clicks = detector_step_diagonals(experiment, config, dt)
    _, y = _coordinates(experiment)
    upper = (y.ravel() >= config.pointer_y)
    if np.all(upper) or not np.any(upper):
        raise ValueError("Pointer must partition the spatial grid")
    null = initial_spatial_state(experiment)
    selected = np.zeros((n, 2, null.size), dtype=complex)
    joint = np.zeros((n+1, n, experiment.y_bins))
    for k in range(n):
        null = _propagate_states(experiment, null, dt/2)
        if k:
            selected[:k] = _propagate_states(experiment, selected[:k], dt/2)
        # Earlier detections precede the choice of this history schedule.
        joint[k, :k] = joint[n, :k]
        selected[k, 0] = null * upper
        selected[k, 1] = null * ~upper
        null = _propagate_states(experiment, null, dt/2)
        selected[:k+1] = _propagate_states(experiment, selected[:k+1], dt/2)
        joint[n, k] = clicks @ np.abs(null)**2
        densities = np.sum(np.abs(selected[:k+1])**2, axis=1)
        joint[:k+1, k] = densities @ clicks.T
        null *= no_click
        selected[:k+1] *= no_click
    final = np.concatenate([selected, np.stack([null, np.zeros_like(null)])[None]])
    total = joint.sum(axis=(1, 2)) + np.sum(np.abs(final)**2, axis=(1, 2))
    if not np.allclose(total, 1, atol=2e-11, rtol=0):
        raise RuntimeError("History detector instrument lost probability")
    return HistoryDynamicsBasis(experiment, config, edges, (edges[:-1]+edges[1:])/2,
                                joint, final)


def evaluate_history_dynamics(basis, envelope, *, front_time, strength=1.0, readout_time=None):
    """Mix schedule-conditioned quantum trajectories without postselection."""
    readout_time = basis.config.duration if readout_time is None else float(readout_time)
    if not np.isfinite(readout_time) or readout_time < basis.config.duration:
        raise ValueError("Record readout must follow the acquisition window")
    weights = selection_time_weights(basis.edges, envelope,
                                     front_time=front_time, strength=strength)
    density = np.einsum('c,cbx->x', weights, np.abs(basis.final_kets)**2)
    ideal = np.einsum('c,cty->ty', weights, basis.clicks)
    # Only the fraction still present at the scheduled selection gets measured.
    null_before = 1 - np.r_[0, np.cumsum(basis.clicks[-1].sum(axis=1))[:-1]]
    formed = weights[:-1] * null_before
    accessible = np.sum(formed * record_survival(readout_time-basis.click_times, envelope))
    return HistoryDynamicsResult(
        weights, density, ideal, float(density.sum()), float(ideal.sum()),
        float(accessible), float(formed.sum()-accessible), float(formed.sum()))


def read_history_detector(basis, result, envelope, *, response=None, readout_time=None):
    """Electronic response and subsequent erasure of time-stamped records.

    Saved detections, erased records, and no record form a complete law. Timing
    jitter may move a timestamp outside the acquisition gate; its mass remains
    in no_record. Fading uses each record's physical creation time, before
    timestamp noise, so it cannot make a future detection accessible early.
    """
    response = DetectorResponse() if response is None else response
    readout_time = basis.config.duration if readout_time is None else float(readout_time)
    if not np.isfinite(readout_time) or readout_time < basis.config.duration:
        raise ValueError("Record readout must follow the acquisition window")
    blur = detector_blur_matrix(basis.experiment.y_bins, response.blur_sigma_bins)
    if response.timing_jitter == 0:
        timing = np.eye(basis.config.steps)
    else:
        cdf = ndtr((basis.edges[:, None]-basis.click_times[None, :])/response.timing_jitter)
        timing = np.diff(cdf, axis=0)
    genuine = response.efficiency * (result.ideal_joint @ blur.T)
    signal_mass = float(np.sum(timing @ genuine))
    # At most one dark record per preparation, in the absence of a genuine record.
    dark_mass = (1-signal_mass)*response.dark_probability
    dark = np.full_like(genuine, dark_mass/genuine.size)
    before = timing @ genuine + dark
    survival = record_survival(readout_time-basis.click_times, envelope)
    saved = timing @ (genuine*survival[:, None]) + dark*survival[:, None]
    erased = float(before.sum()-saved.sum())
    no_record = float(1-before.sum())
    law = HistoryDetectorLaw(saved, erased, no_record, before)
    if np.min(law.probabilities) < -2e-11 or not np.isclose(law.probabilities.sum(), 1, atol=2e-11):
        raise RuntimeError("Record erasure instrument lost probability")
    return law


def sample_history_wavefunction(basis, result, rng):
    """Sample a no-click quantum trajectory at the end of acquisition.

    Return (case, selected_half or None, normalized psi), or None if detected.
    Erasure never reselects a case or a half. This samples the exact finite-grid
    ensemble, including the absorbing outcome, rather than one effective psi
    for the mixed state.
    """
    branch_mass = result.weights[:, None] * np.sum(np.abs(basis.final_kets)**2, axis=2)
    u = rng.random()
    if u >= branch_mass.sum():
        return None
    idx = int(np.searchsorted(np.cumsum(branch_mass.ravel()), u, side='right'))
    case, branch = np.unravel_index(idx, branch_mass.shape)
    ket = basis.final_kets[case, branch]
    return case, (None if case == basis.config.steps else branch), ket/np.linalg.norm(ket)
