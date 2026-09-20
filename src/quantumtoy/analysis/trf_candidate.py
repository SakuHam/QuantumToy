"""Minimal admissible phenomenology for testing a TRF observable.

This is a candidate law, not a derivation from TRF-IT v0.2. It adds one
nonnegative strength lambda to the already calibrated quantum reference.
The width convention alpha is fixed before fitting and sigma_T=alpha*tau_stab.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.special import erf

from analysis.record_environment import detector_kraus, pre_detector_record_vectors


@dataclass(frozen=True)
class HalfGaussianDephasing:
    """Extra path dephasing with cumulative half-Gaussian temporal weight.

    eta(t) = exp[-lambda * erf(t / (sqrt(2) sigma_T))]
    sigma_T = alpha * tau_stab

    The channel preserves path populations and multiplies off-diagonal density
    matrix elements by eta. For lambda >= 0 it is completely positive and
    trace preserving. lambda=0 exactly recovers the quantum reference.
    """

    lambda_strength: float = 0.0
    alpha: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.lambda_strength) or self.lambda_strength < 0:
            raise ValueError("lambda_strength must be finite and nonnegative")
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")

    def cumulative_weight(self, time, tau_stab):
        time = np.asarray(time, dtype=float)
        if np.any(~np.isfinite(time)) or np.any(time < 0):
            raise ValueError("time must be finite and nonnegative")
        if tau_stab is None or not np.isfinite(tau_stab) or tau_stab < 0:
            raise ValueError("A finite nonnegative stabilization latency is required")
        if tau_stab == 0:
            return np.where(time == 0, 0.0, 1.0)
        sigma_t = self.alpha * tau_stab
        return erf(time / (np.sqrt(2) * sigma_t))

    def coherence_factor(self, time, tau_stab):
        return np.exp(-self.lambda_strength * self.cumulative_weight(time, tau_stab))


def dephase_record_states(vectors, factor):
    """Apply a qubit phase-damping channel to unnormalized record states."""
    vectors = np.asarray(vectors, dtype=np.complex128)
    if vectors.ndim < 1 or vectors.shape[-1] != 2 or not np.all(np.isfinite(vectors)):
        raise ValueError("vectors must be finite qubit state vectors")
    factor = float(factor)
    if not np.isfinite(factor) or not 0 <= factor <= 1:
        raise ValueError("factor must lie in [0, 1]")
    states = np.einsum("...i,...j->...ij", vectors, vectors.conj())
    states[..., 0, 1] *= factor
    states[..., 1, 0] *= factor
    return states


def candidate_joint(experiment, g, time, tau_stab, candidate):
    """Return p_theta(a, Y, d) after the candidate channel.

    The channel acts after the weak probe and physical memory formation and
    before the terminal detector. Summing over d therefore preserves p(a,Y).
    """
    return joint_with_dephasing(
        experiment,
        g,
        time,
        candidate.coherence_factor(time, tau_stab),
    )


def joint_with_dephasing(experiment, g, time, coherence_factor, *,
                         fragment_readout_basis="X"):
    """Return the complete joint table for a specified extra dephasing factor."""
    vectors = pre_detector_record_vectors(
        experiment, g, time, readout_basis=fragment_readout_basis)
    states = dephase_record_states(vectors, coherence_factor)
    effects = np.array([k.conj().T @ k for k in detector_kraus(experiment)])
    joint = np.einsum("dij,arji->ard", effects, states).real
    if joint.min() < -1e-12:
        raise ValueError("Candidate produced a negative probability")
    return np.maximum(joint, 0.0)


def detector_visibility(joint):
    """Complete-record detector contrast p(+) - p(-); no-click has score 0."""
    joint = np.asarray(joint, dtype=float)
    if joint.ndim != 3 or joint.shape[-1] != 3:
        raise ValueError("joint must have axes [probe, fragment_record, detector]")
    return float(joint[..., 0].sum() - joint[..., 1].sum())


@dataclass(frozen=True)
class FisherResult:
    parameter_names: tuple[str, ...]
    matrix: np.ndarray
    covariance: np.ndarray
    standard_errors: np.ndarray
    correlation: np.ndarray
    eigenvalues: np.ndarray
    condition_number: float
    rank: int


def design_distributions(experiment, observations, parameters, alpha=1.0):
    """Expected complete joint tables for a synthetic calibration design.

    observations contain (nominal_g, time, nominal_tau_stab). A common g_scale
    changes both the physical coupling and tau_stab by inverse scaling, as in
    the exact record model. ordinary_dephasing_rate is a calibrated nuisance
    channel exp(-gamma*t), included to expose possible parameter degeneracy.
    """
    lambda_strength, g_scale, probe_strength, noise_rate = parameters
    if (lambda_strength < 0 or g_scale <= 0 or not 0 < probe_strength < 1
            or noise_rate < 0):
        raise ValueError("Synthetic design parameters are outside their domain")
    varied = replace(experiment, probe_strength=float(probe_strength))
    candidate = HalfGaussianDephasing(float(lambda_strength), float(alpha))
    distributions = []
    for nominal_g, time, nominal_tau in observations:
        tau = float(nominal_tau) / float(g_scale)
        factor = (candidate.coherence_factor(float(time), tau)
                  * np.exp(-float(noise_rate) * float(time)))
        distributions.append(joint_with_dephasing(
            varied, float(nominal_g) * float(g_scale), float(time), factor))
    return distributions


def expected_fisher_information(experiment, observations, *, lambda_strength,
                                alpha=1.0, g_scale=1.0,
                                ordinary_dephasing_rate=0.02,
                                shots_per_observation=10_000):
    """Expected multinomial Fisher matrix for TRF and calibrated nuisances.

    Parameters are lambda_strength, common g_scale, probe_strength, and an
    ordinary Markovian dephasing rate. Several times and couplings are needed:
    a single terminal distribution generally cannot separate two sources of
    dephasing. This is an identifiability diagnostic, not a fitted result.
    """
    if (not isinstance(shots_per_observation, (int, np.integer))
            or shots_per_observation <= 0):
        raise ValueError("shots_per_observation must be a positive integer")
    observations = tuple(tuple(map(float, row)) for row in observations)
    if not observations or any(len(row) != 3 or row[0] <= 0 or row[1] < 0
                               or row[2] <= 0 for row in observations):
        raise ValueError("observations must be (g>0, time>=0, tau_stab>0)")
    theta = np.array([lambda_strength, g_scale, experiment.probe_strength,
                      ordinary_dephasing_rate], dtype=float)
    if theta[0] <= 0 or theta[3] <= 0:
        raise ValueError("Use positive interior strengths for a central-difference study")
    names = ("lambda_strength", "g_scale", "probe_strength",
             "ordinary_dephasing_rate")
    steps = np.maximum(np.abs(theta) * 1e-4, [1e-6, 1e-6, 1e-6, 1e-6])
    base = design_distributions(experiment, observations, theta, alpha)
    derivatives = []
    for index, step in enumerate(steps):
        upper, lower = theta.copy(), theta.copy()
        upper[index] += step
        lower[index] -= step
        high = design_distributions(experiment, observations, upper, alpha)
        low = design_distributions(experiment, observations, lower, alpha)
        derivatives.append([(a - b) / (2 * step) for a, b in zip(high, low)])

    fisher = np.zeros((len(theta), len(theta)))
    for setting, probabilities in enumerate(base):
        supported = probabilities > 1e-15
        for i in range(len(theta)):
            for j in range(len(theta)):
                fisher[i, j] += shots_per_observation * np.sum(
                    derivatives[i][setting][supported]
                    * derivatives[j][setting][supported]
                    / probabilities[supported])
    fisher = (fisher + fisher.T) / 2
    eigenvalues = np.linalg.eigvalsh(fisher)
    tolerance = max(eigenvalues[-1] * 1e-10, 1e-12)
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    covariance = np.linalg.pinv(fisher, rcond=1e-10)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    denominator = np.outer(standard_errors, standard_errors)
    correlation = np.divide(covariance, denominator,
                            out=np.zeros_like(covariance), where=denominator > 0)
    positive = eigenvalues[eigenvalues > tolerance]
    condition = float("inf") if rank < len(theta) else float(positive[-1] / positive[0])
    return FisherResult(names, fisher, covariance, standard_errors, correlation,
                        eigenvalues, condition, rank)
