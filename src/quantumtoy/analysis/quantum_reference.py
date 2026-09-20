"""Finite-dimensional quantum reference for TRF-IT v0.2, sections 2–5 and 9.

Instruments contain one sequence of Kraus operators per recorded outcome.
Effects describe the complete later record at a common time. No postselection,
loss renormalization, or TRF modification is implicit in this reference.
"""

from __future__ import annotations

import numpy as np


_ATOL = 1e-12


def _operator(value, dimension=None):
    value = np.asarray(value, dtype=np.complex128)
    if (value.ndim != 2 or value.shape[0] == 0
            or value.shape[0] != value.shape[1]
            or not np.all(np.isfinite(value))):
        raise ValueError("Expected a finite, nonempty square operator")
    if dimension is not None and value.shape != (dimension, dimension):
        raise ValueError("Operator dimensions do not match")
    return value


def _positive(value, dimension=None):
    value = _operator(value, dimension)
    if not np.allclose(value, value.conj().T, atol=_ATOL, rtol=0):
        raise ValueError("Operator must be Hermitian")
    if np.linalg.eigvalsh(value).min() < -_ATOL:
        raise ValueError("Operator must be positive semidefinite")
    return value


def _effect(value, dimension=None):
    value = _positive(value, dimension)
    _positive(np.eye(len(value)) - value)
    return value


def _complete(operators, dimension):
    if not np.allclose(sum(operators), np.eye(dimension), atol=_ATOL, rtol=0):
        raise ValueError("Outcomes must be complete; include no-click and loss outcomes")


def _kraus(operators, dimension):
    operators = tuple(_operator(k, dimension) for k in operators)
    if not operators:
        raise ValueError("At least one Kraus operator is required")
    return operators


def _apply(state, operators):
    return sum(k @ state @ k.conj().T for k in operators)


def joint_record_probabilities(rho, instrument, effects, *, channel=None):
    """Return p[a, R] = Tr(E_R T(J_a(rho))) without renormalization.

The optional channel is a complete sequence of Kraus operators propagating
from the intermediate instrument to the later effects. An unread instrument
is retained by summing over a; an absent instrument is ``[[identity]]``.
"""
    rho = _positive(rho)
    dimension = len(rho)
    if not np.isclose(np.trace(rho), 1, atol=_ATOL, rtol=0):
        raise ValueError("Density operator must have unit trace")
    instrument = tuple(_kraus(outcome, dimension) for outcome in instrument)
    if not instrument:
        raise ValueError("Instrument must contain outcomes")
    _complete([k.conj().T @ k for outcome in instrument for k in outcome], dimension)
    effects = tuple(_effect(e, dimension) for e in effects)
    if not effects:
        raise ValueError("Later measurement must contain outcomes")
    _complete(effects, dimension)
    channel = _kraus([np.eye(dimension)] if channel is None else channel, dimension)
    _complete([k.conj().T @ k for k in channel], dimension)

    joint = np.empty((len(instrument), len(effects)), dtype=float)
    for a, outcome in enumerate(instrument):
        state = _apply(_apply(rho, outcome), channel)
        for record, effect in enumerate(effects):
            joint[a, record] = np.trace(effect @ state).real
    # Only remove roundoff-scale negativity; never normalize surviving events.
    if np.min(joint) < -_ATOL:
        raise ValueError("Negative joint probability")
    return np.maximum(joint, 0.0)


def condition_on_record(joint, record):
    """Condition a normalized joint table on a nonzero-probability column."""
    joint = np.asarray(joint, dtype=float)
    if (joint.ndim != 2 or not np.all(np.isfinite(joint))
            or np.any(joint < 0)
            or not np.isclose(joint.sum(), 1, atol=_ATOL, rtol=0)):
        raise ValueError("Expected a normalized nonnegative joint table")
    selected = joint[:, record]
    mass = selected.sum()
    if mass <= 0:
        raise ValueError("Cannot condition on a zero-probability record")
    return selected / mass


def mix_record_effects(effects, weights):
    """Arithmetic mixture of one record's effects for unresolved alternatives.

Each effect is already represented at the same earlier time. Weights are
declared, state-independent prior probabilities, not posterior weights.
"""
    effects = tuple(_effect(e) for e in effects)
    if not effects or any(e.shape != effects[0].shape for e in effects):
        raise ValueError("Effects must have one common dimension")
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != (len(effects),) or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1, atol=_ATOL, rtol=0)):
        raise ValueError("Mixture weights must be nonnegative and sum to one")
    return sum(w * e for w, e in zip(weights, effects))


def rank_one_posterior(prior, likelihood, *, epsilon=0.0, reference_scale=1.0):
    """Section 5 scalar rule for a declared rank-one projective instrument.

The numerical floor changes the posterior but cannot justify an event with
zero original probability. This is not a general weak-measurement rule.
"""
    prior = np.asarray(prior, dtype=float)
    likelihood = np.asarray(likelihood, dtype=float)
    if (prior.ndim != 1 or prior.size == 0 or likelihood.shape != prior.shape
            or not np.all(np.isfinite(prior)) or not np.all(np.isfinite(likelihood))
            or np.any(prior < 0) or np.any(likelihood < 0)
            or not np.isclose(prior.sum(), 1, atol=_ATOL, rtol=0)):
        raise ValueError("Expected a normalized prior and nonnegative likelihood")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    if not np.isfinite(reference_scale) or reference_scale <= 0:
        raise ValueError("reference_scale must be finite and positive")
    if np.dot(prior, likelihood) <= 0:
        raise ValueError("Cannot condition on a zero-probability record")
    weighted = prior * (likelihood / reference_scale + epsilon)
    return weighted / weighted.sum()
