"""Phenomenological future/front/past envelope for the interactive demo.

The model separates irreversible history selection from the accessibility of
its record.  A selected outcome therefore does not become open again when its
local record fades.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, log_ndtr


@dataclass(frozen=True)
class TemporalHistoryEnvelope:
    """Timescales controlling a smooth selection front and record fading."""

    sigma_t: float = 0.2
    retention_time: float = 0.2
    fade_time: float = 0.8
    fade_power: float = 1.5

    def __post_init__(self):
        values = (
            self.sigma_t,
            self.retention_time,
            self.fade_time,
            self.fade_power,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("History-envelope parameters must be finite")
        if self.sigma_t <= 0:
            raise ValueError("sigma_t must be positive")
        if self.retention_time < 0:
            raise ValueError("retention_time must be nonnegative")
        if self.fade_time <= 0:
            raise ValueError("fade_time must be positive")
        if self.fade_power <= 0:
            raise ValueError("fade_power must be positive")


@dataclass(frozen=True)
class TemporalHistoryProfile:
    """Four distinct temporal weights evaluated at one or more ages."""

    selection: np.ndarray
    future_openness: np.ndarray
    present_front: np.ndarray
    accessible_record: np.ndarray


def temporal_history_profile(
    age: np.ndarray | float,
    envelope: TemporalHistoryEnvelope,
) -> TemporalHistoryProfile:
    """Evaluate selection, openness, front, and accessible-record weights.

    ``age`` is measured from the center of the selection front. Negative age
    is the open-future side and positive age is the selected-past side.
    ``selection`` is monotone and independent of record fading.  Only
    ``accessible_record`` decays at old positive ages.
    """

    if not isinstance(envelope, TemporalHistoryEnvelope):
        raise TypeError("envelope must be a TemporalHistoryEnvelope")
    age = np.asarray(age, dtype=float)
    if not np.all(np.isfinite(age)):
        raise ValueError("age must be finite")

    selection = ndtr(age / envelope.sigma_t)
    future = 1.0 - selection
    present = 4.0 * selection * future
    fading_age = np.maximum(age - envelope.retention_time, 0.0)
    survival = np.exp(-np.power(fading_age / envelope.fade_time,
                               envelope.fade_power))
    record = selection * survival
    return TemporalHistoryProfile(
        selection=np.asarray(selection, dtype=float),
        future_openness=np.asarray(future, dtype=float),
        present_front=np.asarray(present, dtype=float),
        accessible_record=np.asarray(record, dtype=float),
    )


def record_survival(age, envelope):
    """Survival of an existing classical record, aged from its creation.

    This erasure probability does not act on the already selected quantum
    branch. Negative ages are rejected: a record cannot precede its creation.
    """
    age = np.asarray(age, dtype=float)
    if not np.all(np.isfinite(age)) or np.any(age < 0):
        raise ValueError("Record ages must be finite and nonnegative")
    with np.errstate(over="ignore"):
        return np.exp(-np.power(
            np.maximum(age - envelope.retention_time, 0) / envelope.fade_time,
            envelope.fade_power))


def selection_time_weights(edges, envelope, *, front_time, strength=1.0):
    """Complete distribution of a single selection time, or no selection.

    A Gaussian clock is conditioned on having no selection before edges[0].
    Each interval receives q * P(selection in interval | selection >= start),
    q=1-exp(-strength). The last element retains all unselected probability,
    including events beyond the simulation horizon. No truncated-tail
    renormalization is used. The clock is independent of the quantum state.
    """
    edges = np.asarray(edges, dtype=float)
    if (edges.ndim != 1 or edges.size < 2
            or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0)):
        raise ValueError("Clock edges must be finite and strictly increasing")
    if not np.isfinite(front_time) or not np.isfinite(strength) or strength < 0:
        raise ValueError("Invalid front center or selection strength")
    # Evaluate survival in log space, including clocks centered before t=0.
    log_tail = log_ndtr((front_time - edges) / envelope.sigma_t)
    log_survival = log_tail - log_tail[0]
    interval = np.exp(log_survival[:-1]) * (-np.expm1(np.diff(log_survival)))
    q = -np.expm1(-strength)
    return np.concatenate([q * interval, [np.exp(-strength) + q * np.exp(log_survival[-1])]])
