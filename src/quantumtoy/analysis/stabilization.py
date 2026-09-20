"""Sampled event-anchored clock for TRF-IT v0.2, section 8.

The caller supplies a predeclared stability predicate combining accessible
record information, redundancy, coherence, and any record-quality criteria.
Passing sampled checks does not prove stability between samples or forever.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StabilizationWindow:
    onset: float
    confirmed_at: float
    latency: float


def sampled_stabilization(times, stable, *, hold_time):
    """Find the first consecutive true run spanning at least hold_time.

The first sample is the declared event time t0. Confirmation uses the first
sample at or beyond onset + hold_time and requires that sample to pass too.
Return None if the observation window cannot establish stabilization; this
does not imply infinite physical latency. Refine sampling to assess error.
"""
    times = np.asarray(times, dtype=float)
    stable = np.asarray(stable)
    if (times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)):
        raise ValueError("times must be finite and strictly increasing")
    if stable.shape != times.shape or stable.dtype != np.dtype(bool):
        raise ValueError("stable must be a boolean array matching times")
    if not np.isfinite(hold_time) or hold_time < 0:
        raise ValueError("hold_time must be finite and nonnegative")
    onset = None
    for time, passes in zip(times, stable):
        if not passes:
            onset = None
        else:
            if onset is None:
                onset = float(time)
            if time - onset >= hold_time:
                return StabilizationWindow(onset, float(time), onset - float(times[0]))
    return None
