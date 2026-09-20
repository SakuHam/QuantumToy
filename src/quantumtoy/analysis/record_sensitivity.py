"""Sensitivity and time-sampling diagnostics for the record clock."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from analysis.record_environment import RecordExperiment, RecordRun, simulate_record_formation
from analysis.stabilization import StabilizationWindow, sampled_stabilization


@dataclass(frozen=True)
class ThresholdResult:
    g: float
    information_deficit: float
    required_copies: int
    coherence_tolerance: float
    hold_time: float
    stabilization: StabilizationWindow | None


def reevaluate_stabilization(run, *, information_deficit, required_copies,
                             coherence_tolerance, hold_time):
    """Apply new predeclared thresholds without rerunning quantum dynamics."""
    if not 0 < information_deficit < 1:
        raise ValueError("information_deficit must lie strictly between 0 and 1")
    if (not isinstance(required_copies, (int, np.integer))
            or not 1 <= required_copies <= run.readout_information.shape[1]):
        raise ValueError("required_copies must fit the physical fragment count")
    if not np.isfinite(coherence_tolerance) or not 0 <= coherence_tolerance <= 1:
        raise ValueError("coherence_tolerance must lie in [0, 1]")
    if not np.isfinite(hold_time) or hold_time < 0:
        raise ValueError("hold_time must be finite and nonnegative")
    threshold = (1 - information_deficit) * run.experiment.branch_entropy
    redundancy = np.count_nonzero(run.readout_information >= threshold, axis=1)
    stable = ((redundancy >= required_copies)
              & (run.coherence <= coherence_tolerance))
    return sampled_stabilization(run.times, stable, hold_time=hold_time)


def threshold_grid(runs, *, information_deficits, required_copies,
                   coherence_tolerances, hold_times):
    """Evaluate a Cartesian threshold grid on already computed record runs."""
    rows = []
    for run in runs:
        for deficit, copies, coherence, hold in product(
                information_deficits, required_copies,
                coherence_tolerances, hold_times):
            window = reevaluate_stabilization(
                run,
                information_deficit=float(deficit),
                required_copies=int(copies),
                coherence_tolerance=float(coherence),
                hold_time=float(hold),
            )
            rows.append(ThresholdResult(
                run.g, float(deficit), int(copies), float(coherence),
                float(hold), window))
    return rows


@dataclass(frozen=True)
class SamplingResult:
    g: float
    dt: float
    latency: float | None
    confirmed_at: float | None
    max_joint_normalization_error: float


def time_sampling_convergence(experiment, g_values, *, duration, time_steps):
    """Repeat the exact model at several sampling intervals.

    The state at each time is analytic. This diagnoses only event-clock
    sampling error; it is not a spatial or propagation convergence test.
    """
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    results = []
    for g, dt in product(g_values, time_steps):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("time_steps must be finite and positive")
        times = np.arange(0, duration + dt / 2, dt)
        run = simulate_record_formation(experiment, float(g), times)
        window = run.stabilization
        results.append(SamplingResult(
            float(g), float(dt),
            None if window is None else window.latency,
            None if window is None else window.confirmed_at,
            float(np.max(np.abs(run.joint.sum(axis=(1, 2, 3)) - 1))),
        ))
    return results


def build_reference_runs(experiment, g_values, *, duration, dt):
    """Build runs once for a threshold study with a shared protocol."""
    times = np.arange(0, duration + dt / 2, dt)
    return [simulate_record_formation(experiment, float(g), times) for g in g_values]
