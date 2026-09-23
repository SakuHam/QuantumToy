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


@dataclass(frozen=True)
class ConventionWidthProfileRow:
    """One shared clock convention and its fixed/calibrated width predictions."""

    information_deficit: float
    required_copies: int
    coherence_tolerance: float
    hold_time: float
    g_values: tuple[float, ...]
    latencies: tuple[float, ...]
    locked_alpha: float
    locked_alpha_widths: tuple[float, ...]
    calibration_profiled_alpha: float
    calibration_profiled_widths: tuple[float, ...]


@dataclass(frozen=True)
class ClockConventionWidthProfile:
    """Discrete convention profile with unresolved choices kept visible."""

    locked_alpha: float
    calibration_g: float
    calibration_sigma_t: float
    g_values: tuple[float, ...]
    rows: tuple[ConventionWidthProfileRow, ...]
    unresolved_conventions: int


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


def profile_clock_convention_widths(
        threshold_rows, *, g_values, locked_alpha,
        calibration_g=1.0, calibration_sigma_t=0.2):
    """Profile one shared threshold convention across all couplings.

    Two policies are reported. ``locked_alpha_widths`` keeps the numerical
    reference alpha fixed while the convention varies. The calibration-
    profiled policy treats the convention as a discrete nuisance and
    recalibrates alpha using only the declared calibration coupling, before
    predicting every held-out coupling. No unresolved convention is silently
    converted to a finite width.
    """
    g_values = tuple(float(value) for value in g_values)
    locked_alpha = float(locked_alpha)
    calibration_g = float(calibration_g)
    calibration_sigma_t = float(calibration_sigma_t)
    if (not g_values or len(set(g_values)) != len(g_values)
            or any(not np.isfinite(value) or value <= 0 for value in g_values)):
        raise ValueError("g_values must be unique, finite, and positive")
    if calibration_g not in g_values:
        raise ValueError("calibration_g must be one of g_values")
    if (not np.isfinite(locked_alpha) or locked_alpha <= 0
            or not np.isfinite(calibration_sigma_t)
            or calibration_sigma_t <= 0):
        raise ValueError("alpha and calibration width must be finite and positive")

    grouped = {}
    for row in threshold_rows:
        if row.g not in g_values:
            continue
        key = (
            row.information_deficit, row.required_copies,
            row.coherence_tolerance, row.hold_time,
        )
        values = grouped.setdefault(key, {})
        if row.g in values:
            raise ValueError("duplicate coupling for one threshold convention")
        values[row.g] = row.stabilization
    if not grouped:
        raise ValueError("threshold_rows contain no requested couplings")

    profile_rows = []
    unresolved = 0
    calibration_index = g_values.index(calibration_g)
    for key, windows in sorted(grouped.items()):
        if set(windows) != set(g_values):
            raise ValueError("every convention must contain every requested coupling")
        if any(windows[g] is None for g in g_values):
            unresolved += 1
            continue
        latencies = tuple(float(windows[g].latency) for g in g_values)
        calibration_alpha = calibration_sigma_t / latencies[calibration_index]
        profile_rows.append(ConventionWidthProfileRow(
            information_deficit=float(key[0]), required_copies=int(key[1]),
            coherence_tolerance=float(key[2]), hold_time=float(key[3]),
            g_values=g_values, latencies=latencies,
            locked_alpha=locked_alpha,
            locked_alpha_widths=tuple(locked_alpha * value for value in latencies),
            calibration_profiled_alpha=float(calibration_alpha),
            calibration_profiled_widths=tuple(
                calibration_alpha * value for value in latencies),
        ))
    return ClockConventionWidthProfile(
        locked_alpha=locked_alpha, calibration_g=calibration_g,
        calibration_sigma_t=calibration_sigma_t, g_values=g_values,
        rows=tuple(profile_rows), unresolved_conventions=unresolved)


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
