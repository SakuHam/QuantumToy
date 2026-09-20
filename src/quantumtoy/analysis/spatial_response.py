"""End-to-end spatial response profile for the measurement-guided theory."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from config import AppConfig
from core.grid import build_grid
from core.packets import PacketFactory
from core.potentials import build_potential
from theories.schrodinger import SchrodingerTheory
from theories.thick_front_measurement_guided import ThickFrontMeasurementGuidedTheory
from theories.thick_front_world_line import ThickFrontWorldLineTheory


@dataclass(frozen=True)
class SpatialResponseExperiment:
    visible_lx: float = 8.0
    visible_ly: float = 6.0
    nx: int = 64
    ny: int = 48
    dt: float = 0.01
    duration: float = 13 / 6
    barrier_x: float = -0.5
    barrier_thickness: float = 0.4
    barrier_smooth: float = 0.25
    screen_x: float = 2.5
    detector_width: float = 0.3
    packet_x: float = -2.5
    packet_sigma: float = 0.5
    packet_kx: float = 3.0
    slit_offset: float = 0.8
    slit_half_height: float = 0.4
    response_strength: float = 1.0
    back_stride: int = 2
    horizon_sigmas: float = 4.0
    normalization_mode: str = "per_run_max"
    normalization_reference_sigma_t: float = 0.1
    front_neighbor_sigma: float = 0.15
    competition_radius: float = 0.375
    gain_blur_sigma: float = 0.125
    competition_blur_sigma: float = 0.0625
    measurement_blur_sigma: float = 0.09375

    def __post_init__(self):
        finite_positive = (
            self.visible_lx, self.visible_ly, self.dt, self.duration,
            self.detector_width, self.packet_sigma, self.packet_kx,
            self.barrier_thickness, self.barrier_smooth,
            self.slit_offset, self.slit_half_height, self.horizon_sigmas,
            self.front_neighbor_sigma, self.competition_radius,
            self.gain_blur_sigma, self.competition_blur_sigma,
            self.measurement_blur_sigma,
        )
        if not np.all(np.isfinite(finite_positive)) or min(finite_positive) <= 0:
            raise ValueError("Spatial response scales must be finite and positive")
        if self.nx < 12 or self.ny < 12:
            raise ValueError("Spatial response grid must be at least 12 by 12")
        if not isinstance(self.back_stride, (int, np.integer)) or self.back_stride < 1:
            raise ValueError("back_stride must be an integer >= 1")
        if not np.isfinite(self.response_strength) or self.response_strength < 0:
            raise ValueError("response_strength must be finite and nonnegative")
        if self.normalization_mode not in {"per_run_max", "fixed", "none"}:
            raise ValueError(
                "normalization_mode must be 'per_run_max', 'fixed', or 'none'")
        if (not np.isfinite(self.normalization_reference_sigma_t)
                or self.normalization_reference_sigma_t <= 0):
            raise ValueError(
                "normalization_reference_sigma_t must be finite and positive")

    @property
    def steps(self):
        return int(round(self.duration / self.dt))


@dataclass(frozen=True)
class SpatialDetectorRun:
    sigma_t: float | None
    response_strength: float
    y: np.ndarray
    y_distribution: np.ndarray
    complete_distribution: np.ndarray
    detector_mass: float
    terminal_norm: float


@dataclass(frozen=True)
class SpatialResponseProfile:
    true_sigma_t: float
    candidate_sigmas: np.ndarray
    kl_divergence: np.ndarray
    expected_deviance: np.ndarray
    best_sigma_t: float
    baseline_total_variation: float
    null_state_max_error: float
    null_distribution_max_error: float
    detector_mass: float
    target_y_distribution: np.ndarray
    target_complete_distribution: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class SpatialNormalizationScales:
    effect: float
    overlap: float


@dataclass(frozen=True)
class CrossGridRecovery:
    reference_grid: tuple[int, int]
    fit_grid: tuple[int, int]
    candidate_sigmas: np.ndarray
    raw_kl_divergence: np.ndarray
    raw_best_sigma_t: float
    excess_weighted_square: np.ndarray
    best_sigma_t: float


def _build_spatial_problem(experiment):
    cfg = AppConfig(
        N_VISIBLE_X=experiment.nx,
        N_VISIBLE_Y=experiment.ny,
        PAD_FACTOR=1,
        VISIBLE_LX=experiment.visible_lx,
        VISIBLE_LY=experiment.visible_ly,
        barrier_center_x=experiment.barrier_x,
        barrier_thickness=experiment.barrier_thickness,
        V_barrier=35.0,
        slit_center_offset=experiment.slit_offset,
        slit_half_height=experiment.slit_half_height,
        CAP_WIDTH=0.6,
        CAP_STRENGTH=1.0,
        screen_center_x=experiment.screen_x,
        screen_eval_width=experiment.detector_width,
        x0=experiment.packet_x,
        y0=0.0,
        sigma0=experiment.packet_sigma,
        k0x=experiment.packet_kx,
        k0y=0.0,
    )
    # These legacy class attributes are not dataclass constructor fields.
    cfg.BARRIER_EDGE_MODE = "smooth"
    cfg.BARRIER_SMOOTH = experiment.barrier_smooth
    grid = build_grid(
        visible_lx=cfg.VISIBLE_LX,
        visible_ly=cfg.VISIBLE_LY,
        n_visible_x=cfg.N_VISIBLE_X,
        n_visible_y=cfg.N_VISIBLE_Y,
        pad_factor=cfg.PAD_FACTOR,
    )
    potential = build_potential(grid, cfg)
    initial = PacketFactory.build_initial_packet(cfg, grid).psi0
    return grid, potential, initial


def _theory_kwargs(experiment, grid, potential):
    pixel_size = max(float(grid.dx), float(grid.dy))
    return dict(
        grid=grid,
        potential=potential,
        m_mass=1.0,
        hbar=1.0,
        front_debug_checks=False,
        front_strength=0.03,
        front_neighbor_sigma=float(experiment.front_neighbor_sigma),
        front_gain_blur_sigma=float(experiment.gain_blur_sigma) / pixel_size,
        front_branch_competition_strength=2.0,
        front_branch_use_flow_direction=False,
        front_branch_competition_radius=max(
            1, int(round(float(experiment.competition_radius) / pixel_size))),
        front_branch_competition_blur_sigma=(
            float(experiment.competition_blur_sigma) / pixel_size),
        worldline_mode="off",
    )


def _guided_theory(experiment, grid, potential, sigma_t, response_strength,
                   *, effect_scale=None, overlap_scale=None):
    normalize = experiment.normalization_mode != "none"
    pixel_size = max(float(grid.dx), float(grid.dy))
    return ThickFrontMeasurementGuidedTheory(
        **_theory_kwargs(experiment, grid, potential),
        measurement_sigma_t=float(sigma_t),
        measurement_response_strength=float(response_strength),
        measurement_back_stride=int(experiment.back_stride),
        measurement_back_horizon_sigmas=float(experiment.horizon_sigmas),
        measurement_detector_center_x=float(experiment.screen_x),
        measurement_detector_width=max(float(experiment.detector_width), 1e-6),
        measurement_debug_print=False,
        measurement_effect_blur_sigma=(
            float(experiment.measurement_blur_sigma) / pixel_size),
        measurement_overlap_blur_sigma=(
            float(experiment.measurement_blur_sigma) / pixel_size),
        measurement_refresh_every_n_steps_pre_init=experiment.steps + 1,
        measurement_refresh_every_n_steps_post_init=experiment.steps + 1,
        measurement_stop_after_worldline_init=False,
        measurement_normalize_effect=normalize,
        measurement_normalize_overlap=normalize,
        measurement_effect_normalization_scale=effect_scale,
        measurement_overlap_normalization_scale=overlap_scale,
    )


@lru_cache(maxsize=16)
def fixed_normalization_scales(experiment):
    """Calibrate fixed field scales on the declared parent-theory reference."""
    calibration = replace(experiment, normalization_mode="none")
    grid, potential, initial = _build_spatial_problem(calibration)
    common = _theory_kwargs(calibration, grid, potential)
    parent = ThickFrontWorldLineTheory(**common)
    probe = _guided_theory(
        calibration, grid, potential,
        calibration.normalization_reference_sigma_t,
        calibration.response_strength)
    state = parent.initialize_state(initial)
    first_base = SchrodingerTheory.step_forward(
        probe, state, calibration.dt).state
    raw_effect = probe._build_measurement_effect_mix(first_base, calibration.dt)
    effect_scale = float(np.max(raw_effect))
    if not np.isfinite(effect_scale) or effect_scale <= probe.front_eps:
        raise RuntimeError("Cannot calibrate a zero measurement effect scale")
    fixed_effect = raw_effect / effect_scale

    overlap_scale = 0.0
    for _ in range(calibration.steps):
        base_state = SchrodingerTheory.step_forward(
            probe, state, calibration.dt).state
        align, rho, _ = probe._coherence_alignment_score(base_state)
        raw_overlap = probe._build_measurement_overlap_score(
            rho, align, fixed_effect)
        overlap_scale = max(overlap_scale, float(np.max(raw_overlap)))
        state = parent.step_forward(state, calibration.dt).state
    if not np.isfinite(overlap_scale) or overlap_scale <= probe.front_eps:
        raise RuntimeError("Cannot calibrate a zero measurement overlap scale")
    return SpatialNormalizationScales(effect_scale, overlap_scale)


def _evolve(experiment, *, sigma_t, response_strength, baseline=False):
    grid, potential, initial = _build_spatial_problem(experiment)
    common = _theory_kwargs(experiment, grid, potential)
    if baseline:
        theory = ThickFrontWorldLineTheory(**common)
    else:
        if not np.isfinite(sigma_t) or sigma_t <= 0:
            raise ValueError("sigma_t must be finite and positive")
        scales = (fixed_normalization_scales(experiment)
                  if experiment.normalization_mode == "fixed" else None)
        theory = _guided_theory(
            experiment, grid, potential, sigma_t, response_strength,
            effect_scale=None if scales is None else scales.effect,
            overlap_scale=None if scales is None else scales.overlap)
    state = theory.initialize_state(initial)
    for _ in range(experiment.steps):
        state = theory.step_forward(state, experiment.dt).state
    return grid, state


def _detector_run(experiment, grid, state, *, sigma_t, response_strength):
    density = np.abs(state) ** 2
    gate = np.exp(
        -0.5 * ((grid.X - experiment.screen_x) / experiment.detector_width) ** 2
    )
    line_mass = np.sum(density * gate, axis=1) * grid.dx
    detector_mass_raw = float(np.sum(line_mass) * grid.dy)
    if not np.isfinite(detector_mass_raw) or detector_mass_raw <= 0:
        raise RuntimeError("Spatial detector distribution has zero mass")
    terminal_norm = float(np.sum(density) * grid.dx * grid.dy)
    detector_mass = detector_mass_raw / terminal_norm
    y_distribution = line_mass / np.sum(line_mass)
    click_bins = line_mass * grid.dy / terminal_norm
    no_click = max(0.0, 1.0 - float(np.sum(click_bins)))
    complete_distribution = np.concatenate([click_bins, [no_click]])
    complete_distribution /= np.sum(complete_distribution)
    return SpatialDetectorRun(
        sigma_t, float(response_strength), grid.Y[:, 0].copy(),
        y_distribution, complete_distribution, detector_mass, terminal_norm,
    )


def run_spatial_detector(experiment, sigma_t, *, response_strength=None):
    strength = (experiment.response_strength if response_strength is None
                else float(response_strength))
    grid, state = _evolve(
        experiment, sigma_t=sigma_t, response_strength=strength)
    return _detector_run(
        experiment, grid, state, sigma_t=float(sigma_t),
        response_strength=strength)


def run_spatial_baseline(experiment):
    grid, state = _evolve(
        experiment, sigma_t=None, response_strength=0.0, baseline=True)
    return _detector_run(
        experiment, grid, state, sigma_t=None, response_strength=0.0)


def _kl_divergence(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    supported = observed > 0
    return float(np.sum(observed[supported] * np.log(
        observed[supported] / np.maximum(predicted[supported], 1e-300))))


def fit_spatial_response(experiment, true_sigma_t, candidate_sigmas, *, shots=100_000):
    """Profile exact terminal distributions over a declared sigma_T grid."""
    candidates = np.asarray(candidate_sigmas, dtype=float)
    if (candidates.ndim != 1 or candidates.size < 2
            or np.any(~np.isfinite(candidates)) or np.any(candidates <= 0)):
        raise ValueError("candidate_sigmas must contain at least two positive values")
    if not isinstance(shots, (int, np.integer)) or shots <= 0:
        raise ValueError("shots must be a positive integer")

    baseline = run_spatial_baseline(experiment)
    target = run_spatial_detector(experiment, true_sigma_t)
    null_grid, null_state = _evolve(
        experiment, sigma_t=true_sigma_t, response_strength=0.0)
    _, base_state = _evolve(
        experiment, sigma_t=None, response_strength=0.0, baseline=True)
    null_run = _detector_run(
        experiment, null_grid, null_state,
        sigma_t=float(true_sigma_t), response_strength=0.0)

    profiles = [run_spatial_detector(experiment, value) for value in candidates]
    kl = np.array([
        _kl_divergence(target.complete_distribution, run.complete_distribution)
        for run in profiles
    ])
    best = float(candidates[int(np.argmin(kl))])
    baseline_tv = 0.5 * float(np.sum(np.abs(
        target.complete_distribution - baseline.complete_distribution)))
    return SpatialResponseProfile(
        true_sigma_t=float(true_sigma_t),
        candidate_sigmas=candidates,
        kl_divergence=kl,
        expected_deviance=2 * shots * kl,
        best_sigma_t=best,
        baseline_total_variation=baseline_tv,
        null_state_max_error=float(np.max(np.abs(null_state - base_state))),
        null_distribution_max_error=float(np.max(np.abs(
            null_run.complete_distribution - baseline.complete_distribution))),
        detector_mass=target.detector_mass,
        target_y_distribution=target.y_distribution,
        target_complete_distribution=target.complete_distribution,
        y=target.y,
    )


def _project_complete_distribution(run, target_y):
    """Project a complete detector law onto another y grid."""
    target_y = np.asarray(target_y, dtype=float)
    interpolated = np.interp(
        target_y, run.y, run.y_distribution / np.gradient(run.y),
        left=0.0, right=0.0)
    interpolated = np.maximum(interpolated * np.gradient(target_y), 0.0)
    interpolated /= np.sum(interpolated)
    click_bins = interpolated * run.detector_mass
    complete = np.concatenate([click_bins, [1.0 - run.detector_mass]])
    complete /= np.sum(complete)
    return complete


def _interpolated_total_variation(reference, comparison):
    complete = _project_complete_distribution(comparison, reference.y)
    return 0.5 * float(np.sum(np.abs(
        reference.complete_distribution - complete)))


def _refined_grid_shape(experiment):
    def scaled_even(value):
        refined = max(value + 4, int(round(value * 5 / 4)))
        return refined if refined % 2 == 0 else refined + 1
    return scaled_even(experiment.nx), scaled_even(experiment.ny)


def fit_cross_grid_response(experiment, true_sigma_t, candidate_sigmas):
    """Generate on a 5/4 grid and fit raw and baseline-corrected responses."""
    candidates = np.asarray(candidate_sigmas, dtype=float)
    if (candidates.ndim != 1 or candidates.size < 2
            or np.any(~np.isfinite(candidates)) or np.any(candidates <= 0)):
        raise ValueError("candidate_sigmas must contain at least two positive values")
    refined_nx, refined_ny = _refined_grid_shape(experiment)
    refined = replace(experiment, nx=refined_nx, ny=refined_ny)
    target = run_spatial_detector(refined, true_sigma_t)
    target_baseline = run_spatial_baseline(refined)
    fit_baseline = run_spatial_baseline(experiment)
    profiles = [run_spatial_detector(experiment, value) for value in candidates]
    projected = _project_complete_distribution(target, profiles[0].y)
    projected_baseline = _project_complete_distribution(
        target_baseline, profiles[0].y)
    raw_kl = np.array([
        _kl_divergence(projected, run.complete_distribution)
        for run in profiles
    ])
    target_excess = projected - projected_baseline
    excess_score = np.array([
        np.sum((target_excess - (
            run.complete_distribution - fit_baseline.complete_distribution
        )) ** 2 / np.maximum(projected_baseline, 1e-12))
        for run in profiles
    ])
    return CrossGridRecovery(
        reference_grid=(refined.nx, refined.ny),
        fit_grid=(experiment.nx, experiment.ny),
        candidate_sigmas=candidates,
        raw_kl_divergence=raw_kl,
        raw_best_sigma_t=float(candidates[int(np.argmin(raw_kl))]),
        excess_weighted_square=excess_score,
        best_sigma_t=float(candidates[int(np.argmin(excess_score))]),
    )


def compare_normalization_modes(experiment, true_sigma_t, candidate_sigmas,
                                *, shots=100_000):
    """Profile identical data-generating and fit modes for three scale rules."""
    return {
        mode: fit_spatial_response(
            replace(experiment, normalization_mode=mode),
            true_sigma_t, candidate_sigmas, shots=shots)
        for mode in ("per_run_max", "fixed", "none")
    }


def spatial_convergence(experiment, sigma_t):
    """Compare dt/2, a larger grid, and a one-sigma-longer horizon."""
    reference = run_spatial_detector(experiment, sigma_t)
    dt_refined = replace(
        experiment, dt=experiment.dt / 2,
        duration=experiment.steps * experiment.dt)
    refined_nx, refined_ny = _refined_grid_shape(experiment)
    grid_refined = replace(experiment, nx=refined_nx, ny=refined_ny)
    horizon_refined = replace(
        experiment, horizon_sigmas=experiment.horizon_sigmas + 1)
    comparisons = {
        "dt_half": run_spatial_detector(dt_refined, sigma_t),
        "grid_5_over_4": run_spatial_detector(grid_refined, sigma_t),
        "horizon_plus_one_sigma": run_spatial_detector(horizon_refined, sigma_t),
    }
    raw = {
        name: _interpolated_total_variation(reference, run)
        for name, run in comparisons.items()
    }
    reference_baseline = run_spatial_baseline(experiment)
    reference_excess = (
        reference.complete_distribution
        - reference_baseline.complete_distribution)
    comparison_excess = {}
    comparison_experiments = {
        "dt_half": dt_refined,
        "grid_5_over_4": grid_refined,
        "horizon_plus_one_sigma": horizon_refined,
    }
    for name, varied in comparison_experiments.items():
        varied_baseline = run_spatial_baseline(varied)
        projected_signal = _project_complete_distribution(
            comparisons[name], reference.y)
        projected_baseline = _project_complete_distribution(
            varied_baseline, reference.y)
        comparison_excess[f"{name}_excess"] = 0.5 * float(np.sum(np.abs(
            reference_excess - (projected_signal - projected_baseline))))
    return raw | comparison_excess
