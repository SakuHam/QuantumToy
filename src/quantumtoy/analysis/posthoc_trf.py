from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from analysis.emix import (
    build_Emix_from_phi_tau,
    build_Emix_density_from_phi_tau,
    make_rho,
)
from analysis.trf_branch import (
    CorridorMasks,
    TrfBranchDecision,
    build_trf_corridor_masks_vis,
    choose_trf_branch_by_corridor,
)
from analysis.worldline import (
    WorldlineResult,
    compute_posthoc_worldline_selected_rho,
)


# ============================================================
# Validation helpers
# ============================================================

def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def _assert_finite_scalar(x, name: str):
    _assert(np.isscalar(x), f"{name} must be scalar, got type={type(x)}")
    xf = float(x)
    _assert(np.isfinite(xf), f"{name} must be finite, got {x}")
    return xf


def _assert_positive_scalar(x, name: str):
    xf = _assert_finite_scalar(x, name)
    _assert(xf > 0.0, f"{name} must be > 0, got {x}")
    return xf


def _assert_nonnegative_scalar(x, name: str):
    xf = _assert_finite_scalar(x, name)
    _assert(xf >= 0.0, f"{name} must be >= 0, got {x}")
    return xf


def _assert_times(times: np.ndarray):
    _assert(isinstance(times, np.ndarray), "times must be np.ndarray")
    _assert(times.ndim == 1, f"times must be 1D, got ndim={times.ndim}")
    _assert(times.size > 0, "times must be non-empty")
    _assert(np.all(np.isfinite(times)), "times contains non-finite values")
    if times.size >= 2:
        dt = np.diff(times)
        _assert(np.all(dt > 0.0), "times must be strictly increasing")


def _assert_visible_xy(X_vis: np.ndarray, Y_vis: np.ndarray):
    _assert(isinstance(X_vis, np.ndarray), "X_vis must be np.ndarray")
    _assert(isinstance(Y_vis, np.ndarray), "Y_vis must be np.ndarray")
    _assert(X_vis.ndim == 2, f"X_vis must be 2D, got ndim={X_vis.ndim}")
    _assert(Y_vis.ndim == 2, f"Y_vis must be 2D, got ndim={Y_vis.ndim}")
    _assert(X_vis.shape == Y_vis.shape,
            f"X_vis shape {X_vis.shape} must match Y_vis shape {Y_vis.shape}")
    _assert(np.all(np.isfinite(X_vis)), "X_vis contains non-finite values")
    _assert(np.all(np.isfinite(Y_vis)), "Y_vis contains non-finite values")


# ============================================================
# Dataclass
# ============================================================

@dataclass
class PosthocTRFResult:
    Emix: np.ndarray | None
    Emix_density: np.ndarray | None
    base_rho: np.ndarray

    corridor_masks: CorridorMasks
    trf_info: TrfBranchDecision

    worldline: WorldlineResult | None

    rho_selected: np.ndarray | None


# ============================================================
# Pipeline
# ============================================================

def run_posthoc_trf_selection(
    frames_psi,
    phi_tau_frames: np.ndarray,
    times: np.ndarray,
    t_det: float,
    dx: float,
    dy: float,
    X_vis: np.ndarray,
    Y_vis: np.ndarray,

    x0: float,
    barrier_center_x: float,
    screen_center_x: float,
    slit_center_offset: float,
    v_est: float,

    sigmaT: float,
    tau_step: float,
    K_JITTER: int = 13,

    make_rho_mode: str = "density_product_oldstyle",
    blend_alpha: float = 0.5,

    use_adaptive_ref: bool = True,
    ref_t_min_frac: float = 0.30,
    ref_t_max_frac: float = 0.95,
    corridor_x_frac_start: float = 0.70,
    corridor_y_sigma: float = 1.3,
    corridor_x_weight_power: float = 2.5,
    valid_total_evidence_eps: float = 1e-12,

    use_worldline: bool = True,
    worldline_track_radius_px: int = 20,
    worldline_min_local_rel: float = 0.03,
    worldline_tube_sigma_px: float = 10.0,
    worldline_gain_strength: float = 2.0,
    worldline_outside_damp: float = 0.20,
    worldline_time_ramp_frac: float = 0.12,
) -> PosthocTRFResult:
    """
    Full posthoc TRF pipeline:

        phi_tau_frames + times + t_det -> Emix / Emix_density
        frames_psi × Emix( or Emix_density ) -> base_rho
        base_rho -> corridor branch decision
        branch decision -> optional worldline refinement

    Notes
    -----
    - For mode="density_product_oldstyle", Emix_density is required and Emix is not needed.
    - For other modes, Emix is required.
    """
    _assert_times(times)
    _assert_positive_scalar(dx, "dx")
    _assert_positive_scalar(dy, "dy")
    _assert_visible_xy(X_vis, Y_vis)

    _assert_finite_scalar(t_det, "t_det")
    _assert_finite_scalar(x0, "x0")
    _assert_finite_scalar(barrier_center_x, "barrier_center_x")
    _assert_finite_scalar(screen_center_x, "screen_center_x")
    _assert_finite_scalar(slit_center_offset, "slit_center_offset")
    _assert_positive_scalar(v_est, "v_est")
    _assert_finite_scalar(sigmaT, "sigmaT")
    _assert_positive_scalar(tau_step, "tau_step")
    _assert(isinstance(K_JITTER, int) and K_JITTER >= 1,
            f"K_JITTER must be int >= 1, got {K_JITTER}")

    _assert_nonnegative_scalar(ref_t_min_frac, "ref_t_min_frac")
    _assert_nonnegative_scalar(ref_t_max_frac, "ref_t_max_frac")
    _assert_nonnegative_scalar(corridor_x_frac_start, "corridor_x_frac_start")
    _assert_positive_scalar(corridor_y_sigma, "corridor_y_sigma")
    _assert_nonnegative_scalar(corridor_x_weight_power, "corridor_x_weight_power")
    _assert_nonnegative_scalar(valid_total_evidence_eps, "valid_total_evidence_eps")

    Emix = None
    Emix_density = None

    if make_rho_mode == "density_product_oldstyle":
        Emix_density = build_Emix_density_from_phi_tau(
            phi_tau_frames=phi_tau_frames,
            times=times,
            t_det=t_det,
            sigmaT=sigmaT,
            tau_step=tau_step,
            K_JITTER=K_JITTER,
        )
    else:
        Emix = build_Emix_from_phi_tau(
            phi_tau_frames=phi_tau_frames,
            times=times,
            t_det=t_det,
            sigmaT=sigmaT,
            tau_step=tau_step,
            K_JITTER=K_JITTER,
        )

    base_rho = make_rho(
        frames_psi=frames_psi,
        Emix=Emix,
        dx=dx,
        dy=dy,
        mode=make_rho_mode,
        blend_alpha=blend_alpha,
        Emix_density=Emix_density,
    )

    corridor_masks = build_trf_corridor_masks_vis(
        X_vis=X_vis,
        Y_vis=Y_vis,
        barrier_center_x=barrier_center_x,
        screen_center_x=screen_center_x,
        slit_center_offset=slit_center_offset,
        corridor_x_frac_start=corridor_x_frac_start,
        corridor_y_sigma=corridor_y_sigma,
        corridor_x_weight_power=corridor_x_weight_power,
    )

    trf_info = choose_trf_branch_by_corridor(
        base_rho=base_rho,
        times=times,
        masks=corridor_masks,
        x0=x0,
        barrier_center_x=barrier_center_x,
        screen_center_x=screen_center_x,
        slit_center_offset=slit_center_offset,
        v_est=v_est,
        use_adaptive_ref=use_adaptive_ref,
        ref_t_min_frac=ref_t_min_frac,
        ref_t_max_frac=ref_t_max_frac,
        valid_total_evidence_eps=valid_total_evidence_eps,
    )

    wl_result = None
    rho_selected = None

    if use_worldline and trf_info.valid:
        wl_result = compute_posthoc_worldline_selected_rho(
            base_rho=base_rho,
            X_vis=X_vis,
            Y_vis=Y_vis,
            trf_info=trf_info,
            dx=dx,
            dy=dy,
            track_radius_px=worldline_track_radius_px,
            min_local_rel=worldline_min_local_rel,
            tube_sigma_px=worldline_tube_sigma_px,
            gain_strength=worldline_gain_strength,
            outside_damp=worldline_outside_damp,
            time_ramp_frac=worldline_time_ramp_frac,
        )
        rho_selected = wl_result.rho_selected

    return PosthocTRFResult(
        Emix=Emix,
        Emix_density=Emix_density,
        base_rho=base_rho,
        corridor_masks=corridor_masks,
        trf_info=trf_info,
        worldline=wl_result,
        rho_selected=rho_selected,
    )