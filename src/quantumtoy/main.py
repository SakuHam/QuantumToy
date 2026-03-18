from __future__ import annotations

from pathlib import Path

import time

import numpy as np

from config import AppConfig
from core.grid import build_grid
from core.potentials import build_double_slit_and_caps
from core.utils import make_packet, norm_prob, make_packet_scout_main_scalar_seed
from core.simulation_types import PotentialSpec
from theories.registry import build_theory

from detection.factory import build_detector
from detection.debug.flux_batch_sampler import FluxBatchSampler

from analysis.emix import (
    detect_click_from_screen,
    build_backward_library,
    build_Emix_from_phi_tau,
    build_Emix_density_from_phi_tau,
    make_rho,
    make_emix_density,
)
from analysis.ridge import compute_ridge_xy
from analysis.current import alignment_and_diagnostics_from_state_frames
from analysis.bohmian import (
    build_velocity_frames_from_state,
    make_bohmian_initial_points,
    integrate_bohmian_trajectories,
)
from analysis.debug_continuity import continuity_residual_from_state_frames

from file.run_io import save_run_bundle


# ============================================================
# Helpers
# ============================================================

def estimate_slit_separation_from_masks(grid, potential):
    """
    Estimate slit separation d from slit masks.

    Returns
    -------
    d : float
        Distance between slit centers in y-direction.
    """

    slit1 = potential.slit1_mask
    slit2 = potential.slit2_mask

    if not np.any(slit1):
        raise RuntimeError("slit1_mask empty")

    if not np.any(slit2):
        raise RuntimeError("slit2_mask empty")

    y1 = np.mean(grid.Y[slit1])
    y2 = np.mean(grid.Y[slit2])

    d = abs(y2 - y1)

    y1_vals = grid.Y[potential.slit1_mask]
    y2_vals = grid.Y[potential.slit2_mask]

    y1c = np.median(y1_vals)
    y2c = np.median(y2_vals)
    d = abs(y2c - y1c)

    a1 = np.max(y1_vals) - np.min(y1_vals)
    a2 = np.max(y2_vals) - np.min(y2_vals)

    print(f"[SLITS] center1_y={y1:.6f} center2_y={y2:.6f} separation d={d:.6f} slit1 width={a1} slit2 width={a2}")

    return d

def estimate_group_velocity(cfg, theory) -> float:
    """
    Estimate packet group velocity for sigmaT initialization.

    For Dirac-like theories (spinor state), use relativistic estimate:
        v = c^2 p / E,   p = hbar * k0x
        E = sqrt((c p)^2 + (m c^2)^2)

    For scalar / Schrödinger-like theories, fall back to:
        v ~ k0x / m
    """
    theory_name = getattr(cfg, "THEORY_NAME", "").lower()
    class_name = theory.__class__.__name__.lower()

    is_dirac_like = ("dirac" in theory_name) or ("dirac" in class_name)

    if is_dirac_like:
        p0 = float(theory.hbar * cfg.k0x)
        mc2 = float(theory.m_mass * theory.c_light**2)
        E0 = float(np.sqrt((theory.c_light * p0) ** 2 + mc2**2))
        return float((theory.c_light**2 * p0) / (E0 + 1e-30))

    return float(cfg.k0x / (cfg.m_mass + 1e-30))


def _forward_density_from_state_vis_frames(state_vis_frames: np.ndarray) -> np.ndarray:
    if state_vis_frames.ndim == 3:
        return (np.abs(state_vis_frames) ** 2).astype(float)
    if state_vis_frames.ndim == 4:
        return np.sum(np.abs(state_vis_frames) ** 2, axis=1).astype(float)
    raise ValueError(f"Unsupported state_vis_frames ndim={state_vis_frames.ndim}")


def build_Emix_density_reference(
    phi_tau_frames: np.ndarray,
    times: np.ndarray,
    t_det: float,
    sigmaT: float,
    tau_step: float,
    K_JITTER: int = 13,
) -> np.ndarray:
    """
    Reference builder matching the old working logic:
      - gaussian weights normalized once globally
      - valid subset NOT re-normalized
      - nearest saved backward frame via round()
    """
    Nt_ = len(times)
    halfK = K_JITTER // 2
    idx_det2 = int(np.argmin(np.abs(times - t_det)))

    k_inds = np.arange(idx_det2 - halfK, idx_det2 + halfK + 1)
    k_inds = np.clip(k_inds, 0, Nt_ - 1)
    k_inds = np.unique(k_inds)

    Tk = times[k_inds]

    if sigmaT <= 0:
        w = np.zeros_like(Tk, dtype=float)
        w[int(np.argmin(np.abs(Tk - t_det)))] = 1.0
    else:
        w = np.exp(-0.5 * ((Tk - t_det) / sigmaT) ** 2)
        s = np.sum(w)
        if s > 0:
            w = w / s

    if phi_tau_frames.ndim == 3:
        phi_tau_density = (np.abs(phi_tau_frames) ** 2).astype(float)
    elif phi_tau_frames.ndim == 4:
        phi_tau_density = np.sum(np.abs(phi_tau_frames) ** 2, axis=1).astype(float)
    else:
        raise ValueError(f"Unsupported phi_tau_frames ndim={phi_tau_frames.ndim}")

    out = np.zeros_like(phi_tau_density, dtype=float)

    for i, ti in enumerate(times):
        tau = Tk - ti
        valid = tau >= 0.0
        if not np.any(valid):
            continue

        j = np.rint(tau[valid] / tau_step).astype(int)
        j = np.clip(j, 0, Nt_ - 1)

        out[i] = np.sum(w[valid][:, None, None] * phi_tau_density[j], axis=0)

    return out


def make_rho_density_product_oldstyle_reference(
    state_vis_frames: np.ndarray,
    Emix_density: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    rho_fwd = _forward_density_from_state_vis_frames(state_vis_frames)
    out = np.zeros_like(rho_fwd, dtype=float)

    for i in range(rho_fwd.shape[0]):
        rr = rho_fwd[i] * Emix_density[i]
        s = float(np.sum(rr) * dx * dy)
        if s > 0:
            rr = rr / s
        out[i] = rr

    return out


def run_diagnostics(
    cfg,
    grid,
    theory,
    potential,
    frames_density: np.ndarray,
    state_vis_frames: np.ndarray | None,
    times: np.ndarray,
    tau_step: float,
    idx_det: int,
    t_det: float,
    x_click: float,
    y_click: float,
    screen_int: np.ndarray,
    phi_tau_frames: np.ndarray,
    sigma_diag: float,
):
    print("\n================ DIAGNOSTIC BLOCK START ================\n")

    Nt = len(times)

    # --------------------------------------------------------
    # D1) screen_int reference check
    # --------------------------------------------------------
    screen_int_ref = np.array(
        [np.sum(frames_density[i][potential.screen_mask_vis]) * grid.dx * grid.dy for i in range(Nt)],
        dtype=float,
    )

    screen_diff = float(np.max(np.abs(screen_int - screen_int_ref)))
    print(f"[DIAG D1] screen_int max abs diff = {screen_diff:.6e}")
    assert np.allclose(screen_int, screen_int_ref, rtol=1e-12, atol=1e-12), \
        f"screen_int mismatch: max abs diff = {screen_diff}"

    # --------------------------------------------------------
    # D2) click sanity
    # --------------------------------------------------------
    assert potential.screen_mask_vis.shape == frames_density[idx_det].shape, \
        "screen_mask_vis shape mismatch at detection frame"

    click_mask = (
        (np.abs(grid.X_vis - x_click) < 0.5 * grid.dx + 1e-12) &
        (np.abs(grid.Y_vis - y_click) < 0.5 * grid.dy + 1e-12)
    )
    n_click_cells = int(np.count_nonzero(click_mask))
    print(f"[DIAG D2] click cell matches on visible grid = {n_click_cells}")
    assert np.any(click_mask), "Click does not land on any visible-grid cell"

    click_on_screen = bool(np.any(click_mask & potential.screen_mask_vis))
    print(f"[DIAG D2] click lies on detector mask = {click_on_screen}")
    assert click_on_screen, "Click is not on the detector mask"

    dx_screen = abs(x_click - cfg.screen_center_x)
    print(f"[DIAG D2] |x_click - screen_center_x| = {dx_screen:.6e}")
    assert dx_screen < (cfg.screen_eval_width + grid.dx), \
        f"x_click={x_click} seems too far from detector screen center {cfg.screen_center_x}"

    # --------------------------------------------------------
    # D3) backward frame 0 must match initialized click state cropped
    # --------------------------------------------------------
    phi0_full = theory.initialize_click_state(x_click, y_click, cfg.sigma_click)
    if phi0_full.ndim == 2:
        phi0_vis_ref = phi0_full[grid.ys, grid.xs]
    elif phi0_full.ndim == 3:
        phi0_vis_ref = phi0_full[:, grid.ys, grid.xs]
    else:
        raise ValueError(f"Unsupported phi0_full ndim={phi0_full.ndim}")

    phi0_diff = float(np.max(np.abs(phi_tau_frames[0] - phi0_vis_ref)))
    print(f"[DIAG D3] phi_tau_frames[0] max abs diff vs click-state crop = {phi0_diff:.6e}")
    assert np.allclose(phi_tau_frames[0], phi0_vis_ref, rtol=1e-12, atol=1e-12), \
        f"phi_tau_frames[0] mismatch: max abs diff = {phi0_diff}"

    # --------------------------------------------------------
    # D4) backward library first few steps must match direct replay
    # --------------------------------------------------------
    ncheck = min(5, Nt - 1)
    for i in range(ncheck):
        phi_test = theory.initialize_click_state(x_click, y_click, cfg.sigma_click)
        for _ in range((i + 1) * cfg.save_every):
            phi_test = theory.step_backward_adjoint(phi_test, cfg.dt).state

        if phi_test.ndim == 2:
            phi_test_vis = phi_test[grid.ys, grid.xs]
        elif phi_test.ndim == 3:
            phi_test_vis = phi_test[:, grid.ys, grid.xs]
        else:
            raise ValueError(f"Unsupported phi_test ndim={phi_test.ndim}")

        step_diff = float(np.max(np.abs(phi_tau_frames[i + 1] - phi_test_vis)))
        print(f"[DIAG D4] backward replay frame {i+1} max abs diff = {step_diff:.6e}")
        assert np.allclose(phi_tau_frames[i + 1], phi_test_vis, rtol=1e-9, atol=1e-9), \
            f"Backward library mismatch at frame {i+1}: max abs diff = {step_diff}"

    # --------------------------------------------------------
    # D5) Emix density reference check
    # --------------------------------------------------------
    Emix_density_ref = build_Emix_density_reference(
        phi_tau_frames=phi_tau_frames,
        times=times,
        t_det=t_det,
        sigmaT=sigma_diag,
        tau_step=tau_step,
        K_JITTER=cfg.K_JITTER,
    )

    Emix_density_new = build_Emix_density_from_phi_tau(
        phi_tau_frames=phi_tau_frames,
        times=times,
        t_det=t_det,
        sigmaT=sigma_diag,
        tau_step=tau_step,
        K_JITTER=cfg.K_JITTER,
    )

    emix_density_diff = float(np.max(np.abs(Emix_density_new - Emix_density_ref)))
    print(f"[DIAG D5] Emix_density max abs diff vs reference = {emix_density_diff:.6e}")
    assert np.allclose(Emix_density_new, Emix_density_ref, rtol=1e-10, atol=1e-10), \
        f"Emix_density mismatch: max abs diff = {emix_density_diff}"

    # --------------------------------------------------------
    # D6) oldstyle rho reference check
    # --------------------------------------------------------
    if state_vis_frames is not None:
        rho_old_ref = make_rho_density_product_oldstyle_reference(
            state_vis_frames=state_vis_frames,
            Emix_density=Emix_density_ref,
            dx=grid.dx,
            dy=grid.dy,
        )

        rho_old_new = make_rho(
            frames_psi=state_vis_frames,
            Emix=None,
            Emix_density=Emix_density_new,
            dx=grid.dx,
            dy=grid.dy,
            mode="density_product_oldstyle",
        )

        rho_old_diff = float(np.max(np.abs(rho_old_new - rho_old_ref)))
        print(f"[DIAG D6] rho_oldstyle max abs diff vs reference = {rho_old_diff:.6e}")
        assert np.allclose(rho_old_new, rho_old_ref, rtol=1e-10, atol=1e-10), \
            f"rho_oldstyle mismatch: max abs diff = {rho_old_diff}"

    # --------------------------------------------------------
    # D7) checkpoint summaries
    # --------------------------------------------------------
    check_ids = sorted(set([0, Nt // 2, Nt - 1]))
    for i in check_ids:
        fi = frames_density[i]
        assert np.all(np.isfinite(fi)), f"frames_density[{i}] non-finite"
        assert np.max(fi) > 0.0, f"frames_density[{i}] is all zeros"

        argmax_ij = np.unravel_index(np.argmax(fi), fi.shape)
        print(
            f"[DIAG D7/FWD] i={i:4d} t={times[i]:7.3f} "
            f"sum={np.sum(fi) * grid.dx * grid.dy:.6e} "
            f"max={np.max(fi):.6e} "
            f"argmax={argmax_ij}"
        )

    for i in check_ids:
        e_ref = Emix_density_ref[i]
        print(
            f"[DIAG D7/EMIX_DENS_REF] i={i:4d} t={times[i]:7.3f} "
            f"sum={np.sum(e_ref) * grid.dx * grid.dy:.6e} "
            f"max={np.max(e_ref):.6e}"
        )

    if state_vis_frames is not None:
        rho_old_ref = make_rho_density_product_oldstyle_reference(
            state_vis_frames=state_vis_frames,
            Emix_density=Emix_density_ref,
            dx=grid.dx,
            dy=grid.dy,
        )
        for i in check_ids:
            rr = rho_old_ref[i]
            print(
                f"[DIAG D7/RHO_OLD_REF] i={i:4d} t={times[i]:7.3f} "
                f"sum={np.sum(rr) * grid.dx * grid.dy:.6e} "
                f"max={np.max(rr):.6e}"
            )

    print("\n================ DIAGNOSTIC BLOCK END ==================\n")


def build_sigma_dependent_products(
    *,
    cfg,
    theory,
    grid,
    state_vis_frames: np.ndarray,
    phi_tau_frames: np.ndarray,
    times: np.ndarray,
    t_det: float,
    tau_step: float,
):
    def _builder(sigmaT: float):
        Emix = build_Emix_from_phi_tau(
            phi_tau_frames=phi_tau_frames,
            times=times,
            t_det=t_det,
            sigmaT=sigmaT,
            tau_step=tau_step,
            K_JITTER=cfg.K_JITTER,
        )

        Emix_density_old = build_Emix_density_from_phi_tau(
            phi_tau_frames=phi_tau_frames,
            times=times,
            t_det=t_det,
            sigmaT=sigmaT,
            tau_step=tau_step,
            K_JITTER=cfg.K_JITTER,
        )

        rho = make_rho(
            frames_psi=state_vis_frames,
            Emix=Emix,
            Emix_density=Emix_density_old,
            dx=grid.dx,
            dy=grid.dy,
            mode=cfg.RHO_MODE,
            blend_alpha=cfg.RHO_BLEND_ALPHA,
        )

        rx, ry, rs = compute_ridge_xy(
            frames_psi=state_vis_frames,
            Emix=Emix,
            x_vis_1d=grid.x_vis_1d,
            y_vis_1d=grid.y_vis_1d,
            mode=cfg.RIDGE_MODE,
            top_q=cfg.CENTROID_TOP_Q,
            radius=cfg.LOCALMAX_RADIUS,
            alpha_smooth=cfg.LOCALMAX_SMOOTH_ALPHA,
        )

        cos_th = speed = ux = uy = div_v = None
        cos_th, speed, ux, uy, div_v = alignment_and_diagnostics_from_state_frames(
            theory=theory,
            state_vis_frames=state_vis_frames,
            ridge_x=rx,
            ridge_y=ry,
            x_vis_1d=grid.x_vis_1d,
            y_vis_1d=grid.y_vis_1d,
            dx=grid.dx,
            dy=grid.dy,
            enable_divergence=cfg.ENABLE_DIVERGENCE_DIAGNOSTIC,
            arrow_spatial_avg=cfg.ARROW_SPATIAL_AVG,
            arrow_avg_radius=cfg.ARROW_AVG_RADIUS,
            arrow_avg_gauss_sigma=cfg.ARROW_AVG_GAUSS_SIGMA,
            arrow_temporal_smooth=cfg.ARROW_TEMPORAL_SMOOTH,
            arrow_smooth_alpha=cfg.ARROW_SMOOTH_ALPHA,
            align_eps_rho=cfg.ALIGN_EPS_RHO,
            align_eps_speed=cfg.ALIGN_EPS_SPEED,
        )

        return rho, Emix, rx, ry, rs, cos_th, speed, ux, uy, div_v

    return _builder


# ============================================================
# Main
# ============================================================

def main():
    cfg = AppConfig()
    cfg.dump_selected()
    batch_fast_mode = bool(getattr(cfg, "BATCH_FAST_MODE", False))

    # --------------------------------------------------------
    # 1) Build grid / potential / theory / detector
    # --------------------------------------------------------
    grid = build_grid(
        visible_lx=cfg.VISIBLE_LX,
        visible_ly=cfg.VISIBLE_LY,
        n_visible_x=cfg.N_VISIBLE_X,
        n_visible_y=cfg.N_VISIBLE_Y,
        pad_factor=cfg.PAD_FACTOR,
    )

    DEBUG_FREE_CASE = False

    if DEBUG_FREE_CASE:
        zeros = np.zeros_like(grid.X, dtype=float)
        false_mask = np.zeros_like(grid.X, dtype=bool)
        false_mask_vis = np.zeros((grid.n_visible_y, grid.n_visible_x), dtype=bool)

        potential = PotentialSpec(
            V_real=zeros.copy(),
            W=zeros.copy(),
            screen_mask_full=false_mask.copy(),
            screen_mask_vis=false_mask_vis.copy(),
            barrier_core=false_mask.copy(),
            slit1_mask=false_mask.copy(),
            slit2_mask=false_mask.copy(),
        )
    else:
        potential = build_double_slit_and_caps(grid, cfg)

    estimate_slit_separation_from_masks(grid, potential)

    theory = build_theory(cfg, grid, potential)
    detector = build_detector(cfg, grid)

    # --------------------------------------------------------
    # 1b) Optional cheap batch sampler from one forward run
    # --------------------------------------------------------
    enable_flux_batch_sampler = getattr(cfg, "ENABLE_FLUX_BATCH_SAMPLER", True)
    flux_batch_num_samples = int(getattr(cfg, "FLUX_BATCH_NUM_SAMPLES", 10_000))
    flux_batch_rng_seed = getattr(cfg, "FLUX_BATCH_RNG_SEED", 12345)
    flux_batch_sample_sigma_x = float(getattr(cfg, "FLUX_BATCH_SAMPLE_SIGMA_X", 0.0))
    flux_batch_sample_sigma_y = float(getattr(cfg, "FLUX_BATCH_SAMPLE_SIGMA_Y", 0.0))
    break_on_detector_click = bool(getattr(cfg, "BREAK_ON_DETECTOR_CLICK", True))

    batch_sampler = None
    pseudo_clicks = None

    if enable_flux_batch_sampler:
        det_gate_x = float(getattr(detector, "detector_gate_center_x", cfg.screen_center_x))
        det_gate_wx = float(getattr(detector, "detector_gate_width", cfg.screen_eval_width))
        det_gate_y = float(getattr(detector, "detector_gate_center_y", 0.0))
        det_gate_wy = float(getattr(detector, "detector_gate_width_y", -1.0))

        batch_sampler = FluxBatchSampler(
            grid=grid,
            gate_center_x=det_gate_x,
            gate_width_x=det_gate_wx,
            gate_center_y=det_gate_y,
            gate_width_y=det_gate_wy,
            hbar=float(getattr(cfg, "hbar", 1.0)),
            mass=float(getattr(cfg, "m_mass", 1.0)),
            sample_sigma_x=flux_batch_sample_sigma_x,
            sample_sigma_y=flux_batch_sample_sigma_y,
            rng_seed=flux_batch_rng_seed,
        )

        print(
            "[BATCH] enabled: "
            f"samples={flux_batch_num_samples}, "
            f"gate_center_x={det_gate_x}, gate_width_x={det_gate_wx}"
        )

    # --------------------------------------------------------
    # 2) Initial state
    # --------------------------------------------------------
    psi0 = make_packet(
        X=grid.X,
        Y=grid.Y,
        x0=cfg.x0,
        y0=cfg.y0,
        sigma0=cfg.sigma0,
        k0x=cfg.k0x,
        k0y=cfg.k0y,
    )
#    psi0 = make_packet_scout_main_scalar_seed(
#        grid.X,
#        grid.Y,
#        main_x0=-15.0,
#        scout_x0=-6.0,
#        main_kx=4.0,
#        scout_kx=4.0,
#        scout_amp=0.15,
#        dx=grid.dx,
#        dy=grid.dy,
#    )

    state = theory.initialize_state(psi0)
    detector.reset()

    # --------------------------------------------------------
    # 3) Forward simulation
    # --------------------------------------------------------
    frames_density = []
    state_vis_frames = []
    times = []
    norms = []

    detector_diags = []
    detector_clicked = False
    det_result_final = None

    print(
        f"Forward simulation starts... "
        f"theory={cfg.THEORY_NAME}, detector={getattr(cfg, 'DETECTOR_NAME', 'unknown')}"
    )

    if cfg.THEORY_NAME == "dirac":
        vx_est, vy_est, sp_est = theory.expected_group_velocity(cfg.k0x, cfg.k0y)
        print(f"[DIRAC V_EST] vx≈{vx_est:.6f}, vy≈{vy_est:.6f}, |v|≈{sp_est:.6f}")

    t_total_start = time.perf_counter()

    t_step = 0.0
    t_detector = 0.0
    t_batch = 0.0
    t_density = 0.0

    actual_last_step = 0

    for n in range(cfg.n_steps + 1):

        actual_last_step = n
        t_now = n * cfg.dt

        # -------------------------
        # Batch sampler update
        # -------------------------
        if batch_sampler is not None and state.ndim == 2:
            t0 = time.perf_counter()
            batch_sampler.update(state, cfg.dt)
            t_batch += time.perf_counter() - t0

        # -------------------------
        # Detector
        # -------------------------
        t0 = time.perf_counter()
        det_res = detector.step(state, cfg.dt, t=t_now)
        t_detector += time.perf_counter() - t0

        detector_diags.append(det_res.aux if det_res.aux is not None else {})

        need_save = (n % cfg.save_every == 0)

        if need_save:

            # -------------------------
            # Density + norm
            # -------------------------
            t0 = time.perf_counter()

            rho = theory.density(state)
            norm_now = norm_prob(rho, grid.dx, grid.dy)

            frames_density.append(rho[grid.ys, grid.xs].copy())
            times.append(t_now)
            norms.append(norm_now)

            if cfg.SAVE_COMPLEX_STATE_FRAMES:
                if state.ndim == 2:
                    state_vis = state[grid.ys, grid.xs].copy()
                elif state.ndim == 3:
                    state_vis = state[:, grid.ys, grid.xs].copy()
                else:
                    raise ValueError(f"Unsupported state ndim={state.ndim}")

                state_vis_frames.append(state_vis)

            t_density += time.perf_counter() - t0

            if (len(frames_density) % 20) == 0:
                print(
                    f"[FWD] step {n:5d}/{cfg.n_steps}, "
                    f"t={times[-1]:7.3f}, norm≈{norm_now:.6f}, "
                    f"frames={len(frames_density)}"
                )

        # -------------------------
        # Click detection
        # -------------------------
        if det_res.clicked and not detector_clicked:

            detector_clicked = True
            det_result_final = det_res

            print(
                f"[DETECTOR] clicked at step={n}, "
                f"t={t_now:.6f}, "
                f"x={det_res.click_x}, "
                f"y={det_res.click_y}"
            )

            if break_on_detector_click:
                print("[FWD] breaking on detector click.")
                break

        # -------------------------
        # Forward step
        # -------------------------
        if n < cfg.n_steps:

            t0 = time.perf_counter()

            state = theory.step_forward(state, cfg.dt).state

            t_step += time.perf_counter() - t0

    t_total = time.perf_counter() - t_total_start

    print("\n================ PROFILE =================")

    def pct(x):
        return 100.0 * x / t_total if t_total > 0 else 0.0

    print(f"Total runtime      : {t_total:8.3f} s")

    print(
        f"step_forward       : {t_step:8.3f} s "
        f"({pct(t_step):5.1f}%)"
    )

    print(
        f"detector.step      : {t_detector:8.3f} s "
        f"({pct(t_detector):5.1f}%)"
    )

    print(
        f"batch_sampler      : {t_batch:8.3f} s "
        f"({pct(t_batch):5.1f}%)"
    )

    print(
        f"density + norm     : {t_density:8.3f} s "
        f"({pct(t_density):5.1f}%)"
    )

    print("==========================================\n")

    frames_density = np.asarray(frames_density, dtype=float)
    times = np.asarray(times, dtype=float)
    norms = np.asarray(norms, dtype=float)

    if cfg.SAVE_COMPLEX_STATE_FRAMES:
        state_vis_frames = np.asarray(state_vis_frames)
    else:
        state_vis_frames = None

    Nt = len(times)
    tau_step = cfg.save_every * cfg.dt

    print("Forward done.")

    if batch_sampler is not None:
        try:
            pseudo_clicks = batch_sampler.sample_clicks(flux_batch_num_samples)
            print(
                "[BATCH] sampled pseudo-clicks: "
                f"n={len(pseudo_clicks)}, "
                f"updates={batch_sampler.num_updates}, "
                f"total_flux={batch_sampler.total_flux_accum:.6e}"
            )
        except Exception as e:
            pseudo_clicks = None
            print(f"[BATCH] sampling skipped: {e}")

    # --------------------------------------------------------
    # 4) Forward debug checks
    # --------------------------------------------------------
    t_final = actual_last_step * cfg.dt
    print(f"[TIMECHK] dt={cfg.dt}")
    print(f"[TIMECHK] n_steps={cfg.n_steps}")
    print(f"[TIMECHK] save_every={cfg.save_every}")
    print(f"[TIMECHK] t_final={t_final:.6f}")

    v_est_dbg = estimate_group_velocity(cfg, theory)
    x_travel_est = abs(v_est_dbg) * t_final
    dist_to_screen_est = abs(cfg.screen_center_x - cfg.x0)

    print(f"[TIMECHK] v_est≈{v_est_dbg:.6f}")
    print(f"[TIMECHK] estimated travel≈{x_travel_est:.6f}")
    print(f"[TIMECHK] distance x0->screen≈{dist_to_screen_est:.6f}")
    print(f"[TIMECHK] travel/distance≈{x_travel_est / (dist_to_screen_est + 1e-30):.6f}")

    check_ids = sorted(set([0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt - 1]))
    for i in check_ids:
        fi = frames_density[i]
        iy, ix = np.unravel_index(np.argmax(fi), fi.shape)
        x_peak = grid.X_vis[iy, ix]
        y_peak = grid.Y_vis[iy, ix]
        mass_vis = np.sum(fi) * grid.dx * grid.dy
        print(
            f"[FWDCHK] i={i:4d} t={times[i]:8.4f} "
            f"mass_vis={mass_vis:.6e} "
            f"max={np.max(fi):.6e} "
            f"peak=({x_peak:.4f},{y_peak:.4f})"
        )

    if cfg.THEORY_NAME == "dirac" and state_vis_frames is not None:
        theory.debug_packet_summary(
            f"vis frame {check_ids[-1]}",
            state_vis_frames[check_ids[-1]],
            X_like=grid.X_vis,
            Y_like=grid.Y_vis,
        )

    # --------------------------------------------------------
    # 5) Optional continuity debug in free case
    # --------------------------------------------------------
    if DEBUG_FREE_CASE and cfg.SAVE_COMPLEX_STATE_FRAMES and state_vis_frames is not None:
        rms_mean, rms_max, abs_max = continuity_residual_from_state_frames(
            theory=theory,
            state_vis_frames=state_vis_frames,
            dx=grid.dx,
            dy=grid.dy,
            dt=cfg.save_every * cfg.dt,
        )

        print(
            "[CONTINUITY] "
            f"RMS_mean≈{rms_mean:.3e}, "
            f"RMS_max≈{rms_max:.3e}, "
            f"ABS_max≈{abs_max:.3e}"
        )

        x_mean = float(np.sum(rho * grid.X) * grid.dx * grid.dy)
        jx, jy, _ = theory.current(state)
        jx_tot = float(np.sum(jx) * grid.dx * grid.dy)
        jy_tot = float(np.sum(jy) * grid.dx * grid.dy)

        print(
            f"[FREEDBG] step={n:5d} t={n * cfg.dt:7.3f} "
            f"x_mean={x_mean: .4f} jx_tot={jx_tot: .4e} jy_tot={jy_tot: .4e}"
        )

        if not np.any(potential.screen_mask_vis):
            print("[DEBUG] screen_mask_vis empty -> skipping click/backward/Emix analysis.")
            return

    # --------------------------------------------------------
    # 6) Detection time + click
    # --------------------------------------------------------
    if detector_clicked and det_result_final is not None:
        x_click = float(det_result_final.click_x)
        y_click = float(det_result_final.click_y)

        idx_det = int(
            np.argmin(
                np.abs(
                    times - float(
                        det_result_final.click_time
                        if det_result_final.click_time is not None
                        else times[-1]
                    )
                )
            )
        )
        t_det = float(times[idx_det])

        screen_int = np.array(
            [np.sum(frames_density[i][potential.screen_mask_vis]) * grid.dx * grid.dy for i in range(Nt)],
            dtype=float,
        )

        print(
            f"[CLICK] detector-driven click: "
            f"t_det≈{t_det:.3f}, click=({x_click:.3f}, {y_click:.3f}), "
            f"detector={getattr(cfg, 'DETECTOR_NAME', 'unknown')}"
        )
    else:
        idx_det, t_det, x_click, y_click, screen_int = detect_click_from_screen(
            frames_density=frames_density,
            times=times,
            screen_mask_vis=potential.screen_mask_vis,
            dx=grid.dx,
            dy=grid.dy,
            X_vis=grid.X_vis,
            Y_vis=grid.Y_vis,
            rng_seed=cfg.CLICK_RNG_SEED,
            click_mode=cfg.CLICK_MODE,
            force_click_x=cfg.FORCE_CLICK_X,
            force_click_y=cfg.FORCE_CLICK_Y,
        )

        print(
            f"[CLICK] fallback screen click: "
            f"t_det≈{t_det:.3f}, click=({x_click:.3f}, {y_click:.3f}), "
            f"click_mode={cfg.CLICK_MODE}"
        )

    print(
        f"[SCREEN] max={np.max(screen_int):.6e} "
        f"argmax_i={np.argmax(screen_int)} "
        f"t_argmax={times[int(np.argmax(screen_int))]:.6f} "
        f"first={screen_int[0]:.6e} last={screen_int[-1]:.6e}"
    )

    # --------------------------------------------------------
    # 6b) Fast batch-only exit
    # --------------------------------------------------------
    if batch_fast_mode:
        output_prefix = getattr(cfg, "OUTPUT_PREFIX", None)
        if not output_prefix:
            output_mp4 = getattr(cfg, "OUTPUT_MP4", "output.mp4")
            output_prefix = str(Path(output_mp4).with_suffix(""))

        if batch_sampler is not None:
            batch_summary_path = f"{output_prefix}_flux_summary.json"
            batch_clicks_json_path = f"{output_prefix}_pseudo_clicks.json"
            batch_clicks_jsonl_path = f"{output_prefix}_pseudo_clicks.jsonl"

            batch_sampler.save_summary_json(batch_summary_path)

            if pseudo_clicks is not None:
                batch_sampler.save_clicks_json(batch_clicks_json_path, pseudo_clicks)
                batch_sampler.append_clicks_jsonl(
                    batch_clicks_jsonl_path,
                    pseudo_clicks,
                    run_id=str(Path(output_prefix).name),
                    theory_name=str(cfg.THEORY_NAME),
                    detector_name=str(getattr(cfg, "DETECTOR_NAME", "unknown")),
                )

                print(
                    "[BATCH_FAST_MODE] saved: "
                    f"{batch_summary_path}, "
                    f"{batch_clicks_json_path}, "
                    f"{batch_clicks_jsonl_path}"
                )
            else:
                print(f"[BATCH_FAST_MODE] saved summary only: {batch_summary_path}")

        print("[BATCH_FAST_MODE] Skipping backward/Emix/Bohmian pipeline.")
        print("Simulation done.")
        return
    
    # --------------------------------------------------------
    # 7) Backward library
    # --------------------------------------------------------
    print("Backward library: computing phi_tau frames...")
    phi_tau_frames = build_backward_library(
        theory=theory,
        grid=grid,
        times=times,
        tau_step=tau_step,
        x_click=x_click,
        y_click=y_click,
        sigma_click=cfg.sigma_click,
        save_every=cfg.save_every,
        print_every_frames=20,
    )
    print("Backward library done.")

    # --------------------------------------------------------
    # 8) Diagnostic block
    # --------------------------------------------------------
    v_est_diag = estimate_group_velocity(cfg, theory)
    L_gap_diag = cfg.screen_center_x - cfg.barrier_center_x
    t_gap_diag = L_gap_diag / (abs(v_est_diag) + 1e-12)
    sigma_diag = 0.60 * t_gap_diag

    run_diagnostics(
        cfg=cfg,
        grid=grid,
        theory=theory,
        potential=potential,
        frames_density=frames_density,
        state_vis_frames=state_vis_frames,
        times=times,
        tau_step=tau_step,
        idx_det=idx_det,
        t_det=t_det,
        x_click=x_click,
        y_click=y_click,
        screen_int=screen_int,
        phi_tau_frames=phi_tau_frames,
        sigma_diag=sigma_diag,
    )

    # --------------------------------------------------------
    # 9) sigmaT-dependent products
    # --------------------------------------------------------
    if not cfg.SAVE_COMPLEX_STATE_FRAMES or state_vis_frames is None:
        raise RuntimeError(
            "This pipeline requires SAVE_COMPLEX_STATE_FRAMES=True "
            "for amplitude-level Emix/ridge analysis and later visualization."
        )

    build_all_for_sigma = build_sigma_dependent_products(
        cfg=cfg,
        theory=theory,
        grid=grid,
        state_vis_frames=state_vis_frames,
        phi_tau_frames=phi_tau_frames,
        times=times,
        t_det=t_det,
        tau_step=tau_step,
    )

    v_est = estimate_group_velocity(cfg, theory)
    L_gap = cfg.screen_center_x - cfg.barrier_center_x
    t_gap = L_gap / (abs(v_est) + 1e-12)

    sigma_init = 0.60 * t_gap

    (
        rho_init,
        Emix_init,
        ridge_x_init,
        ridge_y_init,
        ridge_s_init,
        cos_th_init,
        speed_init,
        ux_init,
        uy_init,
        div_v_init,
    ) = build_all_for_sigma(sigma_init)

    if cfg.USE_FIXED_DISPLAY_SCALE:
        vref = float(np.quantile(rho_init, cfg.DISPLAY_Q))
        if vref <= 0:
            vref = 1.0
    else:
        vref = 1.0

    if speed_init is not None:
        vv = speed_init[np.isfinite(speed_init)]
        speed_ref = float(np.quantile(vv, 0.80)) if vv.size > 0 else 1.0
    else:
        speed_ref = 1.0

    # --------------------------------------------------------
    # 10) Bohmian precompute
    # --------------------------------------------------------
    bohm_traj_x = bohm_traj_y = bohm_traj_alive = None
    bohm_init_points = []

    if cfg.ENABLE_BOHMIAN_OVERLAY:
        print("Bohmian overlay: computing velocity frames...")

        vx_frames, vy_frames, rho_frames = build_velocity_frames_from_state(
            theory=theory,
            state_vis_frames=state_vis_frames,
            eps_rho=cfg.ALIGN_EPS_RHO,
        )

        bohm_init_points = make_bohmian_initial_points(
            mode=cfg.BOHMIAN_INIT_MODE,
            ntraj=cfg.BOHMIAN_N_TRAJ,
            custom_points=cfg.BOHMIAN_CUSTOM_POINTS,
            ridge_x0=ridge_x_init[0],
            ridge_y0=ridge_y_init[0],
            x0_packet=cfg.x0,
            y0_packet=cfg.y0,
            psi0_vis=state_vis_frames[0],
            x_vis_1d=grid.x_vis_1d,
            y_vis_1d=grid.y_vis_1d,
            jitter=cfg.BOHMIAN_INIT_JITTER,
            rng_seed=cfg.BOHMIAN_RNG_SEED,
            with_replacement=cfg.BOHMIAN_WITH_REPLACEMENT,
        )

        print(
            f"Bohmian overlay: integrating {len(bohm_init_points)} trajectories "
            f"({'RK4' if cfg.BOHMIAN_USE_RK4 else 'Euler'})..."
        )

        bohm_traj_x, bohm_traj_y, bohm_traj_alive = integrate_bohmian_trajectories(
            vx_frames=vx_frames,
            vy_frames=vy_frames,
            rho_frames=rho_frames,
            times=times,
            tau_step=tau_step,
            x_vis_1d=grid.x_vis_1d,
            y_vis_1d=grid.y_vis_1d,
            dx=grid.dx,
            dy=grid.dy,
            init_points=bohm_init_points,
            stop_outside_visible=cfg.BOHMIAN_STOP_OUTSIDE_VISIBLE,
            x_vis_min=grid.x_vis_min,
            x_vis_max=grid.x_vis_max,
            y_vis_min=grid.y_vis_min,
            y_vis_max=grid.y_vis_max,
            stop_on_low_rho=cfg.BOHMIAN_STOP_ON_LOW_RHO,
            min_rho=cfg.BOHMIAN_MIN_RHO,
            use_rk4=cfg.BOHMIAN_USE_RK4,
        )

    # --------------------------------------------------------
    # 11) Summary stats
    # --------------------------------------------------------
    if cfg.PRINT_ALIGNMENT_STATS and cos_th_init is not None:
        valid = np.isfinite(cos_th_init)
        if np.any(valid):
            mean_c = float(np.mean(cos_th_init[valid]))
            med_c = float(np.median(cos_th_init[valid]))
            frac_pos = float(np.mean(cos_th_init[valid] > 0.0))
            frac_hi = float(np.mean(cos_th_init[valid] > 0.7))
            print(
                f"[ALIGN] mean cosθ≈{mean_c:.3f}, "
                f"median≈{med_c:.3f}, "
                f"frac(cosθ>0)≈{frac_pos:.3f}, "
                f"frac(cosθ>0.7)≈{frac_hi:.3f}"
            )
        else:
            print("[ALIGN] no valid cosθ.")

    if cfg.ENABLE_DIVERGENCE_DIAGNOSTIC and cfg.PRINT_DIVERGENCE_STATS and div_v_init is not None:
        valid = np.isfinite(div_v_init)
        if np.any(valid):
            mean_div = float(np.mean(div_v_init[valid]))
            med_div = float(np.median(div_v_init[valid]))
            frac_neg = float(np.mean(div_v_init[valid] < 0.0))
            frac_pos = float(np.mean(div_v_init[valid] > 0.0))
            print(
                f"[DIV] mean div(v)≈{mean_div:.3e}, "
                f"median≈{med_div:.3e}, "
                f"frac(<0)≈{frac_neg:.3f}, "
                f"frac(>0)≈{frac_pos:.3f}"
            )
        else:
            print("[DIV] no valid div(v) values.")

    if cfg.ENABLE_BOHMIAN_OVERLAY and cfg.PRINT_BOHMIAN_STATS and bohm_traj_alive is not None:
        alive_counts = np.sum(bohm_traj_alive, axis=1)

        for k in range(bohm_traj_alive.shape[0]):
            if alive_counts[k] > 0:
                i_last = int(alive_counts[k] - 1)
                print(
                    f"[BOHM] traj {k}: steps={alive_counts[k]}, "
                    f"start=({bohm_traj_x[k, 0]:.3f},{bohm_traj_y[k, 0]:.3f}), "
                    f"end=({bohm_traj_x[k, i_last]:.3f},{bohm_traj_y[k, i_last]:.3f})"
                )
            else:
                print(f"[BOHM] traj {k}: no valid steps")

    if detector_clicked and det_result_final is not None:
        print(
            f"[DETECTOR FINAL] clicked=True "
            f"x={det_result_final.click_x}, y={det_result_final.click_y}, "
            f"t={det_result_final.click_time}"
        )
    else:
        print("[DETECTOR FINAL] clicked=False (fallback click may have been used)")

    # --------------------------------------------------------
    # 12) Save run bundle
    # --------------------------------------------------------
    output_prefix = getattr(cfg, "OUTPUT_PREFIX", None)
    if not output_prefix:
        output_mp4 = getattr(cfg, "OUTPUT_MP4", "output.mp4")
        output_prefix = str(Path(output_mp4).with_suffix(""))

    # --------------------------------------------------------
    # 12a) Save cheap batch-sampler outputs
    # --------------------------------------------------------
    if batch_sampler is not None:
        batch_summary_path = f"{output_prefix}_flux_summary.json"
        batch_clicks_json_path = f"{output_prefix}_pseudo_clicks.json"
        batch_clicks_jsonl_path = f"{output_prefix}_pseudo_clicks.jsonl"

        batch_sampler.save_summary_json(batch_summary_path)

        if pseudo_clicks is not None:
            batch_sampler.save_clicks_json(batch_clicks_json_path, pseudo_clicks)
            batch_sampler.append_clicks_jsonl(
                batch_clicks_jsonl_path,
                pseudo_clicks,
                run_id=str(Path(output_prefix).name),
                theory_name=str(cfg.THEORY_NAME),
                detector_name=str(getattr(cfg, "DETECTOR_NAME", "unknown")),
            )

            print(
                "[BATCH] saved: "
                f"{batch_summary_path}, "
                f"{batch_clicks_json_path}, "
                f"{batch_clicks_jsonl_path}"
            )
        else:
            print(f"[BATCH] saved summary only: {batch_summary_path}")

    save_run_bundle(
        output_prefix=output_prefix,
        cfg=cfg,
        grid=grid,
        potential=potential,
        debug_free_case=DEBUG_FREE_CASE,
        times=times,
        frames_density=frames_density,
        state_vis_frames=state_vis_frames,
        norms=norms,
        screen_int=screen_int,
        phi_tau_frames=phi_tau_frames,
        x_click=x_click,
        y_click=y_click,
        t_det=t_det,
        idx_det=idx_det,
        detector_clicked=detector_clicked,
        sigma_init=sigma_init,
        ridge_x_init=ridge_x_init,
        ridge_y_init=ridge_y_init,
        ridge_s_init=ridge_s_init,
        cos_th_init=cos_th_init,
        speed_init=speed_init,
        ux_init=ux_init,
        uy_init=uy_init,
        div_v_init=div_v_init,
        vref=vref,
        speed_ref=speed_ref,
        bohm_traj_x=bohm_traj_x,
        bohm_traj_y=bohm_traj_y,
        bohm_traj_alive=bohm_traj_alive,
        bohm_init_points=bohm_init_points,
    )

    # --------------------------------------------------------
    # 13) Final summaries
    # --------------------------------------------------------
    print("frame0 rho max:", np.max(frames_density[0]))
    print("frame0 Emix density max:", np.max(make_emix_density(Emix_init)[0]))
    print("frame0 overlap rho max:", np.max(rho_init[0]))
    print("Simulation done.")
    print("Use visualize.py for animation, slider and debug plots.")


if __name__ == "__main__":
    main()