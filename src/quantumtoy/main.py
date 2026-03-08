from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

from config import AppConfig
from core.grid import build_grid
from core.potentials import build_double_slit_and_caps
from core.utils import make_packet, norm_prob
from theories.registry import build_theory

from analysis.emix import (
    detect_click_from_screen,
    build_backward_library,
    build_Emix_from_phi_tau,
    make_rho,
)
from analysis.ridge import compute_ridge_xy
from analysis.current import alignment_and_diagnostics_from_state_frames
from analysis.bohmian import (
    build_velocity_frames_from_state,
    make_bohmian_initial_points,
    integrate_bohmian_trajectories,
)
from analysis.debug_continuity import continuity_residual_from_state_frames
from core.simulation_types import PotentialSpec
from analysis.emix import make_emix_density
from analysis.emix import build_Emix_density_from_phi_tau


# ============================================================
# Display helper
# ============================================================

def gamma_display(
    arr: np.ndarray,
    vref: float,
    gamma: float = 0.5,
    use_fixed_scale: bool = True,
):
    if use_fixed_scale:
        disp = np.clip(arr / (vref + 1e-30), 0.0, 1.0)
        return disp ** gamma

    m = float(np.max(arr))
    if m <= 0:
        return arr
    return (arr / m) ** gamma


# ============================================================
# Helpers
# ============================================================

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
        v_est = float((theory.c_light**2 * p0) / (E0 + 1e-30))
        return v_est

    return float(cfg.k0x / (cfg.m_mass + 1e-30))


# ============================================================
# Diagnostic reference helpers
# ============================================================

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
            f"sum={np.sum(fi)*grid.dx*grid.dy:.6e} "
            f"max={np.max(fi):.6e} "
            f"argmax={argmax_ij}"
        )

    for i in check_ids:
        e_ref = Emix_density_ref[i]
        print(
            f"[DIAG D7/EMIX_DENS_REF] i={i:4d} t={times[i]:7.3f} "
            f"sum={np.sum(e_ref)*grid.dx*grid.dy:.6e} "
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
                f"sum={np.sum(rr)*grid.dx*grid.dy:.6e} "
                f"max={np.max(rr):.6e}"
            )

    print("\n================ DIAGNOSTIC BLOCK END ==================\n")


# ============================================================
# Main
# ============================================================

def main():
    cfg = AppConfig()

    # --------------------------------------------------------
    # 1) Build grid / potential / theory
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

    theory = build_theory(cfg, grid, potential)

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

    state = theory.initialize_state(psi0)

    # --------------------------------------------------------
    # 3) Forward simulation
    # --------------------------------------------------------
    frames_density = []
    state_vis_frames = []
    times = []
    norms = []

    print(f"Forward simulation starts... theory={cfg.THEORY_NAME}")
    if cfg.THEORY_NAME == "dirac":
        vx_est, vy_est, sp_est = theory.expected_group_velocity(cfg.k0x, cfg.k0y)
        print(f"[DIRAC V_EST] vx≈{vx_est:.6f}, vy≈{vy_est:.6f}, |v|≈{sp_est:.6f}")

    for n in range(cfg.n_steps + 1):
        rho = theory.density(state)
        norm_now = norm_prob(rho, grid.dx, grid.dy)

        if n % cfg.save_every == 0:
            frames_density.append(rho[grid.ys, grid.xs].copy())
            times.append(n * cfg.dt)
            norms.append(norm_now)

            if cfg.SAVE_COMPLEX_STATE_FRAMES:
                if state.ndim == 2:
                    state_vis = state[grid.ys, grid.xs].copy()
                elif state.ndim == 3:
                    state_vis = state[:, grid.ys, grid.xs].copy()
                else:
                    raise ValueError(f"Unsupported state ndim={state.ndim}")

                state_vis_frames.append(state_vis)

            if (len(frames_density) % 20) == 0:
                print(
                    f"[FWD] step {n:5d}/{cfg.n_steps}, "
                    f"t={times[-1]:7.3f}, norm≈{norm_now:.6f}, "
                    f"frames={len(frames_density)}"
                )

        if n < cfg.n_steps:
            res = theory.step_forward(state, cfg.dt)
            state = res.state

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

    # --------------------------------------------------------
    # Forward debug checks
    # --------------------------------------------------------
    t_final = cfg.n_steps * cfg.dt
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
    print(f"[TIMECHK] travel/distance≈{x_travel_est/(dist_to_screen_est+1e-30):.6f}")

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

    if cfg.THEORY_NAME == "dirac":
        if state_vis_frames is not None:
            theory.debug_packet_summary(
                f"vis frame {i}",
                state_vis_frames[i],
                X_like=grid.X_vis,
                Y_like=grid.Y_vis,
            )
    # --------------------------------------------------------
    # Continuity equation debug
    # --------------------------------------------------------
    if DEBUG_FREE_CASE:
        if cfg.SAVE_COMPLEX_STATE_FRAMES and state_vis_frames is not None:

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
                f"[FREEDBG] step={n:5d} t={n*cfg.dt:7.3f} "
                f"x_mean={x_mean: .4f} jx_tot={jx_tot: .4e} jy_tot={jy_tot: .4e}"
            )

            if not np.any(potential.screen_mask_vis):
                print("[DEBUG] screen_mask_vis empty -> skipping click/backward/Emix analysis.")
                plt.figure(figsize=(8, 5))
                plt.imshow(
                    frames_density[-1],
                    extent=(
                        grid.x_vis_min, grid.x_vis_max,
                        grid.y_vis_min, grid.y_vis_max,
                    ),
                    origin="lower",
                    cmap="magma",
                    aspect="auto",
                )
                plt.colorbar(label="rho")
                plt.title(f"Forward density only (debug free case), t={times[-1]:.3f}")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.show()
                return

    # --------------------------------------------------------
    # 4) Detection time + click
    # --------------------------------------------------------
    idx_det, t_det, x_click, y_click, screen_int = detect_click_from_screen(
        frames_density=frames_density,
        times=times,
        screen_mask_vis=potential.screen_mask_vis,
        dx=grid.dx,
        dy=grid.dy,
        X_vis=grid.X_vis,
        Y_vis=grid.Y_vis,
        rng_seed=cfg.CLICK_RNG_SEED,
    )

    print(f"t_det≈{t_det:.3f}, click=({x_click:.3f}, {y_click:.3f})")
    print(
        f"[SCREEN] max={np.max(screen_int):.6e} "
        f"argmax_i={idx_det} t_det={t_det:.6f} "
        f"first={screen_int[0]:.6e} last={screen_int[-1]:.6e}"
    )

    # --------------------------------------------------------
    # 5) Backward library
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
    # 5.5) Diagnostic block
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
    # 6) Builder for sigmaT-dependent objects
    # --------------------------------------------------------
    def build_all_for_sigma(sigmaT: float):
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

        if not cfg.SAVE_COMPLEX_STATE_FRAMES or state_vis_frames is None:
            raise RuntimeError(
                "Amplitude-level Emix/ridge analysis requires SAVE_COMPLEX_STATE_FRAMES=True"
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

        if state_vis_frames is not None:
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

    # --------------------------------------------------------
    # 7) sigmaT setup
    # --------------------------------------------------------
    v_est = estimate_group_velocity(cfg, theory)
    L_gap = cfg.screen_center_x - cfg.barrier_center_x
    t_gap = L_gap / (abs(v_est) + 1e-12)

    sigma_min = 0.05 * t_gap
    sigma_max = 2.00 * t_gap
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
    # 8) Bohmian precompute
    # --------------------------------------------------------
    bohm_traj_x = bohm_traj_y = bohm_traj_alive = None
    bohm_init_points = []

    if cfg.ENABLE_BOHMIAN_OVERLAY:
        if not cfg.SAVE_COMPLEX_STATE_FRAMES or state_vis_frames is None:
            raise RuntimeError(
                "ENABLE_BOHMIAN_OVERLAY requires SAVE_COMPLEX_STATE_FRAMES=True"
            )

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
    # 9) Visualization setup
    # --------------------------------------------------------
    extent = (
        -cfg.VISIBLE_LX / 2,
        cfg.VISIBLE_LX / 2,
        -cfg.VISIBLE_LY / 2,
        cfg.VISIBLE_LY / 2,
    )

    fig = plt.figure(figsize=(10.8, 7.2))
    ax = fig.add_axes([0.07, 0.18, 0.86, 0.78])

    im = ax.imshow(
        gamma_display(
            rho_init[0],
            vref=vref,
            gamma=cfg.GAMMA,
            use_fixed_scale=cfg.USE_FIXED_DISPLAY_SCALE,
        ),
        extent=extent,
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        cmap="magma",
        interpolation=cfg.IM_INTERPOLATION,
    )

    ax.axvline(cfg.barrier_center_x, color="white", linestyle="--", alpha=0.6)
    ax.axvline(cfg.screen_center_x, color="cyan", linestyle="--", alpha=0.4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    title = ax.set_title(
        rf"ρ(t): σT={sigma_init:.3f}, t={times[0]:.3f}, "
        rf"ridge={cfg.RIDGE_MODE}, theory={cfg.THEORY_NAME}"
    )

    ridge_marker, = ax.plot(
        [ridge_x_init[0]],
        [ridge_y_init[0]],
        marker="o",
        markersize=7,
        linestyle="None",
        color="lime",
        alpha=0.9,
        label=f"ridge ({cfg.RIDGE_MODE})",
    )

    ridge_trail, = ax.plot(
        [],
        [],
        linestyle="-",
        linewidth=1.5,
        color="lime",
        alpha=0.5,
    )

    flow_quiver = None
    if cfg.DRAW_FLOW_ARROW and cfg.SAVE_COMPLEX_STATE_FRAMES and (ux_init is not None):
        flow_quiver = ax.quiver(
            [ridge_x_init[0]],
            [ridge_y_init[0]],
            [0.0],
            [0.0],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="cyan",
            alpha=0.9,
            width=0.006,
        )

    bohm_lines = []
    bohm_heads = []

    if cfg.ENABLE_BOHMIAN_OVERLAY and bohm_traj_x is not None:
        for k in range(bohm_traj_x.shape[0]):
            line_k, = ax.plot(
                [],
                [],
                linestyle="-",
                linewidth=cfg.BOHMIAN_LINEWIDTH,
                color=cfg.BOHMIAN_COLOR,
                alpha=0.85,
                label="Bohmian traj" if k == 0 else None,
            )
            bohm_lines.append(line_k)

            if cfg.BOHMIAN_SHOW_HEAD:
                head_k, = ax.plot(
                    [],
                    [],
                    marker="o",
                    markersize=cfg.BOHMIAN_HEAD_SIZE,
                    linestyle="None",
                    color=cfg.BOHMIAN_HEAD_COLOR,
                    alpha=0.95,
                )
            else:
                head_k = None

            bohm_heads.append(head_k)

    ax.legend(loc="upper right", framealpha=0.35)

    # --------------------------------------------------------
    # 10) Slider state containers
    # --------------------------------------------------------
    ax_sigma = fig.add_axes([0.10, 0.08, 0.80, 0.04])
    sigma_slider = Slider(
        ax=ax_sigma,
        label="sigmaT (time thickness)",
        valmin=sigma_min,
        valmax=sigma_max,
        valinit=sigma_init,
    )

    rho_current = [rho_init]
    sigma_current = [sigma_init]
    ridge_x = [ridge_x_init]
    ridge_y = [ridge_y_init]
    ridge_s = [ridge_s_init]
    cos_th = [cos_th_init]
    speed = [speed_init]
    ux = [ux_init]
    uy = [uy_init]
    div_v_ridge = [div_v_init]

    arrow_state = {"ux": np.nan, "uy": np.nan, "spd": np.nan}

    # --------------------------------------------------------
    # 11) Overlay update helpers
    # --------------------------------------------------------
    def update_flow_arrow(i: int):
        if flow_quiver is None or ux[0] is None or speed[0] is None:
            return

        uxi = ux[0][i]
        uyi = uy[0][i]
        spd = speed[0][i]

        valid = (
            np.isfinite(uxi)
            and np.isfinite(uyi)
            and np.isfinite(spd)
            and (spd > cfg.ALIGN_EPS_SPEED)
        )

        if not valid:
            if cfg.ARROW_HIDE_WHEN_INVALID:
                flow_quiver.set_offsets([[ridge_x[0][i], ridge_y[0][i]]])
                flow_quiver.set_UVC([0.0], [0.0])
                return

            if (
                cfg.ARROW_HOLD_LAST_WHEN_INVALID
                and np.isfinite(arrow_state["ux"])
                and np.isfinite(arrow_state["uy"])
            ):
                uxi = arrow_state["ux"]
                uyi = arrow_state["uy"]
                spd = arrow_state["spd"] if np.isfinite(arrow_state["spd"]) else 0.0
            else:
                flow_quiver.set_offsets([[ridge_x[0][i], ridge_y[0][i]]])
                flow_quiver.set_UVC([0.0], [0.0])
                return

        arrow_state["ux"] = uxi
        arrow_state["uy"] = uyi
        arrow_state["spd"] = spd

        flow_quiver.set_offsets([[ridge_x[0][i], ridge_y[0][i]]])

        L = cfg.ARROW_SCALE * float(
            np.clip(spd / (speed_ref + 1e-30), 0.0, 2.5)
        )

        flow_quiver.set_UVC([L * uxi], [L * uyi])

    def update_bohmian_overlay(i: int):
        if not cfg.ENABLE_BOHMIAN_OVERLAY or bohm_traj_x is None:
            return

        for k in range(bohm_traj_x.shape[0]):
            alive = bohm_traj_alive[k]

            if not np.any(alive[: i + 1]):
                bohm_lines[k].set_data([], [])
                if bohm_heads[k] is not None:
                    bohm_heads[k].set_data([], [])
                continue

            if cfg.BOHMIAN_SHOW_FULL_PATH_EACH_FRAME:
                j0 = 0
            else:
                j0 = max(0, i - cfg.BOHMIAN_TRAIL_LEN + 1)

            mask = alive[j0 : i + 1]
            xs_seg = bohm_traj_x[k, j0 : i + 1][mask]
            ys_seg = bohm_traj_y[k, j0 : i + 1][mask]
            bohm_lines[k].set_data(xs_seg, ys_seg)

            if cfg.BOHMIAN_SHOW_HEAD and bohm_heads[k] is not None:
                alive_idx = np.where(alive[: i + 1])[0]
                if alive_idx.size > 0:
                    ilast = int(alive_idx[-1])
                    bohm_heads[k].set_data(
                        [bohm_traj_x[k, ilast]],
                        [bohm_traj_y[k, ilast]],
                    )
                else:
                    bohm_heads[k].set_data([], [])

    def make_title(i: int):
        parts = [
            rf"ρ(t): σT={sigma_current[0]:.3f}",
            rf"t={times[i]:.3f}",
            rf"ridge={cfg.RIDGE_MODE}",
            rf"theory={cfg.THEORY_NAME}",
            rf"norm≈{norms[i]:.4f}",
            rf"Γ≈{ridge_s[0][i]:.3e}",
        ]

        if cos_th[0] is not None and np.isfinite(cos_th[0][i]):
            parts.append(rf"cosθ≈{cos_th[0][i]:.3f}")

        if (
            cfg.ENABLE_DIVERGENCE_DIAGNOSTIC
            and div_v_ridge[0] is not None
            and np.isfinite(div_v_ridge[0][i])
        ):
            parts.append(rf"div v≈{div_v_ridge[0][i]:.3e}")

        if cfg.ENABLE_BOHMIAN_OVERLAY:
            parts.append("Bohm=RK4" if cfg.BOHMIAN_USE_RK4 else "Bohm=Euler")

        return " | ".join(parts)

    # --------------------------------------------------------
    # 12) Slider callback
    # --------------------------------------------------------
    def on_sigma_change(_val):
        new_sigma = float(sigma_slider.val)
        sigma_current[0] = new_sigma

        (
            rho_new,
            _Emix,
            rx,
            ry,
            rs,
            cth,
            spd,
            uxx,
            uyy,
            divv,
        ) = build_all_for_sigma(new_sigma)

        rho_current[0] = rho_new
        ridge_x[0] = rx
        ridge_y[0] = ry
        ridge_s[0] = rs
        cos_th[0] = cth
        speed[0] = spd
        ux[0] = uxx
        uy[0] = uyy
        div_v_ridge[0] = divv

        i = getattr(on_sigma_change, "last_i", 0)

        im.set_data(
            gamma_display(
                rho_current[0][i],
                vref=vref,
                gamma=cfg.GAMMA,
                use_fixed_scale=cfg.USE_FIXED_DISPLAY_SCALE,
            )
        )

        ridge_marker.set_data([ridge_x[0][i]], [ridge_y[0][i]])

        if cfg.SHOW_TRAIL:
            j0 = max(0, i - cfg.TRAIL_LEN + 1)
            ridge_trail.set_data(
                ridge_x[0][j0 : i + 1],
                ridge_y[0][j0 : i + 1],
            )

        update_flow_arrow(i)
        update_bohmian_overlay(i)
        title.set_text(make_title(i))
        fig.canvas.draw_idle()

    sigma_slider.on_changed(on_sigma_change)

    # --------------------------------------------------------
    # 13) Animation callback
    # --------------------------------------------------------
    def update(i: int):
        on_sigma_change.last_i = i

        im.set_data(
            gamma_display(
                rho_current[0][i],
                vref=vref,
                gamma=cfg.GAMMA,
                use_fixed_scale=cfg.USE_FIXED_DISPLAY_SCALE,
            )
        )

        ridge_marker.set_data([ridge_x[0][i]], [ridge_y[0][i]])

        if cfg.SHOW_TRAIL:
            j0 = max(0, i - cfg.TRAIL_LEN + 1)
            ridge_trail.set_data(
                ridge_x[0][j0 : i + 1],
                ridge_y[0][j0 : i + 1],
            )

        update_flow_arrow(i)
        update_bohmian_overlay(i)
        title.set_text(make_title(i))

        artists = [im, ridge_marker, ridge_trail]

        if flow_quiver is not None:
            artists.append(flow_quiver)

        artists.extend([obj for obj in bohm_lines if obj is not None])
        artists.extend([obj for obj in bohm_heads if obj is not None])

        return tuple(artists)

    ani = FuncAnimation(fig, update, frames=Nt, interval=40, blit=False)

    # --------------------------------------------------------
    # 14) Summary stats
    # --------------------------------------------------------
    if cfg.PRINT_ALIGNMENT_STATS and cfg.SAVE_COMPLEX_STATE_FRAMES and (cos_th_init is not None):
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

    if cfg.ENABLE_DIVERGENCE_DIAGNOSTIC and cfg.PRINT_DIVERGENCE_STATS and (div_v_init is not None):
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

    if cfg.ENABLE_BOHMIAN_OVERLAY and cfg.PRINT_BOHMIAN_STATS and (bohm_traj_alive is not None):
        alive_counts = np.sum(bohm_traj_alive, axis=1)

        for k in range(bohm_traj_alive.shape[0]):
            if alive_counts[k] > 0:
                i_last = int(alive_counts[k] - 1)
                print(
                    f"[BOHM] traj {k}: steps={alive_counts[k]}, "
                    f"start=({bohm_traj_x[k,0]:.3f},{bohm_traj_y[k,0]:.3f}), "
                    f"end=({bohm_traj_x[k,i_last]:.3f},{bohm_traj_y[k,i_last]:.3f})"
                )
            else:
                print(f"[BOHM] traj {k}: no valid steps")

    # --------------------------------------------------------
    # 15) Save + show
    # --------------------------------------------------------
    ani.save(cfg.OUTPUT_MP4, writer="ffmpeg", fps=25, dpi=150)
    plt.show()
    print("frame0 rho max:", np.max(frames_density[0]))
    print("frame0 Emix density max:", np.max(make_emix_density(Emix_init)[0]))
    print("frame0 overlap rho max:", np.max(rho_init[0]))


if __name__ == "__main__":
    main()