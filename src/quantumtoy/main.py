from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
import os

import numpy as np

from config import AppConfig
from core.grid import build_grid
from core.packets import PacketFactory
from core.utils import norm_prob
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
from analysis.posthoc_trf import (
    PosthocTRFConfig,
    PosthocTRFResult,
    run_posthoc_trf,
    build_trf_corridor_masks_vis,
    make_rho_from_density_product,
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
from core.potentials import build_potential


# ============================================================
# Standalone helpers
# ============================================================

def estimate_slit_separation_from_masks(grid, potential):
    """
    Estimate slit separation d from slit masks.
    Returns distance between slit centers in y-direction.
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

    print(
        f"[SLITS] center1_y={y1:.6f} center2_y={y2:.6f} "
        f"separation d={d:.6f} slit1 width={a1} slit2 width={a2}"
    )

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

    screen_int_ref = np.array(
        [np.sum(frames_density[i][potential.screen_mask_vis]) * grid.dx * grid.dy for i in range(Nt)],
        dtype=float,
    )

    screen_diff = float(np.max(np.abs(screen_int - screen_int_ref)))
    print(f"[DIAG D1] screen_int max abs diff = {screen_diff:.6e}")
    assert np.allclose(screen_int, screen_int_ref, rtol=1e-12, atol=1e-12), \
        f"screen_int mismatch: max abs diff = {screen_diff}"

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


# ============================================================
# Runtime data classes
# ============================================================

@dataclass
class SimulationSetup:
    cfg: AppConfig
    grid: any
    potential: PotentialSpec
    theory: any
    detector: any
    debug_free_case: bool


@dataclass
class BatchSamplerConfig:
    enabled: bool
    num_samples: int
    rng_seed: int | None
    sample_sigma_x: float
    sample_sigma_y: float
    break_on_detector_click: bool


@dataclass
class BatchSamplerRuntime:
    sampler: FluxBatchSampler | None = None
    pseudo_clicks: list[dict] | None = None


@dataclass
class ForwardRunResult:
    frames_density: np.ndarray
    state_vis_frames: np.ndarray | None
    posthoc_gamma_like_frames: np.ndarray | None
    times: np.ndarray
    norms: np.ndarray
    detector_diags: list[dict]
    detector_clicked: bool
    det_result_final: any
    actual_last_step: int
    batch_runtime: BatchSamplerRuntime


@dataclass
class ClickResolution:
    idx_det: int
    t_det: float
    x_click: float
    y_click: float
    screen_int: np.ndarray
    used_detector_click: bool


@dataclass
class SigmaProducts:
    rho_init: np.ndarray
    Emix_init: np.ndarray
    ridge_x_init: np.ndarray
    ridge_y_init: np.ndarray
    ridge_s_init: np.ndarray
    cos_th_init: np.ndarray | None
    speed_init: np.ndarray | None
    ux_init: np.ndarray | None
    uy_init: np.ndarray | None
    div_v_init: np.ndarray | None
    sigma_init: float
    vref: float
    speed_ref: float


@dataclass
class BohmianResult:
    bohm_traj_x: np.ndarray | None = None
    bohm_traj_y: np.ndarray | None = None
    bohm_traj_alive: np.ndarray | None = None
    bohm_init_points: list = field(default_factory=list)


@dataclass
class PosthocProducts:
    result: PosthocTRFResult | None = None
    base_rho: np.ndarray | None = None
    rho_selected: np.ndarray | None = None
    corridor_upper_mask: np.ndarray | None = None
    corridor_lower_mask: np.ndarray | None = None


# ============================================================
# Main application class
# ============================================================

class QuantumSimulationApp:

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    def build_setup(self) -> SimulationSetup:
        cfg = self.cfg
        cfg.dump_selected()

        if os.environ.get("DT") is not None:
            cfg.dt = float(os.environ["DT"])

        if os.environ.get("SAVE_EVERY") is not None:
            cfg.save_every = int(os.environ["SAVE_EVERY"])

        if os.environ.get("N_STEPS") is not None:
            cfg.n_steps = int(os.environ["N_STEPS"])

        grid = build_grid(
            visible_lx=cfg.VISIBLE_LX,
            visible_ly=cfg.VISIBLE_LY,
            n_visible_x=cfg.N_VISIBLE_X,
            n_visible_y=cfg.N_VISIBLE_Y,
            pad_factor=cfg.PAD_FACTOR,
        )

        debug_free_case = bool(getattr(cfg, "DEBUG_FREE_CASE", False))

        if debug_free_case:
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
            potential = build_potential(grid, cfg)
            for comp in potential.components:
                print(comp.name, comp.kind, comp.V_real.shape)

        estimate_slit_separation_from_masks(grid, potential)

        theory = build_theory(cfg, grid, potential)
        detector = build_detector(cfg, grid)

        return SimulationSetup(
            cfg=cfg,
            grid=grid,
            potential=potential,
            theory=theory,
            detector=detector,
            debug_free_case=debug_free_case,
        )

    # --------------------------------------------------------
    # Batch sampler
    # --------------------------------------------------------

    def build_batch_sampler_config(self) -> BatchSamplerConfig:
        cfg = self.cfg
        return BatchSamplerConfig(
            enabled=bool(getattr(cfg, "ENABLE_FLUX_BATCH_SAMPLER", True)),
            num_samples=int(getattr(cfg, "FLUX_BATCH_NUM_SAMPLES", 10_000)),
            rng_seed=getattr(cfg, "FLUX_BATCH_RNG_SEED", 12345),
            sample_sigma_x=float(getattr(cfg, "FLUX_BATCH_SAMPLE_SIGMA_X", 0.0)),
            sample_sigma_y=float(getattr(cfg, "FLUX_BATCH_SAMPLE_SIGMA_Y", 0.0)),
            break_on_detector_click=bool(getattr(cfg, "BREAK_ON_DETECTOR_CLICK", True)),
        )

    def maybe_build_batch_sampler(self, setup: SimulationSetup) -> BatchSamplerRuntime:
        cfg = setup.cfg
        detector = setup.detector
        bcfg = self.build_batch_sampler_config()

        runtime = BatchSamplerRuntime()

        if not bcfg.enabled:
            return runtime

        det_gate_x = float(getattr(detector, "detector_gate_center_x", cfg.screen_center_x))
        det_gate_wx = float(getattr(detector, "detector_gate_width", cfg.screen_eval_width))
        det_gate_y = float(getattr(detector, "detector_gate_center_y", 0.0))
        det_gate_wy = float(getattr(detector, "detector_gate_width_y", -1.0))

        runtime.sampler = FluxBatchSampler(
            grid=setup.grid,
            gate_center_x=det_gate_x,
            gate_width_x=det_gate_wx,
            gate_center_y=det_gate_y,
            gate_width_y=det_gate_wy,
            hbar=float(getattr(cfg, "hbar", 1.0)),
            mass=float(getattr(cfg, "m_mass", 1.0)),
            sample_sigma_x=bcfg.sample_sigma_x,
            sample_sigma_y=bcfg.sample_sigma_y,
            rng_seed=bcfg.rng_seed,
        )

        print(
            "[BATCH] enabled: "
            f"samples={bcfg.num_samples}, "
            f"gate_center_x={det_gate_x}, gate_width_x={det_gate_wx}"
        )

        return runtime

    # --------------------------------------------------------
    # Forward simulation
    # --------------------------------------------------------

    def run_forward(self, setup: SimulationSetup) -> ForwardRunResult:
        cfg = setup.cfg
        grid = setup.grid
        theory = setup.theory
        detector = setup.detector

        packet = PacketFactory.build_initial_packet(cfg, grid)
        print(f"[PACKET] initial packet mode={packet.packet_name}")

        state = theory.initialize_state(packet.psi0)
        detector.reset()

        batch_runtime = self.maybe_build_batch_sampler(setup)
        batch_cfg = self.build_batch_sampler_config()

        save_posthoc_gamma_like = bool(getattr(cfg, "POSTHOC_SAVE_GAMMA_LIKE", True))
        can_export_posthoc_fields = hasattr(theory, "compute_posthoc_support_fields")

        frames_density = []
        state_vis_frames = []
        posthoc_gamma_like_frames = []
        times = []
        norms = []

        detector_diags = []
        detector_clicked = False
        det_result_final = None

        print(
            f"Forward simulation starts... "
            f"theory={cfg.THEORY_NAME}, "
            f"detector={getattr(cfg, 'DETECTOR_NAME', 'unknown')}"
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

            if batch_runtime.sampler is not None and state.ndim == 2:
                t0 = time.perf_counter()
                batch_runtime.sampler.update(state, cfg.dt)
                t_batch += time.perf_counter() - t0

            t0 = time.perf_counter()
            det_res = detector.step(state, cfg.dt, t=t_now)
            t_detector += time.perf_counter() - t0

            detector_diags.append(det_res.aux if det_res.aux is not None else {})

            need_save = (n % cfg.save_every == 0)
            if need_save:
                t0 = time.perf_counter()

                rho = theory.density(state)
                norm_now = norm_prob(rho, grid.dx, grid.dy)

                rho_vis = rho[grid.ys, grid.xs].copy()
                frames_density.append(rho_vis)
                times.append(t_now)
                norms.append(norm_now)

                state_vis = None
                if cfg.SAVE_COMPLEX_STATE_FRAMES:
                    if state.ndim == 2:
                        state_vis = state[grid.ys, grid.xs].copy()
                    elif state.ndim == 3:
                        state_vis = state[:, grid.ys, grid.xs].copy()
                    else:
                        raise ValueError(f"Unsupported state ndim={state.ndim}")

                    state_vis_frames.append(state_vis)

                if save_posthoc_gamma_like and can_export_posthoc_fields:
                    if state_vis is None:
                        if state.ndim == 2:
                            state_vis_for_posthoc = state[grid.ys, grid.xs]
                        elif state.ndim == 3:
                            state_vis_for_posthoc = state[:, grid.ys, grid.xs]
                        else:
                            raise ValueError(f"Unsupported state ndim={state.ndim}")
                    else:
                        state_vis_for_posthoc = state_vis

                    try:
                        support = theory.compute_posthoc_support_fields(state_vis_for_posthoc)
                        gamma_like = support.get("gamma_like", None)
                        if gamma_like is not None:
                            posthoc_gamma_like_frames.append(np.asarray(gamma_like, dtype=np.float32))
                    except Exception as e:
                        print(f"[POSTHOC SUPPORT] skipped at save step {n}: {e}")

                t_density += time.perf_counter() - t0

                if (len(frames_density) % 20) == 0:
                    print(
                        f"[FWD] step {n:5d}/{cfg.n_steps}, "
                        f"t={times[-1]:7.3f}, norm≈{norm_now:.6f}, "
                        f"frames={len(frames_density)}"
                    )

            if det_res.clicked and not detector_clicked:
                detector_clicked = True
                det_result_final = det_res

                print(
                    f"[DETECTOR] clicked at step={n}, "
                    f"t={t_now:.6f}, "
                    f"x={det_res.click_x}, "
                    f"y={det_res.click_y}"
                )

                if batch_cfg.break_on_detector_click:
                    print("[FWD] breaking on detector click.")
                    break

            if n < cfg.n_steps:
                t0 = time.perf_counter()
                state = theory.step_forward(state, cfg.dt).state
                t_step += time.perf_counter() - t0

        t_total = time.perf_counter() - t_total_start

        print("\n================ PROFILE =================")

        def pct(x):
            return 100.0 * x / t_total if t_total > 0 else 0.0

        print(f"Total runtime      : {t_total:8.3f} s")
        print(f"step_forward       : {t_step:8.3f} s ({pct(t_step):5.1f}%)")
        print(f"detector.step      : {t_detector:8.3f} s ({pct(t_detector):5.1f}%)")
        print(f"batch_sampler      : {t_batch:8.3f} s ({pct(t_batch):5.1f}%)")
        print(f"density + norm     : {t_density:8.3f} s ({pct(t_density):5.1f}%)")
        print("==========================================\n")

        frames_density = np.asarray(frames_density, dtype=float)
        times = np.asarray(times, dtype=float)
        norms = np.asarray(norms, dtype=float)

        if cfg.SAVE_COMPLEX_STATE_FRAMES:
            state_vis_frames = np.asarray(state_vis_frames)
        else:
            state_vis_frames = None

        if len(posthoc_gamma_like_frames) > 0:
            posthoc_gamma_like_frames = np.asarray(posthoc_gamma_like_frames, dtype=np.float32)
        else:
            posthoc_gamma_like_frames = None

        print("Forward done.")

        if batch_runtime.sampler is not None:
            try:
                batch_runtime.pseudo_clicks = batch_runtime.sampler.sample_clicks(batch_cfg.num_samples)
                print(
                    "[BATCH] sampled pseudo-clicks: "
                    f"n={len(batch_runtime.pseudo_clicks)}, "
                    f"updates={batch_runtime.sampler.num_updates}, "
                    f"total_flux={batch_runtime.sampler.total_flux_accum:.6e}"
                )
            except Exception as e:
                batch_runtime.pseudo_clicks = None
                print(f"[BATCH] sampling skipped: {e}")

        return ForwardRunResult(
            frames_density=frames_density,
            state_vis_frames=state_vis_frames,
            posthoc_gamma_like_frames=posthoc_gamma_like_frames,
            times=times,
            norms=norms,
            detector_diags=detector_diags,
            detector_clicked=detector_clicked,
            det_result_final=det_result_final,
            actual_last_step=actual_last_step,
            batch_runtime=batch_runtime,
        )

    # --------------------------------------------------------
    # Forward debug
    # --------------------------------------------------------

    def print_forward_debug_checks(self, setup: SimulationSetup, forward: ForwardRunResult):
        cfg = setup.cfg
        grid = setup.grid
        theory = setup.theory

        frames_density = forward.frames_density
        times = forward.times

        t_final = forward.actual_last_step * cfg.dt
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

        Nt = len(times)
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

        if cfg.THEORY_NAME == "dirac" and forward.state_vis_frames is not None:
            theory.debug_packet_summary(
                f"vis frame {check_ids[-1]}",
                forward.state_vis_frames[check_ids[-1]],
                X_like=grid.X_vis,
                Y_like=grid.Y_vis,
            )

    # --------------------------------------------------------
    # Optional continuity debug
    # --------------------------------------------------------

    def maybe_run_free_case_debug(self, setup: SimulationSetup, forward: ForwardRunResult):
        cfg = setup.cfg
        grid = setup.grid
        theory = setup.theory
        potential = setup.potential

        if not setup.debug_free_case:
            return

        if not cfg.SAVE_COMPLEX_STATE_FRAMES:
            return

        if forward.state_vis_frames is None:
            return

        rms_mean, rms_max, abs_max = continuity_residual_from_state_frames(
            theory=theory,
            state_vis_frames=forward.state_vis_frames,
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

        if not np.any(potential.screen_mask_vis):
            print("[DEBUG] screen_mask_vis empty -> skipping click/backward/Emix analysis.")

    # --------------------------------------------------------
    # Click resolution
    # --------------------------------------------------------

    def resolve_click(self, setup: SimulationSetup, forward: ForwardRunResult) -> ClickResolution:
        cfg = setup.cfg
        grid = setup.grid
        potential = setup.potential

        frames_density = forward.frames_density
        times = forward.times
        Nt = len(times)

        if forward.detector_clicked and forward.det_result_final is not None:
            x_click = float(forward.det_result_final.click_x)
            y_click = float(forward.det_result_final.click_y)

            idx_det = int(
                np.argmin(
                    np.abs(
                        times - float(
                            forward.det_result_final.click_time
                            if forward.det_result_final.click_time is not None
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

            return ClickResolution(
                idx_det=idx_det,
                t_det=t_det,
                x_click=x_click,
                y_click=y_click,
                screen_int=screen_int,
                used_detector_click=True,
            )

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

        return ClickResolution(
            idx_det=idx_det,
            t_det=t_det,
            x_click=x_click,
            y_click=y_click,
            screen_int=screen_int,
            used_detector_click=False,
        )

    # --------------------------------------------------------
    # Fast batch-only exit
    # --------------------------------------------------------

    def maybe_run_batch_fast_exit(self, setup: SimulationSetup, forward: ForwardRunResult) -> bool:
        cfg = setup.cfg

        batch_fast_mode = bool(getattr(cfg, "BATCH_FAST_MODE", False))
        if not batch_fast_mode:
            return False

        output_prefix = getattr(cfg, "OUTPUT_PREFIX", None)
        if not output_prefix:
            output_mp4 = getattr(cfg, "OUTPUT_MP4", "output.mp4")
            output_prefix = str(Path(output_mp4).with_suffix(""))

        batch_runtime = forward.batch_runtime
        if batch_runtime.sampler is not None:
            batch_summary_path = f"{output_prefix}_flux_summary.json"
            batch_clicks_json_path = f"{output_prefix}_pseudo_clicks.json"
            batch_clicks_jsonl_path = f"{output_prefix}_pseudo_clicks.jsonl"

            batch_runtime.sampler.save_summary_json(batch_summary_path)

            if batch_runtime.pseudo_clicks is not None:
                batch_runtime.sampler.save_clicks_json(batch_clicks_json_path, batch_runtime.pseudo_clicks)
                batch_runtime.sampler.append_clicks_jsonl(
                    batch_clicks_jsonl_path,
                    batch_runtime.pseudo_clicks,
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
        return True

    # --------------------------------------------------------
    # Sigma products
    # --------------------------------------------------------

    def build_sigma_dependent_products_builder(
        self,
        *,
        theory,
        grid,
        state_vis_frames: np.ndarray,
        phi_tau_frames: np.ndarray,
        times: np.ndarray,
        t_det: float,
        tau_step: float,
    ):
        cfg = self.cfg

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

    def build_sigma_products(
        self,
        setup: SimulationSetup,
        forward: ForwardRunResult,
        phi_tau_frames: np.ndarray,
        click: ClickResolution,
    ) -> SigmaProducts:
        cfg = setup.cfg
        theory = setup.theory
        grid = setup.grid

        if not cfg.SAVE_COMPLEX_STATE_FRAMES or forward.state_vis_frames is None:
            raise RuntimeError(
                "This pipeline requires SAVE_COMPLEX_STATE_FRAMES=True "
                "for amplitude-level Emix/ridge analysis and later visualization."
            )

        tau_step = cfg.save_every * cfg.dt

        build_all_for_sigma = self.build_sigma_dependent_products_builder(
            theory=theory,
            grid=grid,
            state_vis_frames=forward.state_vis_frames,
            phi_tau_frames=phi_tau_frames,
            times=forward.times,
            t_det=click.t_det,
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

        return SigmaProducts(
            rho_init=rho_init,
            Emix_init=Emix_init,
            ridge_x_init=ridge_x_init,
            ridge_y_init=ridge_y_init,
            ridge_s_init=ridge_s_init,
            cos_th_init=cos_th_init,
            speed_init=speed_init,
            ux_init=ux_init,
            uy_init=uy_init,
            div_v_init=div_v_init,
            sigma_init=sigma_init,
            vref=vref,
            speed_ref=speed_ref,
        )

    # --------------------------------------------------------
    # Posthoc TRF
    # --------------------------------------------------------

    def build_posthoc_products(
        self,
        setup: SimulationSetup,
        forward: ForwardRunResult,
        phi_tau_frames: np.ndarray,
        click: ClickResolution,
    ) -> PosthocProducts:
        cfg = setup.cfg
        grid = setup.grid
        theory = setup.theory

        if not cfg.SAVE_COMPLEX_STATE_FRAMES or forward.state_vis_frames is None:
            print("[POSTHOC] skipped: SAVE_COMPLEX_STATE_FRAMES=False or state_vis_frames missing")
            return PosthocProducts(result=None)

        tau_step = cfg.save_every * cfg.dt

        v_est = estimate_group_velocity(cfg, theory)
        L_gap = cfg.screen_center_x - cfg.barrier_center_x
        t_gap = L_gap / (abs(v_est) + 1e-12)
        sigma_init = 0.60 * t_gap

        sigmaT_env = os.environ.get("POSTHOC_TRF_SIGMAT", None)
        sigmaT_value = float(sigmaT_env) if sigmaT_env is not None else sigma_init

        base_field_mode = str(getattr(cfg, "POSTHOC_TRF_BASE_FIELD", "density")).lower()
        rho_mode = getattr(cfg, "POSTHOC_TRF_RHO_MODE", "density_product_oldstyle")
        use_worldline = bool(getattr(cfg, "POSTHOC_USE_WORLDLINE", True))

        Emix_amp = build_Emix_from_phi_tau(
            phi_tau_frames=phi_tau_frames,
            times=forward.times,
            t_det=click.t_det,
            sigmaT=sigmaT_value,
            tau_step=tau_step,
            K_JITTER=int(getattr(cfg, "K_JITTER", 13)),
        )

        Emix_density = build_Emix_density_from_phi_tau(
            phi_tau_frames=phi_tau_frames,
            times=forward.times,
            t_det=click.t_det,
            sigmaT=sigmaT_value,
            tau_step=tau_step,
            K_JITTER=int(getattr(cfg, "K_JITTER", 13)),
        )

        if base_field_mode == "gamma_like":
            if forward.posthoc_gamma_like_frames is None:
                print("[POSTHOC] gamma_like requested but no posthoc_gamma_like_frames available; falling back to density.")
                base_field_mode = "density"
            else:
                base_rho = make_rho_from_density_product(
                    frames_density=forward.posthoc_gamma_like_frames,
                    Emix_density=Emix_density,
                    dx=grid.dx,
                    dy=grid.dy,
                )
        if base_field_mode == "density":
            base_rho = make_rho(
                frames_psi=forward.state_vis_frames,
                Emix=Emix_amp,
                Emix_density=Emix_density,
                dx=grid.dx,
                dy=grid.dy,
                mode=rho_mode,
                blend_alpha=float(getattr(cfg, "RHO_BLEND_ALPHA", 0.5)),
            )

        posthoc_cfg = PosthocTRFConfig(
            enabled=True,
            use_adaptive_ref=bool(getattr(cfg, "POSTHOC_TRF_USE_ADAPTIVE_REF", True)),
            ref_t_min_frac=float(getattr(cfg, "POSTHOC_TRF_REF_T_MIN_FRAC", 0.30)),
            ref_t_max_frac=float(getattr(cfg, "POSTHOC_TRF_REF_T_MAX_FRAC", 0.95)),
            corridor_x_frac_start=float(getattr(cfg, "POSTHOC_TRF_CORRIDOR_X_FRAC_START", 0.70)),
            corridor_y_sigma=float(getattr(cfg, "POSTHOC_TRF_CORRIDOR_Y_SIGMA", 1.3)),
            corridor_x_weight_power=float(getattr(cfg, "POSTHOC_TRF_CORRIDOR_X_WEIGHT_POWER", 2.5)),
            valid_total_evidence_eps=float(getattr(cfg, "POSTHOC_TRF_VALID_TOTAL_EVIDENCE_EPS", 1e-12)),
            use_posthoc_worldline=use_worldline,
            wl_track_radius_px=int(getattr(cfg, "POSTHOC_WL_TRACK_RADIUS_PX", 20)),
            wl_min_local_rel=float(getattr(cfg, "POSTHOC_WL_MIN_LOCAL_REL", 0.03)),
            wl_tube_sigma_px=float(getattr(cfg, "POSTHOC_WL_TUBE_SIGMA_PX", 10.0)),
            wl_gain_strength=float(getattr(cfg, "POSTHOC_WL_GAIN_STRENGTH", 2.0)),
            wl_outside_damp=float(getattr(cfg, "POSTHOC_WL_OUTSIDE_DAMP", 0.20)),
            wl_time_ramp_frac=float(getattr(cfg, "POSTHOC_WL_TIME_RAMP_FRAC", 0.12)),
        )

        result = run_posthoc_trf(
            base_rho=base_rho,
            times=forward.times,
            X_vis=grid.X_vis,
            Y_vis=grid.Y_vis,
            barrier_center_x=cfg.barrier_center_x,
            screen_center_x=cfg.screen_center_x,
            slit_center_offset=cfg.slit_center_offset,
            dx=grid.dx,
            dy=grid.dy,
            cfg=posthoc_cfg,
        )

        corridor_upper_mask, corridor_lower_mask, _ = build_trf_corridor_masks_vis(
            X_vis=grid.X_vis,
            Y_vis=grid.Y_vis,
            barrier_center_x=cfg.barrier_center_x,
            screen_center_x=cfg.screen_center_x,
            slit_center_offset=cfg.slit_center_offset,
            cfg=posthoc_cfg,
        )

        rho_selected = result.aux.get("rho_worldline", None)
        if rho_selected is not None:
            rho_selected = np.asarray(rho_selected)

        return PosthocProducts(
            result=result,
            base_rho=np.asarray(base_rho),
            rho_selected=rho_selected,
            corridor_upper_mask=np.asarray(corridor_upper_mask),
            corridor_lower_mask=np.asarray(corridor_lower_mask),
        )

    def print_posthoc_summary(self, posthoc: PosthocProducts):
        if posthoc.result is None:
            print("[POSTHOC] no result")
            return

        res = posthoc.result

        print(
            "[POSTHOC TRF] "
            f"valid={res.valid} "
            f"chosen={res.chosen_side} "
            f"ref_idx={res.ref_idx} "
            f"ref_time={res.ref_time:.6f} "
            f"upper_ev={res.upper_evidence:.6e} "
            f"lower_ev={res.lower_evidence:.6e} "
            f"total_ev={res.total_evidence:.6e} "
            f"abs_margin={res.abs_margin:.6e} "
            f"rel_margin={res.rel_margin:.6f} "
            f"dominance={res.dominance:.6f} "
            f"ratio={res.ratio:.6f} "
            f"adaptive_score={res.adaptive_score:.6e}"
        )

        if res.worldline_used:
            print(
                "[POSTHOC WL] "
                f"used=True "
                f"seed_side={res.worldline_seed_side} "
                f"seed_x={res.worldline_seed_x:.6f} "
                f"seed_y={res.worldline_seed_y:.6f}"
            )
        else:
            print("[POSTHOC WL] used=False")

    # --------------------------------------------------------
    # Bohmian
    # --------------------------------------------------------

    def build_bohmian_overlay(
        self,
        setup: SimulationSetup,
        forward: ForwardRunResult,
        sigma_products: SigmaProducts,
    ) -> BohmianResult:
        cfg = setup.cfg
        grid = setup.grid
        theory = setup.theory

        if not cfg.ENABLE_BOHMIAN_OVERLAY:
            return BohmianResult()

        if forward.state_vis_frames is None:
            return BohmianResult()

        print("Bohmian overlay: computing velocity frames...")

        vx_frames, vy_frames, rho_frames = build_velocity_frames_from_state(
            theory=theory,
            state_vis_frames=forward.state_vis_frames,
            eps_rho=cfg.ALIGN_EPS_RHO,
        )

        bohm_init_points = make_bohmian_initial_points(
            mode=cfg.BOHMIAN_INIT_MODE,
            ntraj=cfg.BOHMIAN_N_TRAJ,
            custom_points=cfg.BOHMIAN_CUSTOM_POINTS,
            ridge_x0=sigma_products.ridge_x_init[0],
            ridge_y0=sigma_products.ridge_y_init[0],
            x0_packet=cfg.x0,
            y0_packet=cfg.y0,
            psi0_vis=forward.state_vis_frames[0],
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
            times=forward.times,
            tau_step=cfg.save_every * cfg.dt,
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

        return BohmianResult(
            bohm_traj_x=bohm_traj_x,
            bohm_traj_y=bohm_traj_y,
            bohm_traj_alive=bohm_traj_alive,
            bohm_init_points=bohm_init_points,
        )

    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------

    def print_summary_stats(
        self,
        setup: SimulationSetup,
        forward: ForwardRunResult,
        sigma_products: SigmaProducts,
        bohm: BohmianResult,
    ):
        cfg = setup.cfg

        if cfg.PRINT_ALIGNMENT_STATS and sigma_products.cos_th_init is not None:
            valid = np.isfinite(sigma_products.cos_th_init)
            if np.any(valid):
                mean_c = float(np.mean(sigma_products.cos_th_init[valid]))
                med_c = float(np.median(sigma_products.cos_th_init[valid]))
                frac_pos = float(np.mean(sigma_products.cos_th_init[valid] > 0.0))
                frac_hi = float(np.mean(sigma_products.cos_th_init[valid] > 0.7))
                print(
                    f"[ALIGN] mean cosθ≈{mean_c:.3f}, "
                    f"median≈{med_c:.3f}, "
                    f"frac(cosθ>0)≈{frac_pos:.3f}, "
                    f"frac(cosθ>0.7)≈{frac_hi:.3f}"
                )
            else:
                print("[ALIGN] no valid cosθ.")

        if (
            cfg.ENABLE_DIVERGENCE_DIAGNOSTIC
            and cfg.PRINT_DIVERGENCE_STATS
            and sigma_products.div_v_init is not None
        ):
            valid = np.isfinite(sigma_products.div_v_init)
            if np.any(valid):
                mean_div = float(np.mean(sigma_products.div_v_init[valid]))
                med_div = float(np.median(sigma_products.div_v_init[valid]))
                frac_neg = float(np.mean(sigma_products.div_v_init[valid] < 0.0))
                frac_pos = float(np.mean(sigma_products.div_v_init[valid] > 0.0))
                print(
                    f"[DIV] mean div(v)≈{mean_div:.3e}, "
                    f"median≈{med_div:.3e}, "
                    f"frac(<0)≈{frac_neg:.3f}, "
                    f"frac(>0)≈{frac_pos:.3f}"
                )
            else:
                print("[DIV] no valid div(v) values.")

        if cfg.ENABLE_BOHMIAN_OVERLAY and cfg.PRINT_BOHMIAN_STATS and bohm.bohm_traj_alive is not None:
            alive_counts = np.sum(bohm.bohm_traj_alive, axis=1)

            for k in range(bohm.bohm_traj_alive.shape[0]):
                if alive_counts[k] > 0:
                    i_last = int(alive_counts[k] - 1)
                    print(
                        f"[BOHM] traj {k}: steps={alive_counts[k]}, "
                        f"start=({bohm.bohm_traj_x[k, 0]:.3f},{bohm.bohm_traj_y[k, 0]:.3f}), "
                        f"end=({bohm.bohm_traj_x[k, i_last]:.3f},{bohm.bohm_traj_y[k, i_last]:.3f})"
                    )
                else:
                    print(f"[BOHM] traj {k}: no valid steps")

        if forward.detector_clicked and forward.det_result_final is not None:
            print(
                f"[DETECTOR FINAL] clicked=True "
                f"x={forward.det_result_final.click_x}, y={forward.det_result_final.click_y}, "
                f"t={forward.det_result_final.click_time}"
            )
        else:
            print("[DETECTOR FINAL] clicked=False (fallback click may have been used)")

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    def get_output_prefix(self) -> str:
        cfg = self.cfg
        output_prefix = getattr(cfg, "OUTPUT_PREFIX", None)
        if output_prefix:
            return str(output_prefix)

        output_mp4 = getattr(cfg, "OUTPUT_MP4", "output.mp4")
        return str(Path(output_mp4).with_suffix(""))

    def save_batch_outputs_if_any(self, forward: ForwardRunResult):
        cfg = self.cfg
        output_prefix = self.get_output_prefix()
        batch_runtime = forward.batch_runtime

        if batch_runtime.sampler is None:
            return

        batch_summary_path = f"{output_prefix}_flux_summary.json"
        batch_clicks_json_path = f"{output_prefix}_pseudo_clicks.json"
        batch_clicks_jsonl_path = f"{output_prefix}_pseudo_clicks.jsonl"

        batch_runtime.sampler.save_summary_json(batch_summary_path)

        if batch_runtime.pseudo_clicks is not None:
            batch_runtime.sampler.save_clicks_json(batch_clicks_json_path, batch_runtime.pseudo_clicks)
            batch_runtime.sampler.append_clicks_jsonl(
                batch_clicks_jsonl_path,
                batch_runtime.pseudo_clicks,
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

    def save_run_outputs(
        self,
        setup: SimulationSetup,
        forward: ForwardRunResult,
        click: ClickResolution,
        phi_tau_frames: np.ndarray,
        sigma_products: SigmaProducts,
        bohm: BohmianResult,
        posthoc: PosthocProducts,
    ):
        self.save_batch_outputs_if_any(forward)

        posthoc_base_rho = None
        posthoc_selected_rho = None
        posthoc_corridor_upper_mask = None
        posthoc_corridor_lower_mask = None
        posthoc_trf_info = None
        posthoc_worldline_info = None

        if posthoc.result is not None:
            res = posthoc.result

            posthoc_base_rho = posthoc.base_rho
            posthoc_selected_rho = posthoc.rho_selected
            posthoc_corridor_upper_mask = posthoc.corridor_upper_mask
            posthoc_corridor_lower_mask = posthoc.corridor_lower_mask

            posthoc_trf_info = {
                "valid": bool(res.valid),
                "ref_idx": None if res.ref_idx is None else int(res.ref_idx),
                "ref_time": None if res.ref_time is None else float(res.ref_time),
                "chosen_side": res.chosen_side,

                "upper_evidence": float(res.upper_evidence),
                "lower_evidence": float(res.lower_evidence),
                "total_evidence": float(res.total_evidence),

                "abs_margin": float(res.abs_margin),
                "rel_margin": float(res.rel_margin),
                "ratio": float(res.ratio),
                "dominance": float(res.dominance),
                "adaptive_score": float(res.adaptive_score),

                "upper_seed_x": None if res.upper_seed_x is None else float(res.upper_seed_x),
                "upper_seed_y": None if res.upper_seed_y is None else float(res.upper_seed_y),
                "lower_seed_x": None if res.lower_seed_x is None else float(res.lower_seed_x),
                "lower_seed_y": None if res.lower_seed_y is None else float(res.lower_seed_y),
            }

            posthoc_worldline_info = {
                "used": bool(res.worldline_used),
                "seed_x": None if res.worldline_seed_x is None else float(res.worldline_seed_x),
                "seed_y": None if res.worldline_seed_y is None else float(res.worldline_seed_y),
                "seed_side": res.worldline_seed_side,
            }

        save_run_bundle(
            output_prefix=self.get_output_prefix(),
            cfg=setup.cfg,
            grid=setup.grid,
            potential=setup.potential,
            debug_free_case=setup.debug_free_case,
            times=forward.times,
            frames_density=forward.frames_density,
            state_vis_frames=forward.state_vis_frames,
            norms=forward.norms,
            screen_int=click.screen_int,
            phi_tau_frames=phi_tau_frames,
            x_click=click.x_click,
            y_click=click.y_click,
            t_det=click.t_det,
            idx_det=click.idx_det,
            detector_clicked=forward.detector_clicked,
            sigma_init=sigma_products.sigma_init,
            ridge_x_init=sigma_products.ridge_x_init,
            ridge_y_init=sigma_products.ridge_y_init,
            ridge_s_init=sigma_products.ridge_s_init,
            cos_th_init=sigma_products.cos_th_init,
            speed_init=sigma_products.speed_init,
            ux_init=sigma_products.ux_init,
            uy_init=sigma_products.uy_init,
            div_v_init=sigma_products.div_v_init,
            vref=sigma_products.vref,
            speed_ref=sigma_products.speed_ref,
            bohm_traj_x=bohm.bohm_traj_x,
            bohm_traj_y=bohm.bohm_traj_y,
            bohm_traj_alive=bohm.bohm_traj_alive,
            bohm_init_points=bohm.bohm_init_points,
            posthoc_base_rho=posthoc_base_rho,
            posthoc_selected_rho=posthoc_selected_rho,
            posthoc_corridor_upper_mask=posthoc_corridor_upper_mask,
            posthoc_corridor_lower_mask=posthoc_corridor_lower_mask,
            posthoc_trf_info=posthoc_trf_info,
            posthoc_worldline_info=posthoc_worldline_info,
        )

    # --------------------------------------------------------
    # Full run
    # --------------------------------------------------------

    def run(self):
        setup = self.build_setup()
        forward = self.run_forward(setup)

        self.print_forward_debug_checks(setup, forward)
        self.maybe_run_free_case_debug(setup, forward)

        click = self.resolve_click(setup, forward)

        print(
            f"[SCREEN] max={np.max(click.screen_int):.6e} "
            f"argmax_i={np.argmax(click.screen_int)} "
            f"t_argmax={forward.times[int(np.argmax(click.screen_int))]:.6f} "
            f"first={click.screen_int[0]:.6e} last={click.screen_int[-1]:.6e}"
        )

        if self.maybe_run_batch_fast_exit(setup, forward):
            return

        print("Backward library: computing phi_tau frames...")
        phi_tau_frames = build_backward_library(
            theory=setup.theory,
            grid=setup.grid,
            times=forward.times,
            tau_step=self.cfg.save_every * self.cfg.dt,
            x_click=click.x_click,
            y_click=click.y_click,
            sigma_click=self.cfg.sigma_click,
            save_every=self.cfg.save_every,
            print_every_frames=20,
        )
        print("Backward library done.")

        v_est_diag = estimate_group_velocity(self.cfg, setup.theory)
        L_gap_diag = self.cfg.screen_center_x - self.cfg.barrier_center_x
        t_gap_diag = L_gap_diag / (abs(v_est_diag) + 1e-12)
        sigma_diag = 0.60 * t_gap_diag

        run_diagnostics(
            cfg=self.cfg,
            grid=setup.grid,
            theory=setup.theory,
            potential=setup.potential,
            frames_density=forward.frames_density,
            state_vis_frames=forward.state_vis_frames,
            times=forward.times,
            tau_step=self.cfg.save_every * self.cfg.dt,
            idx_det=click.idx_det,
            t_det=click.t_det,
            x_click=click.x_click,
            y_click=click.y_click,
            screen_int=click.screen_int,
            phi_tau_frames=phi_tau_frames,
            sigma_diag=sigma_diag,
        )

        posthoc = self.build_posthoc_products(
            setup=setup,
            forward=forward,
            phi_tau_frames=phi_tau_frames,
            click=click,
        )
        self.print_posthoc_summary(posthoc)
        if posthoc.result is not None and posthoc.base_rho is not None:
            print(
                "[POSTHOC SAVE READY] "
                f"base_rho_shape={posthoc.base_rho.shape} "
                f"selected_rho={'yes' if posthoc.rho_selected is not None else 'no'}"
            )

        sigma_products = self.build_sigma_products(
            setup=setup,
            forward=forward,
            phi_tau_frames=phi_tau_frames,
            click=click,
        )

        bohm = self.build_bohmian_overlay(
            setup=setup,
            forward=forward,
            sigma_products=sigma_products,
        )

        self.print_summary_stats(
            setup=setup,
            forward=forward,
            sigma_products=sigma_products,
            bohm=bohm,
        )

        self.save_run_outputs(
            setup=setup,
            forward=forward,
            click=click,
            phi_tau_frames=phi_tau_frames,
            sigma_products=sigma_products,
            bohm=bohm,
            posthoc=posthoc,
        )

        print("frame0 rho max:", np.max(forward.frames_density[0]))
        print("frame0 Emix density max:", np.max(make_emix_density(sigma_products.Emix_init)[0]))
        print("frame0 overlap rho max:", np.max(sigma_products.rho_init[0]))

        if posthoc.base_rho is not None:
            print("frame0 posthoc base_rho max:", np.max(posthoc.base_rho[0]))
            if posthoc.rho_selected is not None:
                print("frame0 posthoc selected_rho max:", np.max(posthoc.rho_selected[0]))

        print("Simulation done.")
        print("Use visualize.py for animation, slider and debug plots.")


# ============================================================
# Entry point
# ============================================================

def main():
    cfg = AppConfig()
    app = QuantumSimulationApp(cfg)
    app.run()


if __name__ == "__main__":
    main()