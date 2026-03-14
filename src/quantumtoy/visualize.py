from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.colors import PowerNorm

from config import AppConfig
from core.grid import build_grid
from core.potentials import build_double_slit_and_caps
from core.simulation_types import PotentialSpec
from theories.registry import build_theory

from analysis.emix import (
    build_Emix_from_phi_tau,
    build_Emix_density_from_phi_tau,
    make_rho,
)
from analysis.ridge import compute_ridge_xy
from analysis.current import alignment_and_diagnostics_from_state_frames

from file.run_io import load_run_bundle, apply_cfg_dict
from viz.visual_debug import (
    debug_plot_phase_density_composite_vis,
    debug_plot_phase_density_composite_with_contours_vis,
    debug_plot_phase_winding_vis,
)


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


def estimate_group_velocity(cfg, theory) -> float:
    theory_name = getattr(cfg, "THEORY_NAME", "").lower()
    class_name = theory.__class__.__name__.lower()

    is_dirac_like = ("dirac" in theory_name) or ("dirac" in class_name)

    if is_dirac_like:
        p0 = float(theory.hbar * cfg.k0x)
        mc2 = float(theory.m_mass * theory.c_light**2)
        E0 = float(np.sqrt((theory.c_light * p0) ** 2 + mc2**2))
        return float((theory.c_light**2 * p0) / (E0 + 1e-30))

    return float(cfg.k0x / (cfg.m_mass + 1e-30))


def build_cfg_from_meta(meta: dict):
    cfg = AppConfig()
    return apply_cfg_dict(cfg, meta["config"])


def build_potential_from_flag(grid, cfg, debug_free_case: bool):
    if debug_free_case:
        zeros = np.zeros_like(grid.X, dtype=float)
        false_mask = np.zeros_like(grid.X, dtype=bool)
        false_mask_vis = np.zeros((grid.n_visible_y, grid.n_visible_x), dtype=bool)

        return PotentialSpec(
            V_real=zeros.copy(),
            W=zeros.copy(),
            screen_mask_full=false_mask.copy(),
            screen_mask_vis=false_mask_vis.copy(),
            barrier_core=false_mask.copy(),
            slit1_mask=false_mask.copy(),
            slit2_mask=false_mask.copy(),
        )

    return build_double_slit_and_caps(grid, cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", help="Path to saved .npz run bundle")
    parser.add_argument("--meta", default=None, help="Optional path to matching .json meta")
    parser.add_argument("--save-mp4", action="store_true", help="Save animation to mp4")
    parser.add_argument("--output-mp4", default=None, help="Override mp4 output path")

    parser.add_argument("--debug-phase", action="store_true")
    parser.add_argument("--debug-phase-contours", action="store_true")
    parser.add_argument("--debug-winding", action="store_true")
    parser.add_argument("--debug-frame", type=int, default=-1, help="Frame index for debug plots, default last")

    args = parser.parse_args()

    bundle = load_run_bundle(args.npz_path, args.meta)
    meta = bundle["meta"]

    cfg = build_cfg_from_meta(meta)

    grid = build_grid(
        visible_lx=cfg.VISIBLE_LX,
        visible_ly=cfg.VISIBLE_LY,
        n_visible_x=cfg.N_VISIBLE_X,
        n_visible_y=cfg.N_VISIBLE_Y,
        pad_factor=cfg.PAD_FACTOR,
    )

    potential = build_potential_from_flag(
        grid,
        cfg,
        debug_free_case=bool(meta.get("debug_free_case", False)),
    )
    theory = build_theory(cfg, grid, potential)

    times = bundle["times"]
    frames_density = bundle["frames_density"]
    state_vis_frames = bundle["state_vis_frames"]
    norms = bundle["norms"]
    phi_tau_frames = bundle["phi_tau_frames"]

    x_click = bundle["x_click"]
    y_click = bundle["y_click"]
    t_det = bundle["t_det"]

    sigma_init = bundle["sigma_init"]
    vref = bundle["vref"]
    speed_ref = bundle["speed_ref"]

    bohm_traj_x = bundle["bohm_traj_x"]
    bohm_traj_y = bundle["bohm_traj_y"]
    bohm_traj_alive = bundle["bohm_traj_alive"]

    Nt = len(times)
    tau_step = cfg.save_every * cfg.dt

    extent = (
        grid.x_vis_min,
        grid.x_vis_max,
        grid.y_vis_min,
        grid.y_vis_max,
    )

    if state_vis_frames is not None:
        dbg_i = args.debug_frame if args.debug_frame >= 0 else (len(state_vis_frames) - 1)
        dbg_i = max(0, min(dbg_i, len(state_vis_frames) - 1))
        state_dbg = state_vis_frames[dbg_i]

        if args.debug_phase:
            debug_plot_phase_density_composite_vis(
                state_vis=state_dbg,
                extent=extent,
                title=f"Phase + density composite (frame {dbg_i}, t={times[dbg_i]:.3f})",
            )

        if args.debug_phase_contours:
            debug_plot_phase_density_composite_with_contours_vis(
                state_vis=state_dbg,
                X_vis=grid.X_vis,
                Y_vis=grid.Y_vis,
                extent=extent,
                title=f"Phase + density composite + contours (frame {dbg_i}, t={times[dbg_i]:.3f})",
            )

        if args.debug_winding:
            debug_plot_phase_winding_vis(
                state_vis=state_dbg,
                extent=extent,
                dx=grid.dx,
                dy=grid.dy,
                title=f"Phase winding (frame {dbg_i}, t={times[dbg_i]:.3f})",
            )

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
            raise RuntimeError("Visualization of sigma-dependent overlap requires complex state frames")

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

        return rho, rx, ry, rs, cos_th, speed, ux, uy, div_v

    v_est = estimate_group_velocity(cfg, theory)
    L_gap = cfg.screen_center_x - cfg.barrier_center_x
    t_gap = L_gap / (abs(v_est) + 1e-12)

    sigma_min = 0.05 * t_gap
    sigma_max = 2.00 * t_gap

    rho_init, ridge_x_init, ridge_y_init, ridge_s_init, cos_th_init, speed_init, ux_init, uy_init, div_v_init = build_all_for_sigma(sigma_init)

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
        cmap="magma",
        interpolation=cfg.IM_INTERPOLATION,
        norm=PowerNorm(gamma=0.35),
    )

    ax.axvline(cfg.barrier_center_x, color="white", linestyle="--", alpha=0.6)
    ax.axvline(cfg.screen_center_x, color="cyan", linestyle="--", alpha=0.4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    title = ax.set_title(
        rf"ρ(t): σT={sigma_init:.3f}, t={times[0]:.3f}, "
        rf"ridge={cfg.RIDGE_MODE}, theory={cfg.THEORY_NAME}, "
        rf"detector={getattr(cfg, 'DETECTOR_NAME', 'unknown')}"
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

    click_marker, = ax.plot(
        [x_click],
        [y_click],
        marker="x",
        markersize=9,
        linestyle="None",
        color="yellow",
        alpha=0.9,
        label="click",
    )

    ridge_trail, = ax.plot([], [], linestyle="-", linewidth=1.5, color="lime", alpha=0.5)

    flow_quiver = None
    if cfg.DRAW_FLOW_ARROW and (ux_init is not None):
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

            if cfg.ARROW_HOLD_LAST_WHEN_INVALID and np.isfinite(arrow_state["ux"]) and np.isfinite(arrow_state["uy"]):
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

        L = cfg.ARROW_SCALE * float(np.clip(spd / (speed_ref + 1e-30), 0.0, 2.5))
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

            j0 = 0 if cfg.BOHMIAN_SHOW_FULL_PATH_EACH_FRAME else max(0, i - cfg.BOHMIAN_TRAIL_LEN + 1)

            mask = alive[j0 : i + 1]
            xs_seg = bohm_traj_x[k, j0 : i + 1][mask]
            ys_seg = bohm_traj_y[k, j0 : i + 1][mask]
            bohm_lines[k].set_data(xs_seg, ys_seg)

            if cfg.BOHMIAN_SHOW_HEAD and bohm_heads[k] is not None:
                alive_idx = np.where(alive[: i + 1])[0]
                if alive_idx.size > 0:
                    ilast = int(alive_idx[-1])
                    bohm_heads[k].set_data([bohm_traj_x[k, ilast]], [bohm_traj_y[k, ilast]])
                else:
                    bohm_heads[k].set_data([], [])

    def make_title(i: int):
        parts = [
            rf"ρ(t): σT={sigma_current[0]:.3f}",
            rf"t={times[i]:.3f}",
            rf"ridge={cfg.RIDGE_MODE}",
            rf"theory={cfg.THEORY_NAME}",
            rf"detector={getattr(cfg, 'DETECTOR_NAME', 'unknown')}",
            rf"norm≈{norms[i]:.4f}",
            rf"Γ≈{ridge_s[0][i]:.3e}",
        ]

        if cos_th[0] is not None and np.isfinite(cos_th[0][i]):
            parts.append(rf"cosθ≈{cos_th[0][i]:.3f}")

        if cfg.ENABLE_DIVERGENCE_DIAGNOSTIC and div_v_ridge[0] is not None and np.isfinite(div_v_ridge[0][i]):
            parts.append(rf"div v≈{div_v_ridge[0][i]:.3e}")

        if cfg.ENABLE_BOHMIAN_OVERLAY:
            parts.append("Bohm=RK4" if cfg.BOHMIAN_USE_RK4 else "Bohm=Euler")

        return " | ".join(parts)

    def on_sigma_change(_val):
        new_sigma = float(sigma_slider.val)
        sigma_current[0] = new_sigma

        rho_new, rx, ry, rs, cth, spd, uxx, uyy, divv = build_all_for_sigma(new_sigma)

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

        im.set_data(gamma_display(rho_current[0][i], vref=vref, gamma=cfg.GAMMA, use_fixed_scale=cfg.USE_FIXED_DISPLAY_SCALE))
        ridge_marker.set_data([ridge_x[0][i]], [ridge_y[0][i]])

        if cfg.SHOW_TRAIL:
            j0 = max(0, i - cfg.TRAIL_LEN + 1)
            ridge_trail.set_data(ridge_x[0][j0 : i + 1], ridge_y[0][j0 : i + 1])

        update_flow_arrow(i)
        update_bohmian_overlay(i)
        title.set_text(make_title(i))
        fig.canvas.draw_idle()

    sigma_slider.on_changed(on_sigma_change)

    def update(i: int):
        on_sigma_change.last_i = i

        im.set_data(gamma_display(rho_current[0][i], vref=vref, gamma=cfg.GAMMA, use_fixed_scale=cfg.USE_FIXED_DISPLAY_SCALE))
        ridge_marker.set_data([ridge_x[0][i]], [ridge_y[0][i]])

        if cfg.SHOW_TRAIL:
            j0 = max(0, i - cfg.TRAIL_LEN + 1)
            ridge_trail.set_data(ridge_x[0][j0 : i + 1], ridge_y[0][j0 : i + 1])

        update_flow_arrow(i)
        update_bohmian_overlay(i)
        title.set_text(make_title(i))

        artists = [im, ridge_marker, click_marker, ridge_trail]
        if flow_quiver is not None:
            artists.append(flow_quiver)
        artists.extend([obj for obj in bohm_lines if obj is not None])
        artists.extend([obj for obj in bohm_heads if obj is not None])
        return tuple(artists)

    ani = FuncAnimation(fig, update, frames=Nt, interval=40, blit=False)

    if args.save_mp4:
        output_mp4 = args.output_mp4
        if output_mp4 is None:
            output_mp4 = getattr(cfg, "OUTPUT_MP4", str(Path(args.npz_path).with_suffix(".mp4")))
        print(f"[SAVE] animation -> {output_mp4}")
        ani.save(output_mp4, writer="ffmpeg", fps=25, dpi=150)

    plt.show()


if __name__ == "__main__":
    main()