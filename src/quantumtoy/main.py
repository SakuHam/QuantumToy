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

    for n in range(cfg.n_steps + 1):
        rho = theory.density(state)
        norm_now = norm_prob(rho, grid.dx, grid.dy)

        if n % cfg.save_every == 0:
            frames_density.append(rho[grid.ys, grid.xs].copy())
            times.append(n * cfg.dt)
            norms.append(norm_now)

            if cfg.SAVE_COMPLEX_STATE_FRAMES:
                state_vis_frames.append(state[grid.ys, grid.xs].copy())

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

        rho = make_rho(
            frames_psi=frames_density,
            Emix=Emix,
            dx=grid.dx,
            dy=grid.dy,
        )

        rx, ry, rs = compute_ridge_xy(
            frames_psi=frames_density,
            Emix=Emix,
            x_vis_1d=grid.x_vis_1d,
            y_vis_1d=grid.y_vis_1d,
            mode=cfg.RIDGE_MODE,
            top_q=cfg.CENTROID_TOP_Q,
            radius=cfg.LOCALMAX_RADIUS,
            alpha_smooth=cfg.LOCALMAX_SMOOTH_ALPHA,
        )

        cos_th = speed = ux = uy = div_v = None

        if cfg.SAVE_COMPLEX_STATE_FRAMES and state_vis_frames is not None:
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
    v_est = cfg.k0x / cfg.m_mass
    L_gap = cfg.screen_center_x - cfg.barrier_center_x
    t_gap = L_gap / (v_est + 1e-12)

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


if __name__ == "__main__":
    main()