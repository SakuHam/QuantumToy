from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider


class SimulationAnimator:

    def __init__(
        self,
        cfg,
        grid,
        times,
        norms,
        rho_init,
        ridge_x_init,
        ridge_y_init,
        ridge_s_init,
        cos_th_init,
        speed_init,
        ux_init,
        uy_init,
        div_v_init,
        vref,
        speed_ref,
        bohm_traj_x=None,
        bohm_traj_y=None,
        bohm_traj_alive=None,
    ):

        self.cfg = cfg
        self.grid = grid
        self.times = times
        self.norms = norms

        self.rho_current = [rho_init]
        self.ridge_x = [ridge_x_init]
        self.ridge_y = [ridge_y_init]
        self.ridge_s = [ridge_s_init]

        self.cos_th = [cos_th_init]
        self.speed = [speed_init]
        self.ux = [ux_init]
        self.uy = [uy_init]
        self.div_v_ridge = [div_v_init]

        self.vref = vref
        self.speed_ref = speed_ref

        self.bohm_traj_x = bohm_traj_x
        self.bohm_traj_y = bohm_traj_y
        self.bohm_traj_alive = bohm_traj_alive

        self.arrow_state = {"ux": np.nan, "uy": np.nan, "spd": np.nan}

        self._setup_plot()

    # ============================================================
    # Display helper
    # ============================================================

    def gamma_display(self, arr):

        if self.cfg.USE_FIXED_DISPLAY_SCALE:
            disp = np.clip(arr / (self.vref + 1e-30), 0.0, 1.0)
            return disp ** self.cfg.GAMMA

        m = float(np.max(arr))
        if m <= 0:
            return arr

        return (arr / m) ** self.cfg.GAMMA

    # ============================================================
    # Plot setup
    # ============================================================

    def _setup_plot(self):

        extent = (
            -self.cfg.VISIBLE_LX / 2,
            self.cfg.VISIBLE_LX / 2,
            -self.cfg.VISIBLE_LY / 2,
            self.cfg.VISIBLE_LY / 2,
        )

        self.fig = plt.figure(figsize=(10.8, 7.2))
        self.ax = self.fig.add_axes([0.07, 0.18, 0.86, 0.78])

        self.im = self.ax.imshow(
            self.gamma_display(self.rho_current[0][0]),
            extent=extent,
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
            interpolation=self.cfg.IM_INTERPOLATION,
        )

        self.ax.axvline(self.cfg.barrier_center_x, color="white", linestyle="--", alpha=0.6)
        self.ax.axvline(self.cfg.screen_center_x, color="cyan", linestyle="--", alpha=0.4)

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        self.title = self.ax.set_title("ρ(t)")

        # ridge marker
        self.ridge_marker, = self.ax.plot(
            [self.ridge_x[0][0]],
            [self.ridge_y[0][0]],
            marker="o",
            markersize=7,
            linestyle="None",
            color="lime",
            alpha=0.9,
        )

        # ridge trail
        self.ridge_trail, = self.ax.plot(
            [],
            [],
            linestyle="-",
            linewidth=1.5,
            color="lime",
            alpha=0.5,
        )

        # flow arrow
        self.flow_quiver = None

        if self.cfg.DRAW_FLOW_ARROW and self.ux[0] is not None:

            self.flow_quiver = self.ax.quiver(
                [self.ridge_x[0][0]],
                [self.ridge_y[0][0]],
                [0.0],
                [0.0],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="cyan",
                alpha=0.9,
                width=0.006,
            )

        # Bohmian trajectories
        self.bohm_lines = []
        self.bohm_heads = []

        if self.cfg.ENABLE_BOHMIAN_OVERLAY and self.bohm_traj_x is not None:

            for k in range(self.bohm_traj_x.shape[0]):

                line_k, = self.ax.plot(
                    [],
                    [],
                    linestyle="-",
                    linewidth=self.cfg.BOHMIAN_LINEWIDTH,
                    color=self.cfg.BOHMIAN_COLOR,
                    alpha=0.85,
                )

                self.bohm_lines.append(line_k)

                if self.cfg.BOHMIAN_SHOW_HEAD:
                    head_k, = self.ax.plot(
                        [],
                        [],
                        marker="o",
                        markersize=self.cfg.BOHMIAN_HEAD_SIZE,
                        linestyle="None",
                        color=self.cfg.BOHMIAN_HEAD_COLOR,
                    )
                else:
                    head_k = None

                self.bohm_heads.append(head_k)

        # slider
        self.ax_sigma = self.fig.add_axes([0.10, 0.08, 0.80, 0.04])

        self.sigma_slider = Slider(
            ax=self.ax_sigma,
            label="sigmaT (time thickness)",
            valmin=0.001,
            valmax=5.0,
            valinit=1.0,
        )

    # ============================================================
    # Arrow update
    # ============================================================

    def update_flow_arrow(self, i):

        if self.flow_quiver is None:
            return

        uxi = self.ux[0][i]
        uyi = self.uy[0][i]
        spd = self.speed[0][i]

        valid = (
            np.isfinite(uxi)
            and np.isfinite(uyi)
            and np.isfinite(spd)
            and (spd > self.cfg.ALIGN_EPS_SPEED)
        )

        if not valid:
            return

        self.arrow_state["ux"] = uxi
        self.arrow_state["uy"] = uyi
        self.arrow_state["spd"] = spd

        self.flow_quiver.set_offsets([[self.ridge_x[0][i], self.ridge_y[0][i]]])

        L = self.cfg.ARROW_SCALE * float(
            np.clip(spd / (self.speed_ref + 1e-30), 0.0, 2.5)
        )

        self.flow_quiver.set_UVC([L * uxi], [L * uyi])

    # ============================================================
    # Bohmian overlay
    # ============================================================

    def update_bohmian_overlay(self, i):

        if not self.cfg.ENABLE_BOHMIAN_OVERLAY:
            return

        for k in range(self.bohm_traj_x.shape[0]):

            alive = self.bohm_traj_alive[k]

            if not np.any(alive[: i + 1]):
                continue

            xs = self.bohm_traj_x[k, : i + 1]
            ys = self.bohm_traj_y[k, : i + 1]

            self.bohm_lines[k].set_data(xs, ys)

            if self.bohm_heads[k] is not None:

                idx = np.where(alive[: i + 1])[0]

                if idx.size > 0:
                    j = int(idx[-1])
                    self.bohm_heads[k].set_data(
                        [self.bohm_traj_x[k, j]],
                        [self.bohm_traj_y[k, j]],
                    )

    # ============================================================
    # Frame update
    # ============================================================

    def update(self, i):

        self.im.set_data(self.gamma_display(self.rho_current[0][i]))

        self.ridge_marker.set_data(
            [self.ridge_x[0][i]],
            [self.ridge_y[0][i]],
        )

        if self.cfg.SHOW_TRAIL:

            j0 = max(0, i - self.cfg.TRAIL_LEN + 1)

            self.ridge_trail.set_data(
                self.ridge_x[0][j0 : i + 1],
                self.ridge_y[0][j0 : i + 1],
            )

        self.update_flow_arrow(i)
        self.update_bohmian_overlay(i)

        self.title.set_text(
            f"t={self.times[i]:.3f} | norm≈{self.norms[i]:.4f}"
        )

        return (self.im, self.ridge_marker, self.ridge_trail)

    # ============================================================
    # Run animation
    # ============================================================

    def run(self):

        ani = FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.times),
            interval=40,
            blit=False,
        )

        ani.save(self.cfg.OUTPUT_MP4, writer="ffmpeg", fps=25, dpi=150)

        plt.show()