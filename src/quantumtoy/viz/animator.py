from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider


class SimulationAnimator:
    """
    Improved animator with structured static geometry rendering.

    Expected optional cfg flags
    ---------------------------
    BARRIER_PLOT_MODE:
        "potential" | "mask" | "contour_only" | "off"
        default: "potential"

    SHOW_SCREEN_OVERLAY:
        bool, default True

    SHOW_CAP_OVERLAY:
        bool, default True

    SHOW_COMPONENT_LABELS:
        bool, default False

    GEOMETRY_CONTOUR_LEVELS:
        list[float] or tuple[float, ...]
        relative levels for per-component contours in potential mode,
        default: (0.15, 0.5, 0.85)

    BARRIER_MASK_ALPHA:
        float, default 0.26

    BARRIER_POTENTIAL_ALPHA_MIN:
        float, default 0.18

    BARRIER_POTENTIAL_ALPHA_MAX:
        float, default 0.55

    SCREEN_ALPHA:
        float, default 0.16

    CAP_ALPHA:
        float, default 0.10
    """

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
        potential_spec=None,
        bohm_traj_x=None,
        bohm_traj_y=None,
        bohm_traj_alive=None,
    ):
        self.cfg = cfg
        self.grid = grid
        self.times = times
        self.norms = norms
        self.potential_spec = potential_spec

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

        self._static_artists = []
        self._component_labels = []

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
    # Config helpers
    # ============================================================

    def _cfg_get(self, name, default):
        return getattr(self.cfg, name, default)

    def _visible_extent(self):
        return (
            -self.cfg.VISIBLE_LX / 2.0,
            self.cfg.VISIBLE_LX / 2.0,
            -self.cfg.VISIBLE_LY / 2.0,
            self.cfg.VISIBLE_LY / 2.0,
        )

    def _get_visible_field(self, arr2d: np.ndarray) -> np.ndarray:
        return arr2d[self.grid.ys, self.grid.xs]

    def _barrier_plot_mode(self) -> str:
        return str(self._cfg_get("BARRIER_PLOT_MODE", "mask")).lower().strip()

    # ============================================================
    # Geometry helpers
    # ============================================================

    def _iter_components(self):
        """
        Yield barrier components from potential_spec if available.
        Fallback to legacy single combined component built from barrier_core/slit masks.
        """
        if self.potential_spec is None:
            return []

        comps = getattr(self.potential_spec, "components", None)
        if comps:
            return list(comps)

        # Legacy fallback
        barrier_core = getattr(self.potential_spec, "barrier_core", None)
        slit1_mask = getattr(self.potential_spec, "slit1_mask", None)
        slit2_mask = getattr(self.potential_spec, "slit2_mask", None)
        V_real = getattr(self.potential_spec, "V_real", None)

        if barrier_core is None or V_real is None:
            return []

        return [
            _LegacyBarrierComponent(
                name="legacy_barrier",
                kind="legacy",
                V_real=V_real,
                barrier_core=barrier_core,
                slit_masks={
                    "slit1": slit1_mask if slit1_mask is not None else np.zeros_like(barrier_core, dtype=bool),
                    "slit2": slit2_mask if slit2_mask is not None else np.zeros_like(barrier_core, dtype=bool),
                },
            )
        ]

    def _component_center_xy(self, comp):
        core = comp.barrier_core
        if core is None or not np.any(core):
            return None

        x = float(np.mean(self.grid.X[core]))
        y = float(np.mean(self.grid.Y[core]))
        return x, y

    def _draw_component_label(self, comp):
        if not self._cfg_get("SHOW_COMPONENT_LABELS", False):
            return

        center = self._component_center_xy(comp)
        if center is None:
            return

        x, y = center
        txt = self.ax.text(
            x,
            y,
            comp.name,
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            alpha=0.75,
            zorder=12,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.35, edgecolor="none"),
        )
        self._component_labels.append(txt)

    def _add_artist(self, artist):
        if artist is None:
            return
        self._static_artists.append(artist)

    def _draw_component_mask_mode(self, comp, extent):
        core_vis = self._get_visible_field(comp.barrier_core).astype(float)
        slit_union = np.zeros_like(core_vis, dtype=bool)

        for _, mask in comp.slit_masks.items():
            if mask is not None:
                slit_union |= self._get_visible_field(mask)

        wall_vis = np.where(core_vis > 0.5, 1.0, 0.0)
        wall_vis[slit_union] = 0.0

        alpha_base = float(self._cfg_get("BARRIER_MASK_ALPHA", 0.26))
        alpha = alpha_base * wall_vis

        artist = self.ax.imshow(
            np.ones_like(wall_vis),
            extent=extent,
            origin="lower",
            cmap="gray",
            interpolation="nearest",
            alpha=alpha,
            zorder=4,
        )
        self._add_artist(artist)

        if np.any(wall_vis > 0.5):
            try:
                cs = self.ax.contour(
                    wall_vis,
                    levels=[0.5],
                    extent=extent,
                    origin="lower",
                    colors=["white"],
                    linewidths=[1.2],
                    alpha=0.9,
                    zorder=6,
                )
                self._add_artist(cs)
            except Exception:
                pass

    def _draw_component_potential_mode(self, comp, extent):
        V_vis = self._get_visible_field(comp.V_real)
        vmax = float(np.max(V_vis))
        if vmax <= 0.0:
            return

        Vn = V_vis / (vmax + 1e-30)

        alpha_min = float(self._cfg_get("BARRIER_POTENTIAL_ALPHA_MIN", 0.18))
        alpha_max = float(self._cfg_get("BARRIER_POTENTIAL_ALPHA_MAX", 0.55))
        alpha = np.where(Vn > 1e-8, alpha_min + (alpha_max - alpha_min) * Vn, 0.0)

        artist = self.ax.imshow(
            np.ones_like(Vn),
            extent=extent,
            origin="lower",
            cmap="gray",
            interpolation="nearest",
            alpha=alpha,
            zorder=4,
        )
        self._add_artist(artist)

        levels_rel = self._cfg_get("GEOMETRY_CONTOUR_LEVELS", (0.15, 0.5, 0.85))
        levels_abs = [float(lv) * vmax for lv in levels_rel if 0.0 < float(lv) < 1.0]

        if levels_abs:
            try:
                cs = self.ax.contour(
                    V_vis,
                    levels=levels_abs,
                    extent=extent,
                    origin="lower",
                    colors=["white"] * len(levels_abs),
                    linewidths=np.linspace(0.8, 1.2, len(levels_abs)),
                    alpha=0.85,
                    zorder=6,
                )
                self._add_artist(cs)
            except Exception:
                pass

    def _draw_component_contour_only_mode(self, comp, extent):
        V_vis = self._get_visible_field(comp.V_real)
        vmax = float(np.max(V_vis))

        if vmax > 0.0:
            level = 0.5 * vmax
            try:
                cs = self.ax.contour(
                    V_vis,
                    levels=[level],
                    extent=extent,
                    origin="lower",
                    colors=["white"],
                    linewidths=[1.4],
                    alpha=0.92,
                    zorder=6,
                )
                self._add_artist(cs)
            except Exception:
                pass
            return

        # fallback to carved hard mask
        core_vis = self._get_visible_field(comp.barrier_core).astype(float)
        slit_union = np.zeros_like(core_vis, dtype=bool)
        for _, mask in comp.slit_masks.items():
            if mask is not None:
                slit_union |= self._get_visible_field(mask)

        wall_vis = np.where(core_vis > 0.5, 1.0, 0.0)
        wall_vis[slit_union] = 0.0

        if np.any(wall_vis > 0.5):
            try:
                cs = self.ax.contour(
                    wall_vis,
                    levels=[0.5],
                    extent=extent,
                    origin="lower",
                    colors=["white"],
                    linewidths=[1.4],
                    alpha=0.92,
                    zorder=6,
                )
                self._add_artist(cs)
            except Exception:
                pass

    def _draw_barrier_components(self):
        if self.potential_spec is None:
            self.ax.axvline(
                self.cfg.barrier_center_x,
                color="white",
                linestyle="--",
                alpha=0.6,
                linewidth=1.1,
                zorder=5,
            )
            self.ax.axvline(
                self.cfg.screen_center_x,
                color="cyan",
                linestyle="--",
                alpha=0.45,
                linewidth=1.0,
                zorder=5,
            )
            return

        extent = self._visible_extent()
        mode = self._barrier_plot_mode()

        if mode == "off":
            return

        for comp in self._iter_components():
            if mode == "mask":
                self._draw_component_mask_mode(comp, extent)
            elif mode == "contour_only":
                self._draw_component_contour_only_mode(comp, extent)
            else:
                self._draw_component_potential_mode(comp, extent)

            self._draw_component_label(comp)

    def _draw_screen_overlay(self):
        if self.potential_spec is None:
            return

        if not self._cfg_get("SHOW_SCREEN_OVERLAY", True):
            return

        extent = self._visible_extent()
        screen_vis = self.potential_spec.screen_mask_vis.astype(float)

        if not np.any(screen_vis > 0.5):
            return

        alpha_scale = float(self._cfg_get("SCREEN_ALPHA", 0.16))

        artist = self.ax.imshow(
            screen_vis,
            extent=extent,
            origin="lower",
            cmap="Blues",
            interpolation="nearest",
            alpha=alpha_scale * screen_vis,
            zorder=3,
        )
        self._add_artist(artist)

        try:
            cs = self.ax.contour(
                screen_vis,
                levels=[0.5],
                extent=extent,
                origin="lower",
                colors=["cyan"],
                linewidths=[1.0],
                alpha=0.70,
                zorder=6,
            )
            self._add_artist(cs)
        except Exception:
            pass

    def _draw_cap_overlay(self):
        if self.potential_spec is None:
            return

        if not self._cfg_get("SHOW_CAP_OVERLAY", True):
            return

        extent = self._visible_extent()
        W_vis = self._get_visible_field(self.potential_spec.W)

        wmax = float(np.max(W_vis))
        if wmax <= 0.0:
            return

        Wn = W_vis / (wmax + 1e-30)
        alpha_scale = float(self._cfg_get("CAP_ALPHA", 0.10))

        artist = self.ax.imshow(
            Wn,
            extent=extent,
            origin="lower",
            cmap="Greens",
            interpolation="nearest",
            alpha=alpha_scale * Wn,
            zorder=2,
        )
        self._add_artist(artist)

        try:
            cs = self.ax.contour(
                Wn,
                levels=[0.2, 0.5, 0.8],
                extent=extent,
                origin="lower",
                colors=["lime", "lime", "lime"],
                linewidths=[0.5, 0.7, 0.9],
                alpha=0.35,
                zorder=5,
            )
            self._add_artist(cs)
        except Exception:
            pass

    def _draw_static_geometry(self):
        self._draw_screen_overlay()
        self._draw_cap_overlay()
        self._draw_barrier_components()

    # ============================================================
    # Plot setup
    # ============================================================

    def _setup_plot(self):
        extent = self._visible_extent()

        self.fig = plt.figure(figsize=(11.0, 7.3))
        self.ax = self.fig.add_axes([0.07, 0.18, 0.86, 0.78])

        self.im = self.ax.imshow(
            self.gamma_display(self.rho_current[0][0]),
            extent=extent,
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
            interpolation=self.cfg.IM_INTERPOLATION,
            zorder=1,
        )

        self._draw_static_geometry()

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_aspect("equal")
        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])

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
            zorder=10,
        )

        # ridge trail
        self.ridge_trail, = self.ax.plot(
            [],
            [],
            linestyle="-",
            linewidth=1.5,
            color="lime",
            alpha=0.5,
            zorder=9,
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
                alpha=0.92,
                width=0.006,
                zorder=11,
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
                    zorder=8,
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
                        zorder=9,
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

        if self.bohm_traj_x is None:
            return

        for k in range(self.bohm_traj_x.shape[0]):
            alive = self.bohm_traj_alive[k]

            if not np.any(alive[: i + 1]):
                self.bohm_lines[k].set_data([], [])
                if self.bohm_heads[k] is not None:
                    self.bohm_heads[k].set_data([], [])
                continue

            mask = alive[: i + 1]
            xs = self.bohm_traj_x[k, : i + 1][mask]
            ys = self.bohm_traj_y[k, : i + 1][mask]
            self.bohm_lines[k].set_data(xs, ys)

            if self.bohm_heads[k] is not None:
                idx = np.where(alive[: i + 1])[0]
                if idx.size > 0:
                    j = int(idx[-1])
                    self.bohm_heads[k].set_data(
                        [self.bohm_traj_x[k, j]],
                        [self.bohm_traj_y[k, j]],
                    )
                else:
                    self.bohm_heads[k].set_data([], [])

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
        else:
            self.ridge_trail.set_data([], [])

        self.update_flow_arrow(i)
        self.update_bohmian_overlay(i)

        self.title.set_text(
            f"t={self.times[i]:.3f} | norm≈{self.norms[i]:.4f}"
        )

        artists = [self.im, self.ridge_marker, self.ridge_trail]
        if self.flow_quiver is not None:
            artists.append(self.flow_quiver)
        artists.extend(self.bohm_lines)
        artists.extend([h for h in self.bohm_heads if h is not None])

        return tuple(artists)

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


class _LegacyBarrierComponent:
    def __init__(self, name, kind, V_real, barrier_core, slit_masks):
        self.name = name
        self.kind = kind
        self.V_real = V_real
        self.barrier_core = barrier_core
        self.slit_masks = slit_masks