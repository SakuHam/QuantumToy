from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.colors import PowerNorm, hsv_to_rgb

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
    debug_plot_scalar_field_vis,
    debug_plot_metric_fields_vis,
)

RENDER_MODES = (
    "density",                  # current visible rho (current pipeline)
    "forward_density",          # |psi_fwd|^2 only
    "backward_density",         # |phi_tau|^2 only
    "overlap_density",          # |psi_fwd * conj(Emix)|^2 style amplitude
    "posthoc_base_rho",         # saved posthoc base rho
    "posthoc_selected_rho",     # saved posthoc worldline-selected rho
    "phase",
    "phase_contours",
    "ridge_phase",
    "ridge_phase_contours",
)


# ============================================================
# Geometry overlay helpers
# ============================================================

class _LegacyBarrierComponent:
    def __init__(self, name, kind, V_real, barrier_core, wall_mask, slit_masks):
        self.name = name
        self.kind = kind
        self.V_real = V_real
        self.barrier_core = barrier_core
        self.wall_mask = wall_mask
        self.slit_masks = slit_masks


def _cfg_get(cfg, name, default):
    return getattr(cfg, name, default)


def _get_visible_field(grid, arr2d: np.ndarray) -> np.ndarray:
    return arr2d[grid.ys, grid.xs]


def _build_wall_mask_from_legacy(grid, barrier_core, slit_masks):
    wall_mask = barrier_core.copy()
    for _, mask in slit_masks.items():
        if mask is not None:
            wall_mask &= (~mask)
    return wall_mask


def _iter_barrier_components(potential):
    comps = getattr(potential, "components", None)
    if comps:
        return list(comps)

    barrier_core = getattr(potential, "barrier_core", None)
    slit1_mask = getattr(potential, "slit1_mask", None)
    slit2_mask = getattr(potential, "slit2_mask", None)
    V_real = getattr(potential, "V_real", None)

    if barrier_core is None or V_real is None:
        return []

    slit_masks = {
        "slit1": slit1_mask if slit1_mask is not None else np.zeros_like(barrier_core, dtype=bool),
        "slit2": slit2_mask if slit2_mask is not None else np.zeros_like(barrier_core, dtype=bool),
    }

    wall_mask = _build_wall_mask_from_legacy(None, barrier_core, slit_masks)

    return [
        _LegacyBarrierComponent(
            name="legacy_barrier",
            kind="legacy",
            V_real=V_real,
            barrier_core=barrier_core,
            wall_mask=wall_mask,
            slit_masks=slit_masks,
        )
    ]


def _component_center_xy(grid, comp):
    core = getattr(comp, "barrier_core", None)
    if core is None or not np.any(core):
        wall_mask = getattr(comp, "wall_mask", None)
        if wall_mask is None or not np.any(wall_mask):
            return None
        x = float(np.mean(grid.X[wall_mask]))
        y = float(np.mean(grid.Y[wall_mask]))
        return x, y

    x = float(np.mean(grid.X[core]))
    y = float(np.mean(grid.Y[core]))
    return x, y


def _draw_component_label(ax, grid, comp, cfg):
    if not _cfg_get(cfg, "SHOW_COMPONENT_LABELS", False):
        return None

    center = _component_center_xy(grid, comp)
    if center is None:
        return None

    x, y = center
    txt = ax.text(
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
    return txt


def _get_component_wall_vis(grid, comp):
    wall_mask = getattr(comp, "wall_mask", None)
    if wall_mask is not None:
        return _get_visible_field(grid, wall_mask).astype(float)

    core = getattr(comp, "barrier_core", None)
    if core is None:
        return None

    core_vis = _get_visible_field(grid, core).astype(float)
    slit_union = np.zeros_like(core_vis, dtype=bool)

    slit_masks = getattr(comp, "slit_masks", {})
    for _, mask in slit_masks.items():
        if mask is not None:
            slit_union |= _get_visible_field(grid, mask)

    wall_vis = np.where(core_vis > 0.5, 1.0, 0.0)
    wall_vis[slit_union] = 0.0
    return wall_vis


def _draw_component_mask_mode(ax, grid, comp, extent, cfg):
    artists = []

    wall_vis = _get_component_wall_vis(grid, comp)
    if wall_vis is None:
        return artists

    alpha_base = float(_cfg_get(cfg, "BARRIER_MASK_ALPHA", 0.26))
    alpha = alpha_base * wall_vis

    im = ax.imshow(
        np.ones_like(wall_vis),
        extent=extent,
        origin="lower",
        cmap="gray",
        interpolation="nearest",
        alpha=alpha,
        zorder=4,
    )
    artists.append(im)

    if np.any(wall_vis > 0.5):
        try:
            cs = ax.contour(
                wall_vis,
                levels=[0.5],
                extent=extent,
                origin="lower",
                colors=["white"],
                linewidths=[1.2],
                alpha=0.9,
                zorder=6,
            )
            artists.extend(cs.collections)
        except Exception:
            pass

    txt = _draw_component_label(ax, grid, comp, cfg)
    if txt is not None:
        artists.append(txt)

    return artists


def _draw_component_potential_mode(ax, grid, comp, extent, cfg):
    """
    Potential mode now uses wall_mask geometry for solid shape, but still
    overlays contours from V_real if desired. This avoids wide smooth tails
    appearing as giant blocks.
    """
    artists = []

    wall_vis = _get_component_wall_vis(grid, comp)
    if wall_vis is None:
        return artists

    alpha_min = float(_cfg_get(cfg, "BARRIER_POTENTIAL_ALPHA_MIN", 0.18))
    alpha_max = float(_cfg_get(cfg, "BARRIER_POTENTIAL_ALPHA_MAX", 0.55))
    alpha = np.where(wall_vis > 0.5, alpha_max, 0.0)

    im = ax.imshow(
        np.ones_like(wall_vis),
        extent=extent,
        origin="lower",
        cmap="gray",
        interpolation="nearest",
        alpha=alpha,
        zorder=4,
    )
    artists.append(im)

    V_real = getattr(comp, "V_real", None)
    if V_real is not None:
        V_vis = _get_visible_field(grid, V_real)
        vmax = float(np.max(V_vis))

        if vmax > 0.0:
            levels_rel = _cfg_get(cfg, "GEOMETRY_CONTOUR_LEVELS", (0.15, 0.5, 0.85))
            levels_abs = [float(lv) * vmax for lv in levels_rel if 0.0 < float(lv) < 1.0]

            if levels_abs:
                try:
                    cs = ax.contour(
                        V_vis,
                        levels=levels_abs,
                        extent=extent,
                        origin="lower",
                        colors=["white"] * len(levels_abs),
                        linewidths=np.linspace(0.8, 1.2, len(levels_abs)),
                        alpha=0.65,
                        zorder=6,
                    )
                    artists.extend(cs.collections)
                except Exception:
                    pass

    # Hard wall outline on top
    if np.any(wall_vis > 0.5):
        try:
            cs2 = ax.contour(
                wall_vis,
                levels=[0.5],
                extent=extent,
                origin="lower",
                colors=["white"],
                linewidths=[1.4],
                alpha=0.95,
                zorder=7,
            )
            artists.extend(cs2.collections)
        except Exception:
            pass

    txt = _draw_component_label(ax, grid, comp, cfg)
    if txt is not None:
        artists.append(txt)

    return artists


def _draw_component_contour_only_mode(ax, grid, comp, extent, cfg):
    artists = []

    wall_vis = _get_component_wall_vis(grid, comp)
    if wall_vis is None:
        return artists

    if np.any(wall_vis > 0.5):
        try:
            cs = ax.contour(
                wall_vis,
                levels=[0.5],
                extent=extent,
                origin="lower",
                colors=["white"],
                linewidths=[1.4],
                alpha=0.92,
                zorder=6,
            )
            artists.extend(cs.collections)
        except Exception:
            pass

    txt = _draw_component_label(ax, grid, comp, cfg)
    if txt is not None:
        artists.append(txt)

    return artists


def draw_static_geometry(ax, grid, potential, cfg, extent):
    """
    Draw screen/CAP/barrier geometry using structured barrier components when available.
    Uses wall_mask for solid barrier visualization.
    Returns a list of static matplotlib artists.
    """
    artists = []

    if potential is None:
        ax.axvline(cfg.barrier_center_x, color="white", linestyle="--", alpha=0.6, linewidth=1.1, zorder=5)
        ax.axvline(cfg.screen_center_x, color="cyan", linestyle="--", alpha=0.45, linewidth=1.0, zorder=5)
        return artists

    if _cfg_get(cfg, "SHOW_SCREEN_OVERLAY", True):
        screen_vis = potential.screen_mask_vis.astype(float)
        if np.any(screen_vis > 0.5):
            alpha_scale = float(_cfg_get(cfg, "SCREEN_ALPHA", 0.16))

            im = ax.imshow(
                screen_vis,
                extent=extent,
                origin="lower",
                cmap="Blues",
                interpolation="nearest",
                alpha=alpha_scale * screen_vis,
                zorder=3,
            )
            artists.append(im)

            try:
                cs = ax.contour(
                    screen_vis,
                    levels=[0.5],
                    extent=extent,
                    origin="lower",
                    colors=["cyan"],
                    linewidths=[1.0],
                    alpha=0.70,
                    zorder=6,
                )
                artists.extend(cs.collections)
            except Exception:
                pass

    if _cfg_get(cfg, "SHOW_CAP_OVERLAY", True):
        W_vis = _get_visible_field(grid, potential.W)
        wmax = float(np.max(W_vis))
        if wmax > 0.0:
            Wn = W_vis / (wmax + 1e-30)
            alpha_scale = float(_cfg_get(cfg, "CAP_ALPHA", 0.10))

            im = ax.imshow(
                Wn,
                extent=extent,
                origin="lower",
                cmap="Greens",
                interpolation="nearest",
                alpha=alpha_scale * Wn,
                zorder=2,
            )
            artists.append(im)

            try:
                cs = ax.contour(
                    Wn,
                    levels=[0.2, 0.5, 0.8],
                    extent=extent,
                    origin="lower",
                    colors=["lime", "lime", "lime"],
                    linewidths=[0.5, 0.7, 0.9],
                    alpha=0.35,
                    zorder=5,
                )
                artists.extend(cs.collections)
            except Exception:
                pass

    mode = str(_cfg_get(cfg, "BARRIER_PLOT_MODE", "potential")).lower().strip()

    if mode != "off":
        for comp in _iter_barrier_components(potential):
            if mode == "mask":
                artists.extend(_draw_component_mask_mode(ax, grid, comp, extent, cfg))
            elif mode == "contour_only":
                artists.extend(_draw_component_contour_only_mode(ax, grid, comp, extent, cfg))
            else:
                artists.extend(_draw_component_potential_mode(ax, grid, comp, extent, cfg))

    return artists


# ============================================================
# General helpers
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
            components=[],
        )

    return build_double_slit_and_caps(grid, cfg)


def density_from_state_vis(state_vis: np.ndarray) -> np.ndarray:
    if state_vis.ndim == 2:
        return np.abs(state_vis) ** 2
    if state_vis.ndim == 3:
        return np.sum(np.abs(state_vis) ** 2, axis=0)
    raise ValueError(f"Unsupported state_vis ndim={state_vis.ndim}")


def phase_from_state_vis(state_vis: np.ndarray) -> np.ndarray:
    if state_vis.ndim == 2:
        return np.angle(state_vis)
    if state_vis.ndim == 3:
        return np.angle(state_vis[0])
    raise ValueError(f"Unsupported state_vis ndim={state_vis.ndim}")


def forward_density_frames_from_state_vis(state_vis_frames: np.ndarray) -> np.ndarray:
    if state_vis_frames.ndim == 3:
        return (np.abs(state_vis_frames) ** 2).astype(float)
    if state_vis_frames.ndim == 4:
        return np.sum(np.abs(state_vis_frames) ** 2, axis=1).astype(float)
    raise ValueError(f"Unsupported state_vis_frames ndim={state_vis_frames.ndim}")


def backward_density_frames_from_emix(emix_frames: np.ndarray) -> np.ndarray:
    if emix_frames.ndim == 3:
        return (np.abs(emix_frames) ** 2).astype(float)
    if emix_frames.ndim == 4:
        return np.sum(np.abs(emix_frames) ** 2, axis=1).astype(float)
    raise ValueError(f"Unsupported emix_frames ndim={emix_frames.ndim}")


def overlap_density_frames(
    state_vis_frames: np.ndarray,
    emix_frames: np.ndarray,
) -> np.ndarray:
    z = make_overlap_complex_frames(state_vis_frames, emix_frames)
    return (np.abs(z) ** 2).astype(float)


def make_phase_density_rgb(
    state_vis: np.ndarray,
    density_gamma: float = 0.35,
    density_floor: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    rho_vis = density_from_state_vis(state_vis)
    phase_vis = phase_from_state_vis(state_vis)

    rho_norm = rho_vis / (np.max(rho_vis) + 1e-30)
    value = np.clip(rho_norm, 0.0, 1.0) ** density_gamma
    value[rho_norm < density_floor] = 0.0

    hue = (phase_vis + np.pi) / (2.0 * np.pi)
    sat = np.ones_like(hue)

    hsv = np.stack([hue, sat, value], axis=-1)
    rgb = hsv_to_rgb(hsv)
    return rgb, rho_norm


def make_overlap_complex_frames(
    state_vis_frames: np.ndarray,
    emix_frames: np.ndarray,
) -> np.ndarray:
    if state_vis_frames.ndim == 3 and emix_frames.ndim == 3:
        return state_vis_frames * np.conj(emix_frames)

    if state_vis_frames.ndim == 4 and emix_frames.ndim == 4:
        return np.sum(state_vis_frames * np.conj(emix_frames), axis=1)

    raise ValueError(
        f"State/Emix ndim mismatch: {state_vis_frames.ndim=} {emix_frames.ndim=}"
    )


def make_phase_density_rgb_from_complex_frames(
    complex_frames: np.ndarray,
    i: int,
    density_gamma: float = 0.35,
    density_floor: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    z = complex_frames[i]
    amp2 = np.abs(z) ** 2
    phase = np.angle(z)

    rho_norm = amp2 / (np.max(amp2) + 1e-30)
    value = np.clip(rho_norm, 0.0, 1.0) ** density_gamma
    value[rho_norm < density_floor] = 0.0

    hue = (phase + np.pi) / (2.0 * np.pi)
    sat = np.ones_like(hue)

    hsv = np.stack([hue, sat, value], axis=-1)
    rgb = hsv_to_rgb(hsv)
    return rgb, rho_norm


def _default_frame_vref(frames: np.ndarray) -> float:
    if frames is None:
        return 1.0
    m = float(np.max(frames[0])) if len(frames) > 0 else 0.0
    return max(m, 1e-30)


def build_render_image(
    *,
    mode: str,
    i: int,
    rho_current: np.ndarray,
    forward_density_current: np.ndarray | None,
    backward_density_current: np.ndarray | None,
    overlap_density_current: np.ndarray | None,
    posthoc_base_rho: np.ndarray | None,
    posthoc_selected_rho: np.ndarray | None,
    state_vis_frames: np.ndarray | None,
    ridge_complex_current: np.ndarray | None,
    vref: float,
    cfg,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    if mode == "density":
        img = gamma_display(
            rho_current[i],
            vref=vref,
            gamma=cfg.GAMMA,
            use_fixed_scale=cfg.USE_FIXED_DISPLAY_SCALE,
        )
        return img, None, "density"

    if mode == "forward_density":
        if forward_density_current is None:
            raise RuntimeError("forward_density render mode requires forward density frames")
        img = gamma_display(
            forward_density_current[i],
            vref=_default_frame_vref(forward_density_current),
            gamma=cfg.GAMMA,
            use_fixed_scale=False,
        )
        return img, None, "density"

    if mode == "backward_density":
        if backward_density_current is None:
            raise RuntimeError("backward_density render mode requires backward density frames")
        img = gamma_display(
            backward_density_current[i],
            vref=_default_frame_vref(backward_density_current),
            gamma=cfg.GAMMA,
            use_fixed_scale=False,
        )
        return img, None, "density"

    if mode == "overlap_density":
        if overlap_density_current is None:
            raise RuntimeError("overlap_density render mode requires overlap density frames")
        img = gamma_display(
            overlap_density_current[i],
            vref=_default_frame_vref(overlap_density_current),
            gamma=cfg.GAMMA,
            use_fixed_scale=False,
        )
        return img, None, "density"

    if mode == "posthoc_base_rho":
        if posthoc_base_rho is None:
            raise RuntimeError("posthoc_base_rho render mode requires saved posthoc_base_rho")
        img = gamma_display(
            posthoc_base_rho[i],
            vref=_default_frame_vref(posthoc_base_rho),
            gamma=cfg.GAMMA,
            use_fixed_scale=False,
        )
        return img, None, "density"

    if mode == "posthoc_selected_rho":
        if posthoc_selected_rho is None:
            raise RuntimeError("posthoc_selected_rho render mode requires saved posthoc_selected_rho")
        img = gamma_display(
            posthoc_selected_rho[i],
            vref=_default_frame_vref(posthoc_selected_rho),
            gamma=cfg.GAMMA,
            use_fixed_scale=False,
        )
        return img, None, "density"

    if mode == "phase":
        if state_vis_frames is None:
            raise RuntimeError("phase render mode requires state_vis_frames")
        rgb, _rho_norm = make_phase_density_rgb(
            state_vis_frames[i],
            density_gamma=0.35,
            density_floor=1e-12,
        )
        return rgb, None, "rgb"

    if mode == "phase_contours":
        if state_vis_frames is None:
            raise RuntimeError("phase_contours render mode requires state_vis_frames")
        rgb, rho_norm = make_phase_density_rgb(
            state_vis_frames[i],
            density_gamma=0.35,
            density_floor=1e-12,
        )
        return rgb, rho_norm, "rgb"

    if mode == "ridge_phase":
        if ridge_complex_current is None:
            raise RuntimeError("ridge_phase render mode requires overlap complex frames")
        rgb, _rho_norm = make_phase_density_rgb_from_complex_frames(
            ridge_complex_current,
            i=i,
            density_gamma=0.35,
            density_floor=1e-12,
        )
        return rgb, None, "rgb"

    if mode == "ridge_phase_contours":
        if ridge_complex_current is None:
            raise RuntimeError("ridge_phase_contours render mode requires overlap complex frames")
        rgb, rho_norm = make_phase_density_rgb_from_complex_frames(
            ridge_complex_current,
            i=i,
            density_gamma=0.35,
            density_floor=1e-12,
        )
        return rgb, rho_norm, "rgb"

    raise ValueError(f"Unsupported render mode: {mode}")


def mode_has_contours(mode: str) -> bool:
    return mode in ("phase_contours", "ridge_phase_contours")


def is_finite_scalar(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def compute_click_frame_idx(times: np.ndarray, t_det) -> int | None:
    if not is_finite_scalar(t_det):
        return None

    t_det = float(t_det)

    if len(times) == 0:
        return None

    if t_det <= float(times[0]):
        return 0
    if t_det > float(times[-1]):
        return None

    return int(np.searchsorted(times, t_det, side="left"))


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

    parser.add_argument(
        "--render-mode",
        choices=RENDER_MODES,
        default="density",
        help="Single-view render mode",
    )
    parser.add_argument(
        "--split-view",
        action="store_true",
        help="Show two render modes side by side",
    )
    parser.add_argument(
        "--left-mode",
        choices=RENDER_MODES,
        default="phase_contours",
        help="Left panel mode in split view",
    )
    parser.add_argument(
        "--right-mode",
        choices=RENDER_MODES,
        default="ridge_phase_contours",
        help="Right panel mode in split view",
    )
    parser.add_argument(
        "--contour-levels",
        type=float,
        nargs="*",
        default=[0.05, 0.10, 0.20, 0.40, 0.70],
        help="Contour levels for contour render modes",
    )

    parser.add_argument(
        "--ridge-source",
        choices=("overlap", "forward"),
        default="overlap",
        help="Use overlap-based ridge (current behavior) or forward-density-only ridge",
    )

    parser.add_argument(
        "--show-ridge-source-label",
        action="store_true",
        help="Append ridge source to plot titles",
    )

    parser.add_argument("--debug-metric", action="store_true")
    parser.add_argument("--debug-metric-alpha", action="store_true")
    parser.add_argument("--debug-metric-a", action="store_true")
    parser.add_argument("--debug-metric-v", action="store_true")

    args = parser.parse_args()

    bundle = load_run_bundle(args.npz_path, args.meta)
    meta = bundle["meta"]

    cfg = build_cfg_from_meta(meta)

    if not hasattr(cfg, "BARRIER_PLOT_MODE"):
        cfg.BARRIER_PLOT_MODE = "mask"
    if not hasattr(cfg, "SHOW_SCREEN_OVERLAY"):
        cfg.SHOW_SCREEN_OVERLAY = True
    if not hasattr(cfg, "SHOW_CAP_OVERLAY"):
        cfg.SHOW_CAP_OVERLAY = False
    if not hasattr(cfg, "SHOW_COMPONENT_LABELS"):
        cfg.SHOW_COMPONENT_LABELS = False
    if not hasattr(cfg, "GEOMETRY_CONTOUR_LEVELS"):
        cfg.GEOMETRY_CONTOUR_LEVELS = (0.15, 0.50, 0.85)
    if not hasattr(cfg, "BARRIER_MASK_ALPHA"):
        cfg.BARRIER_MASK_ALPHA = 0.26
    if not hasattr(cfg, "BARRIER_POTENTIAL_ALPHA_MIN"):
        cfg.BARRIER_POTENTIAL_ALPHA_MIN = 0.18
    if not hasattr(cfg, "BARRIER_POTENTIAL_ALPHA_MAX"):
        cfg.BARRIER_POTENTIAL_ALPHA_MAX = 0.55
    if not hasattr(cfg, "SCREEN_ALPHA"):
        cfg.SCREEN_ALPHA = 0.16
    if not hasattr(cfg, "CAP_ALPHA"):
        cfg.CAP_ALPHA = 0.10

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

    extent = (
        grid.x_vis_min,
        grid.x_vis_max,
        grid.y_vis_min,
        grid.y_vis_max,
    )

    if hasattr(theory, "alpha_metric") and hasattr(theory, "a_metric") and hasattr(theory, "V_metric"):
        alpha_vis = theory.alpha_metric[grid.ys, grid.xs]
        a_vis = theory.a_metric[grid.ys, grid.xs]
        V_metric_vis = theory.V_metric[grid.ys, grid.xs]

        if args.debug_metric:
            debug_plot_metric_fields_vis(
                alpha_vis=alpha_vis,
                a_vis=a_vis,
                V_metric_vis=V_metric_vis,
                extent=extent,
                title_prefix=f"{cfg.THEORY_NAME}",
            )

        if args.debug_metric_alpha:
            debug_plot_scalar_field_vis(
                field_vis=alpha_vis,
                extent=extent,
                title=f"{cfg.THEORY_NAME}: alpha_metric",
                cmap="viridis",
                colorbar_label="alpha",
            )

        if args.debug_metric_a:
            debug_plot_scalar_field_vis(
                field_vis=a_vis,
                extent=extent,
                title=f"{cfg.THEORY_NAME}: a_metric",
                cmap="plasma",
                colorbar_label="a(x,y)",
            )

        if args.debug_metric_v:
            debug_plot_scalar_field_vis(
                field_vis=V_metric_vis,
                extent=extent,
                title=f"{cfg.THEORY_NAME}: V_metric",
                cmap="magma",
                colorbar_label="V_metric",
            )

    times = bundle["times"]
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

    posthoc_base_rho_saved = bundle.get("posthoc_base_rho", None)
    posthoc_selected_rho_saved = bundle.get("posthoc_selected_rho", None)
    posthoc_corridor_upper_mask = bundle.get("posthoc_corridor_upper_mask", None)
    posthoc_corridor_lower_mask = bundle.get("posthoc_corridor_lower_mask", None)
    posthoc_trf_info = bundle.get("posthoc_trf_info", None)
    posthoc_worldline_info = bundle.get("posthoc_worldline_info", None)

    print(
        "[POSTHOC LOAD] "
        f"base_rho={'yes' if posthoc_base_rho_saved is not None else 'no'} "
        f"selected_rho={'yes' if posthoc_selected_rho_saved is not None else 'no'} "
        f"trf_info={'yes' if posthoc_trf_info is not None else 'no'} "
        f"worldline_info={'yes' if posthoc_worldline_info is not None else 'no'}"
    )

    Nt = len(times)
    tau_step = cfg.save_every * cfg.dt

    click_has_position = is_finite_scalar(x_click) and is_finite_scalar(y_click)
    click_frame_idx = compute_click_frame_idx(times, t_det)

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

        ridge_source = args.ridge_source

        if ridge_source == "overlap":
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
        elif ridge_source == "forward":
            rx, ry, rs = compute_ridge_xy(
                frames_psi=state_vis_frames,
                Emix=None,
                x_vis_1d=grid.x_vis_1d,
                y_vis_1d=grid.y_vis_1d,
                mode=cfg.RIDGE_MODE,
                top_q=cfg.CENTROID_TOP_Q,
                radius=cfg.LOCALMAX_RADIUS,
                alpha_smooth=cfg.LOCALMAX_SMOOTH_ALPHA,
            )
        else:
            raise ValueError(f"Unsupported ridge_source={ridge_source!r}")

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

    v_est = estimate_group_velocity(cfg, theory)
    L_gap = cfg.screen_center_x - cfg.barrier_center_x
    t_gap = L_gap / (abs(v_est) + 1e-12)

    sigma_min = 0.05 * t_gap
    sigma_max = 2.00 * t_gap

    (
        rho_init,
        emix_init,
        ridge_x_init,
        ridge_y_init,
        ridge_s_init,
        cos_th_init,
        speed_init,
        ux_init,
        uy_init,
        div_v_init,
    ) = build_all_for_sigma(sigma_init)

    ridge_complex_init = make_overlap_complex_frames(state_vis_frames, emix_init)

    forward_density_init = forward_density_frames_from_state_vis(state_vis_frames)
    backward_density_init = backward_density_frames_from_emix(emix_init)
    overlap_density_init = overlap_density_frames(state_vis_frames, emix_init)

    split_view = bool(args.split_view)
    left_mode = args.left_mode if split_view else args.render_mode
    right_mode = args.right_mode if split_view else None

    if split_view:
        fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.2))
        plt.subplots_adjust(left=0.06, right=0.98, bottom=0.18, top=0.92, wspace=0.10)
        ax_left, ax_right = axes
        axes_list = [ax_left, ax_right]
        panel_modes = [left_mode, right_mode]
    else:
        fig = plt.figure(figsize=(10.8, 7.2))
        ax_single = fig.add_axes([0.07, 0.18, 0.86, 0.74])
        axes_list = [ax_single]
        panel_modes = [left_mode]

    panel_states = []

    for ax, mode in zip(axes_list, panel_modes):
        img0, rho_norm0, img_kind = build_render_image(
            mode=mode,
            i=0,
            rho_current=rho_init,
            forward_density_current=forward_density_init,
            backward_density_current=backward_density_init,
            overlap_density_current=overlap_density_init,
            posthoc_base_rho=posthoc_base_rho_saved,
            posthoc_selected_rho=posthoc_selected_rho_saved,
            state_vis_frames=state_vis_frames,
            ridge_complex_current=ridge_complex_init,
            vref=vref,
            cfg=cfg,
        )

        if img_kind == "density":
            im = ax.imshow(
                img0,
                extent=extent,
                origin="lower",
                cmap="magma",
                interpolation=cfg.IM_INTERPOLATION,
                norm=PowerNorm(gamma=0.35),
                zorder=1,
            )
        else:
            im = ax.imshow(
                img0,
                extent=extent,
                origin="lower",
                aspect="equal",
                interpolation=cfg.IM_INTERPOLATION,
                zorder=1,
            )

        static_geometry_artists = draw_static_geometry(
            ax=ax,
            grid=grid,
            potential=potential,
            cfg=cfg,
            extent=extent,
        )

        contour_artists = []
        if mode_has_contours(mode) and rho_norm0 is not None:
            cs = ax.contour(
                grid.X_vis,
                grid.Y_vis,
                rho_norm0,
                levels=args.contour_levels,
                colors="white",
                linewidths=0.7,
                alpha=0.7,
                zorder=7,
            )
            contour_artists = list(cs.collections)

        # Optional posthoc corridor overlay for posthoc render modes
        posthoc_corridor_artists = []
        if mode in {"posthoc_base_rho", "posthoc_selected_rho"}:
            if posthoc_corridor_upper_mask is not None:
                im_u = ax.imshow(
                    posthoc_corridor_upper_mask,
                    extent=extent,
                    origin="lower",
                    cmap="Blues",
                    interpolation="nearest",
                    alpha=0.10 * np.clip(posthoc_corridor_upper_mask, 0.0, 1.0),
                    zorder=3,
                )
                posthoc_corridor_artists.append(im_u)

            if posthoc_corridor_lower_mask is not None:
                im_l = ax.imshow(
                    posthoc_corridor_lower_mask,
                    extent=extent,
                    origin="lower",
                    cmap="Greens",
                    interpolation="nearest",
                    alpha=0.10 * np.clip(posthoc_corridor_lower_mask, 0.0, 1.0),
                    zorder=3,
                )
                posthoc_corridor_artists.append(im_l)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        if split_view:
            title = ax.set_title(f"{mode} | t={times[0]:.3f} | σT={sigma_init:.3f}")
        else:
            title = ax.set_title(
                rf"ρ(t): σT={sigma_init:.3f}, t={times[0]:.3f}, "
                rf"ridge={cfg.RIDGE_MODE}, theory={cfg.THEORY_NAME}, "
                rf"detector={getattr(cfg, 'DETECTOR_NAME', 'unknown')}, "
                rf"render={mode}"
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
            zorder=10,
        )

        click_marker, = ax.plot(
            [],
            [],
            marker="x",
            markersize=9,
            linestyle="None",
            color="yellow",
            alpha=0.9,
            label="click",
            visible=False,
            zorder=10,
        )

        ridge_trail, = ax.plot(
            [],
            [],
            linestyle="-",
            linewidth=1.5,
            color="lime",
            alpha=0.5,
            zorder=9,
        )

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
                zorder=11,
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
                    zorder=8,
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
                        zorder=9,
                    )
                else:
                    head_k = None

                bohm_heads.append(head_k)

        ax.legend(loc="upper right", framealpha=0.35)

        panel_states.append(
            {
                "ax": ax,
                "mode": mode,
                "im": im,
                "title": title,
                "static_geometry_artists": static_geometry_artists,
                "posthoc_corridor_artists": posthoc_corridor_artists,
                "contour_artists": contour_artists,
                "ridge_marker": ridge_marker,
                "click_marker": click_marker,
                "ridge_trail": ridge_trail,
                "flow_quiver": flow_quiver,
                "bohm_lines": bohm_lines,
                "bohm_heads": bohm_heads,
                "arrow_state": {"ux": np.nan, "uy": np.nan, "spd": np.nan},
            }
        )

    ax_sigma = fig.add_axes([0.10, 0.08, 0.80, 0.04])
    sigma_slider = Slider(
        ax=ax_sigma,
        label="sigmaT (time thickness)",
        valmin=sigma_min,
        valmax=sigma_max,
        valinit=sigma_init,
    )

    rho_current = [rho_init]
    emix_current = [emix_init]
    ridge_complex_current = [ridge_complex_init]

    forward_density_current = [forward_density_init]
    backward_density_current = [backward_density_init]
    overlap_density_current = [overlap_density_init]

    sigma_current = [sigma_init]
    ridge_x = [ridge_x_init]
    ridge_y = [ridge_y_init]
    ridge_s = [ridge_s_init]
    cos_th = [cos_th_init]
    speed = [speed_init]
    ux = [ux_init]
    uy = [uy_init]
    div_v_ridge = [div_v_init]

    def clear_contours(panel: dict):
        for artist in panel["contour_artists"]:
            try:
                artist.remove()
            except Exception:
                pass
        panel["contour_artists"] = []

    def update_panel_image(panel: dict, i: int):
        mode = panel["mode"]
        img, rho_norm, _img_kind = build_render_image(
            mode=mode,
            i=i,
            rho_current=rho_current[0],
            forward_density_current=forward_density_current[0],
            backward_density_current=backward_density_current[0],
            overlap_density_current=overlap_density_current[0],
            posthoc_base_rho=posthoc_base_rho_saved,
            posthoc_selected_rho=posthoc_selected_rho_saved,
            state_vis_frames=state_vis_frames,
            ridge_complex_current=ridge_complex_current[0],
            vref=vref,
            cfg=cfg,
        )
        panel["im"].set_data(img)

        clear_contours(panel)
        if mode_has_contours(mode) and rho_norm is not None:
            cs = panel["ax"].contour(
                grid.X_vis,
                grid.Y_vis,
                rho_norm,
                levels=args.contour_levels,
                colors="white",
                linewidths=0.7,
                alpha=0.7,
                zorder=7,
            )
            panel["contour_artists"] = list(cs.collections)

    def update_flow_arrow(panel: dict, i: int):
        flow_quiver = panel["flow_quiver"]
        arrow_state = panel["arrow_state"]

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

        L = cfg.ARROW_SCALE * float(np.clip(spd / (speed_ref + 1e-30), 0.0, 2.5))
        flow_quiver.set_UVC([L * uxi], [L * uyi])

    def update_bohmian_overlay(panel: dict, i: int):
        if not cfg.ENABLE_BOHMIAN_OVERLAY or bohm_traj_x is None:
            return

        bohm_lines = panel["bohm_lines"]
        bohm_heads = panel["bohm_heads"]

        for k in range(bohm_traj_x.shape[0]):
            alive = bohm_traj_alive[k]

            if not np.any(alive[: i + 1]):
                bohm_lines[k].set_data([], [])
                if bohm_heads[k] is not None:
                    bohm_heads[k].set_data([], [])
                continue

            j0 = 0 if cfg.BOHMIAN_SHOW_FULL_PATH_EACH_FRAME else max(0, i - cfg.BOHMIAN_TRAIL_LEN + 1)

            mask = alive[j0: i + 1]
            xs_seg = bohm_traj_x[k, j0: i + 1][mask]
            ys_seg = bohm_traj_y[k, j0: i + 1][mask]
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

    def update_click_marker(panel: dict, i: int):
        marker = panel["click_marker"]

        if (not click_has_position) or (click_frame_idx is None) or (i < click_frame_idx):
            marker.set_data([], [])
            marker.set_visible(False)
            return

        marker.set_data([float(x_click)], [float(y_click)])
        marker.set_visible(True)

    def make_main_title(i: int, mode: str):
        parts = [
            rf"ρ(t): σT={sigma_current[0]:.3f}",
            rf"t={times[i]:.3f}",
            rf"ridge={cfg.RIDGE_MODE}",
            rf"theory={cfg.THEORY_NAME}",
            rf"detector={getattr(cfg, 'DETECTOR_NAME', 'unknown')}",
            rf"render={mode}",
            rf"norm≈{norms[i]:.4f}",
            rf"Γ≈{ridge_s[0][i]:.3e}",
        ]

        if mode == "posthoc_base_rho" and posthoc_trf_info is not None:
            parts.append(f"trf={posthoc_trf_info.get('chosen_side', None)}")
            if "ref_time" in posthoc_trf_info:
                parts.append(f"t_ref≈{float(posthoc_trf_info['ref_time']):.3f}")

        if mode == "posthoc_selected_rho" and posthoc_worldline_info is not None:
            parts.append(f"wl_used={posthoc_worldline_info.get('used', False)}")
            if posthoc_worldline_info.get("seed_side", None) is not None:
                parts.append(f"wl_seed={posthoc_worldline_info.get('seed_side')}")

        if args.show_ridge_source_label:
            parts.append(rf"ridge_src={args.ridge_source}")

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

    def make_split_title(i: int, mode: str):
        parts = [
            f"{mode}",
            f"t={times[i]:.3f}",
            f"σT={sigma_current[0]:.3f}",
            f"Γ≈{ridge_s[0][i]:.3e}",
        ]

        if mode == "posthoc_base_rho" and posthoc_trf_info is not None:
            parts.append(f"trf={posthoc_trf_info.get('chosen_side', None)}")

        if mode == "posthoc_selected_rho" and posthoc_worldline_info is not None:
            parts.append(f"wl={posthoc_worldline_info.get('used', False)}")

        return " | ".join(parts)

    def refresh_titles(i: int):
        for panel in panel_states:
            if split_view:
                panel["title"].set_text(make_split_title(i, panel["mode"]))
            else:
                panel["title"].set_text(make_main_title(i, panel["mode"]))

    def refresh_overlays(i: int):
        for panel in panel_states:
            panel["ridge_marker"].set_data([ridge_x[0][i]], [ridge_y[0][i]])

            if cfg.SHOW_TRAIL:
                j0 = max(0, i - cfg.TRAIL_LEN + 1)
                panel["ridge_trail"].set_data(
                    ridge_x[0][j0: i + 1],
                    ridge_y[0][j0: i + 1],
                )
            else:
                panel["ridge_trail"].set_data([], [])

            update_click_marker(panel, i)
            update_flow_arrow(panel, i)
            update_bohmian_overlay(panel, i)

    def on_sigma_change(_val):
        new_sigma = float(sigma_slider.val)
        sigma_current[0] = new_sigma

        (
            rho_new,
            emix_new,
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
        emix_current[0] = emix_new
        ridge_complex_current[0] = make_overlap_complex_frames(state_vis_frames, emix_new)

        forward_density_current[0] = forward_density_frames_from_state_vis(state_vis_frames)
        backward_density_current[0] = backward_density_frames_from_emix(emix_new)
        overlap_density_current[0] = overlap_density_frames(state_vis_frames, emix_new)

        ridge_x[0] = rx
        ridge_y[0] = ry
        ridge_s[0] = rs
        cos_th[0] = cth
        speed[0] = spd
        ux[0] = uxx
        uy[0] = uyy
        div_v_ridge[0] = divv

        i = getattr(on_sigma_change, "last_i", 0)

        for panel in panel_states:
            update_panel_image(panel, i)

        refresh_overlays(i)
        refresh_titles(i)
        fig.canvas.draw_idle()

    sigma_slider.on_changed(on_sigma_change)

    def update(i: int):
        on_sigma_change.last_i = i

        artists = []

        for panel in panel_states:
            update_panel_image(panel, i)
            artists.append(panel["im"])

        refresh_overlays(i)
        refresh_titles(i)

        for panel in panel_states:
            artists.extend(panel["static_geometry_artists"])
            artists.extend(panel["posthoc_corridor_artists"])
            artists.append(panel["ridge_marker"])
            artists.append(panel["click_marker"])
            artists.append(panel["ridge_trail"])

            if panel["flow_quiver"] is not None:
                artists.append(panel["flow_quiver"])

            artists.extend([obj for obj in panel["bohm_lines"] if obj is not None])
            artists.extend([obj for obj in panel["bohm_heads"] if obj is not None])
            artists.extend(panel["contour_artists"])

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