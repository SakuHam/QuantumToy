from __future__ import annotations

import numpy as np
from core.simulation_types import PotentialSpec


# ============================================================
# Validation helpers
# ============================================================

def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def _require_attr(obj, name: str):
    _assert(hasattr(obj, name), f"Missing required attribute: {name}")
    return getattr(obj, name)


def _assert_finite_scalar(x, name: str):
    _assert(np.isscalar(x), f"{name} must be a scalar, got type={type(x)}")
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


def _assert_array_shape(arr: np.ndarray, shape: tuple[int, ...], name: str):
    _assert(isinstance(arr, np.ndarray), f"{name} must be np.ndarray")
    _assert(arr.shape == shape, f"{name}.shape {arr.shape} != expected {shape}")


def _assert_finite_array(arr: np.ndarray, name: str):
    _assert(np.all(np.isfinite(arr)), f"{name} contains non-finite values")


# ============================================================
# CAP helper
# ============================================================

def smooth_cap_edge(X, Y, Lx, Ly, cap_width=8.0, strength=2.0, power=4):
    _assert(isinstance(X, np.ndarray), "X must be np.ndarray")
    _assert(isinstance(Y, np.ndarray), "Y must be np.ndarray")
    _assert(X.ndim == 2, f"X must be 2D, got ndim={X.ndim}")
    _assert(Y.ndim == 2, f"Y must be 2D, got ndim={Y.ndim}")
    _assert(X.shape == Y.shape, f"X.shape {X.shape} != Y.shape {Y.shape}")
    _assert_finite_array(X, "X")
    _assert_finite_array(Y, "Y")

    Lx = _assert_positive_scalar(Lx, "Lx")
    Ly = _assert_positive_scalar(Ly, "Ly")
    cap_width = _assert_nonnegative_scalar(cap_width, "cap_width")
    strength = _assert_nonnegative_scalar(strength, "strength")
    power = _assert_positive_scalar(power, "power")

    dist_to_x = (Lx / 2.0) - np.abs(X)
    dist_to_y = (Ly / 2.0) - np.abs(Y)
    dist_to_edge = np.minimum(dist_to_x, dist_to_y)

    _assert_array_shape(dist_to_edge, X.shape, "dist_to_edge")
    _assert_finite_array(dist_to_edge, "dist_to_edge")

    W = np.zeros_like(X, dtype=float)
    _assert_array_shape(W, X.shape, "W(init)")

    if cap_width == 0.0 or strength == 0.0:
        _assert(np.all(W == 0.0), "W should remain zero when cap_width==0 or strength==0")
        return W

    mask = dist_to_edge < cap_width
    _assert_array_shape(mask, X.shape, "mask")
    _assert(np.issubdtype(mask.dtype, np.bool_), "mask must be boolean")

    s = (cap_width - dist_to_edge[mask]) / cap_width
    _assert(np.all(np.isfinite(s)), "s contains non-finite values")
    _assert(np.all(s >= 0.0), "s contains negative values")
    _assert(np.all(s <= 1.0 + 1e-12), "s exceeds 1 unexpectedly")

    W[mask] = strength * (s ** power)

    _assert_array_shape(W, X.shape, "W")
    _assert_finite_array(W, "W")
    _assert(np.all(W >= -1e-14), "W contains significantly negative values")

    # Outside mask, W should be zero
    if np.any(~mask):
        _assert(np.allclose(W[~mask], 0.0), "W must be zero outside CAP mask")

    return W


# ============================================================
# Potential builder
# ============================================================

def build_double_slit_and_caps(grid, cfg) -> PotentialSpec:
    # --------------------------------------------------------
    # Validate grid
    # --------------------------------------------------------
    for attr in (
        "X", "Y", "Nx", "Ny", "Lx", "Ly", "xs", "ys",
        "n_visible_x", "n_visible_y"
    ):
        _require_attr(grid, attr)

    X = grid.X
    Y = grid.Y

    _assert_array_shape(X, (grid.Ny, grid.Nx), "grid.X")
    _assert_array_shape(Y, (grid.Ny, grid.Nx), "grid.Y")
    _assert_finite_array(X, "grid.X")
    _assert_finite_array(Y, "grid.Y")

    # --------------------------------------------------------
    # Validate cfg params
    # --------------------------------------------------------
    barrier_center_x = _assert_finite_scalar(_require_attr(cfg, "barrier_center_x"), "cfg.barrier_center_x")
    barrier_thickness = _assert_positive_scalar(_require_attr(cfg, "barrier_thickness"), "cfg.barrier_thickness")
    V_barrier = _assert_nonnegative_scalar(_require_attr(cfg, "V_barrier"), "cfg.V_barrier")

    slit_center_offset = _assert_finite_scalar(_require_attr(cfg, "slit_center_offset"), "cfg.slit_center_offset")
    slit_half_height = _assert_positive_scalar(_require_attr(cfg, "slit_half_height"), "cfg.slit_half_height")

    BARRIER_SMOOTH = _assert_finite_scalar(_require_attr(cfg, "BARRIER_SMOOTH"), "cfg.BARRIER_SMOOTH")
    CAP_WIDTH = _assert_nonnegative_scalar(_require_attr(cfg, "CAP_WIDTH"), "cfg.CAP_WIDTH")
    CAP_STRENGTH = _assert_nonnegative_scalar(_require_attr(cfg, "CAP_STRENGTH"), "cfg.CAP_STRENGTH")
    CAP_POWER = _assert_positive_scalar(_require_attr(cfg, "CAP_POWER"), "cfg.CAP_POWER")

    screen_center_x = _assert_finite_scalar(_require_attr(cfg, "screen_center_x"), "cfg.screen_center_x")
    screen_eval_width = _assert_positive_scalar(_require_attr(cfg, "screen_eval_width"), "cfg.screen_eval_width")

    USE_SCREEN_CAP = _require_attr(cfg, "USE_SCREEN_CAP")
    _assert(isinstance(USE_SCREEN_CAP, bool), f"cfg.USE_SCREEN_CAP must be bool, got {type(USE_SCREEN_CAP)}")

    SCREEN_CAP_STRENGTH = _assert_nonnegative_scalar(
        _require_attr(cfg, "SCREEN_CAP_STRENGTH"),
        "cfg.SCREEN_CAP_STRENGTH",
    )

    # --------------------------------------------------------
    # Barrier + slits
    # --------------------------------------------------------
    # Existing downstream double slit
    barrier_core = np.abs(X - barrier_center_x) < (barrier_thickness / 2.0)
    slit1_mask = np.abs(Y - slit_center_offset) < slit_half_height
    slit2_mask = np.abs(Y + slit_center_offset) < slit_half_height

    _assert_array_shape(barrier_core, (grid.Ny, grid.Nx), "barrier_core")
    _assert_array_shape(slit1_mask, (grid.Ny, grid.Nx), "slit1_mask")
    _assert_array_shape(slit2_mask, (grid.Ny, grid.Nx), "slit2_mask")
    _assert(np.issubdtype(barrier_core.dtype, np.bool_), "barrier_core must be boolean")
    _assert(np.issubdtype(slit1_mask.dtype, np.bool_), "slit1_mask must be boolean")
    _assert(np.issubdtype(slit2_mask.dtype, np.bool_), "slit2_mask must be boolean")

    # --------------------------------------------------------
    # NEW: upstream single slit (hardcoded quick test)
    # --------------------------------------------------------
    USE_UPSTREAM_SINGLE_SLIT = True

    # Put the single slit to the left of the current double slit.
    # Tune these three numbers for quick experiments.
    single_slit_center_x = barrier_center_x - 1.325
    single_slit_thickness = barrier_thickness
    single_slit_half_height = 1.4

    single_barrier_core = np.abs(X - single_slit_center_x) < (single_slit_thickness / 2.0)
    single_slit_mask = np.abs(Y) < single_slit_half_height

    _assert_array_shape(single_barrier_core, (grid.Ny, grid.Nx), "single_barrier_core")
    _assert_array_shape(single_slit_mask, (grid.Ny, grid.Nx), "single_slit_mask")
    _assert(np.issubdtype(single_barrier_core.dtype, np.bool_), "single_barrier_core must be boolean")
    _assert(np.issubdtype(single_slit_mask.dtype, np.bool_), "single_slit_mask must be boolean")

    if BARRIER_SMOOTH <= 0.0:
        barrier_mask = barrier_core.copy()
        barrier_mask[slit1_mask] = False
        barrier_mask[slit2_mask] = False

        if USE_UPSTREAM_SINGLE_SLIT:
            single_barrier_mask = single_barrier_core.copy()
            single_barrier_mask[single_slit_mask] = False
            barrier_mask |= single_barrier_mask

        _assert_array_shape(barrier_mask, (grid.Ny, grid.Nx), "barrier_mask")
        _assert(np.issubdtype(barrier_mask.dtype, np.bool_), "barrier_mask must be boolean")

        V_real = np.zeros_like(X, dtype=float)
        V_real[barrier_mask] = V_barrier
    else:
        # Smooth downstream double slit wall
        dist = np.abs(X - barrier_center_x) - (barrier_thickness / 2.0)
        _assert_array_shape(dist, (grid.Ny, grid.Nx), "dist")
        _assert_finite_array(dist, "dist")

        wall = 1.0 / (1.0 + np.exp(dist / BARRIER_SMOOTH))
        _assert_array_shape(wall, (grid.Ny, grid.Nx), "wall")
        _assert_finite_array(wall, "wall")
        _assert(np.all(wall >= 0.0), "wall contains negative values")
        _assert(np.all(wall <= 1.0 + 1e-12), "wall exceeds 1 unexpectedly")

        wall[slit1_mask] = 0.0
        wall[slit2_mask] = 0.0

        V_real = V_barrier * wall

        # NEW: add smooth upstream single slit wall
        if USE_UPSTREAM_SINGLE_SLIT:
            dist_single = np.abs(X - single_slit_center_x) - (single_slit_thickness / 2.0)
            _assert_array_shape(dist_single, (grid.Ny, grid.Nx), "dist_single")
            _assert_finite_array(dist_single, "dist_single")

            wall_single = 1.0 / (1.0 + np.exp(dist_single / BARRIER_SMOOTH))
            _assert_array_shape(wall_single, (grid.Ny, grid.Nx), "wall_single")
            _assert_finite_array(wall_single, "wall_single")
            _assert(np.all(wall_single >= 0.0), "wall_single contains negative values")
            _assert(np.all(wall_single <= 1.0 + 1e-12), "wall_single exceeds 1 unexpectedly")

            wall_single[single_slit_mask] = 0.0

            V_real += V_barrier * wall_single
            V_real = np.minimum(V_real, V_barrier)

    _assert_array_shape(V_real, (grid.Ny, grid.Nx), "V_real")
    _assert_finite_array(V_real, "V_real")
    _assert(np.all(V_real >= -1e-14), "V_real contains significantly negative values")

    # --------------------------------------------------------
    # Edge CAP
    # --------------------------------------------------------
    W_edge = smooth_cap_edge(
        X, Y, grid.Lx, grid.Ly,
        cap_width=CAP_WIDTH,
        strength=CAP_STRENGTH,
        power=CAP_POWER,
    )

    _assert_array_shape(W_edge, (grid.Ny, grid.Nx), "W_edge")
    _assert_finite_array(W_edge, "W_edge")
    _assert(np.all(W_edge >= -1e-14), "W_edge contains significantly negative values")

    # --------------------------------------------------------
    # Screen mask
    # --------------------------------------------------------
    screen_mask_full = np.abs(X - screen_center_x) < screen_eval_width
    screen_mask_vis = screen_mask_full[grid.ys, grid.xs]

    _assert_array_shape(screen_mask_full, (grid.Ny, grid.Nx), "screen_mask_full")
    _assert_array_shape(screen_mask_vis, (grid.n_visible_y, grid.n_visible_x), "screen_mask_vis")
    _assert(np.issubdtype(screen_mask_full.dtype, np.bool_), "screen_mask_full must be boolean")
    _assert(np.issubdtype(screen_mask_vis.dtype, np.bool_), "screen_mask_vis must be boolean")

    _assert(
        np.array_equal(screen_mask_vis, screen_mask_full[grid.ys, grid.xs]),
        "screen_mask_vis must equal screen_mask_full cropped to visible window",
    )

    # --------------------------------------------------------
    # Optional screen CAP
    # --------------------------------------------------------
    W_screen = np.zeros_like(X, dtype=float)
    _assert_array_shape(W_screen, (grid.Ny, grid.Nx), "W_screen(init)")

    if USE_SCREEN_CAP:
        W_screen[screen_mask_full] = SCREEN_CAP_STRENGTH

    _assert_array_shape(W_screen, (grid.Ny, grid.Nx), "W_screen")
    _assert_finite_array(W_screen, "W_screen")
    _assert(np.all(W_screen >= -1e-14), "W_screen contains significantly negative values")

    # --------------------------------------------------------
    # Total imaginary absorber
    # --------------------------------------------------------
    W = W_edge + W_screen
    _assert_array_shape(W, (grid.Ny, grid.Nx), "W")
    _assert_finite_array(W, "W")
    _assert(np.all(W >= -1e-14), "W contains significantly negative values")

    # --------------------------------------------------------
    # Extra structural sanity checks
    # --------------------------------------------------------
    _assert(np.any(barrier_core), "barrier_core is empty; barrier may be outside the grid")
    _assert(np.any(screen_mask_full), "screen_mask_full is empty; detector screen may be outside the grid")

    if USE_UPSTREAM_SINGLE_SLIT:
        _assert(np.any(single_barrier_core), "single_barrier_core is empty; upstream single slit barrier may be outside the grid")

    if BARRIER_SMOOTH <= 0.0:
        barrier_region_slits = barrier_core & (slit1_mask | slit2_mask)
        if np.any(barrier_region_slits):
            _assert(
                np.allclose(V_real[barrier_region_slits], 0.0),
                "Hard-wall slit region inside barrier must have V_real=0",
            )

        if USE_UPSTREAM_SINGLE_SLIT:
            single_barrier_region_slit = single_barrier_core & single_slit_mask
            if np.any(single_barrier_region_slit):
                _assert(
                    np.allclose(V_real[single_barrier_region_slit], 0.0),
                    "Hard-wall upstream single slit region inside barrier must have V_real=0",
                )

    _assert(
        float(np.max(V_real)) <= V_barrier + 1e-10,
        f"V_real max {np.max(V_real)} exceeds V_barrier={V_barrier}",
    )

    return PotentialSpec(
        V_real=V_real,
        W=W,
        screen_mask_full=screen_mask_full,
        screen_mask_vis=screen_mask_vis,
        barrier_core=barrier_core,
        slit1_mask=slit1_mask,
        slit2_mask=slit2_mask,
    )