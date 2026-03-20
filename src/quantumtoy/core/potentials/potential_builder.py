from __future__ import annotations

import numpy as np

from core.simulation_types import PotentialSpec
from .validation import (
    _assert,
    _require_attr,
    _assert_array_shape,
    _assert_finite_array,
    _assert_finite_scalar,
    _assert_positive_scalar,
    _assert_nonnegative_scalar,
)
from .cap import smooth_cap_edge
from .barriers.single_slit import build_single_slit_barrier
from .barriers.double_slit import build_double_slit_barrier
from .barriers.composite import combine_barriers


def build_barrier_system_and_caps(grid, cfg) -> PotentialSpec:
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

    barrier_thickness = _assert_positive_scalar(_require_attr(cfg, "barrier_thickness"), "cfg.barrier_thickness")
    V_barrier = _assert_nonnegative_scalar(_require_attr(cfg, "V_barrier"), "cfg.V_barrier")
    BARRIER_SMOOTH = _assert_finite_scalar(_require_attr(cfg, "BARRIER_SMOOTH"), "cfg.BARRIER_SMOOTH")

    components = []

    use_upstream_single_slit = getattr(cfg, "USE_UPSTREAM_SINGLE_SLIT", False)
    if use_upstream_single_slit:
        single = build_single_slit_barrier(
            X=X,
            Y=Y,
            center_x=_assert_finite_scalar(_require_attr(cfg, "single_slit_center_x"), "cfg.single_slit_center_x"),
            thickness=barrier_thickness,
            slit_half_height=_assert_positive_scalar(_require_attr(cfg, "single_slit_half_height"), "cfg.single_slit_half_height"),
            V_barrier=V_barrier,
            barrier_smooth=BARRIER_SMOOTH,
        )
        components.append(single)

    double = build_double_slit_barrier(
        X=X,
        Y=Y,
        center_x=_assert_finite_scalar(_require_attr(cfg, "barrier_center_x"), "cfg.barrier_center_x"),
        thickness=barrier_thickness,
        slit_center_offset=_assert_finite_scalar(_require_attr(cfg, "slit_center_offset"), "cfg.slit_center_offset"),
        slit_half_height=_assert_positive_scalar(_require_attr(cfg, "slit_half_height"), "cfg.slit_half_height"),
        V_barrier=V_barrier,
        barrier_smooth=BARRIER_SMOOTH,
    )
    components.append(double)

    V_real = combine_barriers((grid.Ny, grid.Nx), components, V_barrier=V_barrier)

    CAP_WIDTH = _assert_nonnegative_scalar(_require_attr(cfg, "CAP_WIDTH"), "cfg.CAP_WIDTH")
    CAP_STRENGTH = _assert_nonnegative_scalar(_require_attr(cfg, "CAP_STRENGTH"), "cfg.CAP_STRENGTH")
    CAP_POWER = _assert_positive_scalar(_require_attr(cfg, "CAP_POWER"), "cfg.CAP_POWER")

    W_edge = smooth_cap_edge(
        X, Y, grid.Lx, grid.Ly,
        cap_width=CAP_WIDTH,
        strength=CAP_STRENGTH,
        power=CAP_POWER,
    )

    screen_center_x = _assert_finite_scalar(_require_attr(cfg, "screen_center_x"), "cfg.screen_center_x")
    screen_eval_width = _assert_positive_scalar(_require_attr(cfg, "screen_eval_width"), "cfg.screen_eval_width")
    USE_SCREEN_CAP = _require_attr(cfg, "USE_SCREEN_CAP")
    SCREEN_CAP_STRENGTH = _assert_nonnegative_scalar(
        _require_attr(cfg, "SCREEN_CAP_STRENGTH"),
        "cfg.SCREEN_CAP_STRENGTH",
    )

    screen_mask_full = np.abs(X - screen_center_x) < screen_eval_width
    screen_mask_vis = screen_mask_full[grid.ys, grid.xs]

    W_screen = np.zeros_like(X, dtype=float)
    if USE_SCREEN_CAP:
        W_screen[screen_mask_full] = SCREEN_CAP_STRENGTH

    W = W_edge + W_screen

    return PotentialSpec(
        V_real=V_real,
        W=W,
        screen_mask_full=screen_mask_full,
        screen_mask_vis=screen_mask_vis,
        barrier_core=double.barrier_core,
        slit1_mask=double.slit_masks["slit1"],
        slit2_mask=double.slit_masks["slit2"],
    )