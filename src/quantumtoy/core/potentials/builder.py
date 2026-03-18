from __future__ import annotations

import numpy as np

from core.simulation_types import PotentialSpec, BarrierComponentSpec

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
from .barriers import (
    SingleSlitBarrier,
    DoubleSlitBarrier,
    CompositeBarrierSystem,
)


class PotentialBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, grid) -> PotentialSpec:
        cfg = self.cfg

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

        barrier_system = CompositeBarrierSystem(V_clip_max=V_barrier)

        use_upstream_single_slit = bool(getattr(cfg, "USE_UPSTREAM_SINGLE_SLIT", False))
        if use_upstream_single_slit:
            single_slit_center_x = _assert_finite_scalar(
                _require_attr(cfg, "single_slit_center_x"),
                "cfg.single_slit_center_x",
            )
            single_slit_half_height = _assert_positive_scalar(
                _require_attr(cfg, "single_slit_half_height"),
                "cfg.single_slit_half_height",
            )

            barrier_system.add(
                SingleSlitBarrier(
                    center_x=single_slit_center_x,
                    thickness=barrier_thickness,
                    slit_half_height=single_slit_half_height,
                    V_barrier=V_barrier,
                    barrier_smooth=BARRIER_SMOOTH,
                    name="upstream_single_slit",
                )
            )

        barrier_system.add(
            DoubleSlitBarrier(
                center_x=barrier_center_x,
                thickness=barrier_thickness,
                slit_center_offset=slit_center_offset,
                slit_half_height=slit_half_height,
                V_barrier=V_barrier,
                barrier_smooth=BARRIER_SMOOTH,
                name="downstream_double_slit",
            )
        )

        barrier_result = barrier_system.build(X, Y)
        V_real = barrier_result.V_real

        _assert_array_shape(V_real, (grid.Ny, grid.Nx), "V_real")
        _assert_finite_array(V_real, "V_real")
        _assert(np.all(V_real >= -1e-14), "V_real contains significantly negative values")
        _assert(
            float(np.max(V_real)) <= V_barrier + 1e-10,
            f"V_real max {np.max(V_real)} exceeds V_barrier={V_barrier}",
        )

        W_edge = smooth_cap_edge(
            X, Y, grid.Lx, grid.Ly,
            cap_width=CAP_WIDTH,
            strength=CAP_STRENGTH,
            power=CAP_POWER,
        )

        screen_mask_full = np.abs(X - screen_center_x) < screen_eval_width
        screen_mask_vis = screen_mask_full[grid.ys, grid.xs]

        W_screen = np.zeros_like(X, dtype=float)
        if USE_SCREEN_CAP:
            W_screen[screen_mask_full] = SCREEN_CAP_STRENGTH

        W = W_edge + W_screen

        _assert_array_shape(W, (grid.Ny, grid.Nx), "W")
        _assert_finite_array(W, "W")
        _assert(np.all(W >= -1e-14), "W contains significantly negative values")

        double_comp = next(
            comp for comp in barrier_result.components
            if comp.name == "downstream_double_slit"
        )

        component_specs = [
            BarrierComponentSpec(
                name=comp.name,
                kind=comp.kind,
                V_real=comp.V_real,
                barrier_core=comp.barrier_core,
                slit_masks=comp.slit_masks,
            )
            for comp in barrier_result.components
        ]

        return PotentialSpec(
            V_real=V_real,
            W=W,
            screen_mask_full=screen_mask_full,
            screen_mask_vis=screen_mask_vis,

            # legacy compatibility
            barrier_core=double_comp.barrier_core,
            slit1_mask=double_comp.slit_masks["slit1"],
            slit2_mask=double_comp.slit_masks["slit2"],

            # new structure
            components=component_specs,
        )


def build_potential(grid, cfg) -> PotentialSpec:
    return PotentialBuilder(cfg).build(grid)


def build_double_slit_and_caps(grid, cfg) -> PotentialSpec:
    return build_potential(grid, cfg)