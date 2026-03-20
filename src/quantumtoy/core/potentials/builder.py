from __future__ import annotations

import numpy as np

from core.simulation_types import PotentialSpec, PotentialComponentSpec

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
    SimpleBarrier,
    MicroBlackHole,
    HybridBlackHole,
    CompositeBarrierSystem,
)


class PotentialBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, grid) -> PotentialSpec:
        cfg = self.cfg
        X = grid.X
        Y = grid.Y

        barrier_center_x = _assert_finite_scalar(_require_attr(cfg, "barrier_center_x"), "cfg.barrier_center_x")
        barrier_thickness = _assert_positive_scalar(_require_attr(cfg, "barrier_thickness"), "cfg.barrier_thickness")
        V_barrier = _assert_nonnegative_scalar(_require_attr(cfg, "V_barrier"), "cfg.V_barrier")
        slit_center_offset = _assert_finite_scalar(_require_attr(cfg, "slit_center_offset"), "cfg.slit_center_offset")
        slit_half_height = _assert_positive_scalar(_require_attr(cfg, "slit_half_height"), "cfg.slit_half_height")

        BARRIER_SMOOTH = _assert_finite_scalar(_require_attr(cfg, "BARRIER_SMOOTH"), "cfg.BARRIER_SMOOTH")
        barrier_edge_mode = str(getattr(cfg, "BARRIER_EDGE_MODE", "smooth")).lower().strip()
        barrier_sharp_smooth = float(
            getattr(cfg, "BARRIER_SHARP_SMOOTH", max(1e-12, 0.25 * BARRIER_SMOOTH))
        )

        CAP_WIDTH = _assert_nonnegative_scalar(_require_attr(cfg, "CAP_WIDTH"), "cfg.CAP_WIDTH")
        CAP_STRENGTH = _assert_nonnegative_scalar(_require_attr(cfg, "CAP_STRENGTH"), "cfg.CAP_STRENGTH")
        CAP_POWER = _assert_positive_scalar(_require_attr(cfg, "CAP_POWER"), "cfg.CAP_POWER")

        screen_center_x = _assert_finite_scalar(_require_attr(cfg, "screen_center_x"), "cfg.screen_center_x")
        screen_eval_width = _assert_positive_scalar(_require_attr(cfg, "screen_eval_width"), "cfg.screen_eval_width")
        USE_SCREEN_CAP = bool(_require_attr(cfg, "USE_SCREEN_CAP"))
        SCREEN_CAP_STRENGTH = _assert_nonnegative_scalar(
            _require_attr(cfg, "SCREEN_CAP_STRENGTH"),
            "cfg.SCREEN_CAP_STRENGTH",
        )

        system = CompositeBarrierSystem(V_clip_max=V_barrier)

        if bool(getattr(cfg, "USE_UPSTREAM_SINGLE_SLIT", False)):
            system.add(
                SingleSlitBarrier(
                    center_x=float(_require_attr(cfg, "single_slit_center_x")),
                    thickness=barrier_thickness,
                    slit_half_height=float(_require_attr(cfg, "single_slit_half_height")),
                    V_barrier=V_barrier,
                    barrier_smooth=BARRIER_SMOOTH,
                    sharp_smooth_width=barrier_sharp_smooth,
                    edge_mode=barrier_edge_mode,
                    name="upstream_single_slit",
                )
            )

        if bool(getattr(cfg, "USE_SIMPLE_BARRIER", False)):
            system.add(
                SimpleBarrier(
                    center_x=float(_require_attr(cfg, "simple_barrier_center_x")),
                    center_y=float(getattr(cfg, "simple_barrier_center_y", 0.0)),
                    thickness=barrier_thickness,
                    half_height=float(_require_attr(cfg, "simple_barrier_half_height")),
                    V_barrier=V_barrier,
                    barrier_smooth=BARRIER_SMOOTH,
                    sharp_smooth_width=barrier_sharp_smooth,
                    edge_mode=barrier_edge_mode,
                    name="simple_barrier",
                )
            )

        if bool(getattr(cfg, "USE_MICRO_BLACK_HOLE", False)):
            system.add(
                MicroBlackHole(
                    center_x=float(_require_attr(cfg, "micro_bh_center_x")),
                    center_y=float(_require_attr(cfg, "micro_bh_center_y")),
                    sigma_V=float(getattr(cfg, "micro_bh_sigma_V", 0.6)),
                    sigma_W=float(getattr(cfg, "micro_bh_sigma_W", 0.25)),
                    V_strength=float(getattr(cfg, "micro_bh_V_strength", 5.0)),
                    W_strength=float(getattr(cfg, "micro_bh_W_strength", 1.5)),
                    name="micro_black_hole",
                )
            )

        if bool(getattr(cfg, "USE_HYBRID_BLACK_HOLE", False)):
            system.add(
                HybridBlackHole(
                    center_x=float(_require_attr(cfg, "hybrid_bh_center_x")),
                    center_y=float(_require_attr(cfg, "hybrid_bh_center_y")),
                    sigma_V=float(getattr(cfg, "hybrid_bh_sigma_V", 1.2)),
                    V_strength=float(getattr(cfg, "hybrid_bh_V_strength", 1.5)),
                    horizon_radius=float(getattr(cfg, "hybrid_bh_horizon_radius", 1.2)),
                    horizon_width=float(getattr(cfg, "hybrid_bh_horizon_width", 0.35)),
                    W_horizon=float(getattr(cfg, "hybrid_bh_W_horizon", 2.5)),
                    core_radius=float(getattr(cfg, "hybrid_bh_core_radius", 0.7)),
                    W_core=float(getattr(cfg, "hybrid_bh_W_core", 1.5)),
                    name="hybrid_black_hole",
                )
            )
            
        system.add(
            DoubleSlitBarrier(
                center_x=barrier_center_x,
                thickness=barrier_thickness,
                slit_center_offset=slit_center_offset,
                slit_half_height=slit_half_height,
                V_barrier=V_barrier,
                barrier_smooth=BARRIER_SMOOTH,
                sharp_smooth_width=barrier_sharp_smooth,
                edge_mode=barrier_edge_mode,
                name="downstream_double_slit",
            )
        )

        component_result = system.build(X, Y)

        V_real = component_result.V_real
        W_components = component_result.W

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

        W_total = W_edge + W_screen + W_components

        double_comp = next(
            comp for comp in component_result.components
            if comp.name == "downstream_double_slit"
        )

        component_specs = [
            PotentialComponentSpec(
                name=comp.name,
                kind=comp.kind,
                V_real=comp.V_real,
                W=comp.W,
                barrier_core=comp.barrier_core,
                wall_mask=comp.wall_mask,
                slit_masks=comp.slit_masks,
            )
            for comp in component_result.components
        ]

        return PotentialSpec(
            V_real=V_real,
            W=W_total,
            screen_mask_full=screen_mask_full,
            screen_mask_vis=screen_mask_vis,
            barrier_core=double_comp.barrier_core,
            slit1_mask=double_comp.slit_masks["slit1"],
            slit2_mask=double_comp.slit_masks["slit2"],
            components=component_specs,
        )


def build_potential(grid, cfg) -> PotentialSpec:
    return PotentialBuilder(cfg).build(grid)


def build_double_slit_and_caps(grid, cfg) -> PotentialSpec:
    return build_potential(grid, cfg)