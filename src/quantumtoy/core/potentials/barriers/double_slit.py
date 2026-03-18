from __future__ import annotations

import numpy as np

from .base import BarrierComponent
from ..validation import (
    _assert,
    _assert_array_shape,
    _assert_finite_array,
)


class DoubleSlitBarrier:
    """
    Double-slit barrier with selectable edge model.

    edge_mode:
        - "hard"
        - "sharp_smooth"
        - "smooth"
    """

    VALID_EDGE_MODES = ("hard", "sharp_smooth", "smooth")

    def __init__(
        self,
        *,
        center_x: float,
        thickness: float,
        slit_center_offset: float,
        slit_half_height: float,
        V_barrier: float,
        barrier_smooth: float,
        sharp_smooth_width: float | None = None,
        edge_mode: str = "smooth",
        name: str = "double_slit",
    ):
        self.center_x = float(center_x)
        self.thickness = float(thickness)
        self.slit_center_offset = float(slit_center_offset)
        self.slit_half_height = float(slit_half_height)
        self.V_barrier = float(V_barrier)
        self.barrier_smooth = float(barrier_smooth)
        self.sharp_smooth_width = (
            float(sharp_smooth_width)
            if sharp_smooth_width is not None
            else max(1e-12, 0.25 * float(barrier_smooth))
        )
        self.edge_mode = str(edge_mode).lower().strip()
        self.name = name

        _assert(
            self.edge_mode in self.VALID_EDGE_MODES,
            f"Invalid edge_mode={self.edge_mode!r}, expected one of {self.VALID_EDGE_MODES}",
        )

    def _build_hard(self, X: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
        V_real = np.zeros_like(X, dtype=float)
        V_real[wall_mask] = self.V_barrier
        return V_real

    def _build_smooth_like(
        self,
        X: np.ndarray,
        slit1_mask: np.ndarray,
        slit2_mask: np.ndarray,
        smooth_width: float,
    ) -> np.ndarray:
        _assert(smooth_width > 0.0, f"smooth_width must be > 0, got {smooth_width}")

        dist = np.abs(X - self.center_x) - (self.thickness / 2.0)
        wall = 1.0 / (1.0 + np.exp(dist / smooth_width))
        wall = wall.astype(float, copy=False)
        wall[slit1_mask] = 0.0
        wall[slit2_mask] = 0.0
        return self.V_barrier * wall

    def build(self, X: np.ndarray, Y: np.ndarray) -> BarrierComponent:
        barrier_core = np.abs(X - self.center_x) < (self.thickness / 2.0)
        slit1_mask = np.abs(Y - self.slit_center_offset) < self.slit_half_height
        slit2_mask = np.abs(Y + self.slit_center_offset) < self.slit_half_height
        wall_mask = barrier_core & (~slit1_mask) & (~slit2_mask)

        _assert_array_shape(barrier_core, X.shape, "double.barrier_core")
        _assert_array_shape(slit1_mask, X.shape, "double.slit1_mask")
        _assert_array_shape(slit2_mask, X.shape, "double.slit2_mask")
        _assert_array_shape(wall_mask, X.shape, "double.wall_mask")

        if self.edge_mode == "hard":
            V_real = self._build_hard(X, wall_mask)
        elif self.edge_mode == "sharp_smooth":
            V_real = self._build_smooth_like(
                X,
                slit1_mask,
                slit2_mask,
                self.sharp_smooth_width,
            )
        elif self.edge_mode == "smooth":
            V_real = self._build_smooth_like(
                X,
                slit1_mask,
                slit2_mask,
                self.barrier_smooth,
            )
        else:
            raise AssertionError(f"Unhandled edge_mode={self.edge_mode!r}")

        _assert_finite_array(V_real, "double.V_real")

        return BarrierComponent(
            name=self.name,
            kind="double_slit",
            V_real=V_real,
            barrier_core=barrier_core,
            wall_mask=wall_mask,
            slit_masks={
                "slit1": slit1_mask,
                "slit2": slit2_mask,
            },
        )