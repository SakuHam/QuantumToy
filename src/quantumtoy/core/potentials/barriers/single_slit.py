from __future__ import annotations

import numpy as np

from .base import PotentialComponent
from ..validation import (
    _assert,
    _assert_array_shape,
    _assert_finite_array,
)


class SingleSlitBarrier:
    VALID_EDGE_MODES = ("hard", "sharp_smooth", "smooth")

    def __init__(
        self,
        *,
        center_x: float,
        thickness: float,
        slit_half_height: float,
        V_barrier: float,
        barrier_smooth: float,
        sharp_smooth_width: float | None = None,
        edge_mode: str = "smooth",
        name: str = "single_slit",
    ):
        self.center_x = float(center_x)
        self.thickness = float(thickness)
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
        slit_mask: np.ndarray,
        smooth_width: float,
    ) -> np.ndarray:
        _assert(smooth_width > 0.0, f"smooth_width must be > 0, got {smooth_width}")

        dist = np.abs(X - self.center_x) - (self.thickness / 2.0)
        wall = 1.0 / (1.0 + np.exp(dist / smooth_width))
        wall = wall.astype(float, copy=False)
        wall[slit_mask] = 0.0
        return self.V_barrier * wall

    def build(self, X: np.ndarray, Y: np.ndarray) -> PotentialComponent:
        barrier_core = np.abs(X - self.center_x) < (self.thickness / 2.0)
        slit_mask = np.abs(Y) < self.slit_half_height
        wall_mask = barrier_core & (~slit_mask)

        _assert_array_shape(barrier_core, X.shape, "single.barrier_core")
        _assert_array_shape(slit_mask, X.shape, "single.slit_mask")
        _assert_array_shape(wall_mask, X.shape, "single.wall_mask")

        if self.edge_mode == "hard":
            V_real = self._build_hard(X, wall_mask)
        elif self.edge_mode == "sharp_smooth":
            V_real = self._build_smooth_like(X, slit_mask, self.sharp_smooth_width)
        elif self.edge_mode == "smooth":
            V_real = self._build_smooth_like(X, slit_mask, self.barrier_smooth)
        else:
            raise AssertionError(f"Unhandled edge_mode={self.edge_mode!r}")

        W = np.zeros_like(X, dtype=float)

        _assert_finite_array(V_real, "single.V_real")
        _assert_finite_array(W, "single.W")

        return PotentialComponent(
            name=self.name,
            kind="single_slit",
            V_real=V_real,
            W=W,
            barrier_core=barrier_core,
            wall_mask=wall_mask,
            slit_masks={"slit": slit_mask},
        )