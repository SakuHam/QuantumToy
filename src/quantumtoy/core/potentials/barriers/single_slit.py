from __future__ import annotations

import numpy as np

from .base import BarrierComponent
from ..validation import (
    _assert_array_shape,
    _assert_finite_array,
)


class SingleSlitBarrier:
    def __init__(
        self,
        *,
        center_x: float,
        thickness: float,
        slit_half_height: float,
        V_barrier: float,
        barrier_smooth: float,
        name: str = "single_slit",
    ):
        self.center_x = float(center_x)
        self.thickness = float(thickness)
        self.slit_half_height = float(slit_half_height)
        self.V_barrier = float(V_barrier)
        self.barrier_smooth = float(barrier_smooth)
        self.name = name

    def build(self, X: np.ndarray, Y: np.ndarray) -> BarrierComponent:
        barrier_core = np.abs(X - self.center_x) < (self.thickness / 2.0)
        slit_mask = np.abs(Y) < self.slit_half_height

        _assert_array_shape(barrier_core, X.shape, "single.barrier_core")
        _assert_array_shape(slit_mask, X.shape, "single.slit_mask")

        if self.barrier_smooth <= 0.0:
            barrier_mask = barrier_core.copy()
            barrier_mask[slit_mask] = False
            V_real = np.zeros_like(X, dtype=float)
            V_real[barrier_mask] = self.V_barrier
        else:
            dist = np.abs(X - self.center_x) - (self.thickness / 2.0)
            wall = 1.0 / (1.0 + np.exp(dist / self.barrier_smooth))
            wall[slit_mask] = 0.0
            V_real = self.V_barrier * wall

        _assert_finite_array(V_real, "single.V_real")

        return BarrierComponent(
            name=self.name,
            kind="single_slit",
            V_real=V_real,
            barrier_core=barrier_core,
            slit_masks={"slit": slit_mask},
        )