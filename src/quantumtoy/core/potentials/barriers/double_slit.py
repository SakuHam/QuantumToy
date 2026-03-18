from __future__ import annotations

import numpy as np

from .base import BarrierComponent
from ..validation import (
    _assert_array_shape,
    _assert_finite_array,
)


class DoubleSlitBarrier:
    def __init__(
        self,
        *,
        center_x: float,
        thickness: float,
        slit_center_offset: float,
        slit_half_height: float,
        V_barrier: float,
        barrier_smooth: float,
        name: str = "double_slit",
    ):
        self.center_x = float(center_x)
        self.thickness = float(thickness)
        self.slit_center_offset = float(slit_center_offset)
        self.slit_half_height = float(slit_half_height)
        self.V_barrier = float(V_barrier)
        self.barrier_smooth = float(barrier_smooth)
        self.name = name

    def build(self, X: np.ndarray, Y: np.ndarray) -> BarrierComponent:
        barrier_core = np.abs(X - self.center_x) < (self.thickness / 2.0)
        slit1_mask = np.abs(Y - self.slit_center_offset) < self.slit_half_height
        slit2_mask = np.abs(Y + self.slit_center_offset) < self.slit_half_height

        _assert_array_shape(barrier_core, X.shape, "double.barrier_core")
        _assert_array_shape(slit1_mask, X.shape, "double.slit1_mask")
        _assert_array_shape(slit2_mask, X.shape, "double.slit2_mask")

        if self.barrier_smooth <= 0.0:
            barrier_mask = barrier_core.copy()
            barrier_mask[slit1_mask] = False
            barrier_mask[slit2_mask] = False
            V_real = np.zeros_like(X, dtype=float)
            V_real[barrier_mask] = self.V_barrier
        else:
            dist = np.abs(X - self.center_x) - (self.thickness / 2.0)
            wall = 1.0 / (1.0 + np.exp(dist / self.barrier_smooth))
            wall[slit1_mask] = 0.0
            wall[slit2_mask] = 0.0
            V_real = self.V_barrier * wall

        _assert_finite_array(V_real, "double.V_real")

        return BarrierComponent(
            name=self.name,
            kind="double_slit",
            V_real=V_real,
            barrier_core=barrier_core,
            slit_masks={
                "slit1": slit1_mask,
                "slit2": slit2_mask,
            },
        )