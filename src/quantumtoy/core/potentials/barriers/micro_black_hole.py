from __future__ import annotations

import numpy as np

from .base import PotentialComponent
from ..validation import _assert, _assert_array_shape, _assert_finite_array


class MicroBlackHole:
    """
    Effective toy-model micro black hole:
      - attractive real potential well
      - absorbing imaginary core
    """

    def __init__(
        self,
        *,
        center_x: float,
        center_y: float,
        sigma_V: float = 0.6,
        sigma_W: float = 0.25,
        V_strength: float = 5.0,
        W_strength: float = 1.5,
        name: str = "micro_black_hole",
    ):
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.sigma_V = float(sigma_V)
        self.sigma_W = float(sigma_W)
        self.V_strength = float(V_strength)
        self.W_strength = float(W_strength)
        self.name = name

        _assert(self.sigma_V > 0.0, "sigma_V must be > 0")
        _assert(self.sigma_W > 0.0, "sigma_W must be > 0")
        _assert(self.V_strength >= 0.0, "V_strength must be >= 0")
        _assert(self.W_strength >= 0.0, "W_strength must be >= 0")

    def build(self, X: np.ndarray, Y: np.ndarray) -> PotentialComponent:
        dx = X - self.center_x
        dy = Y - self.center_y
        r2 = dx * dx + dy * dy

        V_real = -self.V_strength * np.exp(-r2 / (2.0 * self.sigma_V**2))
        W = self.W_strength * np.exp(-r2 / (2.0 * self.sigma_W**2))

        barrier_core = r2 < (self.sigma_V**2)
        wall_mask = r2 < (self.sigma_W**2)

        _assert_array_shape(V_real, X.shape, "mbh.V_real")
        _assert_array_shape(W, X.shape, "mbh.W")
        _assert_array_shape(barrier_core, X.shape, "mbh.barrier_core")
        _assert_array_shape(wall_mask, X.shape, "mbh.wall_mask")

        _assert_finite_array(V_real, "mbh.V_real")
        _assert_finite_array(W, "mbh.W")

        return PotentialComponent(
            name=self.name,
            kind="micro_black_hole",
            V_real=V_real,
            W=W,
            barrier_core=barrier_core,
            wall_mask=wall_mask,
            slit_masks={},
        )