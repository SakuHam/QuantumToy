from __future__ import annotations

import numpy as np

from .base import PotentialComponent
from ..validation import _assert, _assert_array_shape, _assert_finite_array


class HybridBlackHole:
    """
    Hybrid toy black hole component.

    Contributes:
      - moderate attractive real potential well
      - horizon-like absorbing shell
      - optional inner absorbing core

    This is designed to work especially well together with
    MetricAwareSchrodingerTheory:
      - metric handles curved propagation / slowing
      - this component handles capture / horizon loss

    Parameters
    ----------
    center_x, center_y:
        Black-hole center.

    sigma_V:
        Width of attractive real well.

    V_strength:
        Strength of attractive well.
        Produces:
            V_real = -V_strength * exp(-r^2 / (2 sigma_V^2))

    horizon_radius:
        Radius of horizon-like absorbing shell.

    horizon_width:
        Thickness of the shell.

    W_horizon:
        Strength of shell absorption.

    core_radius:
        Radius of inner absorber.

    W_core:
        Strength of inner absorption.
    """

    def __init__(
        self,
        *,
        center_x: float,
        center_y: float,
        sigma_V: float = 1.2,
        V_strength: float = 1.5,
        horizon_radius: float = 1.2,
        horizon_width: float = 0.35,
        W_horizon: float = 2.5,
        core_radius: float = 0.7,
        W_core: float = 1.5,
        name: str = "hybrid_black_hole",
    ):
        self.center_x = float(center_x)
        self.center_y = float(center_y)

        self.sigma_V = float(sigma_V)
        self.V_strength = float(V_strength)

        self.horizon_radius = float(horizon_radius)
        self.horizon_width = float(horizon_width)
        self.W_horizon = float(W_horizon)

        self.core_radius = float(core_radius)
        self.W_core = float(W_core)

        self.name = name

        _assert(self.sigma_V > 0.0, "sigma_V must be > 0")
        _assert(self.V_strength >= 0.0, "V_strength must be >= 0")
        _assert(self.horizon_radius > 0.0, "horizon_radius must be > 0")
        _assert(self.horizon_width > 0.0, "horizon_width must be > 0")
        _assert(self.W_horizon >= 0.0, "W_horizon must be >= 0")
        _assert(self.core_radius >= 0.0, "core_radius must be >= 0")
        _assert(self.W_core >= 0.0, "W_core must be >= 0")

    def build(self, X: np.ndarray, Y: np.ndarray) -> PotentialComponent:
        dx = X - self.center_x
        dy = Y - self.center_y
        r = np.sqrt(dx * dx + dy * dy)
        r2 = dx * dx + dy * dy

        # ---------------------------------------------
        # Attractive real well (keep moderate!)
        # ---------------------------------------------
        V_real = -self.V_strength * np.exp(-r2 / (2.0 * self.sigma_V**2))

        # ---------------------------------------------
        # Horizon shell absorption
        # strongest around r ~ horizon_radius
        # ---------------------------------------------
        W_shell = self.W_horizon * np.exp(
            -((r - self.horizon_radius) ** 2) / (2.0 * self.horizon_width**2)
        )

        # ---------------------------------------------
        # Inner core absorption
        # logistic-like inside absorber
        # ---------------------------------------------
        if self.core_radius > 0.0 and self.W_core > 0.0:
            core_transition = max(1e-12, 0.35 * self.horizon_width)
            W_core = self.W_core / (1.0 + np.exp((r - self.core_radius) / core_transition))
        else:
            W_core = np.zeros_like(r, dtype=float)

        W = W_shell + W_core

        # ---------------------------------------------
        # Visualization/debug geometry
        # ---------------------------------------------
        barrier_core = r <= self.horizon_radius
        wall_mask = np.abs(r - self.horizon_radius) <= max(self.horizon_width, 0.15)

        _assert_array_shape(V_real, X.shape, "hybrid_bh.V_real")
        _assert_array_shape(W, X.shape, "hybrid_bh.W")
        _assert_array_shape(barrier_core, X.shape, "hybrid_bh.barrier_core")
        _assert_array_shape(wall_mask, X.shape, "hybrid_bh.wall_mask")

        _assert_finite_array(V_real, "hybrid_bh.V_real")
        _assert_finite_array(W, "hybrid_bh.W")

        return PotentialComponent(
            name=self.name,
            kind="hybrid_black_hole",
            V_real=V_real,
            W=W,
            barrier_core=barrier_core,
            wall_mask=wall_mask,
            slit_masks={},
        )