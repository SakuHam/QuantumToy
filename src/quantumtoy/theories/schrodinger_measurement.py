from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from theories.base import TheoryStepResult
from theories.schrodinger import SchrodingerTheory
from core.utils import normalize_unit


@dataclass
class SchrodingerMeasurementTheory(SchrodingerTheory):
    kappa_meas: float = 0.02
    rng_seed: int = 1234
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.rng = np.random.default_rng(self.rng_seed)

    def _expval_xy_unitnorm(self, psi_unit: np.ndarray):
        p = np.abs(psi_unit) ** 2
        norm = float(np.sum(p) * self.grid.dx * self.grid.dy)

        if norm <= 0:
            return 0.0, 0.0

        mx = float(np.sum(p * self.grid.X) * self.grid.dx * self.grid.dy / norm)
        my = float(np.sum(p * self.grid.Y) * self.grid.dx * self.grid.dy / norm)
        return mx, my

    def _continuous_measurement_update_preserve_norm(
        self,
        psi: np.ndarray,
        dt: float,
    ):
        if self.kappa_meas <= 0:
            return psi, {"dWx": 0.0, "dWy": 0.0, "mx": 0.0, "my": 0.0}

        psi_u, n0 = normalize_unit(psi, self.grid.dx, self.grid.dy)
        if n0 <= 0:
            return psi, {"dWx": 0.0, "dWy": 0.0, "mx": 0.0, "my": 0.0}

        mx, my = self._expval_xy_unitnorm(psi_u)

        Xc = self.grid.X - mx
        Yc = self.grid.Y - my

        dWx = self.rng.normal(0.0, np.sqrt(dt))
        dWy = self.rng.normal(0.0, np.sqrt(dt))

        drift = -0.5 * self.kappa_meas * (Xc**2 + Yc**2) * dt
        stoch = np.sqrt(self.kappa_meas) * (Xc * dWx + Yc * dWy)

        psi_u2 = psi_u * np.exp(drift + stoch)
        psi_u2, _ = normalize_unit(psi_u2, self.grid.dx, self.grid.dy)

        psi_new = psi_u2 * n0

        return psi_new, {
            "dWx": float(dWx),
            "dWy": float(dWy),
            "mx": float(mx),
            "my": float(my),
        }

    def step_forward(self, state: np.ndarray, dt: float) -> TheoryStepResult:
        base = super().step_forward(state, dt)

        psi2, meas_aux = self._continuous_measurement_update_preserve_norm(
            base.state,
            dt,
        )

        aux = dict(base.aux) if base.aux is not None else {}
        aux["measurement"] = meas_aux

        return TheoryStepResult(
            state=psi2,
            aux=aux,
        )