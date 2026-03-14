from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .base import DetectorModel, DetectorStepResult


@dataclass
class BornDetector(DetectorModel):
    """
    Simple detector that samples a click from rho = |psi|^2
    in a detector-gated region.

    This is not emergent dynamics; it is useful as a baseline detector.
    """

    grid: any

    detector_gate_center_x: float = 10.0
    detector_gate_width: float = 0.75
    detector_gate_center_y: float = 0.0
    detector_gate_width_y: float = -1.0
    detector_min_total_weight: float = 1e-6
    detector_min_peak_weight: float = 1e-8
    detector_latch_click: bool = True

    click_time_mode: str = "first_call"   # or "provided_time"
    rng_seed: int | None = None

    _clicked: bool = field(init=False, default=False, repr=False)
    _click_ix: int | None = field(init=False, default=None, repr=False)
    _click_iy: int | None = field(init=False, default=None, repr=False)
    _click_time: float | None = field(init=False, default=None, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _gate_cache: np.ndarray | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.rng_seed)

    def reset(self, **kwargs) -> None:
        self._clicked = False
        self._click_ix = None
        self._click_iy = None
        self._click_time = None
        if "rng_seed" in kwargs:
            self.rng_seed = kwargs["rng_seed"]
        self._rng = np.random.default_rng(self.rng_seed)

    def has_clicked(self) -> bool:
        return self._clicked

    def _detector_gate(self) -> np.ndarray:
        if self._gate_cache is not None:
            return self._gate_cache

        X = self.grid.X
        Y = self.grid.Y

        gx = np.exp(
            -((X - float(self.detector_gate_center_x)) ** 2)
            / (2.0 * float(self.detector_gate_width) ** 2)
        )

        if self.detector_gate_width_y is not None and self.detector_gate_width_y > 0.0:
            gy = np.exp(
                -((Y - float(self.detector_gate_center_y)) ** 2)
                / (2.0 * float(self.detector_gate_width_y) ** 2)
            )
            gate = gx * gy
        else:
            gate = gx

        self._gate_cache = gate.astype(float)
        return self._gate_cache

    def step(
        self,
        psi: np.ndarray,
        dt: float,
        t: float | None = None,
    ) -> DetectorStepResult:
        if self._clicked:
            return DetectorStepResult(
                clicked=True,
                click_ix=self._click_ix,
                click_iy=self._click_iy,
                click_x=None if self._click_ix is None else float(self.grid.X[self._click_iy, self._click_ix]),
                click_y=None if self._click_iy is None else float(self.grid.Y[self._click_iy, self._click_ix]),
                click_time=self._click_time,
                aux={"mode": "latched"},
            )

        rho = (np.abs(psi) ** 2).astype(float)
        gate = self._detector_gate()
        weights = rho * gate

        total = float(np.sum(weights) * self.grid.dx * self.grid.dy)
        peak = float(np.max(weights))

        if total < float(self.detector_min_total_weight):
            return DetectorStepResult(
                clicked=False,
                aux={
                    "rho_gate_sum": total,
                    "rho_gate_max": peak,
                    "armed": False,
                    "reason": "total_weight_below_threshold",
                },
            )

        if peak < float(self.detector_min_peak_weight):
            return DetectorStepResult(
                clicked=False,
                aux={
                    "rho_gate_sum": total,
                    "rho_gate_max": peak,
                    "armed": False,
                    "reason": "peak_weight_below_threshold",
                },
            )

        flat = weights.ravel()
        s = float(np.sum(flat))
        if s <= 0.0:
            return DetectorStepResult(
                clicked=False,
                aux={
                    "rho_gate_sum": total,
                    "rho_gate_max": peak,
                    "armed": False,
                    "reason": "zero_weight_after_checks",
                },
            )

        flat = flat / s
        idx = int(self._rng.choice(flat.size, p=flat))
        iy, ix = np.unravel_index(idx, weights.shape)

        self._clicked = True
        self._click_ix = int(ix)
        self._click_iy = int(iy)
        self._click_time = t

        return DetectorStepResult(
            clicked=True,
            click_ix=self._click_ix,
            click_iy=self._click_iy,
            click_x=float(self.grid.X[iy, ix]),
            click_y=float(self.grid.Y[iy, ix]),
            click_time=self._click_time,
            aux={
                "rho_gate_sum": total,
                "rho_gate_max": peak,
                "armed": True,
            },
        )