from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .base import DetectorModel, DetectorStepResult


@dataclass
class BornDetector(DetectorModel):
    """
    Simple detector that samples a click from rho in a detector-gated region.

    Supports both:
      - scalar wavefunction: psi.shape == (Ny, Nx)
      - 2-component Dirac spinor: psi.shape == (2, Ny, Nx)

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
        self._validate_params()
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

    def _validate_params(self) -> None:
        if self.detector_gate_width <= 0.0:
            raise ValueError("detector_gate_width must be > 0")
        if self.detector_min_total_weight < 0.0:
            raise ValueError("detector_min_total_weight must be >= 0")
        if self.detector_min_peak_weight < 0.0:
            raise ValueError("detector_min_peak_weight must be >= 0")
        if self.click_time_mode not in ("first_call", "provided_time"):
            raise ValueError("click_time_mode must be 'first_call' or 'provided_time'")

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

    def _compute_rho(self, psi: np.ndarray) -> np.ndarray:
        """
        Return a 2D probability density.

        Scalar Schrödinger:
            rho = |psi|^2

        2-component Dirac:
            rho = |psi[0]|^2 + |psi[1]|^2
        """
        psi = np.asarray(psi)

        if psi.ndim == 2:
            rho = np.abs(psi) ** 2
            return rho.astype(float)

        if psi.ndim == 3:
            if psi.shape[0] != 2:
                raise ValueError(
                    f"Dirac psi must have shape (2, Ny, Nx), got {psi.shape}"
                )
            rho = np.abs(psi[0]) ** 2 + np.abs(psi[1]) ** 2
            return rho.astype(float)

        raise ValueError(
            f"Unsupported psi shape {psi.shape}; expected (Ny, Nx) or (2, Ny, Nx)"
        )

    def _validate_psi_shape(self, psi: np.ndarray) -> None:
        psi = np.asarray(psi)

        if psi.ndim == 2:
            spatial_shape = psi.shape
        elif psi.ndim == 3:
            spatial_shape = psi.shape[-2:]
        else:
            raise ValueError(
                f"Unsupported psi shape {psi.shape}; expected (Ny, Nx) or (2, Ny, Nx)"
            )

        expected = (self.grid.Ny, self.grid.Nx)
        if spatial_shape != expected:
            raise ValueError(
                f"psi spatial shape {spatial_shape} does not match grid shape {expected}; "
                f"full psi shape was {psi.shape}"
            )

    def _latched_result(self) -> DetectorStepResult:
        return DetectorStepResult(
            clicked=True,
            click_ix=self._click_ix,
            click_iy=self._click_iy,
            click_x=None if self._click_ix is None else float(self.grid.X[self._click_iy, self._click_ix]),
            click_y=None if self._click_iy is None else float(self.grid.Y[self._click_iy, self._click_ix]),
            click_time=self._click_time,
            aux={"mode": "latched"},
        )

    def step(
        self,
        psi: np.ndarray,
        dt: float,
        t: float | None = None,
    ) -> DetectorStepResult:
        if not np.isfinite(dt):
            raise ValueError(f"dt is not finite: {dt}")
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")

        psi = np.asarray(psi, dtype=np.complex128)
        self._validate_psi_shape(psi)

        if self._clicked and self.detector_latch_click:
            return self._latched_result()

        rho = self._compute_rho(psi)
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
        if s <= 0.0 or not np.isfinite(s):
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

        if self.click_time_mode == "provided_time":
            self._click_time = None if t is None else float(t)
        else:
            self._click_time = 0.0 if t is None else float(t)

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