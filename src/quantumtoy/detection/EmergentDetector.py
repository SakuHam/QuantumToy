from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import maximum_filter, gaussian_filter

from .base import DetectorModel, DetectorStepResult

@dataclass
class EmergentDetector(DetectorModel):
    
    """
    Emergent detector model.

    Core idea:
      - Detector has an internal activation field D(x,y)
      - Wave density rho drives the detector locally
      - Detector activation leaks away over time
      - Neighboring detector sites compete via lateral inhibition
      - A click is declared when activation crosses a threshold

    This avoids explicit Born sampling. Instead, the click emerges as a
    dynamical winner-take-all detector event.
    """

    # --------------------------------------------------------
    # Geometry / grid
    # --------------------------------------------------------

    grid: any

    # --------------------------------------------------------
    # Detector gate / screen coupling
    # --------------------------------------------------------

    # Detector is mainly active near x = center_x
    detector_gate_center_x: float = 10.0

    # Gaussian width in x of the detector region
    detector_gate_width: float = 0.75

    # Optional extra Gaussian confinement in y.
    # Set to <= 0 to disable.
    detector_gate_center_y: float = 0.0
    detector_gate_width_y: float = -1.0

    # --------------------------------------------------------
    # Dynamics
    # --------------------------------------------------------

    # Drive from rho into detector state
    detector_gain: float = 4.0

    # Leak / forgetting term
    detector_leak: float = 0.5

    # Lateral inhibition strength
    detector_competition_strength: float = 6.0

    # Radius in pixels for local competition
    detector_competition_radius: int = 10

    # Optional Gaussian smoothing of D after update
    detector_state_blur_sigma: float = 0.0

    # Clamp state nonnegative
    detector_nonnegative: bool = True

    # --------------------------------------------------------
    # Click decision
    # --------------------------------------------------------

    # Threshold for declaring a click
    detector_click_threshold: float = 1.0

    # Require click hotspot to be sufficiently isolated?
    detector_require_local_max: bool = True

    # If True, once clicked, keep the click forever
    detector_latch_click: bool = True

    # --------------------------------------------------------
    # Optional stochastic tie-breaking
    # --------------------------------------------------------

    detector_noise_strength: float = 0.0
    detector_noise_seed: int | None = None

    # --------------------------------------------------------
    # Internal state
    # --------------------------------------------------------

    _state: np.ndarray = field(init=False, repr=False)
    _clicked: bool = field(init=False, default=False, repr=False)
    _click_ix: int | None = field(init=False, default=None, repr=False)
    _click_iy: int | None = field(init=False, default=None, repr=False)
    _click_time: float | None = field(init=False, default=None, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _gate_cache: np.ndarray | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self._state = np.zeros((self.grid.Ny, self.grid.Nx), dtype=float)
        self._clicked = False
        self._click_ix = None
        self._click_iy = None
        self._click_time = None
        self._rng = np.random.default_rng(self.detector_noise_seed)

        if self.detector_gate_width <= 0.0:
            raise ValueError("detector_gate_width must be > 0")
        if self.detector_competition_radius < 1:
            raise ValueError("detector_competition_radius must be >= 1")
        if self.detector_gain < 0.0:
            raise ValueError("detector_gain must be >= 0")
        if self.detector_leak < 0.0:
            raise ValueError("detector_leak must be >= 0")
        if self.detector_competition_strength < 0.0:
            raise ValueError("detector_competition_strength must be >= 0")
        if self.detector_click_threshold <= 0.0:
            raise ValueError("detector_click_threshold must be > 0")
        if self.detector_state_blur_sigma < 0.0:
            raise ValueError("detector_state_blur_sigma must be >= 0")
        if self.detector_noise_strength < 0.0:
            raise ValueError("detector_noise_strength must be >= 0")

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def reset(self, noise_seed: int | None = None):
        self._state.fill(0.0)
        self._clicked = False
        self._click_ix = None
        self._click_iy = None
        self._click_time = None
        if noise_seed is not None:
            self.detector_noise_seed = noise_seed
        self._rng = np.random.default_rng(self.detector_noise_seed)

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def clicked(self) -> bool:
        return self._clicked

    @property
    def click_index(self) -> tuple[int, int] | None:
        if self._click_iy is None or self._click_ix is None:
            return None
        return self._click_iy, self._click_ix

    @property
    def click_position(self) -> tuple[float, float] | None:
        if self._click_iy is None or self._click_ix is None:
            return None
        y = float(self.grid.Y[self._click_iy, self._click_ix])
        x = float(self.grid.X[self._click_iy, self._click_ix])
        return x, y

    @property
    def click_time(self) -> float | None:
        return self._click_time

    def has_clicked(self) -> bool:
        return self._clicked

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    def _detector_gate(self) -> np.ndarray:
        """
        Spatial coupling gate telling where detector can respond.

        By default this is a Gaussian strip near a screen x-position.
        Optionally also confined in y.
        """
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

    def _lateral_inhibition(self, D: np.ndarray) -> np.ndarray:
        """
        Winner-take-all style inhibition.

        For each pixel:
            inhibition = max(local_max(D) - D, 0)

        Interpretation:
          - if a nearby detector site is stronger, this point gets suppressed
          - local winners are relatively protected
        """
        radius = int(self.detector_competition_radius)
        size = 2 * radius + 1

        local_max = maximum_filter(D, size=(size, size), mode="nearest")
        inhibition = np.maximum(local_max - D, 0.0)
        return inhibition.astype(float)

    def _is_valid_click_site(self, D: np.ndarray, iy: int, ix: int) -> bool:
        """
        Optional local-max requirement for click declaration.
        """
        if not self.detector_require_local_max:
            return True

        radius = int(self.detector_competition_radius)
        y0 = max(0, iy - radius)
        y1 = min(D.shape[0], iy + radius + 1)
        x0 = max(0, ix - radius)
        x1 = min(D.shape[1], ix + radius + 1)

        patch = D[y0:y1, x0:x1]
        return D[iy, ix] >= float(np.max(patch)) - 1e-15

    # --------------------------------------------------------
    # Main update
    # --------------------------------------------------------

    def step(
        self,
        psi: np.ndarray,
        dt: float,
        t: float | None = None,
    ) -> dict:
        """
        Advance the detector by one time step.

        Inputs:
          psi : complex wavefield
          dt  : time step
          t   : optional simulation time, stored if click happens

        Returns diagnostics dict.
        """
        if self._clicked and self.detector_latch_click:
            return {
                "clicked": True,
                "click_ix": self._click_ix,
                "click_iy": self._click_iy,
                "click_x": None if self._click_ix is None else float(self.grid.X[self._click_iy, self._click_ix]),
                "click_y": None if self._click_iy is None else float(self.grid.Y[self._click_iy, self._click_ix]),
                "click_time": self._click_time,
                "detector_state_max": float(np.max(self._state)),
                "detector_state_mean": float(np.mean(self._state)),
                "detector_gate_mean": float(np.mean(self._detector_gate())),
                "detector_gate_max": float(np.max(self._detector_gate())),
                "detector_drive_mean": 0.0,
                "detector_drive_max": 0.0,
                "detector_inhibition_mean": 0.0,
                "detector_inhibition_max": 0.0,
            }

        rho = (np.abs(psi) ** 2).astype(float)
        gate = self._detector_gate()

        # ----------------------------------------------------
        # 1) Drive from wave density
        # ----------------------------------------------------
        drive = self.detector_gain * gate * rho

        # ----------------------------------------------------
        # 2) Leak
        # ----------------------------------------------------
        leak = self.detector_leak * self._state

        # ----------------------------------------------------
        # 3) Lateral inhibition
        # ----------------------------------------------------
        inhibition = self.detector_competition_strength * self._lateral_inhibition(self._state)

        # ----------------------------------------------------
        # 4) Optional noise
        # ----------------------------------------------------
        noise = 0.0
        if self.detector_noise_strength > 0.0:
            noise = (
                self.detector_noise_strength
                * np.sqrt(max(dt, 0.0))
                * self._rng.standard_normal(self._state.shape)
                * gate
            )

        # ----------------------------------------------------
        # 5) Euler update
        # ----------------------------------------------------
        D_new = self._state + dt * (drive - leak - inhibition) + noise

        if self.detector_state_blur_sigma > 0.0:
            D_new = gaussian_filter(
                D_new,
                sigma=float(self.detector_state_blur_sigma),
                mode="nearest",
            )

        if self.detector_nonnegative:
            D_new = np.maximum(D_new, 0.0)

        self._state = D_new.astype(float)

        # ----------------------------------------------------
        # 6) Check click
        # ----------------------------------------------------
        iy, ix = np.unravel_index(np.argmax(self._state), self._state.shape)
        peak = float(self._state[iy, ix])

        if peak >= float(self.detector_click_threshold):
            if self._is_valid_click_site(self._state, iy, ix):
                self._clicked = True
                self._click_ix = int(ix)
                self._click_iy = int(iy)
                self._click_time = None if t is None else float(t)

        return DetectorStepResult(
            clicked=bool(self._clicked),
            click_ix=self._click_ix,
            click_iy=self._click_iy,
            click_x=None if self._click_ix is None else float(self.grid.X[self._click_iy, self._click_ix]),
            click_y=None if self._click_iy is None else float(self.grid.Y[self._click_iy, self._click_ix]),
            click_time=self._click_time,
            aux={
                "detector_state_max": float(np.max(self._state)),
                "detector_state_mean": float(np.mean(self._state)),
                "detector_gate_mean": float(np.mean(gate)),
                "detector_gate_max": float(np.max(gate)),
                "detector_drive_mean": float(np.mean(drive)),
                "detector_drive_max": float(np.max(drive)),
                "detector_inhibition_mean": float(np.mean(inhibition)),
                "detector_inhibition_max": float(np.max(inhibition)),
            },
        )
