from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import maximum_filter, gaussian_filter

from .base import DetectorModel, DetectorStepResult


@dataclass
class EmergentDetector(DetectorModel):
    """
    Emergent detector model with:
      - flux-based drive
      - hotspot / branch selection
      - finite click localization resolution

    Main ideas:
      - Detector state D(x,y) integrates incoming signal
      - Click is not forced to global argmax(D)
      - Multiple local hotspots can compete as branch candidates
      - Final click location is sampled from local detector hotspot
      - Optional finite spatial resolution adds Heisenberg-like localization blur
    """

    # --------------------------------------------------------
    # Geometry / grid
    # --------------------------------------------------------

    grid: any

    # --------------------------------------------------------
    # Detector gate / screen coupling
    # --------------------------------------------------------

    detector_gate_center_x: float = 10.0
    detector_gate_width: float = 0.75

    detector_gate_center_y: float = 0.0
    detector_gate_width_y: float = -1.0

    # --------------------------------------------------------
    # Drive model
    # --------------------------------------------------------

    detector_drive_mode: str = "flux_x_positive"  # "flux_x_positive", "flux_x_abs", "rho"
    detector_hbar: float = 1.0
    detector_mass: float = 1.0

    # --------------------------------------------------------
    # Dynamics
    # --------------------------------------------------------

    detector_gain: float = 4.0
    detector_leak: float = 0.5
    detector_competition_strength: float = 6.0
    detector_competition_radius: int = 10
    detector_state_blur_sigma: float = 0.0
    detector_nonnegative: bool = True

    # --------------------------------------------------------
    # Click decision
    # --------------------------------------------------------

    detector_click_threshold: float = 1.0
    detector_require_local_max: bool = True
    detector_latch_click: bool = True

    # --------------------------------------------------------
    # Branch / hotspot selection
    # --------------------------------------------------------

    detector_branch_selection_mode: str = "sample_hotspots"
    # "argmax"            : old behavior
    # "sample_hotspots"   : choose among local hotspots weighted by strength
    # "sample_patch"      : sample directly from superthreshold patch

    detector_hotspot_min_fraction_of_peak: float = 0.35
    detector_hotspot_separation_radius: int = 6
    detector_click_patch_radius: int = 6

    # --------------------------------------------------------
    # Finite localization / Heisenberg-like resolution
    # --------------------------------------------------------

    detector_sample_click_position: bool = True
    detector_click_position_sigma_x: float = 0.0
    detector_click_position_sigma_y: float = 0.0

    detector_use_quantum_min_resolution: bool = True
    detector_quantum_resolution_scale: float = 0.5
    # Effective extra blur from local wavelength:
    # sigma_q ~ scale / max(|k_x|, k_floor)

    detector_k_floor: float = 1e-6

    # --------------------------------------------------------
    # Optional stochastic tie-breaking
    # --------------------------------------------------------

    detector_noise_strength: float = 0.0
    detector_noise_seed: int | None = None

    # --------------------------------------------------------
    # Debug
    # --------------------------------------------------------

    detector_debug: bool = False
    detector_debug_every_n_steps: int = 1

    # --------------------------------------------------------
    # Internal state
    # --------------------------------------------------------

    _state: np.ndarray = field(init=False, repr=False)
    _clicked: bool = field(init=False, default=False, repr=False)
    _click_ix: int | None = field(init=False, default=None, repr=False)
    _click_iy: int | None = field(init=False, default=None, repr=False)
    _click_time: float | None = field(init=False, default=None, repr=False)
    _click_x_override: float | None = field(init=False, default=None, repr=False)
    _click_y_override: float | None = field(init=False, default=None, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _gate_cache: np.ndarray | None = field(init=False, default=None, repr=False)
    _debug_step_counter: int = field(init=False, default=0, repr=False)

    def __post_init__(self):
        self._validate_params()

        self._state = np.zeros((self.grid.Ny, self.grid.Nx), dtype=float)
        self._clicked = False
        self._click_ix = None
        self._click_iy = None
        self._click_time = None
        self._click_x_override = None
        self._click_y_override = None
        self._rng = np.random.default_rng(self.detector_noise_seed)
        self._gate_cache = None
        self._debug_step_counter = 0

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def reset(self, noise_seed: int | None = None):
        self._state.fill(0.0)
        self._clicked = False
        self._click_ix = None
        self._click_iy = None
        self._click_time = None
        self._click_x_override = None
        self._click_y_override = None
        self._debug_step_counter = 0

        if noise_seed is not None:
            self.detector_noise_seed = noise_seed
        self._rng = np.random.default_rng(self.detector_noise_seed)

    def invalidate_gate_cache(self):
        self._gate_cache = None

    def set_gate(
        self,
        *,
        center_x: float | None = None,
        width_x: float | None = None,
        center_y: float | None = None,
        width_y: float | None = None,
    ):
        if center_x is not None:
            self.detector_gate_center_x = float(center_x)
        if width_x is not None:
            self.detector_gate_width = float(width_x)
        if center_y is not None:
            self.detector_gate_center_y = float(center_y)
        if width_y is not None:
            self.detector_gate_width_y = float(width_y)

        self._validate_params()
        self.invalidate_gate_cache()

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
        if self._click_ix is None or self._click_iy is None:
            return None
        x = self._click_x_override
        y = self._click_y_override
        if x is None:
            x = float(self.grid.X[self._click_iy, self._click_ix])
        if y is None:
            y = float(self.grid.Y[self._click_iy, self._click_ix])
        return x, y

    @property
    def click_time(self) -> float | None:
        return self._click_time

    def has_clicked(self) -> bool:
        return self._clicked

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def _validate_params(self):
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
        if self.detector_debug_every_n_steps < 1:
            raise ValueError("detector_debug_every_n_steps must be >= 1")
        if self.detector_mass <= 0.0:
            raise ValueError("detector_mass must be > 0")
        if self.detector_drive_mode not in ("flux_x_positive", "flux_x_abs", "rho"):
            raise ValueError("invalid detector_drive_mode")
        if self.detector_branch_selection_mode not in ("argmax", "sample_hotspots", "sample_patch"):
            raise ValueError("invalid detector_branch_selection_mode")
        if self.detector_hotspot_min_fraction_of_peak <= 0.0:
            raise ValueError("detector_hotspot_min_fraction_of_peak must be > 0")
        if self.detector_hotspot_separation_radius < 1:
            raise ValueError("detector_hotspot_separation_radius must be >= 1")
        if self.detector_click_patch_radius < 1:
            raise ValueError("detector_click_patch_radius must be >= 1")
        if self.detector_click_position_sigma_x < 0.0 or self.detector_click_position_sigma_y < 0.0:
            raise ValueError("click position sigmas must be >= 0")
        if self.detector_quantum_resolution_scale < 0.0:
            raise ValueError("detector_quantum_resolution_scale must be >= 0")
        if self.detector_k_floor <= 0.0:
            raise ValueError("detector_k_floor must be > 0")

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

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

    def _lateral_inhibition(self, D: np.ndarray) -> np.ndarray:
        radius = int(self.detector_competition_radius)
        size = 2 * radius + 1
        local_max = maximum_filter(D, size=(size, size), mode="nearest")
        inhibition = np.maximum(local_max - D, 0.0)
        return inhibition.astype(float)

    def _compute_rho(self, psi: np.ndarray) -> np.ndarray:
        return (np.abs(psi) ** 2).astype(float)

    def _compute_jx(self, psi: np.ndarray) -> np.ndarray:
        dpsi_dx = np.gradient(psi, float(self.grid.dx), axis=1)
        jx = (float(self.detector_hbar) / float(self.detector_mass)) * np.imag(
            np.conjugate(psi) * dpsi_dx
        )
        return jx.astype(float)

    def _compute_local_kx(self, psi: np.ndarray) -> np.ndarray:
        phase = np.angle(psi)
        dphi_dx = np.gradient(phase, float(self.grid.dx), axis=1)
        amp = np.abs(psi)
        mask = amp > 1e-12
        kx = np.zeros_like(dphi_dx, dtype=float)
        kx[mask] = dphi_dx[mask]
        return kx

    def _compute_drive(self, psi: np.ndarray, gate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho = self._compute_rho(psi)

        if self.detector_drive_mode == "rho":
            drive_base = rho
        else:
            jx = self._compute_jx(psi)
            if self.detector_drive_mode == "flux_x_positive":
                drive_base = np.maximum(jx, 0.0)
            else:
                drive_base = np.abs(jx)

        drive = self.detector_gain * gate * drive_base
        return drive.astype(float), rho.astype(float), drive_base.astype(float)

    def _find_hotspots(self, D: np.ndarray) -> list[tuple[int, int, float]]:
        peak = float(np.max(D))
        if peak <= 0.0:
            return []

        radius = int(self.detector_hotspot_separation_radius)
        size = 2 * radius + 1
        local_max = maximum_filter(D, size=(size, size), mode="nearest")
        thresh = float(self.detector_hotspot_min_fraction_of_peak) * peak

        mask = (D >= local_max - 1e-15) & (D >= thresh)
        ys, xs = np.where(mask)

        hotspots: list[tuple[int, int, float]] = []
        for iy, ix in zip(ys, xs):
            hotspots.append((int(iy), int(ix), float(D[iy, ix])))

        return hotspots

    def _sample_hotspot(self, D: np.ndarray) -> tuple[int, int]:
        hotspots = self._find_hotspots(D)
        if not hotspots:
            iy, ix = np.unravel_index(np.argmax(D), D.shape)
            return int(iy), int(ix)

        weights = np.array([max(h[2], 0.0) for h in hotspots], dtype=float)
        s = float(np.sum(weights))
        if s <= 0.0:
            iy, ix = np.unravel_index(np.argmax(D), D.shape)
            return int(iy), int(ix)

        probs = weights / s
        idx = int(self._rng.choice(len(hotspots), p=probs))
        iy, ix, _ = hotspots[idx]
        return iy, ix

    def _patch_bounds(self, iy: int, ix: int, r: int) -> tuple[int, int, int, int]:
        y0 = max(0, iy - r)
        y1 = min(self.grid.Ny, iy + r + 1)
        x0 = max(0, ix - r)
        x1 = min(self.grid.Nx, ix + r + 1)
        return y0, y1, x0, x1

    def _sample_from_patch(self, D: np.ndarray, iy0: int, ix0: int) -> tuple[int, int]:
        r = int(self.detector_click_patch_radius)
        y0, y1, x0, x1 = self._patch_bounds(iy0, ix0, r)

        patch = np.maximum(D[y0:y1, x0:x1], 0.0)
        s = float(np.sum(patch))
        if s <= 0.0:
            return iy0, ix0

        probs = (patch / s).ravel()
        flat_idx = int(self._rng.choice(probs.size, p=probs))
        py, px = np.unravel_index(flat_idx, patch.shape)
        return y0 + int(py), x0 + int(px)

    def _centroid_from_patch(self, D: np.ndarray, iy0: int, ix0: int) -> tuple[float, float]:
        r = int(self.detector_click_patch_radius)
        y0, y1, x0, x1 = self._patch_bounds(iy0, ix0, r)

        patch = np.maximum(D[y0:y1, x0:x1], 0.0)
        s = float(np.sum(patch))
        if s <= 0.0:
            return float(self.grid.X[iy0, ix0]), float(self.grid.Y[iy0, ix0])

        Xp = self.grid.X[y0:y1, x0:x1]
        Yp = self.grid.Y[y0:y1, x0:x1]

        x = float(np.sum(patch * Xp) / s)
        y = float(np.sum(patch * Yp) / s)
        return x, y

    def _apply_click_resolution(self, x: float, y: float, psi: np.ndarray, iy: int, ix: int) -> tuple[float, float]:
        sigma_x = float(self.detector_click_position_sigma_x)
        sigma_y = float(self.detector_click_position_sigma_y)

        if self.detector_use_quantum_min_resolution:
            kx_local = abs(float(self._compute_local_kx(psi)[iy, ix]))
            sigma_q = float(self.detector_quantum_resolution_scale) / max(kx_local, float(self.detector_k_floor))
            sigma_x = max(sigma_x, sigma_q)
            sigma_y = max(sigma_y, sigma_q)

        if sigma_x > 0.0:
            x = float(self._rng.normal(x, sigma_x))
        if sigma_y > 0.0:
            y = float(self._rng.normal(y, sigma_y))

        return x, y

    def _choose_click_site(self, D: np.ndarray, psi: np.ndarray) -> tuple[int, int, float, float]:
        if self.detector_branch_selection_mode == "argmax":
            iy, ix = np.unravel_index(np.argmax(D), D.shape)
            iy = int(iy)
            ix = int(ix)
        elif self.detector_branch_selection_mode == "sample_hotspots":
            iy, ix = self._sample_hotspot(D)
        elif self.detector_branch_selection_mode == "sample_patch":
            iy0, ix0 = np.unravel_index(np.argmax(D), D.shape)
            iy, ix = self._sample_from_patch(D, int(iy0), int(ix0))
        else:
            raise RuntimeError("Unexpected detector_branch_selection_mode")

        if self.detector_sample_click_position:
            x, y = self._centroid_from_patch(D, iy, ix)
            x, y = self._apply_click_resolution(x, y, psi, iy, ix)
        else:
            x = float(self.grid.X[iy, ix])
            y = float(self.grid.Y[iy, ix])

        return iy, ix, x, y

    def _is_valid_click_site(self, D: np.ndarray, iy: int, ix: int) -> bool:
        if not self.detector_require_local_max:
            return True

        radius = int(self.detector_competition_radius)
        y0 = max(0, iy - radius)
        y1 = min(D.shape[0], iy + radius + 1)
        x0 = max(0, ix - radius)
        x1 = min(D.shape[1], ix + radius + 1)

        patch = D[y0:y1, x0:x1]
        return D[iy, ix] >= float(np.max(patch)) - 1e-15

    def _build_result(
        self,
        gate: np.ndarray,
        drive: np.ndarray | None = None,
        inhibition: np.ndarray | None = None,
        rho: np.ndarray | None = None,
        drive_base: np.ndarray | None = None,
    ) -> DetectorStepResult:
        click_x = self._click_x_override
        click_y = self._click_y_override

        if click_x is None and self._click_ix is not None and self._click_iy is not None:
            click_x = float(self.grid.X[self._click_iy, self._click_ix])
        if click_y is None and self._click_ix is not None and self._click_iy is not None:
            click_y = float(self.grid.Y[self._click_iy, self._click_ix])

        aux = {
            "detector_state_max": float(np.max(self._state)),
            "detector_state_mean": float(np.mean(self._state)),
            "detector_gate_mean": float(np.mean(gate)),
            "detector_gate_max": float(np.max(gate)),
            "detector_drive_mean": 0.0 if drive is None else float(np.mean(drive)),
            "detector_drive_max": 0.0 if drive is None else float(np.max(drive)),
            "detector_inhibition_mean": 0.0 if inhibition is None else float(np.mean(inhibition)),
            "detector_inhibition_max": 0.0 if inhibition is None else float(np.max(inhibition)),
            "detector_drive_mode": self.detector_drive_mode,
            "detector_branch_selection_mode": self.detector_branch_selection_mode,
        }

        if rho is not None:
            rho_iy, rho_ix = np.unravel_index(np.argmax(rho), rho.shape)
            aux.update(
                {
                    "rho_max": float(np.max(rho)),
                    "rho_mean": float(np.mean(rho)),
                    "rho_peak_x": float(self.grid.X[rho_iy, rho_ix]),
                    "rho_peak_y": float(self.grid.Y[rho_iy, rho_ix]),
                    "gate_at_rho_peak": float(gate[rho_iy, rho_ix]),
                }
            )

        if drive_base is not None:
            db_iy, db_ix = np.unravel_index(np.argmax(drive_base), drive_base.shape)
            aux.update(
                {
                    "detector_source_max": float(np.max(drive_base)),
                    "detector_source_mean": float(np.mean(drive_base)),
                    "detector_source_peak_x": float(self.grid.X[db_iy, db_ix]),
                    "detector_source_peak_y": float(self.grid.Y[db_iy, db_ix]),
                    "gate_at_source_peak": float(gate[db_iy, db_ix]),
                }
            )

        if self._clicked:
            aux.update(
                {
                    "click_x_final": None if click_x is None else float(click_x),
                    "click_y_final": None if click_y is None else float(click_y),
                }
            )

        return DetectorStepResult(
            clicked=bool(self._clicked),
            click_ix=self._click_ix,
            click_iy=self._click_iy,
            click_x=click_x,
            click_y=click_y,
            click_time=self._click_time,
            aux=aux,
        )

    def _maybe_debug_print(self, *, t: float | None, dt: float, rho: np.ndarray, gate: np.ndarray, drive: np.ndarray, drive_base: np.ndarray):
        if not self.detector_debug:
            return

        self._debug_step_counter += 1
        if (self._debug_step_counter - 1) % self.detector_debug_every_n_steps != 0:
            return

        rho_iy, rho_ix = np.unravel_index(np.argmax(rho), rho.shape)
        rho_peak_x = float(self.grid.X[rho_iy, rho_ix])
        rho_peak_y = float(self.grid.Y[rho_iy, rho_ix])

        print(
            f"[EmergentDetector] "
            f"t={t} dt={dt:.6g} "
            f"mode={self.detector_drive_mode} "
            f"branch_mode={self.detector_branch_selection_mode} "
            f"rho_peak=(x={rho_peak_x:.6g}, y={rho_peak_y:.6g}) "
            f"gate_center_x={self.detector_gate_center_x:.6g} "
            f"source_max={float(np.max(drive_base)):.6g} "
            f"drive_max={float(np.max(drive)):.6g} "
            f"state_max={float(np.max(self._state)):.6g} "
            f"threshold={self.detector_click_threshold:.6g} "
            f"clicked={self._clicked}"
        )

    # --------------------------------------------------------
    # Main update
    # --------------------------------------------------------

    def step(self, psi: np.ndarray, dt: float, t: float | None = None) -> DetectorStepResult:
        if not np.isfinite(dt):
            raise ValueError(f"dt is not finite: {dt}")
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if psi.shape != self._state.shape:
            raise ValueError(f"psi shape {psi.shape} does not match detector state shape {self._state.shape}")

        gate = self._detector_gate()

        if self._clicked and self.detector_latch_click:
            return self._build_result(gate=gate)

        drive, rho, drive_base = self._compute_drive(psi, gate)

        leak = self.detector_leak * self._state
        inhibition = self.detector_competition_strength * self._lateral_inhibition(self._state)

        noise = 0.0
        if self.detector_noise_strength > 0.0:
            noise = (
                self.detector_noise_strength
                * np.sqrt(dt)
                * self._rng.standard_normal(self._state.shape)
                * gate
            )

        D_new = self._state + dt * (drive - leak - inhibition) + noise

        if self.detector_state_blur_sigma > 0.0:
            D_new = gaussian_filter(D_new, sigma=float(self.detector_state_blur_sigma), mode="nearest")

        if self.detector_nonnegative:
            D_new = np.maximum(D_new, 0.0)

        self._state = D_new.astype(float)

        self._maybe_debug_print(
            t=t,
            dt=dt,
            rho=rho,
            gate=gate,
            drive=drive,
            drive_base=drive_base,
        )

        peak = float(np.max(self._state))
        if peak >= float(self.detector_click_threshold):
            iy, ix, x_click, y_click = self._choose_click_site(self._state, psi)

            if self._is_valid_click_site(self._state, iy, ix):
                self._clicked = True
                self._click_ix = int(ix)
                self._click_iy = int(iy)
                self._click_x_override = float(x_click)
                self._click_y_override = float(y_click)
                self._click_time = None if t is None else float(t)

        return self._build_result(
            gate=gate,
            drive=drive,
            inhibition=inhibition,
            rho=rho,
            drive_base=drive_base,
        )