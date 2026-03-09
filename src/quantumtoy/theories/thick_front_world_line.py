from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from theories.base import TheoryStepResult
from theories.thick_front_optimized import ThickFrontOptimizedTheory
from core.utils import normalize_unit


@dataclass
class ThickFrontWorldLineTheory(ThickFrontOptimizedTheory):
    """
    Thick-front theory with a persistent worldline-style symmetry-breaking bias.

    This theory keeps ThickFrontOptimizedTheory intact and adds a separate
    worldline-oriented mechanism:

      - a smooth random bias field is generated once
      - the field stays fixed in time ("frozen")
      - it gives a tiny persistent preference to some branch-like regions
      - nearly symmetric competing jets can then resolve differently
        across runs, without frame-by-frame re-randomization

    This is not yet a full explicit branch-ID lock-in model.
    It is a first worldline-oriented extension that introduces persistent
    branch preference while keeping dynamics local and modular.
    """

    # --------------------------------------------------------
    # Worldline / frozen-bias mode
    # --------------------------------------------------------

    # Mode:
    #   "off"         -> disabled
    #   "gain"        -> tiny additive bias to local thick-front gain
    #   "competition" -> tiny modulation of local competition damping
    #   "both"        -> apply to both
    worldline_mode: str = "both"

    # Strength of frozen bias effect.
    # Keep small. Typical range might be 0.001 ... 0.02
    worldline_bias_strength: float = 0.05

    # Correlation length of random field in physical units.
    # Larger => smoother / broader bias structures.
    worldline_bias_sigma: float = 2.0 #1.0

    # Optional RNG seed for reproducible branch preference.
    worldline_bias_seed: int = 7, #| None = None

    # If True, bias mainly acts where branch-like structure exists.
    worldline_bias_gate_by_gamma: bool = True

    # Exponent for gamma gate
    worldline_bias_gamma_power: float = 1.0

    # Extra alignment gate exponent
    worldline_bias_align_power: float = 1.0

    # Optional global scale for the gate itself
    worldline_bias_gate_scale: float = 1.0

    # --------------------------------------------------------
    # Internal state
    # --------------------------------------------------------

    _worldline_rng: np.random.Generator = field(init=False, repr=False)
    _worldline_bias_field: np.ndarray | None = field(
        init=False, default=None, repr=False
    )

    # --------------------------------------------------------
    # Init / validation
    # --------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()

        allowed_modes = {"off", "gain", "competition", "both"}
        if self.worldline_mode not in allowed_modes:
            raise ValueError(
                f"worldline_mode must be one of {allowed_modes}, "
                f"got {self.worldline_mode!r}"
            )

        if self.worldline_bias_strength < 0.0:
            raise ValueError("worldline_bias_strength must be >= 0")
        if self.worldline_bias_sigma < 0.0:
            raise ValueError("worldline_bias_sigma must be >= 0")
        if self.worldline_bias_gamma_power < 0.0:
            raise ValueError("worldline_bias_gamma_power must be >= 0")
        if self.worldline_bias_align_power < 0.0:
            raise ValueError("worldline_bias_align_power must be >= 0")
        if self.worldline_bias_gate_scale < 0.0:
            raise ValueError("worldline_bias_gate_scale must be >= 0")

        self._worldline_rng = np.random.default_rng(self.worldline_bias_seed)
        self._worldline_bias_field = None

    # --------------------------------------------------------
    # Public helper
    # --------------------------------------------------------

    def reset_worldline_bias(self, seed: int | None = None):
        """
        Reset the frozen worldline bias field.

        Use this between runs if you want a new random persistent preference.
        If seed is provided, it also replaces the stored seed.
        """
        if seed is not None:
            self.worldline_bias_seed = seed

        self._worldline_rng = np.random.default_rng(self.worldline_bias_seed)
        self._worldline_bias_field = None

    # --------------------------------------------------------
    # Frozen bias helpers
    # --------------------------------------------------------

    def _fft_gaussian_smooth_periodic(
        self,
        arr: np.ndarray,
        sigma_x: float,
        sigma_y: float,
    ) -> np.ndarray:
        """
        Periodic Gaussian smoothing via FFT.

        sigma_x and sigma_y are in physical units.
        """
        arr = np.asarray(arr, dtype=float)

        if sigma_x <= self.front_eps and sigma_y <= self.front_eps:
            return arr.copy()

        kx = 2.0 * np.pi * np.fft.fftfreq(self.grid.Nx, d=self.grid.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.grid.Ny, d=self.grid.dy)
        KX, KY = np.meshgrid(kx, ky)

        filt = np.exp(
            -0.5 * (
                (sigma_x ** 2) * (KX ** 2)
                + (sigma_y ** 2) * (KY ** 2)
            )
        )

        arr_k = np.fft.fft2(arr)
        smoothed = np.fft.ifft2(arr_k * filt).real
        return smoothed.astype(float)

    def _build_worldline_bias_field(self) -> np.ndarray:
        """
        Build a smooth zero-mean unit-std frozen random field.

        The field is generated once and then kept fixed over time.
        """
        raw = self._worldline_rng.standard_normal((self.grid.Ny, self.grid.Nx))

        sigma = float(self.worldline_bias_sigma)
        bias = self._fft_gaussian_smooth_periodic(
            raw,
            sigma_x=sigma,
            sigma_y=sigma,
        )

        bias -= float(np.mean(bias))
        std = float(np.std(bias))

        if std > self.front_eps:
            bias = bias / std
        else:
            bias = np.zeros_like(bias, dtype=float)

        return bias.astype(float)

    def _get_worldline_bias_field(self) -> np.ndarray | None:
        if self.worldline_mode == "off":
            return None
        if self.worldline_bias_strength <= 0.0:
            return None

        if self._worldline_bias_field is None:
            self._worldline_bias_field = self._build_worldline_bias_field()

        return self._worldline_bias_field

    def _worldline_bias_gate(
        self,
        gamma_like: np.ndarray | None,
        align_real: np.ndarray,
    ) -> np.ndarray:
        """
        Gate the frozen bias so that it mainly acts on meaningful
        branch-like regions, not uniformly everywhere.
        """
        gate = np.ones_like(align_real, dtype=float)

        if self.worldline_bias_gate_by_gamma and gamma_like is not None:
            g = np.maximum(gamma_like, 0.0)
            gp = float(self.worldline_bias_gamma_power)
            if gp != 1.0:
                g = g ** gp
            gate *= g

        ap = float(self.worldline_bias_align_power)
        if ap > 0.0:
            align_pos = np.maximum(align_real, 0.0)
            if ap != 1.0:
                align_pos = align_pos ** ap
            gate *= align_pos

        if self.worldline_bias_gate_scale != 1.0:
            gate *= float(self.worldline_bias_gate_scale)

        return gate.astype(float)

    # --------------------------------------------------------
    # Front operator override
    # --------------------------------------------------------

    def _front_sharpen(self, psi: np.ndarray, dt: float):
        """
        Apply local thick-front sharpening + optional branch competition
        + optional persistent worldline frozen bias.
        """
        align_real, rho, u_local = self._coherence_alignment_score(psi)

        # ----------------------------------------------------
        # Original thick-front sharpening
        # ----------------------------------------------------
        gain = (
            self.front_strength * np.maximum(align_real, 0.0)
            - self.front_misaligned_damp * np.maximum(-align_real, 0.0)
        )

        if self.front_density_weighted:
            rho_mean = float(np.mean(rho))
            if rho_mean > self.front_eps:
                rho_scale = rho / rho_mean
                gain = gain * np.sqrt(np.maximum(rho_scale, 0.0))

        # ----------------------------------------------------
        # Branch competition field from parent logic
        # ----------------------------------------------------
        competition_raw = None
        competition_gate = None
        gamma_like = None
        gamma_blur = None

        if self.front_branch_competition_strength > 0.0:
            gamma_like, gamma_blur, competition_raw, competition_gate = (
                self._branch_competition_field(
                    rho=rho,
                    align_real=align_real,
                )
            )

        # ----------------------------------------------------
        # If worldline bias is active but competition is off,
        # we still need a branch-likeness field for gating
        # ----------------------------------------------------
        if gamma_like is None:
            align_pos = np.maximum(align_real, 0.0)
            gamma_like = (
                np.power(np.maximum(rho, 0.0), float(self.front_branch_density_power))
                * np.power(
                    np.maximum(align_pos, 0.0),
                    float(self.front_branch_align_power),
                )
            )

            if self.front_branch_normalize_gamma:
                gmax = float(np.max(gamma_like))
                if gmax > self.front_eps:
                    gamma_like = gamma_like / gmax

        # ----------------------------------------------------
        # Worldline frozen bias
        # ----------------------------------------------------
        bias_field = self._get_worldline_bias_field()
        bias_gate = None
        bias_effect = None

        if bias_field is not None:
            bias_gate = self._worldline_bias_gate(
                gamma_like=gamma_like,
                align_real=align_real,
            )

            bias_effect = (
                float(self.worldline_bias_strength)
                * bias_field
                * bias_gate
            )

            # Positive bias slightly favors local growth
            if self.worldline_mode in {"gain", "both"}:
                gain = gain + bias_effect

            # Positive bias slightly reduces competition damping;
            # negative bias slightly increases it.
            if (
                self.worldline_mode in {"competition", "both"}
                and competition_raw is not None
            ):
                comp_factor = 1.0 - bias_effect
                comp_factor = np.clip(comp_factor, 0.0, 2.0)
                competition_raw = competition_raw * comp_factor

        # ----------------------------------------------------
        # Apply gain
        # ----------------------------------------------------
        gain_dt = np.clip(gain * dt, -self.front_clip, self.front_clip)
        psi_new = psi * np.exp(gain_dt)

        # ----------------------------------------------------
        # Apply branch competition damping
        # ----------------------------------------------------
        if (
            self.front_branch_competition_strength > 0.0
            and competition_raw is not None
        ):
            comp_dt = np.clip(
                float(self.front_branch_competition_strength) * competition_raw * dt,
                0.0,
                self.front_clip,
            )
            psi_new = psi_new * np.exp(-comp_dt)

        # ----------------------------------------------------
        # Optional tiny phase relaxation
        # ----------------------------------------------------
        if self.front_phase_relax_strength > 0.0:
            amp = np.abs(psi_new)
            u = psi_new / np.maximum(amp, self.front_eps)

            alpha = np.clip(self.front_phase_relax_strength * dt, 0.0, 1.0)
            u_mix = (1.0 - alpha) * u + alpha * u_local
            u_mix /= np.maximum(np.abs(u_mix), self.front_eps)

            psi_new = amp * u_mix

        # ----------------------------------------------------
        # Aux diagnostics
        # ----------------------------------------------------
        aux_front = {
            "align_mean": float(np.mean(align_real)),
            "align_max": float(np.max(align_real)),
            "rho_mean": float(np.mean(rho)),
            "rho_max": float(np.max(rho)),
            "branch_gamma_like_mean": float(np.mean(gamma_like)),
            "branch_gamma_like_max": float(np.max(gamma_like)),
        }

        if gamma_blur is not None:
            aux_front["branch_gamma_blur_mean"] = float(np.mean(gamma_blur))
            aux_front["branch_gamma_blur_max"] = float(np.max(gamma_blur))

        if competition_raw is not None:
            aux_front["branch_competition_mean"] = float(np.mean(competition_raw))
            aux_front["branch_competition_max"] = float(np.max(competition_raw))

        if competition_gate is not None:
            aux_front["branch_gate_mean"] = float(np.mean(competition_gate))
            aux_front["branch_gate_max"] = float(np.max(competition_gate))

        if bias_field is not None:
            aux_front["worldline_bias_mean"] = float(np.mean(bias_field))
            aux_front["worldline_bias_std"] = float(np.std(bias_field))
            aux_front["worldline_bias_min"] = float(np.min(bias_field))
            aux_front["worldline_bias_max"] = float(np.max(bias_field))

        if bias_gate is not None:
            aux_front["worldline_bias_gate_mean"] = float(np.mean(bias_gate))
            aux_front["worldline_bias_gate_max"] = float(np.max(bias_gate))

        if bias_effect is not None:
            aux_front["worldline_bias_effect_mean"] = float(np.mean(bias_effect))
            aux_front["worldline_bias_effect_abs_mean"] = float(
                np.mean(np.abs(bias_effect))
            )
            aux_front["worldline_bias_effect_min"] = float(np.min(bias_effect))
            aux_front["worldline_bias_effect_max"] = float(np.max(bias_effect))

        return psi_new.astype(np.complex128), aux_front

    # --------------------------------------------------------
    # Forward step override
    # --------------------------------------------------------

    def step_forward(self, state: np.ndarray, dt: float) -> TheoryStepResult:
        """
        One forward step:
            Schrödinger step
            + thick-front sharpening
            + optional deterministic branch competition
            + optional frozen worldline bias
        """
        base = super(ThickFrontOptimizedTheory, self).step_forward(state, dt)
        psi = base.state

        psi, aux_front = self._front_sharpen(psi, dt)
        psi, _ = normalize_unit(psi, self.grid.dx, self.grid.dy)

        aux = dict(base.aux) if base.aux is not None else {}
        aux["thick_front_worldline"] = {
            "front_strength": float(self.front_strength),
            "front_misaligned_damp": float(self.front_misaligned_damp),
            "front_diag_weight": float(self.front_diag_weight),
            "front_phase_relax_strength": float(self.front_phase_relax_strength),
            "front_branch_competition_strength": float(
                self.front_branch_competition_strength
            ),
            "front_branch_competition_power": float(
                self.front_branch_competition_power
            ),
            "front_branch_gate_power": float(self.front_branch_gate_power),
            "front_branch_competition_x_weight": float(
                self.front_branch_competition_x_weight
            ),
            "front_branch_competition_y_weight": float(
                self.front_branch_competition_y_weight
            ),
            "front_branch_competition_diag_weight": float(
                self.front_branch_competition_diag_weight
            ),
            "front_branch_density_power": float(self.front_branch_density_power),
            "front_branch_align_power": float(self.front_branch_align_power),
            "front_branch_competition_threshold": float(
                self.front_branch_competition_threshold
            ),
            "front_branch_normalize_gamma": bool(
                self.front_branch_normalize_gamma
            ),
            "worldline_mode": str(self.worldline_mode),
            "worldline_bias_strength": float(self.worldline_bias_strength),
            "worldline_bias_sigma": float(self.worldline_bias_sigma),
            "worldline_bias_seed": self.worldline_bias_seed,
            "worldline_bias_gate_by_gamma": bool(
                self.worldline_bias_gate_by_gamma
            ),
            "worldline_bias_gamma_power": float(self.worldline_bias_gamma_power),
            "worldline_bias_align_power": float(self.worldline_bias_align_power),
            "worldline_bias_gate_scale": float(self.worldline_bias_gate_scale),
            **aux_front,
        }

        return TheoryStepResult(state=psi, aux=aux)

    # --------------------------------------------------------
    # Backward evolution
    # --------------------------------------------------------

    def step_backward_adjoint(self, state: np.ndarray, dt: float) -> TheoryStepResult:
        """
        Backward evolution is kept as plain adjoint Schrödinger evolution.
        """
        return super().step_backward_adjoint(state, dt)

    # --------------------------------------------------------
    # Observables
    # --------------------------------------------------------

    def density(self, state: np.ndarray) -> np.ndarray:
        return (np.abs(state) ** 2).astype(float)

    def current(self, state_vis: np.ndarray):
        """
        Same current definition as Schrödinger.
        """
        dpsi_dx = (
            np.roll(state_vis, -1, axis=1) - np.roll(state_vis, 1, axis=1)
        ) / (2.0 * self.grid.dx)

        dpsi_dy = (
            np.roll(state_vis, -1, axis=0) - np.roll(state_vis, 1, axis=0)
        ) / (2.0 * self.grid.dy)

        rho = (np.abs(state_vis) ** 2).astype(float)

        jx = (
            (self.hbar / self.m_mass)
            * np.imag(np.conjugate(state_vis) * dpsi_dx)
        ).astype(float)

        jy = (
            (self.hbar / self.m_mass)
            * np.imag(np.conjugate(state_vis) * dpsi_dy)
        ).astype(float)

        return jx, jy, rho