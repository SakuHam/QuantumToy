from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from theories.base import TheoryStepResult
from theories.schrodinger import SchrodingerTheory
from core.utils import normalize_unit


@dataclass
class ThickFrontOptimizedTheory(SchrodingerTheory):
    """
    Optimized thick-front theory.

    Base evolution:
        standard Schrödinger split-operator step

    Extra front operator:
        local coherence sharpening based on phase alignment with neighbors

    Added branch competition term:
        nearby weaker coherent branches are damped relative to stronger
        local branches, helping the dominant jet survive more selectively.
    """

    # --------------------------------------------------------
    # Original thick-front parameters
    # --------------------------------------------------------

    # Strength of thick-front sharpening per step
    front_strength: float = 0.03

    # Small damping on badly misaligned regions
    front_misaligned_damp: float = 0.01

    # Mix of axis neighbors vs diagonal neighbors
    front_diag_weight: float = 0.5

    # Use visible local density as extra weight
    front_density_weighted: bool = True

    # Small epsilon for stable division
    front_eps: float = 1e-12

    # Optional soft phase pull toward local coherent phase
    front_phase_relax_strength: float = 0.00

    # Clip exponent argument for stability
    front_clip: float = 0.25

    # --------------------------------------------------------
    # New branch competition / lateral inhibition parameters
    # --------------------------------------------------------

    # Main strength of branch competition damping.
    # 0.0 disables the new term.
    front_branch_competition_strength: float = 0.00

    # Nonlinear contrast for competition term
    front_branch_competition_power: float = 1.00

    # Gate competition by local positive alignment:
    # competition *= max(align_real, 0)^front_branch_gate_power
    front_branch_gate_power: float = 1.00

    # Weight of x-neighbor contribution in competition blur
    front_branch_competition_x_weight: float = 0.35

    # Weight of y-neighbor contribution in competition blur
    # Set larger than x-weight if you want stronger suppression of side-by-side jets.
    front_branch_competition_y_weight: float = 1.00

    # Weight of diagonal neighbors in competition blur
    front_branch_competition_diag_weight: float = 0.35

    # Exponent for gamma_like = rho^a * align_pos^b
    front_branch_density_power: float = 1.0
    front_branch_align_power: float = 1.0

    # Optional threshold before damping starts
    front_branch_competition_threshold: float = 0.00

    # If True, normalize gamma_like by its frame max before competition.
    # This makes the competition term depend more on relative branch strength.
    front_branch_normalize_gamma: bool = True

    # --------------------------------------------------------
    # Small internal validation
    # --------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()

        if self.front_eps <= 0.0:
            raise ValueError(f"front_eps must be > 0, got {self.front_eps}")
        if self.front_clip <= 0.0:
            raise ValueError(f"front_clip must be > 0, got {self.front_clip}")
        if self.front_branch_competition_strength < 0.0:
            raise ValueError(
                "front_branch_competition_strength must be >= 0"
            )
        if self.front_branch_competition_power < 0.0:
            raise ValueError(
                "front_branch_competition_power must be >= 0"
            )
        if self.front_branch_gate_power < 0.0:
            raise ValueError(
                "front_branch_gate_power must be >= 0"
            )
        if self.front_branch_competition_x_weight < 0.0:
            raise ValueError(
                "front_branch_competition_x_weight must be >= 0"
            )
        if self.front_branch_competition_y_weight < 0.0:
            raise ValueError(
                "front_branch_competition_y_weight must be >= 0"
            )
        if self.front_branch_competition_diag_weight < 0.0:
            raise ValueError(
                "front_branch_competition_diag_weight must be >= 0"
            )
        if self.front_branch_density_power < 0.0:
            raise ValueError("front_branch_density_power must be >= 0")
        if self.front_branch_align_power < 0.0:
            raise ValueError("front_branch_align_power must be >= 0")
        if self.front_branch_competition_threshold < 0.0:
            raise ValueError(
                "front_branch_competition_threshold must be >= 0"
            )

    # --------------------------------------------------------
    # Neighborhood helpers
    # --------------------------------------------------------

    def _neighbor_average_complex(self, z: np.ndarray) -> np.ndarray:
        """
        Local complex neighborhood average using nearest and diagonal neighbors.

        Uses periodic np.roll boundaries, matching the style already used elsewhere
        in the project for derivatives.
        """
        z_xp = np.roll(z, -1, axis=1)
        z_xm = np.roll(z,  1, axis=1)
        z_yp = np.roll(z, -1, axis=0)
        z_ym = np.roll(z,  1, axis=0)

        z_d1 = np.roll(z_xp, -1, axis=0)  # (+x, +y)
        z_d2 = np.roll(z_xp,  1, axis=0)  # (+x, -y)
        z_d3 = np.roll(z_xm, -1, axis=0)  # (-x, +y)
        z_d4 = np.roll(z_xm,  1, axis=0)  # (-x, -y)

        axis_sum = z_xp + z_xm + z_yp + z_ym
        diag_sum = z_d1 + z_d2 + z_d3 + z_d4

        w_axis = 1.0
        w_diag = float(self.front_diag_weight)

        denom = 4.0 * w_axis + 4.0 * w_diag
        return (w_axis * axis_sum + w_diag * diag_sum) / max(denom, self.front_eps)

    def _anisotropic_neighbor_average_real(self, arr: np.ndarray) -> np.ndarray:
        """
        Real anisotropic local average for branch competition.

        Designed so that y-neighbor coupling can be stronger than x-neighbor coupling,
        which is useful when competing jets tend to sit side-by-side in y.
        """
        arr = np.asarray(arr, dtype=float)

        a_xp = np.roll(arr, -1, axis=1)
        a_xm = np.roll(arr,  1, axis=1)
        a_yp = np.roll(arr, -1, axis=0)
        a_ym = np.roll(arr,  1, axis=0)

        a_d1 = np.roll(a_xp, -1, axis=0)
        a_d2 = np.roll(a_xp,  1, axis=0)
        a_d3 = np.roll(a_xm, -1, axis=0)
        a_d4 = np.roll(a_xm,  1, axis=0)

        wx = float(self.front_branch_competition_x_weight)
        wy = float(self.front_branch_competition_y_weight)
        wd = float(self.front_branch_competition_diag_weight)

        num = wx * (a_xp + a_xm) + wy * (a_yp + a_ym) + wd * (a_d1 + a_d2 + a_d3 + a_d4)
        den = 2.0 * wx + 2.0 * wy + 4.0 * wd

        if den <= self.front_eps:
            return arr.copy()

        return num / den

    # --------------------------------------------------------
    # Coherence / competition fields
    # --------------------------------------------------------

    def _coherence_alignment_score(self, psi: np.ndarray):
        """
        Returns:
            align_real : roughly in [-1, 1], phase alignment with neighborhood
            rho        : local density
            u_local    : normalized local coherent phase suggestion
        """
        rho = np.abs(psi) ** 2
        amp = np.sqrt(rho)

        u = psi / np.maximum(amp, self.front_eps)
        u_nei = self._neighbor_average_complex(u)

        if self.front_density_weighted:
            # Weight neighbor suggestion by local/nearby amplitude structure.
            amp_nei = self._neighbor_average_complex(amp.astype(np.complex128)).real
            amp_nei = np.maximum(amp_nei, 0.0)
            u_nei = u_nei * (1.0 + amp_nei)

        u_nei_mag = np.abs(u_nei)
        u_local = u_nei / np.maximum(u_nei_mag, self.front_eps)

        # Real part of phase agreement: +1 aligned, -1 opposite
        align_real = np.real(np.conjugate(u) * u_local)

        return align_real.astype(float), rho.astype(float), u_local

    def _branch_competition_field(
        self,
        rho: np.ndarray,
        align_real: np.ndarray,
    ):
        """
        Construct a competition field based on a local ridge-likeness measure.

        gamma_like is large when:
          - local density is large
          - phase alignment is positive and strong

        competition_raw > 0 means:
          neighboring branch evidence is stronger than local branch evidence
          -> damp this point

        A local alignment gate is also applied so that competition acts mainly
        where coherent branch structure already exists.
        """
        align_pos = np.maximum(align_real, 0.0)

        gamma_like = (
            np.power(np.maximum(rho, 0.0), float(self.front_branch_density_power))
            * np.power(np.maximum(align_pos, 0.0), float(self.front_branch_align_power))
        )

        if self.front_branch_normalize_gamma:
            gmax = float(np.max(gamma_like))
            if gmax > self.front_eps:
                gamma_like = gamma_like / gmax

        gamma_blur = self._anisotropic_neighbor_average_real(gamma_like)

        competition_raw = gamma_blur - gamma_like
        competition_raw = np.maximum(
            competition_raw - float(self.front_branch_competition_threshold),
            0.0,
        )

        p = float(self.front_branch_competition_power)
        if p != 1.0:
            competition_raw = competition_raw ** p

        # New gate: only strongly coherent / positively aligned regions
        # engage in strong branch competition.
        gpow = float(self.front_branch_gate_power)
        if gpow > 0.0:
            competition_gate = np.power(np.maximum(align_pos, 0.0), gpow)
            competition_raw = competition_raw * competition_gate
        else:
            competition_gate = np.ones_like(competition_raw, dtype=float)

        return (
            gamma_like.astype(float),
            gamma_blur.astype(float),
            competition_raw.astype(float),
            competition_gate.astype(float),
        )

    # --------------------------------------------------------
    # Front operator
    # --------------------------------------------------------

    def _front_sharpen(self, psi: np.ndarray, dt: float):
        """
        Apply local thick-front sharpening + optional branch competition.

        Mechanism:
        - aligned phase regions get slightly amplified
        - anti-aligned regions get slightly damped
        - optional branch competition damps locally weaker neighboring jets
        - competition is gated by local positive alignment
        - optional tiny phase relaxation toward local coherent phase
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

        gain_dt = np.clip(gain * dt, -self.front_clip, self.front_clip)
        psi_new = psi * np.exp(gain_dt)

        # ----------------------------------------------------
        # New branch competition / lateral inhibition
        # ----------------------------------------------------
        competition_raw = None
        competition_gate = None
        gamma_like = None
        gamma_blur = None

        if self.front_branch_competition_strength > 0.0:
            gamma_like, gamma_blur, competition_raw, competition_gate = self._branch_competition_field(
                rho=rho,
                align_real=align_real,
            )

            comp_dt = np.clip(
                float(self.front_branch_competition_strength) * competition_raw * dt,
                0.0,
                self.front_clip,
            )

            # Pure damping of locally weaker neighboring branches
            psi_new = psi_new * np.exp(-comp_dt)

        # ----------------------------------------------------
        # Optional tiny phase pull toward neighborhood coherent phase
        # ----------------------------------------------------
        if self.front_phase_relax_strength > 0.0:
            amp = np.abs(psi_new)
            u = psi_new / np.maximum(amp, self.front_eps)

            alpha = np.clip(self.front_phase_relax_strength * dt, 0.0, 1.0)
            u_mix = (1.0 - alpha) * u + alpha * u_local
            u_mix /= np.maximum(np.abs(u_mix), self.front_eps)

            psi_new = amp * u_mix

        aux_front = {
            "align_mean": float(np.mean(align_real)),
            "align_max": float(np.max(align_real)),
            "rho_mean": float(np.mean(rho)),
            "rho_max": float(np.max(rho)),
        }

        if competition_raw is not None:
            aux_front["branch_gamma_like_mean"] = float(np.mean(gamma_like))
            aux_front["branch_gamma_like_max"] = float(np.max(gamma_like))
            aux_front["branch_gamma_blur_mean"] = float(np.mean(gamma_blur))
            aux_front["branch_competition_mean"] = float(np.mean(competition_raw))
            aux_front["branch_competition_max"] = float(np.max(competition_raw))
            aux_front["branch_gate_mean"] = float(np.mean(competition_gate))
            aux_front["branch_gate_max"] = float(np.max(competition_gate))

        return psi_new.astype(np.complex128), aux_front

    # --------------------------------------------------------
    # Stepping
    # --------------------------------------------------------

    def step_forward(self, state: np.ndarray, dt: float) -> TheoryStepResult:
        """
        One forward step:
            Schrödinger step + optimized thick-front sharpening
            + optional branch competition / lateral inhibition
        """
        base = super().step_forward(state, dt)
        psi = base.state

        psi, aux_front = self._front_sharpen(psi, dt)
        psi, _ = normalize_unit(psi, self.grid.dx, self.grid.dy)

        aux = dict(base.aux) if base.aux is not None else {}
        aux["thick_front"] = {
            "front_strength": float(self.front_strength),
            "front_misaligned_damp": float(self.front_misaligned_damp),
            "front_diag_weight": float(self.front_diag_weight),
            "front_phase_relax_strength": float(self.front_phase_relax_strength),
            "front_branch_competition_strength": float(self.front_branch_competition_strength),
            "front_branch_competition_power": float(self.front_branch_competition_power),
            "front_branch_gate_power": float(self.front_branch_gate_power),
            "front_branch_competition_x_weight": float(self.front_branch_competition_x_weight),
            "front_branch_competition_y_weight": float(self.front_branch_competition_y_weight),
            "front_branch_competition_diag_weight": float(self.front_branch_competition_diag_weight),
            "front_branch_density_power": float(self.front_branch_density_power),
            "front_branch_align_power": float(self.front_branch_align_power),
            "front_branch_competition_threshold": float(self.front_branch_competition_threshold),
            "front_branch_normalize_gamma": bool(self.front_branch_normalize_gamma),
            **aux_front,
        }

        return TheoryStepResult(state=psi, aux=aux)

    def step_backward_adjoint(self, state: np.ndarray, dt: float) -> TheoryStepResult:
        """
        Backward evolution is kept as plain adjoint Schrödinger evolution.

        That is usually the safer choice for retrodictive / click-backward library use.
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