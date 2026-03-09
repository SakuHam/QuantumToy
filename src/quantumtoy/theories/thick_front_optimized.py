from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import maximum_filter, gaussian_filter

from theories.base import TheoryStepResult
from theories.schrodinger import SchrodingerTheory
from core.utils import normalize_unit


# ============================================================
# Small local validation helpers
# ============================================================

def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def _assert_finite_scalar(x, name: str):
    _assert(np.isscalar(x), f"{name} must be scalar, got {type(x)}")
    xf = float(x)
    _assert(np.isfinite(xf), f"{name} must be finite, got {x}")
    return xf


def _assert_positive_scalar(x, name: str):
    xf = _assert_finite_scalar(x, name)
    _assert(xf > 0.0, f"{name} must be > 0, got {x}")
    return xf


def _assert_complex_array_2d(arr: np.ndarray, name: str):
    _assert(isinstance(arr, np.ndarray), f"{name} must be np.ndarray")
    _assert(arr.ndim == 2, f"{name} must be 2D, got ndim={arr.ndim}")
    _assert(np.all(np.isfinite(arr.real)), f"{name}.real contains non-finite values")
    _assert(np.all(np.isfinite(arr.imag)), f"{name}.imag contains non-finite values")


def _assert_real_array_2d(arr: np.ndarray, name: str):
    _assert(isinstance(arr, np.ndarray), f"{name} must be np.ndarray")
    _assert(arr.ndim == 2, f"{name} must be 2D, got ndim={arr.ndim}")
    _assert(np.all(np.isfinite(arr)), f"{name} contains non-finite values")


@dataclass
class ThickFrontOptimizedTheory(SchrodingerTheory):
    """
    Optimized thick-front theory.

    Base evolution:
        standard Schrödinger split-operator step

    Extra front operator:
        local coherence sharpening based on phase alignment with neighbors

    Branch competition:
        nearby weaker coherent branches are damped relative to stronger
        nearby branches.

    Important updates in this version:
        1) competition is evaluated from psi_tmp, i.e. AFTER the local front
           sharpening step
        2) competition uses full-resolution scipy maximum_filter
        3) optional Gaussian smoothing is applied to gain and competition fields
           to reduce grid anisotropy / square artifacts
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

    # Optional Gaussian blur on gain before exponentiation
    # Helps suppress grid-shaped artifacts and over-sharpening.
    front_gain_blur_sigma: float = 0.0

    # --------------------------------------------------------
    # Branch competition / lateral inhibition parameters
    # --------------------------------------------------------

    # Main strength of branch competition damping.
    # 0.0 disables the new term.
    front_branch_competition_strength: float = 0.00

    # Nonlinear contrast for competition term
    front_branch_competition_power: float = 1.00

    # Gate competition by local positive alignment:
    # competition *= max(align_real, 0)^front_branch_gate_power
    front_branch_gate_power: float = 1.00

    # Kept for compatibility / possible future variants
    front_branch_competition_x_weight: float = 0.35
    front_branch_competition_y_weight: float = 1.00
    front_branch_competition_diag_weight: float = 0.35

    # Exponent for gamma_like = rho^a * align_pos^b
    front_branch_density_power: float = 1.0
    front_branch_align_power: float = 1.0

    # Optional threshold before damping starts
    front_branch_competition_threshold: float = 0.00

    # If True, normalize gamma_like by its frame max before competition.
    front_branch_normalize_gamma: bool = True

    # --------------------------------------------------------
    # Full-resolution SciPy competition parameters
    # --------------------------------------------------------

    # Competition search radius in full-resolution pixels
    front_branch_competition_radius: int = 20

    # Local branch is safe if it is within this factor of the strongest
    # nearby competitor.
    # margin > 1 also suppresses self-competition even if center is included
    # in the maximum filter.
    front_branch_competition_margin: float = 1.01

    # Optional Gaussian blur on competition_raw before comp_dt
    # Helps expand extremely sparse competition and reduce blocky/square artifacts.
    front_branch_competition_blur_sigma: float = 0.0

    # --------------------------------------------------------
    # Debug / safety parameters
    # --------------------------------------------------------

    front_debug_checks: bool = True
    front_norm_tol: float = 1e-8

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()

        np.seterr(divide="raise", over="raise", invalid="raise")

        self.front_eps = _assert_positive_scalar(self.front_eps, "front_eps")
        self.front_clip = _assert_positive_scalar(self.front_clip, "front_clip")
        self.front_norm_tol = _assert_positive_scalar(self.front_norm_tol, "front_norm_tol")

        _assert(self.front_strength >= 0.0, "front_strength must be >= 0")
        _assert(self.front_misaligned_damp >= 0.0, "front_misaligned_damp must be >= 0")
        _assert(self.front_diag_weight >= 0.0, "front_diag_weight must be >= 0")
        _assert(self.front_phase_relax_strength >= 0.0, "front_phase_relax_strength must be >= 0")
        _assert(self.front_gain_blur_sigma >= 0.0, "front_gain_blur_sigma must be >= 0")

        _assert(self.front_branch_competition_strength >= 0.0,
                "front_branch_competition_strength must be >= 0")
        _assert(self.front_branch_competition_power >= 0.0,
                "front_branch_competition_power must be >= 0")
        _assert(self.front_branch_gate_power >= 0.0,
                "front_branch_gate_power must be >= 0")

        _assert(self.front_branch_competition_x_weight >= 0.0,
                "front_branch_competition_x_weight must be >= 0")
        _assert(self.front_branch_competition_y_weight >= 0.0,
                "front_branch_competition_y_weight must be >= 0")
        _assert(self.front_branch_competition_diag_weight >= 0.0,
                "front_branch_competition_diag_weight must be >= 0")

        _assert(self.front_branch_density_power >= 0.0,
                "front_branch_density_power must be >= 0")
        _assert(self.front_branch_align_power >= 0.0,
                "front_branch_align_power must be >= 0")
        _assert(self.front_branch_competition_threshold >= 0.0,
                "front_branch_competition_threshold must be >= 0")

        _assert(isinstance(self.front_branch_competition_radius, int),
                "front_branch_competition_radius must be int")
        _assert(self.front_branch_competition_radius >= 1,
                "front_branch_competition_radius must be >= 1")

        _assert(self.front_branch_competition_margin > 0.0,
                "front_branch_competition_margin must be > 0")
        _assert(self.front_branch_competition_blur_sigma >= 0.0,
                "front_branch_competition_blur_sigma must be >= 0")

    # --------------------------------------------------------
    # Basic utilities
    # --------------------------------------------------------

    def _state_probability(self, psi: np.ndarray) -> float:
        _assert_complex_array_2d(psi, "psi")
        prob = float(np.sum(np.abs(psi) ** 2) * self.grid.dx * self.grid.dy)
        _assert(np.isfinite(prob), "state probability is non-finite")
        _assert(prob >= 0.0, f"state probability must be >= 0, got {prob}")
        return prob

    def _neighbor_average_complex(self, z: np.ndarray) -> np.ndarray:
        """
        Local complex neighborhood average using nearest and diagonal neighbors.

        Uses periodic np.roll boundaries, matching the style already used elsewhere
        in the project for derivatives.
        """
        _assert_complex_array_2d(z, "z")

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
        out = (w_axis * axis_sum + w_diag * diag_sum) / max(denom, self.front_eps)
        _assert_complex_array_2d(out, "_neighbor_average_complex(out)")
        return out

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
        _assert_complex_array_2d(psi, "psi")

        rho = np.abs(psi) ** 2
        amp = np.sqrt(rho)

        _assert_real_array_2d(rho, "rho")
        _assert_real_array_2d(amp, "amp")

        u = psi / np.maximum(amp, self.front_eps)
        _assert_complex_array_2d(u, "u")

        u_nei = self._neighbor_average_complex(u)

        if self.front_density_weighted:
            amp_nei = self._neighbor_average_complex(amp.astype(np.complex128)).real
            amp_nei = np.maximum(amp_nei, 0.0)
            u_nei = u_nei * (1.0 + amp_nei)

        u_nei_mag = np.abs(u_nei)
        u_local = u_nei / np.maximum(u_nei_mag, self.front_eps)

        # Real part of phase agreement: +1 aligned, -1 opposite
        align_real = np.real(np.conjugate(u) * u_local)

        _assert_real_array_2d(align_real, "align_real")
        _assert_complex_array_2d(u_local, "u_local")

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

        Competition is computed directly on the full-resolution grid:
          1) build gamma_like
          2) find strongest nearby competitor via scipy maximum_filter
          3) compute competition on the full grid
          4) optionally smooth competition with a Gaussian
        """
        _assert_real_array_2d(rho, "rho")
        _assert_real_array_2d(align_real, "align_real")
        _assert(rho.shape == align_real.shape, "rho and align_real must have same shape")

        align_pos = np.maximum(align_real, 0.0)

        gamma_like = (
            np.power(np.maximum(rho, 0.0), float(self.front_branch_density_power))
            * np.power(np.maximum(align_pos, 0.0), float(self.front_branch_align_power))
        ).astype(float)

        if self.front_branch_normalize_gamma:
            gmax = float(np.max(gamma_like))
            if gmax > self.front_eps:
                gamma_like = gamma_like / gmax

        _assert_real_array_2d(gamma_like, "gamma_like")
        _assert(np.all(gamma_like >= -1e-14), "gamma_like contains significantly negative values")

        filt_size = 2 * int(self.front_branch_competition_radius) + 1
        neighbor_max = maximum_filter(
            gamma_like,
            size=(filt_size, filt_size),
            mode="wrap",
        ).astype(float)

        _assert_real_array_2d(neighbor_max, "neighbor_max")

        competition_raw = np.maximum(
            neighbor_max - float(self.front_branch_competition_margin) * gamma_like,
            0.0,
        )

        competition_raw = np.maximum(
            competition_raw - float(self.front_branch_competition_threshold),
            0.0,
        )

        p = float(self.front_branch_competition_power)
        if p != 1.0:
            competition_raw = competition_raw ** p

        # Optional smoothing to reduce square/grid artifacts and expand sparse masks
        if self.front_branch_competition_blur_sigma > 0.0:
            competition_raw = gaussian_filter(
                competition_raw,
                sigma=float(self.front_branch_competition_blur_sigma),
                mode="wrap",
            )

        # Gate competition by local positive alignment
        gpow = float(self.front_branch_gate_power)
        if gpow > 0.0:
            competition_gate = np.power(np.maximum(align_pos, 0.0), gpow)
            competition_raw = competition_raw * competition_gate
        else:
            competition_gate = np.ones_like(competition_raw, dtype=float)

        _assert_real_array_2d(competition_raw, "competition_raw")
        _assert_real_array_2d(competition_gate, "competition_gate")
        _assert(np.all(competition_raw >= -1e-14),
                "competition_raw contains significantly negative values")

        return (
            gamma_like.astype(float),
            neighbor_max.astype(float),
            competition_raw.astype(float),
            competition_gate.astype(float),
        )

    # --------------------------------------------------------
    # Front operator
    # --------------------------------------------------------

    def _front_sharpen(self, psi: np.ndarray, dt: float):
        """
        Apply local thick-front sharpening + optional branch competition.

        Updated order:
        1) compute coherence from psi
        2) build gain
        3) optionally smooth gain
        4) form psi_tmp = sharpened field
        5) recompute coherence from psi_tmp
        6) compute competition from psi_tmp-derived fields
        7) optionally smooth competition
        8) apply competition damping
        9) optional phase relaxation
        """
        _assert_complex_array_2d(psi, "psi")
        _assert_finite_scalar(dt, "dt")

        # --------------------------------------------
        # Diagnostics on input norm
        # --------------------------------------------
        prob_in = self._state_probability(psi)

        # --------------------------------------------
        # Coherence from original psi
        # --------------------------------------------
        align_real, rho, u_local = self._coherence_alignment_score(psi)

        gain = (
            self.front_strength * np.maximum(align_real, 0.0)
            - self.front_misaligned_damp * np.maximum(-align_real, 0.0)
        )

        if self.front_density_weighted:
            rho_mean = float(np.mean(rho))
            if rho_mean > self.front_eps:
                rho_scale = rho / rho_mean
                gain = gain * np.sqrt(np.maximum(rho_scale, 0.0))

        # Optional smoothing to reduce grid anisotropy and explosive local growth
        if self.front_gain_blur_sigma > 0.0:
            gain = gaussian_filter(
                gain,
                sigma=float(self.front_gain_blur_sigma),
                mode="wrap",
            )

        _assert_real_array_2d(gain, "gain")

        gain_dt = np.clip(gain * dt, -self.front_clip, self.front_clip)
        _assert_real_array_2d(gain_dt, "gain_dt")

        # --------------------------------------------
        # First stage: sharpening
        # --------------------------------------------
        psi_tmp = psi * np.exp(gain_dt)
        _assert_complex_array_2d(psi_tmp, "psi_tmp")

        prob_after_gain = self._state_probability(psi_tmp)

        # --------------------------------------------
        # Recompute coherence from sharpened psi_tmp
        # --------------------------------------------
        align_real_tmp, rho_tmp, u_local_tmp = self._coherence_alignment_score(psi_tmp)

        competition_raw = None
        competition_gate = None
        gamma_like = None
        neighbor_max = None
        comp_dt = None

        # --------------------------------------------
        # Second stage: competition from psi_tmp
        # --------------------------------------------
        if self.front_branch_competition_strength > 0.0:
            gamma_like, neighbor_max, competition_raw, competition_gate = self._branch_competition_field(
                rho=rho_tmp,
                align_real=align_real_tmp,
            )

            comp_dt = np.clip(
                float(self.front_branch_competition_strength) * competition_raw * dt,
                0.0,
                self.front_clip,
            )
            _assert_real_array_2d(comp_dt, "comp_dt")

            psi_new = psi_tmp * np.exp(-comp_dt)
        else:
            psi_new = psi_tmp

        _assert_complex_array_2d(psi_new, "psi_new(after competition)")
        prob_after_comp = self._state_probability(psi_new)

        # --------------------------------------------
        # Optional tiny phase pull toward sharpened local coherent phase
        # --------------------------------------------
        if self.front_phase_relax_strength > 0.0:
            amp = np.abs(psi_new)
            u = psi_new / np.maximum(amp, self.front_eps)

            alpha = np.clip(self.front_phase_relax_strength * dt, 0.0, 1.0)
            u_mix = (1.0 - alpha) * u + alpha * u_local_tmp
            u_mix /= np.maximum(np.abs(u_mix), self.front_eps)

            psi_new = amp * u_mix
            _assert_complex_array_2d(psi_new, "psi_new(after phase relax)")

        prob_after_relax = self._state_probability(psi_new)

        if self.front_debug_checks:
            _assert(prob_in > 0.0, f"input norm must be > 0, got {prob_in}")
            _assert(prob_after_gain > 0.0, f"prob_after_gain must be > 0, got {prob_after_gain}")
            _assert(prob_after_comp > 0.0, f"prob_after_comp must be > 0, got {prob_after_comp}")
            _assert(prob_after_relax > 0.0, f"prob_after_relax must be > 0, got {prob_after_relax}")

            # Competition should not increase norm if it is pure damping.
            if self.front_branch_competition_strength > 0.0:
                _assert(
                    prob_after_comp <= prob_after_gain + 1e-12,
                    f"competition should not increase norm: after_comp={prob_after_comp}, after_gain={prob_after_gain}"
                )

        aux_front = {
            "align_mean_pre": float(np.mean(align_real)),
            "align_max_pre": float(np.max(align_real)),
            "rho_mean_pre": float(np.mean(rho)),
            "rho_max_pre": float(np.max(rho)),

            "align_mean_postgain": float(np.mean(align_real_tmp)),
            "align_max_postgain": float(np.max(align_real_tmp)),
            "rho_mean_postgain": float(np.mean(rho_tmp)),
            "rho_max_postgain": float(np.max(rho_tmp)),

            "prob_in": float(prob_in),
            "prob_after_gain": float(prob_after_gain),
            "prob_after_comp": float(prob_after_comp),
            "prob_after_relax": float(prob_after_relax),

            "gain_dt_mean": float(np.mean(gain_dt)),
            "gain_dt_max": float(np.max(gain_dt)),
        }

        if competition_raw is not None:
            aux_front["branch_gamma_like_mean"] = float(np.mean(gamma_like))
            aux_front["branch_gamma_like_max"] = float(np.max(gamma_like))
            aux_front["branch_neighbor_max_mean"] = float(np.mean(neighbor_max))
            aux_front["branch_neighbor_max_max"] = float(np.max(neighbor_max))
            aux_front["branch_competition_mean"] = float(np.mean(competition_raw))
            aux_front["branch_competition_max"] = float(np.max(competition_raw))
            aux_front["branch_gate_mean"] = float(np.mean(competition_gate))
            aux_front["branch_gate_max"] = float(np.max(competition_gate))
            aux_front["branch_comp_dt_mean"] = float(np.mean(comp_dt))
            aux_front["branch_comp_dt_max"] = float(np.max(comp_dt))

        return psi_new.astype(np.complex128), aux_front

    # --------------------------------------------------------
    # Stepping
    # --------------------------------------------------------

    def step_forward(self, state: np.ndarray, dt: float) -> TheoryStepResult:
        """
        One forward step:
            Schrödinger step + optimized thick-front sharpening
            + optional branch competition / lateral inhibition
            + final renormalization
        """
        _assert_complex_array_2d(state, "state")
        _assert_finite_scalar(dt, "dt")

        base = super().step_forward(state, dt)
        psi = base.state
        _assert_complex_array_2d(psi, "base.state")

        prob_after_base = self._state_probability(psi)

        psi, aux_front = self._front_sharpen(psi, dt)
        _assert_complex_array_2d(psi, "psi(after _front_sharpen)")

        prob_before_norm = self._state_probability(psi)

        psi, norm_factor = normalize_unit(psi, self.grid.dx, self.grid.dy)
        psi = psi.astype(np.complex128)
        _assert_complex_array_2d(psi, "psi(after normalize_unit)")

        prob_after_norm = self._state_probability(psi)

        if self.front_debug_checks:
            _assert(np.isfinite(float(norm_factor)),
                    f"normalize_unit returned non-finite norm_factor={norm_factor}")
            _assert(prob_before_norm > 0.0,
                    f"prob_before_norm must be > 0, got {prob_before_norm}")
            _assert(
                np.isclose(prob_after_norm, 1.0, atol=self.front_norm_tol),
                f"final normalized probability must be 1, got {prob_after_norm}"
            )

        aux = dict(base.aux) if base.aux is not None else {}
        aux["thick_front"] = {
            "front_strength": float(self.front_strength),
            "front_misaligned_damp": float(self.front_misaligned_damp),
            "front_diag_weight": float(self.front_diag_weight),
            "front_phase_relax_strength": float(self.front_phase_relax_strength),
            "front_gain_blur_sigma": float(self.front_gain_blur_sigma),

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

            "front_branch_competition_radius": int(self.front_branch_competition_radius),
            "front_branch_competition_margin": float(self.front_branch_competition_margin),
            "front_branch_competition_blur_sigma": float(self.front_branch_competition_blur_sigma),

            "prob_after_base": float(prob_after_base),
            "prob_before_norm": float(prob_before_norm),
            "prob_after_norm": float(prob_after_norm),
            "normalize_unit_returned_norm": float(norm_factor),

            **aux_front,
        }

        return TheoryStepResult(state=psi, aux=aux)

    def step_backward_adjoint(self, state: np.ndarray, dt: float) -> TheoryStepResult:
        """
        Backward evolution is kept as plain adjoint Schrödinger evolution.

        That is usually the safer choice for retrodictive / click-backward library use.
        """
        _assert_complex_array_2d(state, "state(backward)")
        _assert_finite_scalar(dt, "dt(backward)")
        return super().step_backward_adjoint(state, dt)

    # --------------------------------------------------------
    # Observables
    # --------------------------------------------------------

    def density(self, state: np.ndarray) -> np.ndarray:
        _assert_complex_array_2d(state, "state(density)")
        rho = (np.abs(state) ** 2).astype(float)
        _assert_real_array_2d(rho, "rho(density)")
        _assert(np.all(rho >= -1e-14), "rho(density) contains significantly negative values")
        return rho

    def current(self, state_vis: np.ndarray):
        """
        Same current definition as Schrödinger.

        NOTE:
        Uses centered finite differences with np.roll, i.e. periodic wrapping.
        """
        _assert_complex_array_2d(state_vis, "state_vis(current)")

        dpsi_dx = (
            np.roll(state_vis, -1, axis=1) - np.roll(state_vis, 1, axis=1)
        ) / (2.0 * self.grid.dx)

        dpsi_dy = (
            np.roll(state_vis, -1, axis=0) - np.roll(state_vis, 1, axis=0)
        ) / (2.0 * self.grid.dy)

        _assert_complex_array_2d(dpsi_dx, "dpsi_dx")
        _assert_complex_array_2d(dpsi_dy, "dpsi_dy")

        rho = (np.abs(state_vis) ** 2).astype(float)
        _assert_real_array_2d(rho, "rho(current)")

        jx = (
            (self.hbar / self.m_mass)
            * np.imag(np.conjugate(state_vis) * dpsi_dx)
        ).astype(float)

        jy = (
            (self.hbar / self.m_mass)
            * np.imag(np.conjugate(state_vis) * dpsi_dy)
        ).astype(float)

        _assert_real_array_2d(jx, "jx")
        _assert_real_array_2d(jy, "jy")

        return jx, jy, rho