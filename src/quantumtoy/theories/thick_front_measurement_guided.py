from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import gaussian_filter

from theories.base import TheoryStepResult
from theories.schrodinger import SchrodingerTheory
from theories.thick_front_world_line import ThickFrontWorldLineTheory
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
class ThickFrontMeasurementGuidedTheory(ThickFrontWorldLineTheory):
    """
    Thick-front theory with measurement-inspired backward effect guidance.

    Main imported idea:

        overlap_score ~ rho_tmp * effect_mix

    where effect_mix is a short backward-adjoint detector-seeded effect field.

    This optimized version avoids rebuilding the effect field every step.
    """

    # --------------------------------------------------------
    # Measurement-guided effect field
    # --------------------------------------------------------

    measurement_guidance_enabled: bool = True

    # Detector gate in x where the effect seed is built
    measurement_detector_center_x: float = 10.0
    measurement_detector_width: float = 1.5

    # Optional y localization of the seed around detector-weighted centroid
    # 0.0 -> no y Gaussian, only x gate
    measurement_seed_y_sigma: float = 0.75

    # Refresh cadence before worldline branch lock-in
    measurement_refresh_every_n_steps_pre_init: int = 4

    # Refresh cadence after worldline branch lock-in
    measurement_refresh_every_n_steps_post_init: int = 32

    # If True, stop refreshing entirely once worldline bias is initialized
    measurement_stop_after_worldline_init: bool = True

    # Backward horizon: number of stored samples in the effect mix
    measurement_back_steps: int = 12

    # Each sample is separated by this many adjoint steps.
    # Larger stride = much cheaper, thicker/coarser backward horizon.
    measurement_back_stride: int = 2

    # Gaussian width in backward-time *steps* of the sampled library
    measurement_sigma_tau_steps: float = 4.0

    # Optional blur of the mixed effect field itself
    measurement_effect_blur_sigma: float = 0.75

    # Optional blur of overlap score
    measurement_overlap_blur_sigma: float = 0.75

    # If True, normalize effect_mix and overlap_score by frame max
    measurement_normalize_effect: bool = True
    measurement_normalize_overlap: bool = True

    # Skip expensive effect refresh if detector region has almost no mass
    measurement_min_detector_mass: float = 1e-8

    # --------------------------------------------------------
    # Use overlap in dynamics
    # --------------------------------------------------------

    # Use overlap_score for forced weaker-branch selection
    measurement_use_overlap_for_forced_selection: bool = True

    # Source field for forced weaker-branch if above is False:
    # "gamma_like" or "rho_tmp"
    measurement_forced_fallback_source: str = "rho_tmp"

    # Add positive gain on overlap-compatible regions
    measurement_gain_strength: float = 0.00

    # Reduce competition damping on overlap-compatible regions
    measurement_competition_relief_strength: float = 0.50
    measurement_competition_factor_min: float = 0.25
    measurement_competition_factor_max: float = 2.0

    # Gate measurement effects by positive local alignment
    measurement_gate_by_align: bool = True
    measurement_align_power: float = 1.0

    # --------------------------------------------------------
    # Debug / printing
    # --------------------------------------------------------

    measurement_debug_print: bool = False

    # --------------------------------------------------------
    # Internal state
    # --------------------------------------------------------

    _measurement_step_counter: int = field(init=False, default=0, repr=False)
    _measurement_effect_mix: np.ndarray | None = field(init=False, default=None, repr=False)
    _measurement_overlap_score: np.ndarray | None = field(init=False, default=None, repr=False)
    _measurement_diag: dict = field(init=False, default_factory=dict, repr=False)
    _measurement_last_refresh_step: int = field(init=False, default=-10**18, repr=False)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()

        allowed = {"gamma_like", "rho_tmp"}
        if self.measurement_forced_fallback_source not in allowed:
            raise ValueError(
                f"measurement_forced_fallback_source must be one of {allowed}, "
                f"got {self.measurement_forced_fallback_source!r}"
            )

        if self.measurement_detector_width <= 0.0:
            raise ValueError("measurement_detector_width must be > 0")
        if self.measurement_seed_y_sigma < 0.0:
            raise ValueError("measurement_seed_y_sigma must be >= 0")
        if self.measurement_refresh_every_n_steps_pre_init < 1:
            raise ValueError("measurement_refresh_every_n_steps_pre_init must be >= 1")
        if self.measurement_refresh_every_n_steps_post_init < 1:
            raise ValueError("measurement_refresh_every_n_steps_post_init must be >= 1")
        if self.measurement_back_steps < 1:
            raise ValueError("measurement_back_steps must be >= 1")
        if self.measurement_back_stride < 1:
            raise ValueError("measurement_back_stride must be >= 1")
        if self.measurement_sigma_tau_steps <= 0.0:
            raise ValueError("measurement_sigma_tau_steps must be > 0")
        if self.measurement_effect_blur_sigma < 0.0:
            raise ValueError("measurement_effect_blur_sigma must be >= 0")
        if self.measurement_overlap_blur_sigma < 0.0:
            raise ValueError("measurement_overlap_blur_sigma must be >= 0")
        if self.measurement_min_detector_mass < 0.0:
            raise ValueError("measurement_min_detector_mass must be >= 0")
        if self.measurement_gain_strength < 0.0:
            raise ValueError("measurement_gain_strength must be >= 0")
        if self.measurement_competition_relief_strength < 0.0:
            raise ValueError("measurement_competition_relief_strength must be >= 0")
        if self.measurement_competition_factor_min < 0.0:
            raise ValueError("measurement_competition_factor_min must be >= 0")
        if self.measurement_competition_factor_max <= 0.0:
            raise ValueError("measurement_competition_factor_max must be > 0")
        if self.measurement_competition_factor_min > self.measurement_competition_factor_max:
            raise ValueError(
                "measurement_competition_factor_min must be <= "
                "measurement_competition_factor_max"
            )
        if self.measurement_align_power < 0.0:
            raise ValueError("measurement_align_power must be >= 0")

        self._measurement_step_counter = 0
        self._measurement_effect_mix = None
        self._measurement_overlap_score = None
        self._measurement_diag = {}
        self._measurement_last_refresh_step = -10**18

    # --------------------------------------------------------
    # Detector / effect helpers
    # --------------------------------------------------------

    def _measurement_detector_gate_x(self) -> np.ndarray:
        X = self.grid.X
        xc = float(self.measurement_detector_center_x)
        sigma = float(self.measurement_detector_width)
        gate = np.exp(-((X - xc) ** 2) / (2.0 * sigma ** 2)).astype(float)
        _assert_real_array_2d(gate, "measurement_detector_gate_x")
        return gate

    def _measurement_detector_weighted_y_center(
        self,
        rho: np.ndarray,
        gate_x: np.ndarray,
    ) -> float:
        _assert_real_array_2d(rho, "rho")
        _assert_real_array_2d(gate_x, "gate_x")

        w = np.maximum(rho, 0.0) * np.maximum(gate_x, 0.0)
        s = float(np.sum(w) * self.grid.dx * self.grid.dy)
        if s <= self.front_eps:
            return 0.0
        y0 = float(np.sum(w * self.grid.Y) * self.grid.dx * self.grid.dy / s)
        return y0

    def _measurement_detector_mass(self, psi: np.ndarray) -> float:
        _assert_complex_array_2d(psi, "psi")
        rho = (np.abs(psi) ** 2).astype(float)
        gate_x = self._measurement_detector_gate_x()
        mass = float(np.sum(rho * gate_x) * self.grid.dx * self.grid.dy)
        return mass

    def _build_measurement_seed_from_state(self, psi: np.ndarray) -> np.ndarray:
        """
        Build a detector-seed amplitude from the current forward state.

        Uses:
            seed_amp ~ sqrt(rho) * detector_x_gate * optional_y_gate
        """
        _assert_complex_array_2d(psi, "psi")

        rho = (np.abs(psi) ** 2).astype(float)
        gate_x = self._measurement_detector_gate_x()

        if self.measurement_seed_y_sigma > 0.0:
            y0 = self._measurement_detector_weighted_y_center(rho, gate_x)
            y_sigma = float(self.measurement_seed_y_sigma)
            gate_y = np.exp(-((self.grid.Y - y0) ** 2) / (2.0 * y_sigma ** 2)).astype(float)
        else:
            y0 = 0.0
            gate_y = np.ones_like(gate_x, dtype=float)

        seed_amp = np.sqrt(np.maximum(rho, 0.0)) * gate_x * gate_y

        # fallback if detector region is still almost empty
        if float(np.max(seed_amp)) <= self.front_eps:
            seed_amp = gate_x * gate_y

        seed = seed_amp.astype(np.complex128)
        seed, seed_norm = normalize_unit(seed, self.grid.dx, self.grid.dy)

        if self.measurement_debug_print:
            print(
                "[measurement-guidance] seed built: "
                f"y0={y0:.6g}, norm_before={seed_norm:.6e}"
            )

        _assert_complex_array_2d(seed, "measurement_seed")
        return seed

    def _gaussian_time_weights(
        self,
        n: int,
        sigma_steps: float,
        stride: int,
    ) -> np.ndarray:
        idx = np.arange(n, dtype=float) * float(stride)
        w = np.exp(-0.5 * (idx / float(sigma_steps)) ** 2)
        s = float(np.sum(w))
        if s > 0.0:
            w /= s
        return w.astype(float)

    def _build_measurement_effect_mix(self, psi_ref: np.ndarray, dt: float) -> np.ndarray:
        """
        Build a mixed backward effect field from a detector-seed.

        Optimized:
          - uses only measurement_back_steps samples
          - each sample is separated by measurement_back_stride adjoint steps
        """
        _assert_complex_array_2d(psi_ref, "psi_ref")
        _assert_finite_scalar(dt, "dt")

        phi = self._build_measurement_seed_from_state(psi_ref)
        n_back = int(self.measurement_back_steps)
        stride = int(self.measurement_back_stride)

        weights = self._gaussian_time_weights(
            n=n_back,
            sigma_steps=float(self.measurement_sigma_tau_steps),
            stride=stride,
        )

        effect_mix = np.zeros((self.grid.Ny, self.grid.Nx), dtype=float)

        for j in range(n_back):
            rho_phi = (np.abs(phi) ** 2).astype(float)
            effect_mix += float(weights[j]) * rho_phi

            if j < n_back - 1:
                for _ in range(stride):
                    phi = SchrodingerTheory.step_backward_adjoint(self, phi, dt).state

        if self.measurement_effect_blur_sigma > 0.0:
            effect_mix = gaussian_filter(
                effect_mix,
                sigma=float(self.measurement_effect_blur_sigma),
                mode="wrap",
            )

        if self.measurement_normalize_effect:
            emax = float(np.max(effect_mix))
            if emax > self.front_eps:
                effect_mix = effect_mix / emax

        _assert_real_array_2d(effect_mix, "effect_mix")
        return effect_mix.astype(float)

    def _measurement_gate_from_align(self, align_real: np.ndarray) -> np.ndarray:
        _assert_real_array_2d(align_real, "align_real")

        if not self.measurement_gate_by_align:
            return np.ones_like(align_real, dtype=float)

        align_pos = np.maximum(align_real, 0.0)
        p = float(self.measurement_align_power)
        if p != 1.0:
            align_pos = align_pos ** p
        return align_pos.astype(float)

    def _build_measurement_overlap_score(
        self,
        rho_tmp: np.ndarray,
        align_real_tmp: np.ndarray,
        effect_mix: np.ndarray | None,
    ) -> np.ndarray | None:
        """
        overlap_score ~ rho_tmp * effect_mix * optional_align_gate
        """
        _assert_real_array_2d(rho_tmp, "rho_tmp")
        _assert_real_array_2d(align_real_tmp, "align_real_tmp")

        if effect_mix is None:
            return None

        _assert_real_array_2d(effect_mix, "effect_mix")

        gate = self._measurement_gate_from_align(align_real_tmp)
        overlap = np.maximum(rho_tmp, 0.0) * np.maximum(effect_mix, 0.0) * gate

        if self.measurement_overlap_blur_sigma > 0.0:
            overlap = gaussian_filter(
                overlap,
                sigma=float(self.measurement_overlap_blur_sigma),
                mode="wrap",
            )

        if self.measurement_normalize_overlap:
            omax = float(np.max(overlap))
            if omax > self.front_eps:
                overlap = overlap / omax

        _assert_real_array_2d(overlap, "overlap_score")
        return overlap.astype(float)

    def _measurement_refresh_interval(self) -> int:
        if self._worldline_bias_initialized:
            return int(self.measurement_refresh_every_n_steps_post_init)
        return int(self.measurement_refresh_every_n_steps_pre_init)

    def _should_refresh_measurement_guidance(self, psi_ref: np.ndarray) -> bool:
        if not self.measurement_guidance_enabled:
            return False

        if self.measurement_stop_after_worldline_init and self._worldline_bias_initialized:
            return False

        # always build once if we do not yet have an effect field
        if self._measurement_effect_mix is None:
            return True

        interval = self._measurement_refresh_interval()
        if (self._measurement_step_counter - self._measurement_last_refresh_step) < interval:
            return False

        detector_mass = self._measurement_detector_mass(psi_ref)
        if detector_mass < float(self.measurement_min_detector_mass):
            # detector still empty -> keep cached field, skip expensive refresh
            self._measurement_diag = {
                "measurement_enabled": True,
                "measurement_refresh_skipped_detector_empty": True,
                "measurement_detector_mass": float(detector_mass),
                "measurement_effect_cached": True,
            }
            return False

        return True

    def _refresh_measurement_guidance_fields(self, psi_ref: np.ndarray, dt: float):
        """
        Refresh effect_mix from psi_ref and clear overlap cache.
        """
        if not self.measurement_guidance_enabled:
            self._measurement_effect_mix = None
            self._measurement_overlap_score = None
            self._measurement_diag = {"measurement_enabled": False}
            return

        detector_mass = self._measurement_detector_mass(psi_ref)

        if detector_mass < float(self.measurement_min_detector_mass) and self._measurement_effect_mix is not None:
            self._measurement_overlap_score = None
            self._measurement_diag = {
                "measurement_enabled": True,
                "measurement_refresh_skipped_detector_empty": True,
                "measurement_detector_mass": float(detector_mass),
                "measurement_effect_cached": True,
                "measurement_effect_mean": float(np.mean(self._measurement_effect_mix)),
                "measurement_effect_max": float(np.max(self._measurement_effect_mix)),
                "measurement_effect_min": float(np.min(self._measurement_effect_mix)),
            }
            return

        effect_mix = self._build_measurement_effect_mix(psi_ref, dt)
        self._measurement_effect_mix = effect_mix
        self._measurement_overlap_score = None
        self._measurement_last_refresh_step = int(self._measurement_step_counter)

        self._measurement_diag = {
            "measurement_enabled": True,
            "measurement_effect_refreshed": True,
            "measurement_detector_mass": float(detector_mass),
            "measurement_effect_mean": float(np.mean(effect_mix)),
            "measurement_effect_max": float(np.max(effect_mix)),
            "measurement_effect_min": float(np.min(effect_mix)),
        }

    # --------------------------------------------------------
    # Shared branch helpers
    # --------------------------------------------------------

    def _gamma_like_from_rho_align(
        self,
        rho: np.ndarray,
        align_real: np.ndarray,
    ) -> np.ndarray:
        _assert_real_array_2d(rho, "rho")
        _assert_real_array_2d(align_real, "align_real")
        _assert(rho.shape == align_real.shape, "rho and align_real must have same shape")

        align_pos = np.maximum(align_real, 0.0)
        gamma_like = (
            np.power(np.maximum(rho, 0.0), float(self.front_branch_density_power))
            * np.power(align_pos, float(self.front_branch_align_power))
        ).astype(float)

        if self.front_branch_normalize_gamma:
            gmax = float(np.max(gamma_like))
            if gmax > self.front_eps:
                gamma_like = gamma_like / gmax

        _assert_real_array_2d(gamma_like, "_gamma_like_from_rho_align(out)")
        return gamma_like

    # --------------------------------------------------------
    # Forced branch selection
    # --------------------------------------------------------

    def _forced_selection_source_field(
        self,
        gamma_like: np.ndarray | None,
        rho_tmp: np.ndarray,
        overlap_score: np.ndarray | None,
    ) -> tuple[np.ndarray, str]:
        _assert_real_array_2d(rho_tmp, "rho_tmp")

        if self.measurement_use_overlap_for_forced_selection and overlap_score is not None:
            return overlap_score.astype(float), "measurement_overlap"

        src = self.measurement_forced_fallback_source
        if src == "rho_tmp":
            return rho_tmp.astype(float), "rho_tmp"

        if gamma_like is None:
            return rho_tmp.astype(float), "rho_tmp(fallback)"
        return gamma_like.astype(float), "gamma_like"

    def _initialize_forced_weaker_branch_bias_from_field(
        self,
        score_field: np.ndarray,
        score_name: str,
    ) -> bool:
        """
        Same forced weaker-branch logic as worldline parent, but source field
        may be measurement_overlap, rho_tmp, or gamma_like.
        """
        _assert_real_array_2d(score_field, "score_field")

        score = self._forced_branch_score_field(score_field)
        candidates = self._detect_top_branch_candidates(score)

        diag = {
            "selection_source": score_name,
            "selection_score_max": float(np.max(score)),
            "selection_score_mean": float(np.mean(score)),
            "selection_num_candidates": int(len(candidates)),
            "selection_initialized_now": False,
        }

        if len(candidates) < 2:
            self._print_forced_branch_candidates(
                candidates=candidates,
                selected_idx=None,
                reason=f"waiting for >=2 candidate peaks in {score_name}",
                score_name=score_name,
            )
            self._last_worldline_init_diag = diag
            return False

        sep = self._peak_pixel_separation(candidates[0], candidates[1])
        diag["selection_top2_separation_px"] = float(sep)

        if sep < float(self.worldline_forced_min_peak_separation_px):
            self._print_forced_branch_candidates(
                candidates=candidates,
                selected_idx=None,
                reason="top-2 peaks found but too close together; waiting for clearer split",
                score_name=score_name,
            )
            self._last_worldline_init_diag = diag
            return False

        selected_idx = 1
        score_sel, iy, ix = candidates[selected_idx]

        self._print_forced_branch_candidates(
            candidates=candidates,
            selected_idx=selected_idx,
            reason=f"selected weaker of top-2 strongest peaks from {score_name}",
            score_name=score_name,
        )

        x_center = float(self.grid.X[iy, ix])
        y_center = float(self.grid.Y[iy, ix])

        self._worldline_bias_field = self._build_gaussian_bump_bias_field(
            x_center=x_center,
            y_center=y_center,
        )
        self._worldline_bias_initialized = True

        diag["selection_initialized_now"] = True
        diag["selection_selected_idx"] = int(selected_idx)
        diag["selection_selected_score"] = float(score_sel)
        diag["selection_selected_ix"] = int(ix)
        diag["selection_selected_iy"] = int(iy)
        diag["selection_selected_x"] = float(x_center)
        diag["selection_selected_y"] = float(y_center)

        self._last_worldline_init_diag = diag
        return True

    # --------------------------------------------------------
    # Front operator override
    # --------------------------------------------------------

    def _front_sharpen(self, psi: np.ndarray, dt: float):
        _assert_complex_array_2d(psi, "psi")
        _assert_finite_scalar(dt, "dt")

        prob_in = self._state_probability(psi)

        # --------------------------------------------
        # Pre-gain coherence
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

        if self.front_gain_blur_sigma > 0.0:
            gain = gaussian_filter(
                gain,
                sigma=float(self.front_gain_blur_sigma),
                mode="wrap",
            )

        _assert_real_array_2d(gain, "gain")

        gain_dt = np.clip(gain * dt, -self.front_clip, self.front_clip)
        _assert_real_array_2d(gain_dt, "gain_dt")

        psi_tmp = psi * np.exp(gain_dt)
        _assert_complex_array_2d(psi_tmp, "psi_tmp")
        prob_after_gain = self._state_probability(psi_tmp)

        # --------------------------------------------
        # Post-gain coherence
        # --------------------------------------------
        align_real_tmp, rho_tmp, u_local_tmp = self._coherence_alignment_score(psi_tmp)

        competition_raw = None
        competition_gate = None
        gamma_like = None
        neighbor_max = None
        comp_dt = None
        detector_gate = None
        local_strength = None

        if self.front_branch_competition_strength > 0.0:
            gamma_like, neighbor_max, competition_raw, competition_gate = self._branch_competition_field(
                rho=rho_tmp,
                align_real=align_real_tmp,
            )

            if self.front_branch_detector_gate_enabled:
                detector_gate = self._detector_competition_gate()
                local_strength = float(self.front_branch_competition_strength) * (
                    1.0 + float(self.front_branch_detector_gate_boost) * detector_gate
                )
            else:
                local_strength = np.full_like(
                    competition_raw,
                    float(self.front_branch_competition_strength),
                    dtype=float,
                )

            _assert_real_array_2d(local_strength, "local_strength")
        else:
            gamma_like = self._gamma_like_from_rho_align(rho_tmp, align_real_tmp)

        # --------------------------------------------
        # Measurement-inspired overlap ridge
        # --------------------------------------------
        effect_mix = self._measurement_effect_mix
        overlap_score = self._build_measurement_overlap_score(
            rho_tmp=rho_tmp,
            align_real_tmp=align_real_tmp,
            effect_mix=effect_mix,
        )
        self._measurement_overlap_score = overlap_score

        # --------------------------------------------
        # Forced weaker-branch initialization
        # --------------------------------------------
        if (
            self.worldline_mode == "forced_weaker_branch"
            and not self._worldline_bias_initialized
        ):
            forced_field, forced_name = self._forced_selection_source_field(
                gamma_like=gamma_like,
                rho_tmp=rho_tmp,
                overlap_score=overlap_score,
            )
            self._initialize_forced_weaker_branch_bias_from_field(
                score_field=forced_field,
                score_name=forced_name,
            )

        # --------------------------------------------
        # Measurement-guided gain
        # --------------------------------------------
        measurement_gain_dt = None
        if overlap_score is not None and self.measurement_gain_strength > 0.0:
            measurement_gain = float(self.measurement_gain_strength) * overlap_score
            measurement_gain_dt = np.clip(
                measurement_gain * dt,
                -self.front_clip,
                self.front_clip,
            )
            psi_tmp = psi_tmp * np.exp(measurement_gain_dt)
            _assert_complex_array_2d(psi_tmp, "psi_tmp(after measurement gain)")

            align_real_tmp, rho_tmp, u_local_tmp = self._coherence_alignment_score(psi_tmp)
            prob_after_gain = self._state_probability(psi_tmp)

            if self.front_branch_competition_strength > 0.0:
                gamma_like, neighbor_max, competition_raw, competition_gate = self._branch_competition_field(
                    rho=rho_tmp,
                    align_real=align_real_tmp,
                )
                if self.front_branch_detector_gate_enabled:
                    detector_gate = self._detector_competition_gate()
                    local_strength = float(self.front_branch_competition_strength) * (
                        1.0 + float(self.front_branch_detector_gate_boost) * detector_gate
                    )
                else:
                    local_strength = np.full_like(
                        competition_raw,
                        float(self.front_branch_competition_strength),
                        dtype=float,
                    )
            else:
                gamma_like = self._gamma_like_from_rho_align(rho_tmp, align_real_tmp)

            overlap_score = self._build_measurement_overlap_score(
                rho_tmp=rho_tmp,
                align_real_tmp=align_real_tmp,
                effect_mix=effect_mix,
            )
            self._measurement_overlap_score = overlap_score

        # --------------------------------------------
        # Parent-style worldline frozen bias
        # --------------------------------------------
        bias_field = self._get_worldline_bias_field()
        bias_gate = None
        bias_effect = None

        if bias_field is not None:
            _assert_real_array_2d(bias_field, "bias_field")

            gamma_like_for_gate = gamma_like if gamma_like is not None else rho_tmp

            bias_gate = self._worldline_bias_gate(
                gamma_like=gamma_like_for_gate,
                align_real=align_real_tmp,
            )

            bias_effect = (
                float(self.worldline_bias_strength)
                * bias_field
                * bias_gate
            )

            if self.worldline_bias_effect_blur_sigma > 0.0:
                bias_effect = gaussian_filter(
                    bias_effect,
                    sigma=float(self.worldline_bias_effect_blur_sigma),
                    mode="wrap",
                )

            _assert_real_array_2d(bias_effect, "bias_effect")

            active_mode = self.worldline_mode
            if active_mode == "forced_weaker_branch":
                active_mode = "both"

            if active_mode in {"gain", "both"}:
                gain_dt_bias = np.clip(bias_effect * dt, -self.front_clip, self.front_clip)
                psi_tmp = psi_tmp * np.exp(gain_dt_bias)
                _assert_complex_array_2d(psi_tmp, "psi_tmp(after worldline gain bias)")

                align_real_tmp, rho_tmp, u_local_tmp = self._coherence_alignment_score(psi_tmp)
                prob_after_gain = self._state_probability(psi_tmp)

                if self.front_branch_competition_strength > 0.0:
                    gamma_like, neighbor_max, competition_raw, competition_gate = self._branch_competition_field(
                        rho=rho_tmp,
                        align_real=align_real_tmp,
                    )
                    if self.front_branch_detector_gate_enabled:
                        detector_gate = self._detector_competition_gate()
                        local_strength = float(self.front_branch_competition_strength) * (
                            1.0 + float(self.front_branch_detector_gate_boost) * detector_gate
                        )
                    else:
                        local_strength = np.full_like(
                            competition_raw,
                            float(self.front_branch_competition_strength),
                            dtype=float,
                        )
                else:
                    gamma_like = self._gamma_like_from_rho_align(rho_tmp, align_real_tmp)

                overlap_score = self._build_measurement_overlap_score(
                    rho_tmp=rho_tmp,
                    align_real_tmp=align_real_tmp,
                    effect_mix=effect_mix,
                )
                self._measurement_overlap_score = overlap_score

            if active_mode in {"competition", "both"} and competition_raw is not None:
                comp_factor = 1.0 - bias_effect
                comp_factor = np.clip(
                    comp_factor,
                    float(self.worldline_competition_factor_min),
                    float(self.worldline_competition_factor_max),
                )
                competition_raw = competition_raw * comp_factor
                _assert_real_array_2d(
                    competition_raw,
                    "competition_raw(after worldline competition bias)"
                )

        # --------------------------------------------
        # Measurement-guided competition relief
        # --------------------------------------------
        measurement_comp_factor = None
        if (
            overlap_score is not None
            and competition_raw is not None
            and self.measurement_competition_relief_strength > 0.0
        ):
            measurement_comp_factor = 1.0 - float(self.measurement_competition_relief_strength) * overlap_score
            measurement_comp_factor = np.clip(
                measurement_comp_factor,
                float(self.measurement_competition_factor_min),
                float(self.measurement_competition_factor_max),
            )
            competition_raw = competition_raw * measurement_comp_factor
            _assert_real_array_2d(competition_raw, "competition_raw(after measurement relief)")

        # --------------------------------------------
        # Apply competition damping
        # --------------------------------------------
        if self.front_branch_competition_strength > 0.0 and competition_raw is not None:
            comp_dt = np.clip(local_strength * competition_raw * dt, 0.0, self.front_clip)
            _assert_real_array_2d(comp_dt, "comp_dt")
            psi_new = psi_tmp * np.exp(-comp_dt)
        else:
            psi_new = psi_tmp

        _assert_complex_array_2d(psi_new, "psi_new(after competition)")
        prob_after_comp = self._state_probability(psi_new)

        # --------------------------------------------
        # Optional phase relaxation
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

        if gamma_like is not None:
            aux_front["branch_gamma_like_mean"] = float(np.mean(gamma_like))
            aux_front["branch_gamma_like_max"] = float(np.max(gamma_like))

        if neighbor_max is not None:
            aux_front["branch_neighbor_max_mean"] = float(np.mean(neighbor_max))
            aux_front["branch_neighbor_max_max"] = float(np.max(neighbor_max))

        if competition_raw is not None:
            aux_front["branch_competition_mean"] = float(np.mean(competition_raw))
            aux_front["branch_competition_max"] = float(np.max(competition_raw))

        if competition_gate is not None:
            aux_front["branch_gate_mean"] = float(np.mean(competition_gate))
            aux_front["branch_gate_max"] = float(np.max(competition_gate))

        if comp_dt is not None:
            aux_front["branch_comp_dt_mean"] = float(np.mean(comp_dt))
            aux_front["branch_comp_dt_max"] = float(np.max(comp_dt))

        if local_strength is not None:
            aux_front["branch_local_strength_mean"] = float(np.mean(local_strength))
            aux_front["branch_local_strength_max"] = float(np.max(local_strength))

        if detector_gate is not None:
            aux_front["branch_detector_gate_mean"] = float(np.mean(detector_gate))
            aux_front["branch_detector_gate_max"] = float(np.max(detector_gate))

        if measurement_gain_dt is not None:
            aux_front["measurement_gain_dt_mean"] = float(np.mean(measurement_gain_dt))
            aux_front["measurement_gain_dt_max"] = float(np.max(measurement_gain_dt))

        if effect_mix is not None:
            aux_front["measurement_effect_mean"] = float(np.mean(effect_mix))
            aux_front["measurement_effect_max"] = float(np.max(effect_mix))
            aux_front["measurement_effect_min"] = float(np.min(effect_mix))

        if overlap_score is not None:
            aux_front["measurement_overlap_mean"] = float(np.mean(overlap_score))
            aux_front["measurement_overlap_max"] = float(np.max(overlap_score))
            aux_front["measurement_overlap_min"] = float(np.min(overlap_score))

        if measurement_comp_factor is not None:
            aux_front["measurement_comp_factor_mean"] = float(np.mean(measurement_comp_factor))
            aux_front["measurement_comp_factor_min"] = float(np.min(measurement_comp_factor))
            aux_front["measurement_comp_factor_max"] = float(np.max(measurement_comp_factor))

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
            aux_front["worldline_bias_effect_abs_mean"] = float(np.mean(np.abs(bias_effect)))
            aux_front["worldline_bias_effect_min"] = float(np.min(bias_effect))
            aux_front["worldline_bias_effect_max"] = float(np.max(bias_effect))

        if self._measurement_diag:
            for k, v in self._measurement_diag.items():
                aux_front[f"measurement_diag_{k}"] = v

        if self._last_worldline_init_diag:
            for k, v in self._last_worldline_init_diag.items():
                aux_front[f"worldline_init_{k}"] = v

        return psi_new.astype(np.complex128), aux_front

    # --------------------------------------------------------
    # Stepping override
    # --------------------------------------------------------

    def step_forward(self, state: np.ndarray, dt: float) -> TheoryStepResult:
        """
        One forward step:
            1) plain Schrödinger step
            2) refresh measurement-inspired backward effect field only when needed
            3) thick-front sharpening + competition + measurement guidance
            4) final renormalization
        """
        _assert_complex_array_2d(state, "state")
        _assert_finite_scalar(dt, "dt")

        base = SchrodingerTheory.step_forward(self, state, dt)
        psi = base.state
        _assert_complex_array_2d(psi, "base.state")

        prob_after_base = self._state_probability(psi)

        if self._should_refresh_measurement_guidance(psi):
            self._refresh_measurement_guidance_fields(psi_ref=psi, dt=dt)
        elif self.measurement_guidance_enabled and self._measurement_effect_mix is not None:
            self._measurement_diag = {
                "measurement_enabled": True,
                "measurement_effect_cached": True,
                "measurement_effect_mean": float(np.mean(self._measurement_effect_mix)),
                "measurement_effect_max": float(np.max(self._measurement_effect_mix)),
                "measurement_effect_min": float(np.min(self._measurement_effect_mix)),
            }

        psi, aux_front = self._front_sharpen(psi, dt)
        _assert_complex_array_2d(psi, "psi(after _front_sharpen)")

        prob_before_norm = self._state_probability(psi)

        psi, norm_factor = normalize_unit(psi, self.grid.dx, self.grid.dy)
        psi = psi.astype(np.complex128)
        _assert_complex_array_2d(psi, "psi(after normalize_unit)")

        prob_after_norm = self._state_probability(psi)

        if self.front_debug_checks:
            _assert(
                np.isfinite(float(norm_factor)),
                f"normalize_unit returned non-finite norm_factor={norm_factor}"
            )
            _assert(prob_before_norm > 0.0, f"prob_before_norm must be > 0, got {prob_before_norm}")
            _assert(
                np.isclose(prob_after_norm, 1.0, atol=self.front_norm_tol),
                f"final normalized probability must be 1, got {prob_after_norm}"
            )

        aux = dict(base.aux) if base.aux is not None else {}
        aux["thick_front_measurement_guided"] = {
            "front_strength": float(self.front_strength),
            "front_misaligned_damp": float(self.front_misaligned_damp),
            "front_diag_weight": float(self.front_diag_weight),
            "front_phase_relax_strength": float(self.front_phase_relax_strength),
            "front_gain_blur_sigma": float(self.front_gain_blur_sigma),

            "front_branch_competition_strength": float(self.front_branch_competition_strength),
            "front_branch_competition_power": float(self.front_branch_competition_power),
            "front_branch_gate_power": float(self.front_branch_gate_power),
            "front_branch_density_power": float(self.front_branch_density_power),
            "front_branch_align_power": float(self.front_branch_align_power),
            "front_branch_competition_threshold": float(self.front_branch_competition_threshold),
            "front_branch_normalize_gamma": bool(self.front_branch_normalize_gamma),
            "front_branch_competition_radius": int(self.front_branch_competition_radius),
            "front_branch_competition_margin": float(self.front_branch_competition_margin),
            "front_branch_competition_blur_sigma": float(self.front_branch_competition_blur_sigma),

            "front_branch_detector_gate_enabled": bool(self.front_branch_detector_gate_enabled),
            "front_branch_detector_gate_center_x": float(self.front_branch_detector_gate_center_x),
            "front_branch_detector_gate_width": float(self.front_branch_detector_gate_width),
            "front_branch_detector_gate_boost": float(self.front_branch_detector_gate_boost),

            "worldline_mode": str(self.worldline_mode),
            "worldline_bias_strength": float(self.worldline_bias_strength),
            "worldline_bias_sigma": float(self.worldline_bias_sigma),
            "worldline_bias_seed": self.worldline_bias_seed,
            "worldline_bias_gate_by_gamma": bool(self.worldline_bias_gate_by_gamma),
            "worldline_bias_gamma_power": float(self.worldline_bias_gamma_power),
            "worldline_bias_align_power": float(self.worldline_bias_align_power),
            "worldline_bias_gate_scale": float(self.worldline_bias_gate_scale),
            "worldline_bias_effect_blur_sigma": float(self.worldline_bias_effect_blur_sigma),
            "worldline_competition_factor_min": float(self.worldline_competition_factor_min),
            "worldline_competition_factor_max": float(self.worldline_competition_factor_max),

            "worldline_forced_localmax_radius": int(self.worldline_forced_localmax_radius),
            "worldline_forced_peak_rel_threshold": float(self.worldline_forced_peak_rel_threshold),
            "worldline_forced_bias_sigma_x": float(self.worldline_forced_bias_sigma_x),
            "worldline_forced_bias_sigma_y": float(self.worldline_forced_bias_sigma_y),
            "worldline_forced_bias_bump_amplitude": float(self.worldline_forced_bias_bump_amplitude),
            "worldline_forced_print_candidates": bool(self.worldline_forced_print_candidates),
            "worldline_forced_score_blur_sigma": float(self.worldline_forced_score_blur_sigma),
            "worldline_forced_min_peak_separation_px": int(self.worldline_forced_min_peak_separation_px),

            "measurement_guidance_enabled": bool(self.measurement_guidance_enabled),
            "measurement_detector_center_x": float(self.measurement_detector_center_x),
            "measurement_detector_width": float(self.measurement_detector_width),
            "measurement_seed_y_sigma": float(self.measurement_seed_y_sigma),
            "measurement_refresh_every_n_steps_pre_init": int(self.measurement_refresh_every_n_steps_pre_init),
            "measurement_refresh_every_n_steps_post_init": int(self.measurement_refresh_every_n_steps_post_init),
            "measurement_stop_after_worldline_init": bool(self.measurement_stop_after_worldline_init),
            "measurement_back_steps": int(self.measurement_back_steps),
            "measurement_back_stride": int(self.measurement_back_stride),
            "measurement_sigma_tau_steps": float(self.measurement_sigma_tau_steps),
            "measurement_effect_blur_sigma": float(self.measurement_effect_blur_sigma),
            "measurement_overlap_blur_sigma": float(self.measurement_overlap_blur_sigma),
            "measurement_normalize_effect": bool(self.measurement_normalize_effect),
            "measurement_normalize_overlap": bool(self.measurement_normalize_overlap),
            "measurement_min_detector_mass": float(self.measurement_min_detector_mass),
            "measurement_use_overlap_for_forced_selection": bool(
                self.measurement_use_overlap_for_forced_selection
            ),
            "measurement_forced_fallback_source": str(self.measurement_forced_fallback_source),
            "measurement_gain_strength": float(self.measurement_gain_strength),
            "measurement_competition_relief_strength": float(self.measurement_competition_relief_strength),
            "measurement_competition_factor_min": float(self.measurement_competition_factor_min),
            "measurement_competition_factor_max": float(self.measurement_competition_factor_max),
            "measurement_gate_by_align": bool(self.measurement_gate_by_align),
            "measurement_align_power": float(self.measurement_align_power),

            "prob_after_base": float(prob_after_base),
            "prob_before_norm": float(prob_before_norm),
            "prob_after_norm": float(prob_after_norm),
            "normalize_unit_returned_norm": float(norm_factor),

            **aux_front,
        }

        self._measurement_step_counter += 1
        return TheoryStepResult(state=psi, aux=aux)

    # --------------------------------------------------------
    # Backward evolution
    # --------------------------------------------------------

    def step_backward_adjoint(self, state: np.ndarray, dt: float) -> TheoryStepResult:
        """
        Backward evolution stays as plain adjoint Schrödinger evolution.
        """
        _assert_complex_array_2d(state, "state(backward)")
        _assert_finite_scalar(dt, "dt(backward)")
        return SchrodingerTheory.step_backward_adjoint(self, state, dt)