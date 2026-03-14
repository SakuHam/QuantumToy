from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from theories.base import TheoryStepResult
from theories.thick_front_optimized import ThickFrontOptimizedTheory
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
class ThickFrontWorldLineTheory(ThickFrontOptimizedTheory):
    """
    Thick-front theory with a persistent worldline-style symmetry-breaking bias.

    Extends ThickFrontOptimizedTheory by adding a frozen bias field that remains
    fixed during the run.

    Modes:
      - "off"                 : disabled
      - "gain"                : frozen bias modulates local growth
      - "competition"         : frozen bias modulates local competition damping
      - "both"                : both of the above
      - "forced_weaker_branch": wait until branch-like structure is visible in
                                gamma_like, detect the strongest local branch
                                peaks, and build a frozen bias centered on the
                                weaker of the top-2 peaks
    """

    # --------------------------------------------------------
    # Worldline / frozen-bias mode
    # --------------------------------------------------------

    # "off", "gain", "competition", "both", "forced_weaker_branch"
    worldline_mode: str = "forced_weaker_branch"

    # Keep modest for normal random-bias modes.
    worldline_bias_strength: float = 0.05

    # Correlation length in physical units for random bias
    worldline_bias_sigma: float = 2.0

    # Reproducible frozen field
    worldline_bias_seed: int | None = 7

    # Gate bias by current branch-like structure
    worldline_bias_gate_by_gamma: bool = True
    worldline_bias_gamma_power: float = 1.0
    worldline_bias_align_power: float = 1.0
    worldline_bias_gate_scale: float = 1.0

    # Optional blur of the bias effect itself
    worldline_bias_effect_blur_sigma: float = 0.0

    # When modulating competition, clamp multiplicative factor into this range
    worldline_competition_factor_min: float = 0.25
    worldline_competition_factor_max: float = 2.0

    # --------------------------------------------------------
    # Forced weaker-branch initialization
    # --------------------------------------------------------

    # Radius in pixels for local-max detection on first frame
    worldline_forced_localmax_radius: int = 5

    # Minimum relative threshold vs frame max for candidate peaks
    worldline_forced_peak_rel_threshold: float = 0.10

    # Gaussian width (physical units) of the forced bias bump
    worldline_forced_bias_sigma_x: float = 1.5
    worldline_forced_bias_sigma_y: float = 1.5

    # Strength multiplier for the forced bump before normalization.
    # This mostly affects shape before zero-mean/std normalization.
    worldline_forced_bias_bump_amplitude: float = 1.0

    # Print detected forced-branch candidates on initialization attempts
    worldline_forced_print_candidates: bool = False

    # Optional additional blur before peak picking in forced mode
    worldline_forced_score_blur_sigma: float = 0.0

    # Minimum separation between the top-2 peaks for accepting initialization
    worldline_forced_min_peak_separation_px: int = 20

    # --------------------------------------------------------
    # Internal state
    # --------------------------------------------------------

    _worldline_rng: np.random.Generator = field(init=False, repr=False)
    _worldline_bias_field: np.ndarray | None = field(init=False, default=None, repr=False)
    _worldline_bias_initialized: bool = field(init=False, default=False, repr=False)
    _last_worldline_init_diag: dict = field(init=False, default_factory=dict, repr=False)

    # --------------------------------------------------------
    # Init / validation
    # --------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()

        allowed_modes = {
            "off",
            "gain",
            "competition",
            "both",
            "forced_weaker_branch",
        }
        if self.worldline_mode not in allowed_modes:
            raise ValueError(
                f"worldline_mode must be one of {allowed_modes}, got {self.worldline_mode!r}"
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
        if self.worldline_bias_effect_blur_sigma < 0.0:
            raise ValueError("worldline_bias_effect_blur_sigma must be >= 0")

        if self.worldline_competition_factor_min < 0.0:
            raise ValueError("worldline_competition_factor_min must be >= 0")
        if self.worldline_competition_factor_max <= 0.0:
            raise ValueError("worldline_competition_factor_max must be > 0")
        if self.worldline_competition_factor_min > self.worldline_competition_factor_max:
            raise ValueError(
                "worldline_competition_factor_min must be <= worldline_competition_factor_max"
            )

        if not isinstance(self.worldline_forced_localmax_radius, int):
            raise ValueError("worldline_forced_localmax_radius must be int")
        if self.worldline_forced_localmax_radius < 1:
            raise ValueError("worldline_forced_localmax_radius must be >= 1")

        if not (0.0 <= self.worldline_forced_peak_rel_threshold <= 1.0):
            raise ValueError("worldline_forced_peak_rel_threshold must be in [0,1]")

        if self.worldline_forced_bias_sigma_x <= 0.0:
            raise ValueError("worldline_forced_bias_sigma_x must be > 0")
        if self.worldline_forced_bias_sigma_y <= 0.0:
            raise ValueError("worldline_forced_bias_sigma_y must be > 0")
        if self.worldline_forced_bias_bump_amplitude <= 0.0:
            raise ValueError("worldline_forced_bias_bump_amplitude must be > 0")

        if self.worldline_forced_score_blur_sigma < 0.0:
            raise ValueError("worldline_forced_score_blur_sigma must be >= 0")
        if self.worldline_forced_min_peak_separation_px < 0:
            raise ValueError("worldline_forced_min_peak_separation_px must be >= 0")

        self._worldline_rng = np.random.default_rng(self.worldline_bias_seed)
        self._worldline_bias_field = None
        self._worldline_bias_initialized = False
        self._last_worldline_init_diag = {}

    # --------------------------------------------------------
    # Public helper
    # --------------------------------------------------------

    def reset_worldline_bias(self, seed: int | None = None):
        """
        Reset the frozen worldline bias field.

        Use this between runs if you want a new persistent preference.
        """
        if seed is not None:
            self.worldline_bias_seed = seed

        self._worldline_rng = np.random.default_rng(self.worldline_bias_seed)
        self._worldline_bias_field = None
        self._worldline_bias_initialized = False
        self._last_worldline_init_diag = {}

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
        _assert_real_array_2d(arr, "arr")

        if sigma_x <= self.front_eps and sigma_y <= self.front_eps:
            return arr.copy()

        kx = 2.0 * np.pi * np.fft.fftfreq(self.grid.Nx, d=self.grid.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.grid.Ny, d=self.grid.dy)
        KX, KY = np.meshgrid(kx, ky)

        filt = np.exp(
            -0.5 * ((sigma_x ** 2) * (KX ** 2) + (sigma_y ** 2) * (KY ** 2))
        )

        arr_k = np.fft.fft2(arr)
        smoothed = np.fft.ifft2(arr_k * filt).real
        out = smoothed.astype(float)
        _assert_real_array_2d(out, "_fft_gaussian_smooth_periodic(out)")
        return out

    def _build_worldline_bias_field(self) -> np.ndarray:
        """
        Build a smooth zero-mean unit-std frozen random field.
        """
        raw = self._worldline_rng.standard_normal((self.grid.Ny, self.grid.Nx))
        sigma = float(self.worldline_bias_sigma)

        bias = self._fft_gaussian_smooth_periodic(raw, sigma_x=sigma, sigma_y=sigma)

        bias -= float(np.mean(bias))
        std = float(np.std(bias))

        if std > self.front_eps:
            bias = bias / std
        else:
            bias = np.zeros_like(bias, dtype=float)

        out = bias.astype(float)
        _assert_real_array_2d(out, "_build_worldline_bias_field(out)")
        return out

    def _build_gaussian_bump_bias_field(
        self,
        x_center: float,
        y_center: float,
    ) -> np.ndarray:
        """
        Build a smooth bias bump centered on a chosen branch candidate.
        """
        Xc = self.grid.X - float(x_center)
        Yc = self.grid.Y - float(y_center)

        bias = float(self.worldline_forced_bias_bump_amplitude) * np.exp(
            -0.5 * (
                (Xc / float(self.worldline_forced_bias_sigma_x)) ** 2
                + (Yc / float(self.worldline_forced_bias_sigma_y)) ** 2
            )
        )

        bias = bias.astype(float)
        bias -= float(np.mean(bias))
        std = float(np.std(bias))

        if std > self.front_eps:
            bias = bias / std
        else:
            bias = np.zeros_like(bias, dtype=float)

        _assert_real_array_2d(bias, "_build_gaussian_bump_bias_field(out)")
        return bias

    # --------------------------------------------------------
    # Forced-branch peak picking helpers
    # --------------------------------------------------------

    def _forced_branch_score_field(
        self,
        gamma_like: np.ndarray,
    ) -> np.ndarray:
        """
        Score field used for forced weaker-branch selection.

        Currently based on gamma_like, optionally blurred to suppress tiny
        noisy peak splitting before true branch formation.
        """
        _assert_real_array_2d(gamma_like, "gamma_like")

        score = np.asarray(gamma_like, dtype=float)

        if self.worldline_forced_score_blur_sigma > 0.0:
            score = gaussian_filter(
                score,
                sigma=float(self.worldline_forced_score_blur_sigma),
                mode="wrap",
            )

        smax = float(np.max(score))
        if smax > self.front_eps:
            score = score / smax

        _assert_real_array_2d(score, "_forced_branch_score_field(out)")
        return score

    def _detect_top_branch_candidates(
        self,
        score_field: np.ndarray,
    ) -> list[tuple[float, int, int]]:
        """
        Detect strong local maxima in score_field and return them sorted descending:
            [(score, iy, ix), ...]
        """
        _assert_real_array_2d(score_field, "score_field")

        radius = int(self.worldline_forced_localmax_radius)
        filt_size = 2 * radius + 1

        local_max = maximum_filter(score_field, size=(filt_size, filt_size), mode="wrap")
        smax = float(np.max(score_field))

        if smax <= self.front_eps:
            return []

        thr = float(self.worldline_forced_peak_rel_threshold) * smax
        mask = (score_field >= local_max - 1e-15) & (score_field >= thr)

        iy_all, ix_all = np.where(mask)
        if iy_all.size == 0:
            return []

        vals = score_field[iy_all, ix_all]
        order = np.argsort(vals)[::-1]

        candidates: list[tuple[float, int, int]] = []
        taken: list[tuple[int, int]] = []

        for idx in order:
            iy = int(iy_all[idx])
            ix = int(ix_all[idx])
            score = float(vals[idx])

            keep = True
            for py, px in taken:
                if abs(iy - py) <= radius and abs(ix - px) <= radius:
                    keep = False
                    break

            if keep:
                candidates.append((score, iy, ix))
                taken.append((iy, ix))

            if len(candidates) >= 8:
                break

        return candidates

    def _peak_pixel_separation(
        self,
        a: tuple[float, int, int],
        b: tuple[float, int, int],
    ) -> float:
        _, iya, ixa = a
        _, iyb, ixb = b
        dy = float(iya - iyb)
        dx = float(ixa - ixb)
        return float(np.hypot(dx, dy))

    def _print_forced_branch_candidates(
        self,
        candidates: list[tuple[float, int, int]],
        selected_idx: int | None,
        reason: str,
        score_name: str = "gamma_like",
    ):
        """
        Print up to top-3 strongest branch candidates and which one was selected.
        """
        if not self.worldline_forced_print_candidates:
            return

        print("[worldline forced_weaker_branch] candidate peak summary")
        print(f"  reason: {reason}")
        print(f"  source field: {score_name}")

        topn = min(3, len(candidates))
        if topn == 0:
            print("  no local-max candidates detected")
        else:
            top1 = float(candidates[0][0])
            for rank in range(topn):
                score, iy, ix = candidates[rank]
                x = float(self.grid.X[iy, ix])
                y = float(self.grid.Y[iy, ix])
                rel = score / max(top1, self.front_eps)
                print(
                    f"  peak #{rank + 1}: "
                    f"ix={ix}, iy={iy}, "
                    f"x={x:.6g}, y={y:.6g}, "
                    f"strength={score:.6e}, "
                    f"rel_to_top1={rel:.6f}"
                )

        if selected_idx is None:
            print("  selected: none yet")
        else:
            score, iy, ix = candidates[selected_idx]
            x = float(self.grid.X[iy, ix])
            y = float(self.grid.Y[iy, ix])
            print(
                f"  selected peak #{selected_idx + 1}: "
                f"ix={ix}, iy={iy}, "
                f"x={x:.6g}, y={y:.6g}, "
                f"strength={score:.6e}"
            )

    def _initialize_forced_weaker_branch_bias(
        self,
        gamma_like: np.ndarray,
    ) -> bool:
        """
        Initialize only when at least two sufficiently separated strong local peaks
        exist in gamma_like. Select the weaker of the top-2 strongest peaks.

        Returns:
            True if initialization happened, False otherwise.
        """
        _assert_real_array_2d(gamma_like, "gamma_like")

        score_field = self._forced_branch_score_field(gamma_like)
        candidates = self._detect_top_branch_candidates(score_field)

        diag = {
            "score_max": float(np.max(score_field)),
            "score_mean": float(np.mean(score_field)),
            "num_candidates": int(len(candidates)),
            "initialized_now": False,
        }

        if len(candidates) < 2:
            self._print_forced_branch_candidates(
                candidates=candidates,
                selected_idx=None,
                reason="waiting for >=2 candidate peaks in gamma_like branch field",
                score_name="gamma_like",
            )
            self._last_worldline_init_diag = diag
            return False

        sep = self._peak_pixel_separation(candidates[0], candidates[1])
        diag["top2_separation_px"] = float(sep)

        if sep < float(self.worldline_forced_min_peak_separation_px):
            self._print_forced_branch_candidates(
                candidates=candidates,
                selected_idx=None,
                reason=(
                    "top-2 peaks found but too close together; "
                    "waiting for clearer branch split"
                ),
                score_name="gamma_like",
            )
            self._last_worldline_init_diag = diag
            return False

        # Sorted descending, so index 1 is the weaker of the two strongest peaks
        selected_idx = 1
        score, iy, ix = candidates[selected_idx]

        self._print_forced_branch_candidates(
            candidates=candidates,
            selected_idx=selected_idx,
            reason="selected weaker of top-2 strongest peaks from gamma_like branch field",
            score_name="gamma_like",
        )

        x_center = float(self.grid.X[iy, ix])
        y_center = float(self.grid.Y[iy, ix])

        self._worldline_bias_field = self._build_gaussian_bump_bias_field(
            x_center=x_center,
            y_center=y_center,
        )
        self._worldline_bias_initialized = True

        diag["initialized_now"] = True
        diag["selected_idx"] = int(selected_idx)
        diag["selected_score"] = float(score)
        diag["selected_ix"] = int(ix)
        diag["selected_iy"] = int(iy)
        diag["selected_x"] = float(x_center)
        diag["selected_y"] = float(y_center)

        self._last_worldline_init_diag = diag
        return True

    def _get_worldline_bias_field(self) -> np.ndarray | None:
        if self.worldline_mode == "off":
            return None
        if self.worldline_bias_strength <= 0.0:
            return None
        if self.worldline_mode == "forced_weaker_branch":
            return self._worldline_bias_field

        if self._worldline_bias_field is None:
            self._worldline_bias_field = self._build_worldline_bias_field()
            self._worldline_bias_initialized = True

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
        _assert_real_array_2d(align_real, "align_real")

        gate = np.ones_like(align_real, dtype=float)

        if self.worldline_bias_gate_by_gamma and gamma_like is not None:
            _assert_real_array_2d(gamma_like, "gamma_like")
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

        out = gate.astype(float)
        _assert_real_array_2d(out, "_worldline_bias_gate(out)")
        return out

    # --------------------------------------------------------
    # Front operator override
    # --------------------------------------------------------

    def _front_sharpen(self, psi: np.ndarray, dt: float):
        """
        Apply local thick-front sharpening + optional branch competition
        + optional persistent frozen worldline bias.

        Updated to be compatible with the newer flow-aware
        ThickFrontOptimizedTheory structure.
        """
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

        # --------------------------------------------
        # First stage: gain application
        # --------------------------------------------
        gain_dt = np.clip(gain * dt, -self.front_clip, self.front_clip)
        _assert_real_array_2d(gain_dt, "gain_dt")

        psi_tmp = psi * np.exp(gain_dt)
        _assert_complex_array_2d(psi_tmp, "psi_tmp")

        prob_after_gain = self._state_probability(psi_tmp)

        # --------------------------------------------
        # Post-gain coherence
        # --------------------------------------------
        align_real_tmp, rho_tmp, u_local_tmp = self._coherence_alignment_score(psi_tmp)

        # --------------------------------------------
        # Flow field from sharpened psi_tmp
        # --------------------------------------------
        jx_tmp = None
        jy_tmp = None
        speed_tmp = None
        ux_tmp = None
        uy_tmp = None
        flow_valid_mask = None

        if self.front_branch_use_flow_direction:
            (
                jx_tmp,
                jy_tmp,
                rho_flow_tmp,
                speed_tmp,
                ux_tmp,
                uy_tmp,
                flow_valid_mask,
            ) = self._flow_direction_field(psi_tmp)

            if self.front_debug_checks:
                _assert(
                    np.allclose(rho_flow_tmp, rho_tmp, atol=1e-12, rtol=1e-10),
                    "rho from flow field and rho from coherence score differ unexpectedly"
                )

        competition_raw = None
        competition_gate = None
        gamma_like = None
        neighbor_max = None
        comp_dt = None
        detector_gate = None
        local_strength = None
        transverse_best = None
        direction_mismatch_best = None

        # --------------------------------------------
        # Competition from post-gain fields
        # --------------------------------------------
        if self.front_branch_competition_strength > 0.0:
            (
                gamma_like,
                neighbor_max,
                competition_raw,
                competition_gate,
                transverse_best,
                direction_mismatch_best,
            ) = self._branch_competition_field(
                rho=rho_tmp,
                align_real=align_real_tmp,
                ux=ux_tmp,
                uy=uy_tmp,
                valid_mask=flow_valid_mask,
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
            align_pos_tmp = np.maximum(align_real_tmp, 0.0)
            gamma_like = (
                np.power(np.maximum(rho_tmp, 0.0), float(self.front_branch_density_power))
                * np.power(np.maximum(align_pos_tmp, 0.0), float(self.front_branch_align_power))
            ).astype(float)

            if self.front_branch_normalize_gamma:
                gmax = float(np.max(gamma_like))
                if gmax > self.front_eps:
                    gamma_like = gamma_like / gmax

        # --------------------------------------------
        # Forced weaker-branch initialization
        # --------------------------------------------
        if (
            self.worldline_mode == "forced_weaker_branch"
            and not self._worldline_bias_initialized
            and gamma_like is not None
        ):
            self._initialize_forced_weaker_branch_bias(gamma_like)

        # --------------------------------------------
        # Frozen worldline bias
        # --------------------------------------------
        bias_field = self._get_worldline_bias_field()
        bias_gate = None
        bias_effect = None

        if bias_field is not None:
            _assert_real_array_2d(bias_field, "bias_field")

            bias_gate = self._worldline_bias_gate(
                gamma_like=gamma_like,
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

            # Positive frozen bias slightly favors local growth
            if active_mode in {"gain", "both"}:
                gain_dt_bias = np.clip(bias_effect * dt, -self.front_clip, self.front_clip)
                psi_tmp = psi_tmp * np.exp(gain_dt_bias)
                _assert_complex_array_2d(psi_tmp, "psi_tmp(after worldline gain bias)")

                # refresh post-gain diagnostics after bias
                align_real_tmp, rho_tmp, u_local_tmp = self._coherence_alignment_score(psi_tmp)
                prob_after_gain = self._state_probability(psi_tmp)

                # refresh flow diagnostics after bias
                if self.front_branch_use_flow_direction:
                    (
                        jx_tmp,
                        jy_tmp,
                        rho_flow_tmp,
                        speed_tmp,
                        ux_tmp,
                        uy_tmp,
                        flow_valid_mask,
                    ) = self._flow_direction_field(psi_tmp)

                    if self.front_debug_checks:
                        _assert(
                            np.allclose(rho_flow_tmp, rho_tmp, atol=1e-12, rtol=1e-10),
                            "rho from flow field and rho from coherence score differ unexpectedly after worldline gain bias"
                        )

                # recompute competition so it matches biased psi_tmp
                if self.front_branch_competition_strength > 0.0:
                    (
                        gamma_like,
                        neighbor_max,
                        competition_raw,
                        competition_gate,
                        transverse_best,
                        direction_mismatch_best,
                    ) = self._branch_competition_field(
                        rho=rho_tmp,
                        align_real=align_real_tmp,
                        ux=ux_tmp,
                        uy=uy_tmp,
                        valid_mask=flow_valid_mask,
                    )

            # Positive frozen bias reduces competition damping on favored branch
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

            "flow_direction_enabled": bool(self.front_branch_use_flow_direction),
        }

        if speed_tmp is not None:
            aux_front["flow_speed_mean"] = float(np.mean(speed_tmp))
            aux_front["flow_speed_max"] = float(np.max(speed_tmp))
            aux_front["flow_valid_fraction"] = float(np.mean(flow_valid_mask.astype(float)))
            aux_front["flow_jx_mean_abs"] = float(np.mean(np.abs(jx_tmp)))
            aux_front["flow_jy_mean_abs"] = float(np.mean(np.abs(jy_tmp)))

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

        if transverse_best is not None:
            aux_front["branch_transverse_best_mean"] = float(np.mean(transverse_best))
            aux_front["branch_transverse_best_max"] = float(np.max(transverse_best))

        if direction_mismatch_best is not None:
            aux_front["branch_direction_mismatch_best_mean"] = float(
                np.mean(direction_mismatch_best)
            )
            aux_front["branch_direction_mismatch_best_max"] = float(
                np.max(direction_mismatch_best)
            )

        if detector_gate is not None:
            aux_front["branch_detector_gate_mean"] = float(np.mean(detector_gate))
            aux_front["branch_detector_gate_max"] = float(np.max(detector_gate))

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

        aux_front["worldline_bias_initialized"] = bool(self._worldline_bias_initialized)

        if self._last_worldline_init_diag:
            for k, v in self._last_worldline_init_diag.items():
                aux_front[f"worldline_init_{k}"] = v

        return psi_new.astype(np.complex128), aux_front

    # --------------------------------------------------------
    # Forward step override
    # --------------------------------------------------------

    def step_forward(self, state: np.ndarray, dt: float) -> TheoryStepResult:
        """
        One forward step:
            pure Schrödinger step
            + thick-front sharpening
            + optional deterministic branch competition
            + optional frozen worldline bias
        """
        base = SchrodingerTheory.step_forward(self, state, dt)
        psi = base.state

        psi, aux_front = self._front_sharpen(psi, dt)
        psi, norm_factor = normalize_unit(psi, self.grid.dx, self.grid.dy)
        psi = psi.astype(np.complex128)

        aux = dict(base.aux) if base.aux is not None else {}
        aux["thick_front_worldline"] = {
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

            "front_branch_use_flow_direction": bool(self.front_branch_use_flow_direction),
            "front_branch_direction_mismatch_power": float(self.front_branch_direction_mismatch_power),
            "front_branch_transverse_weight_power": float(self.front_branch_transverse_weight_power),
            "front_branch_min_speed_fraction": float(self.front_branch_min_speed_fraction),
            "front_branch_min_rho_fraction": float(self.front_branch_min_rho_fraction),

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

            "normalize_unit_returned_norm": float(norm_factor),
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