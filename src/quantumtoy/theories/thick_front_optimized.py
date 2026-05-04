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
        2) competition can use current / flow direction derived from psi_tmp
        3) competition can prefer transverse (cross-flow) competitors rather
           than same-branch along-flow neighbors
        4) optional Gaussian smoothing is applied to gain and competition fields
           to reduce grid anisotropy / square artifacts
        5) competition can be boosted near the detector / screen region
        6) flow competition is sped up by evaluating only active pixels
        7) optional subclass hook can inject worldline / frozen-bias effects
        8) post hoc helper can export lightweight support fields for saved frames
    """

    # --------------------------------------------------------
    # Original thick-front parameters
    # --------------------------------------------------------

    front_strength: float = 0.03
    front_misaligned_damp: float = 0.01
    front_diag_weight: float = 0.5
    front_density_weighted: bool = True
    front_eps: float = 1e-12
    front_phase_relax_strength: float = 0.00
    front_clip: float = 0.25
    front_gain_blur_sigma: float = 1.0

    # --------------------------------------------------------
    # Branch competition / lateral inhibition parameters
    # --------------------------------------------------------

    front_branch_competition_strength: float = 0.20
    front_branch_competition_power: float = 1.00
    front_branch_gate_power: float = 1.00

    # Kept for compatibility / possible future variants
    front_branch_competition_x_weight: float = 0.35
    front_branch_competition_y_weight: float = 1.00
    front_branch_competition_diag_weight: float = 0.35

    front_branch_density_power: float = 1.0
    front_branch_align_power: float = 2.0
    front_branch_competition_threshold: float = 0.00
    front_branch_normalize_gamma: bool = True

    # --------------------------------------------------------
    # Competition search radius / filtering
    # --------------------------------------------------------

    front_branch_competition_radius: int = 20
    front_branch_competition_margin: float = 0.90
    front_branch_competition_blur_sigma: float = 0.5

    # --------------------------------------------------------
    # Flow-based branch competition
    # --------------------------------------------------------

    # If False, falls back to simple local maximum_filter competition
    front_branch_use_flow_direction: bool = True

    # Use Bohmian-like current direction computed from psi_tmp
    front_branch_direction_mismatch_power: float = 1.0

    # Prefer competitors across the local flow direction
    # 0.0 -> no transverse preference
    # 1.0+ -> stronger normal-direction competition
    front_branch_transverse_weight_power: float = 2.0

    # Ignore direction info where local speed is too tiny
    # threshold = fraction * max(speed)
    front_branch_min_speed_fraction: float = 1e-4

    # Optional floor on local density for valid direction competition
    front_branch_min_rho_fraction: float = 1e-6

    # --------------------------------------------------------
    # Detector / screen-local competition boost
    # --------------------------------------------------------

    front_branch_detector_gate_enabled: bool = True
    front_branch_detector_gate_center_x: float = 10.0
    front_branch_detector_gate_width: float = 2.0
    front_branch_detector_gate_boost: float = 10.0

    # --------------------------------------------------------
    # Debug / safety parameters
    # --------------------------------------------------------

    front_debug_checks: bool = True
    front_norm_tol: float = 1e-8

    front_debug_plot_enabled: bool = True
    front_debug_plot_once: bool = True
    front_debug_plot_every: int = 1
    front_debug_quiver_stride: int = 24
    front_debug_abs2_floor: float = 1e-10

    # --------------------------------------------------------
    # Post hoc export helper
    # --------------------------------------------------------

    front_export_posthoc_fields: bool = False

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

        _assert(
            self.front_branch_competition_strength >= 0.0,
            "front_branch_competition_strength must be >= 0",
        )
        _assert(
            self.front_branch_competition_power >= 0.0,
            "front_branch_competition_power must be >= 0",
        )
        _assert(
            self.front_branch_gate_power >= 0.0,
            "front_branch_gate_power must be >= 0",
        )

        _assert(
            self.front_branch_competition_x_weight >= 0.0,
            "front_branch_competition_x_weight must be >= 0",
        )
        _assert(
            self.front_branch_competition_y_weight >= 0.0,
            "front_branch_competition_y_weight must be >= 0",
        )
        _assert(
            self.front_branch_competition_diag_weight >= 0.0,
            "front_branch_competition_diag_weight must be >= 0",
        )

        _assert(
            self.front_branch_density_power >= 0.0,
            "front_branch_density_power must be >= 0",
        )
        _assert(
            self.front_branch_align_power >= 0.0,
            "front_branch_align_power must be >= 0",
        )
        _assert(
            self.front_branch_competition_threshold >= 0.0,
            "front_branch_competition_threshold must be >= 0",
        )

        _assert(
            isinstance(self.front_branch_competition_radius, int),
            "front_branch_competition_radius must be int",
        )
        _assert(
            self.front_branch_competition_radius >= 1,
            "front_branch_competition_radius must be >= 1",
        )

        _assert(
            self.front_branch_competition_margin > 0.0,
            "front_branch_competition_margin must be > 0",
        )
        _assert(
            self.front_branch_competition_blur_sigma >= 0.0,
            "front_branch_competition_blur_sigma must be >= 0",
        )

        _assert(
            self.front_branch_direction_mismatch_power >= 0.0,
            "front_branch_direction_mismatch_power must be >= 0",
        )
        _assert(
            self.front_branch_transverse_weight_power >= 0.0,
            "front_branch_transverse_weight_power must be >= 0",
        )
        _assert(
            self.front_branch_min_speed_fraction >= 0.0,
            "front_branch_min_speed_fraction must be >= 0",
        )
        _assert(
            self.front_branch_min_rho_fraction >= 0.0,
            "front_branch_min_rho_fraction must be >= 0",
        )

        _assert(
            self.front_branch_detector_gate_width > 0.0,
            "front_branch_detector_gate_width must be > 0",
        )
        _assert(
            self.front_branch_detector_gate_boost >= 0.0,
            "front_branch_detector_gate_boost must be >= 0",
        )

        _assert(
            isinstance(self.front_export_posthoc_fields, bool),
            "front_export_posthoc_fields must be bool",
        )

        self._debug_plot_counter = 0
        self._debug_plot_done = False

        self._competition_offsets = self._build_competition_offsets()
        self._detector_gate_cache = None

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
        z_xm = np.roll(z, 1, axis=1)
        z_yp = np.roll(z, -1, axis=0)
        z_ym = np.roll(z, 1, axis=0)

        z_d1 = np.roll(z_xp, -1, axis=0)
        z_d2 = np.roll(z_xp, 1, axis=0)
        z_d3 = np.roll(z_xm, -1, axis=0)
        z_d4 = np.roll(z_xm, 1, axis=0)

        axis_sum = z_xp + z_xm + z_yp + z_ym
        diag_sum = z_d1 + z_d2 + z_d3 + z_d4

        w_axis = 1.0
        w_diag = float(self.front_diag_weight)

        denom = 4.0 * w_axis + 4.0 * w_diag
        out = (w_axis * axis_sum + w_diag * diag_sum) / max(denom, self.front_eps)
        _assert_complex_array_2d(out, "_neighbor_average_complex(out)")
        return out

    def _build_competition_offsets(self):
        """
        Precompute integer offsets inside the competition radius.
        Each entry is (dy, dx, ox, oy), where (ox, oy) is the normalized offset.
        """
        radius = int(self.front_branch_competition_radius)
        offsets = []

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue

                dist2 = dx * dx + dy * dy
                if dist2 > radius * radius:
                    continue

                dist = float(np.sqrt(dist2))
                ox = dx / dist
                oy = dy / dist
                offsets.append((dy, dx, ox, oy))

        return offsets

    def _wrapped_take(self, arr: np.ndarray, ys: np.ndarray, xs: np.ndarray, dy: int, dx: int):
        """
        Gather arr[(ys+dy)%H, (xs+dx)%W] without rolling the full array.
        """
        h, w = arr.shape
        y2 = (ys + dy) % h
        x2 = (xs + dx) % w
        return arr[y2, x2]

    def _detector_competition_gate(self) -> np.ndarray:
        """
        Spatial gate that boosts competition near the detector / screen.

        Uses a Gaussian in x centered at front_branch_detector_gate_center_x.
        Cached because grid/X is static.
        """
        if self._detector_gate_cache is not None:
            return self._detector_gate_cache

        X = self.grid.X
        xc = float(self.front_branch_detector_gate_center_x)
        sigma = float(self.front_branch_detector_gate_width)

        gate = np.exp(-((X - xc) ** 2) / (2.0 * sigma ** 2)).astype(float)
        _assert_real_array_2d(gate, "detector_gate")
        _assert(np.all(gate >= -1e-14), "detector_gate contains significantly negative values")

        self._detector_gate_cache = gate
        return gate

    def _make_gamma_like(self, rho: np.ndarray, align_real: np.ndarray) -> np.ndarray:
        _assert_real_array_2d(rho, "rho(gamma_like)")
        _assert_real_array_2d(align_real, "align_real(gamma_like)")
        _assert(rho.shape == align_real.shape, "rho and align_real shape mismatch in gamma_like")

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
        return gamma_like

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

        align_real = np.real(np.conjugate(u) * u_local)

        _assert_real_array_2d(align_real, "align_real")
        _assert_complex_array_2d(u_local, "u_local")

        return align_real.astype(float), rho.astype(float), u_local

    def _flow_direction_field(self, psi: np.ndarray):
        """
        Compute Bohmian-like local flow direction from current:
            j = (hbar/m) Im(conj(psi) grad psi)
            v = j / rho

        Returns:
            jx, jy, rho, speed, ux, uy, valid_mask
        """
        _assert_complex_array_2d(psi, "psi(flow)")

        dpsi_dx = (
            np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)
        ) / (2.0 * self.grid.dx)

        dpsi_dy = (
            np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)
        ) / (2.0 * self.grid.dy)

        _assert_complex_array_2d(dpsi_dx, "dpsi_dx(flow)")
        _assert_complex_array_2d(dpsi_dy, "dpsi_dy(flow)")

        rho = (np.abs(psi) ** 2).astype(float)
        _assert_real_array_2d(rho, "rho(flow)")

        jx = (
            (self.hbar / self.m_mass)
            * np.imag(np.conjugate(psi) * dpsi_dx)
        ).astype(float)

        jy = (
            (self.hbar / self.m_mass)
            * np.imag(np.conjugate(psi) * dpsi_dy)
        ).astype(float)

        _assert_real_array_2d(jx, "jx(flow)")
        _assert_real_array_2d(jy, "jy(flow)")

        vx = jx / np.maximum(rho, self.front_eps)
        vy = jy / np.maximum(rho, self.front_eps)

        _assert_real_array_2d(vx, "vx(flow)")
        _assert_real_array_2d(vy, "vy(flow)")

        speed = np.sqrt(np.maximum(vx * vx + vy * vy, 0.0)).astype(float)
        _assert_real_array_2d(speed, "speed(flow)")

        speed_max = float(np.max(speed))
        rho_max = float(np.max(rho))

        speed_thr = float(self.front_branch_min_speed_fraction) * max(speed_max, self.front_eps)
        rho_thr = float(self.front_branch_min_rho_fraction) * max(rho_max, self.front_eps)

        valid_mask = (speed > speed_thr) & (rho > rho_thr)
        _assert_real_array_2d(valid_mask.astype(float), "valid_mask(flow)")

        ux = np.zeros_like(speed, dtype=float)
        uy = np.zeros_like(speed, dtype=float)

        denom = np.maximum(speed, self.front_eps)
        ux[valid_mask] = vx[valid_mask] / denom[valid_mask]
        uy[valid_mask] = vy[valid_mask] / denom[valid_mask]

        _assert_real_array_2d(ux, "ux(flow)")
        _assert_real_array_2d(uy, "uy(flow)")

        return (
            jx.astype(float),
            jy.astype(float),
            rho.astype(float),
            speed.astype(float),
            ux.astype(float),
            uy.astype(float),
            valid_mask.astype(bool),
        )

    def _branch_competition_field_simple(
        self,
        rho: np.ndarray,
        align_real: np.ndarray,
    ):
        """
        Old-style simple scalar competition based on local maximum_filter.
        Kept as fallback.
        """
        _assert_real_array_2d(rho, "rho")
        _assert_real_array_2d(align_real, "align_real")
        _assert(rho.shape == align_real.shape, "rho and align_real must have same shape")

        align_pos = np.maximum(align_real, 0.0)

        gamma_like = self._make_gamma_like(rho, align_real)

        filt_size = 2 * int(self.front_branch_competition_radius) + 1
        neighbor_max = maximum_filter(
            gamma_like,
            size=(filt_size, filt_size),
            mode="wrap",
        ).astype(float)

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

        gpow = float(self.front_branch_gate_power)
        if gpow > 0.0:
            competition_gate = np.power(np.maximum(align_pos, 0.0), gpow)
            competition_raw = competition_raw * competition_gate
        else:
            competition_gate = np.ones_like(competition_raw, dtype=float)

        if self.front_branch_competition_blur_sigma > 0.0:
            competition_raw = gaussian_filter(
                competition_raw,
                sigma=float(self.front_branch_competition_blur_sigma),
                mode="wrap",
            )

        return (
            gamma_like.astype(float),
            neighbor_max.astype(float),
            competition_raw.astype(float),
            competition_gate.astype(float),
            None,
            None,
        )

    def _branch_competition_field_flow(
        self,
        rho: np.ndarray,
        align_real: np.ndarray,
        ux: np.ndarray,
        uy: np.ndarray,
        valid_mask: np.ndarray,
    ):
        """
        Faster flow-aware branch competition.

        Main optimization:
          - evaluate competitors only on active pixels (valid_mask),
            instead of rolling full arrays for every offset.
          - gather shifted neighbor values by wrapped indexing.
        """
        _assert_real_array_2d(rho, "rho")
        _assert_real_array_2d(align_real, "align_real")
        _assert_real_array_2d(ux, "ux")
        _assert_real_array_2d(uy, "uy")
        _assert(isinstance(valid_mask, np.ndarray), "valid_mask must be ndarray")
        _assert(valid_mask.ndim == 2, "valid_mask must be 2D")
        _assert(
            rho.shape == align_real.shape == ux.shape == uy.shape == valid_mask.shape,
            "flow competition fields must have same shape"
        )

        align_pos = np.maximum(align_real, 0.0)

        gamma_like = self._make_gamma_like(rho, align_real)

        active = valid_mask.copy()
        active &= (gamma_like > 0.0)

        ys, xs = np.where(active)

        if ys.size == 0:
            zero = np.zeros_like(gamma_like, dtype=float)
            one_gate = np.ones_like(gamma_like, dtype=float)
            return (
                gamma_like.astype(float),
                zero,
                zero,
                one_gate,
                zero,
                zero,
            )

        gamma_a = gamma_like[ys, xs]
        ux_a = ux[ys, xs]
        uy_a = uy[ys, xs]

        nx_a = -uy_a
        ny_a = ux_a

        neighbor_max_a = np.zeros_like(gamma_a, dtype=float)
        competitor_score_a = np.zeros_like(gamma_a, dtype=float)
        transverse_best_a = np.zeros_like(gamma_a, dtype=float)
        direction_mismatch_best_a = np.zeros_like(gamma_a, dtype=float)

        mismatch_pow = float(self.front_branch_direction_mismatch_power)
        transverse_pow = float(self.front_branch_transverse_weight_power)

        for dy, dx, ox, oy in self._competition_offsets:
            gamma_s = self._wrapped_take(gamma_like, ys, xs, dy, dx)
            ux_s = self._wrapped_take(ux, ys, xs, dy, dx)
            uy_s = self._wrapped_take(uy, ys, xs, dy, dx)
            valid_s = self._wrapped_take(valid_mask, ys, xs, dy, dx)

            dot_dir = np.clip(ux_a * ux_s + uy_a * uy_s, -1.0, 1.0)
            dir_mismatch = np.maximum(1.0 - np.abs(dot_dir), 0.0)

            if mismatch_pow != 1.0:
                dir_mismatch = dir_mismatch ** mismatch_pow

            transverse = np.abs(nx_a * ox + ny_a * oy)

            if transverse_pow != 1.0 and transverse_pow > 0.0:
                transverse = transverse ** transverse_pow
            elif transverse_pow == 0.0:
                transverse = np.ones_like(transverse, dtype=float)

            valid_pair = valid_s.astype(float)
            weight = gamma_s * dir_mismatch * transverse * valid_pair

            neighbor_max_a = np.maximum(neighbor_max_a, gamma_s)
            competitor_score_a = np.maximum(competitor_score_a, weight)
            transverse_best_a = np.maximum(transverse_best_a, transverse * valid_pair)
            direction_mismatch_best_a = np.maximum(direction_mismatch_best_a, dir_mismatch * valid_pair)

        competition_raw_a = np.maximum(
            competitor_score_a - float(self.front_branch_competition_margin) * gamma_a,
            0.0,
        )

        competition_raw_a = np.maximum(
            competition_raw_a - float(self.front_branch_competition_threshold),
            0.0,
        )

        p = float(self.front_branch_competition_power)
        if p != 1.0:
            competition_raw_a = competition_raw_a ** p

        gpow = float(self.front_branch_gate_power)
        if gpow > 0.0:
            competition_gate = np.ones_like(gamma_like, dtype=float)
            competition_gate_a = np.power(np.maximum(align_pos[ys, xs], 0.0), gpow)
            competition_raw_a = competition_raw_a * competition_gate_a
            competition_gate[ys, xs] = competition_gate_a
        else:
            competition_gate = np.ones_like(gamma_like, dtype=float)

        neighbor_max = np.zeros_like(gamma_like, dtype=float)
        competition_raw = np.zeros_like(gamma_like, dtype=float)
        transverse_best = np.zeros_like(gamma_like, dtype=float)
        direction_mismatch_best = np.zeros_like(gamma_like, dtype=float)

        neighbor_max[ys, xs] = neighbor_max_a
        competition_raw[ys, xs] = competition_raw_a
        transverse_best[ys, xs] = transverse_best_a
        direction_mismatch_best[ys, xs] = direction_mismatch_best_a

        if self.front_branch_competition_blur_sigma > 0.0:
            competition_raw = gaussian_filter(
                competition_raw,
                sigma=float(self.front_branch_competition_blur_sigma),
                mode="wrap",
            )

        _assert_real_array_2d(competition_raw, "competition_raw(flow)")
        _assert_real_array_2d(competition_gate, "competition_gate(flow)")

        return (
            gamma_like.astype(float),
            neighbor_max.astype(float),
            competition_raw.astype(float),
            competition_gate.astype(float),
            transverse_best.astype(float),
            direction_mismatch_best.astype(float),
        )

    def _branch_competition_field(
        self,
        rho: np.ndarray,
        align_real: np.ndarray,
        ux: np.ndarray | None = None,
        uy: np.ndarray | None = None,
        valid_mask: np.ndarray | None = None,
    ):
        """
        Dispatcher: flow-aware branch competition if direction fields are provided,
        otherwise fallback to simple local maximum competition.
        """
        if (
            self.front_branch_use_flow_direction
            and ux is not None
            and uy is not None
            and valid_mask is not None
        ):
            return self._branch_competition_field_flow(
                rho=rho,
                align_real=align_real,
                ux=ux,
                uy=uy,
                valid_mask=valid_mask,
            )

        return self._branch_competition_field_simple(
            rho=rho,
            align_real=align_real,
        )

    # --------------------------------------------------------
    # Optional subclass hook
    # --------------------------------------------------------

    def _apply_optional_worldline_bias(
        self,
        psi_new: np.ndarray,
        psi_tmp: np.ndarray,
        rho_tmp: np.ndarray,
        align_real_tmp: np.ndarray,
        gamma_like: np.ndarray,
        dt: float,
    ):
        """
        Default no-op hook.

        Subclasses may override this to inject a frozen worldline bias,
        detector-oriented branch preference, or similar effects.

        Returns:
            psi_out, aux_worldline
        """
        _assert_complex_array_2d(psi_new, "psi_new(worldline_hook)")
        _assert_complex_array_2d(psi_tmp, "psi_tmp(worldline_hook)")
        _assert_real_array_2d(rho_tmp, "rho_tmp(worldline_hook)")
        _assert_real_array_2d(align_real_tmp, "align_real_tmp(worldline_hook)")
        _assert_real_array_2d(gamma_like, "gamma_like(worldline_hook)")
        _assert_finite_scalar(dt, "dt(worldline_hook)")
        return psi_new, {}

    # --------------------------------------------------------
    # Post hoc support export
    # --------------------------------------------------------

    def compute_posthoc_support_fields(self, state: np.ndarray) -> dict:
        """
        Compute lightweight fields for post hoc TRF analysis.

        Intended use:
            call this only on saved frames in the main run loop.

        Returned fields are based on the current state, not on hidden internal
        psi_tmp from _front_sharpen. That keeps the API simple and avoids
        bloating TheoryStepResult aux with large 2D arrays every step.
        """
        _assert_complex_array_2d(state, "state(posthoc_support_fields)")

        align_real, rho, _ = self._coherence_alignment_score(state)
        gamma_like = self._make_gamma_like(rho, align_real)

        out = {
            "rho": rho.astype(np.float32),
            "align_real": align_real.astype(np.float32),
            "gamma_like": gamma_like.astype(np.float32),
        }

        if self.front_branch_use_flow_direction:
            (
                _jx,
                _jy,
                _rho2,
                speed,
                ux,
                uy,
                valid_mask,
            ) = self._flow_direction_field(state)

            out["speed"] = speed.astype(np.float32)
            out["ux"] = ux.astype(np.float32)
            out["uy"] = uy.astype(np.float32)
            out["flow_valid_mask"] = valid_mask.astype(np.uint8)

        return out

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
        6) compute flow direction from psi_tmp
        7) compute competition from psi_tmp-derived fields
        8) optionally smooth competition
        9) optionally boost competition near detector
        10) apply competition damping
        11) optional subclass hook may apply worldline/frozen bias
        12) optional phase relaxation
        """
        _assert_complex_array_2d(psi, "psi")
        _assert_finite_scalar(dt, "dt")

        prob_in = self._state_probability(psi)

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

        align_real_tmp, rho_tmp, u_local_tmp = self._coherence_alignment_score(psi_tmp)

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

            comp_dt = np.clip(
                local_strength * competition_raw * dt,
                0.0,
                self.front_clip,
            )
            _assert_real_array_2d(comp_dt, "comp_dt")

            psi_new = psi_tmp * np.exp(-comp_dt)
        else:
            psi_new = psi_tmp
            gamma_like = self._make_gamma_like(rho_tmp, align_real_tmp)

        _assert_complex_array_2d(psi_new, "psi_new(after competition)")
        prob_after_comp = self._state_probability(psi_new)

        psi_new, aux_worldline = self._apply_optional_worldline_bias(
            psi_new=psi_new,
            psi_tmp=psi_tmp,
            rho_tmp=rho_tmp,
            align_real_tmp=align_real_tmp,
            gamma_like=gamma_like,
            dt=dt,
        )
        _assert_complex_array_2d(psi_new, "psi_new(after worldline hook)")
        prob_after_worldline = self._state_probability(psi_new)

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
            _assert(prob_after_worldline > 0.0, f"prob_after_worldline must be > 0, got {prob_after_worldline}")
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
            "prob_after_worldline": float(prob_after_worldline),
            "prob_after_relax": float(prob_after_relax),

            "gain_dt_mean": float(np.mean(gain_dt)),
            "gain_dt_max": float(np.max(gain_dt)),

            "flow_direction_enabled": bool(self.front_branch_use_flow_direction),
        }

        if speed_tmp is not None:
            aux_front["flow_speed_mean"] = float(np.mean(speed_tmp))
            aux_front["flow_speed_max"] = float(np.max(speed_tmp))
            aux_front["flow_valid_fraction"] = float(np.mean(flow_valid_mask.astype(float)))
            aux_front["flow_jx_mean_abs"] = float(np.mean(np.abs(jx_tmp)))
            aux_front["flow_jy_mean_abs"] = float(np.mean(np.abs(jy_tmp)))

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

        if aux_worldline:
            aux_front["worldline"] = aux_worldline

        if self.front_export_posthoc_fields:
            aux_front["posthoc_fields"] = {
                "rho_tmp": rho_tmp.astype(np.float32),
                "align_real_tmp": align_real_tmp.astype(np.float32),
                "gamma_like": gamma_like.astype(np.float32),
            }
            if speed_tmp is not None:
                aux_front["posthoc_fields"]["speed_tmp"] = speed_tmp.astype(np.float32)
                aux_front["posthoc_fields"]["ux_tmp"] = ux_tmp.astype(np.float32)
                aux_front["posthoc_fields"]["uy_tmp"] = uy_tmp.astype(np.float32)
                aux_front["posthoc_fields"]["flow_valid_mask"] = flow_valid_mask.astype(np.uint8)

        if self.front_debug_plot_enabled and (competition_raw is not None):
            comp_max = float(np.max(competition_raw))

            if comp_max > 1e-9:
                self._debug_plot_counter += 1

                should_plot = (
                    (not self.front_debug_plot_once or not self._debug_plot_done)
                    and (self._debug_plot_counter % max(1, self.front_debug_plot_every) == 0)
                )

                if should_plot:
                    self._debug_plot_front_fields(
                        psi_tmp=psi_tmp,
                        rho_tmp=rho_tmp,
                        gain_dt=gain_dt,
                        align_real_tmp=align_real_tmp,
                        speed_tmp=speed_tmp,
                        ux_tmp=ux_tmp,
                        uy_tmp=uy_tmp,
                        flow_valid_mask=flow_valid_mask,
                        gamma_like=gamma_like,
                        competition_raw=competition_raw,
                        transverse_best=transverse_best,
                        direction_mismatch_best=direction_mismatch_best,
                        comp_dt=comp_dt,
                    )
                    self._debug_plot_done = True

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
            _assert(
                np.isfinite(float(norm_factor)),
                f"normalize_unit returned non-finite norm_factor={norm_factor}",
            )
            _assert(
                prob_before_norm > 0.0,
                f"prob_before_norm must be > 0, got {prob_before_norm}",
            )
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

            "front_branch_use_flow_direction": bool(self.front_branch_use_flow_direction),
            "front_branch_direction_mismatch_power": float(self.front_branch_direction_mismatch_power),
            "front_branch_transverse_weight_power": float(self.front_branch_transverse_weight_power),
            "front_branch_min_speed_fraction": float(self.front_branch_min_speed_fraction),
            "front_branch_min_rho_fraction": float(self.front_branch_min_rho_fraction),

            "front_branch_detector_gate_enabled": bool(self.front_branch_detector_gate_enabled),
            "front_branch_detector_gate_center_x": float(self.front_branch_detector_gate_center_x),
            "front_branch_detector_gate_width": float(self.front_branch_detector_gate_width),
            "front_branch_detector_gate_boost": float(self.front_branch_detector_gate_boost),

            "front_export_posthoc_fields": bool(self.front_export_posthoc_fields),

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

    def _debug_plot_front_fields(
        self,
        psi_tmp: np.ndarray,
        rho_tmp: np.ndarray,
        gain_dt: np.ndarray,
        align_real_tmp: np.ndarray,
        speed_tmp: np.ndarray | None = None,
        ux_tmp: np.ndarray | None = None,
        uy_tmp: np.ndarray | None = None,
        flow_valid_mask: np.ndarray | None = None,
        gamma_like: np.ndarray | None = None,
        competition_raw: np.ndarray | None = None,
        transverse_best: np.ndarray | None = None,
        direction_mismatch_best: np.ndarray | None = None,
        comp_dt: np.ndarray | None = None,
    ):
        import matplotlib.pyplot as plt

        nrows, ncols = 2, 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 9))
        axes = axes.ravel()

        def _show(ax, arr, title, cmap="magma", symmetric=False):
            arr = np.asarray(arr)
            if symmetric:
                vmax = float(np.max(np.abs(arr))) if arr.size else 1.0
                if vmax <= 0.0:
                    vmax = 1.0
                im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax)
            else:
                im = ax.imshow(arr, origin="lower", cmap=cmap)
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        abs2 = np.abs(psi_tmp) ** 2
        _show(axes[0], abs2, "|psi_tmp|^2", cmap="magma")
        _show(axes[1], gain_dt, "gain_dt", cmap="RdBu_r", symmetric=True)
        _show(axes[2], align_real_tmp, "align_real_tmp", cmap="RdBu_r", symmetric=True)

        if competition_raw is not None:
            _show(axes[3], competition_raw, "competition_raw", cmap="viridis")
        elif gamma_like is not None:
            _show(axes[3], gamma_like, "gamma_like", cmap="viridis")
        else:
            axes[3].axis("off")

        if comp_dt is not None:
            _show(axes[4], comp_dt, "comp_dt", cmap="viridis")
        elif speed_tmp is not None:
            _show(axes[4], speed_tmp, "flow speed", cmap="viridis")
        else:
            axes[4].axis("off")

        phase = np.angle(psi_tmp)
        mask = abs2 > float(self.front_debug_abs2_floor)
        phase_vis = np.full_like(phase, np.nan, dtype=float)
        phase_vis[mask] = phase[mask]

        im = axes[5].imshow(phase_vis, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[5].set_title("arg(psi_tmp) + flow")
        plt.colorbar(im, ax=axes[5], fraction=0.046, pad=0.04)

        if (
            ux_tmp is not None
            and uy_tmp is not None
            and flow_valid_mask is not None
        ):
            stride = max(1, int(self.front_debug_quiver_stride))
            yy, xx = np.mgrid[0:ux_tmp.shape[0], 0:ux_tmp.shape[1]]

            xs = xx[::stride, ::stride]
            ys = yy[::stride, ::stride]
            uxs = ux_tmp[::stride, ::stride].copy()
            uys = uy_tmp[::stride, ::stride].copy()
            vms = flow_valid_mask[::stride, ::stride]

            uxs[~vms] = 0.0
            uys[~vms] = 0.0

            axes[5].quiver(xs, ys, uxs, uys, color="white", scale=25)

        plt.tight_layout()
        plt.show()


@dataclass
class ThickFrontWorldLineTheory(ThickFrontOptimizedTheory):
    """
    Conservative extension of ThickFrontOptimizedTheory with a persistent
    frozen-bias / worldline-like selection field.

    This is intentionally simpler than the full posthoc corridor+worldline
    pipeline from the experimental script. The goal here is to bring the idea
    into the main project in a form that:
      - keeps current optimized front behavior intact
      - adds one optional persistent branch/path bias
      - can later be refined toward the experimental TRF pipeline

    Typical modes:
      - "off"                  : disabled
      - "gain"                 : only boost chosen branch/tube region
      - "competition"          : only damp the outside / opposite region
      - "both"                 : boost chosen region and damp outside
      - "forced_weaker_branch" : choose second-strongest peak if available,
                                 then apply "both"
    """

    worldline_bias_enabled: bool = True
    worldline_bias_mode: str = "forced_weaker_branch"

    worldline_peak_radius_px: int = 18
    worldline_peak_rel_threshold: float = 0.03
    worldline_top_peaks_to_print: int = 3
    worldline_print_peak_info: bool = True

    worldline_bias_sigma_px: float = 10.0
    worldline_bias_gain_strength: float = 2.0
    worldline_bias_competition_strength: float = 0.20

    worldline_bias_gamma_power: float = 1.0
    worldline_bias_align_power: float = 1.0
    worldline_bias_blur_sigma: float = 1.0

    worldline_bias_persistent: bool = True
    worldline_time_ramp_strength: float = 1.0

    def __post_init__(self):
        super().__post_init__()

        valid_modes = {"off", "gain", "competition", "both", "forced_weaker_branch"}
        _assert(
            self.worldline_bias_mode in valid_modes,
            f"worldline_bias_mode must be one of {sorted(valid_modes)}, got {self.worldline_bias_mode}",
        )

        _assert(isinstance(self.worldline_peak_radius_px, int), "worldline_peak_radius_px must be int")
        _assert(self.worldline_peak_radius_px >= 1, "worldline_peak_radius_px must be >= 1")

        _assert(self.worldline_peak_rel_threshold >= 0.0, "worldline_peak_rel_threshold must be >= 0")
        _assert(self.worldline_top_peaks_to_print >= 1, "worldline_top_peaks_to_print must be >= 1")
        _assert(self.worldline_bias_sigma_px > 0.0, "worldline_bias_sigma_px must be > 0")
        _assert(self.worldline_bias_gain_strength >= 0.0, "worldline_bias_gain_strength must be >= 0")
        _assert(self.worldline_bias_competition_strength >= 0.0, "worldline_bias_competition_strength must be >= 0")
        _assert(self.worldline_bias_gamma_power >= 0.0, "worldline_bias_gamma_power must be >= 0")
        _assert(self.worldline_bias_align_power >= 0.0, "worldline_bias_align_power must be >= 0")
        _assert(self.worldline_bias_blur_sigma >= 0.0, "worldline_bias_blur_sigma must be >= 0")
        _assert(self.worldline_time_ramp_strength >= 0.0, "worldline_time_ramp_strength must be >= 0")

        self._worldline_bias_initialized = False
        self._worldline_bias_field = None
        self._worldline_selected_peak = None
        self._worldline_step_counter = 0

    def reset_runtime_state(self):
        """
        Call this before a fresh run if the same theory instance is reused.
        """
        self._worldline_bias_initialized = False
        self._worldline_bias_field = None
        self._worldline_selected_peak = None
        self._worldline_step_counter = 0
        self._debug_plot_counter = 0
        self._debug_plot_done = False

    def _build_gaussian_mask_px(self, iy_center: int, ix_center: int, shape, sigma_px: float):
        ny, nx = shape
        yy = np.arange(ny)[:, None]
        xx = np.arange(nx)[None, :]
        inv2s2 = 1.0 / max(2.0 * sigma_px * sigma_px, 1e-12)
        mask = np.exp(-((yy - iy_center) ** 2 + (xx - ix_center) ** 2) * inv2s2).astype(float)
        return mask

    def _find_local_peaks(self, gamma_like: np.ndarray):
        _assert_real_array_2d(gamma_like, "gamma_like(find_local_peaks)")

        peak_radius = int(self.worldline_peak_radius_px)
        filt_size = 2 * peak_radius + 1

        local_max = maximum_filter(gamma_like, size=(filt_size, filt_size), mode="wrap")
        is_peak = gamma_like >= local_max - 1e-15

        gmax = float(np.max(gamma_like))
        thr = float(self.worldline_peak_rel_threshold) * max(gmax, self.front_eps)
        is_peak &= (gamma_like >= thr)

        ys, xs = np.where(is_peak)
        peaks = []

        for iy, ix in zip(ys, xs):
            peaks.append({
                "iy": int(iy),
                "ix": int(ix),
                "value": float(gamma_like[iy, ix]),
            })

        peaks.sort(key=lambda rec: rec["value"], reverse=True)
        return peaks

    def _print_peak_info(self, peaks, chosen_idx: int):
        if not self.worldline_print_peak_info:
            return

        k = min(int(self.worldline_top_peaks_to_print), len(peaks))
        print("[WORLDLINE] top peaks:", flush=True)
        for i in range(k):
            rec = peaks[i]
            tag = " <= chosen" if i == chosen_idx else ""
            print(
                f"  #{i+1}: iy={rec['iy']} ix={rec['ix']} value={rec['value']:.6e}{tag}",
                flush=True,
            )

    def _choose_bias_peak(self, peaks):
        if len(peaks) == 0:
            return None, None

        if self.worldline_bias_mode == "forced_weaker_branch":
            chosen_idx = 1 if len(peaks) >= 2 else 0
        else:
            chosen_idx = 0

        chosen = peaks[chosen_idx]
        self._print_peak_info(peaks, chosen_idx)
        return chosen, chosen_idx

    def _initialize_worldline_bias(self, gamma_like: np.ndarray):
        peaks = self._find_local_peaks(gamma_like)
        chosen, _ = self._choose_bias_peak(peaks)

        if chosen is None:
            self._worldline_bias_field = np.zeros_like(gamma_like, dtype=float)
            self._worldline_selected_peak = None
            self._worldline_bias_initialized = True
            return

        iy = int(chosen["iy"])
        ix = int(chosen["ix"])

        bump = self._build_gaussian_mask_px(
            iy_center=iy,
            ix_center=ix,
            shape=gamma_like.shape,
            sigma_px=float(self.worldline_bias_sigma_px),
        )

        if self.worldline_bias_blur_sigma > 0.0:
            bump = gaussian_filter(
                bump,
                sigma=float(self.worldline_bias_blur_sigma),
                mode="wrap",
            )

        field = bump - float(np.mean(bump))

        maxabs = float(np.max(np.abs(field)))
        if maxabs > self.front_eps:
            field = field / maxabs

        _assert_real_array_2d(field, "worldline_bias_field")

        self._worldline_bias_field = field.astype(float)
        self._worldline_selected_peak = {
            "iy": iy,
            "ix": ix,
            "value": float(chosen["value"]),
        }
        self._worldline_bias_initialized = True

    def _worldline_gate(self, gamma_like: np.ndarray, align_real_tmp: np.ndarray):
        _assert_real_array_2d(gamma_like, "gamma_like(worldline_gate)")
        _assert_real_array_2d(align_real_tmp, "align_real_tmp(worldline_gate)")

        align_pos = np.maximum(align_real_tmp, 0.0)

        gate = (
            np.power(np.maximum(gamma_like, 0.0), float(self.worldline_bias_gamma_power))
            * np.power(np.maximum(align_pos, 0.0), float(self.worldline_bias_align_power))
        ).astype(float)

        if self.worldline_bias_blur_sigma > 0.0:
            gate = gaussian_filter(
                gate,
                sigma=float(self.worldline_bias_blur_sigma),
                mode="wrap",
            )

        gmax = float(np.max(gate))
        if gmax > self.front_eps:
            gate = gate / gmax

        _assert_real_array_2d(gate, "worldline_gate")
        return gate

    def _apply_optional_worldline_bias(
        self,
        psi_new: np.ndarray,
        psi_tmp: np.ndarray,
        rho_tmp: np.ndarray,
        align_real_tmp: np.ndarray,
        gamma_like: np.ndarray,
        dt: float,
    ):
        _assert_complex_array_2d(psi_new, "psi_new(worldline)")
        _assert_real_array_2d(gamma_like, "gamma_like(worldline)")
        _assert_real_array_2d(align_real_tmp, "align_real_tmp(worldline)")
        _assert_finite_scalar(dt, "dt(worldline)")

        self._worldline_step_counter += 1

        if (not self.worldline_bias_enabled) or self.worldline_bias_mode == "off":
            return psi_new, {
                "enabled": False,
                "mode": str(self.worldline_bias_mode),
            }

        if (not self._worldline_bias_initialized) or (not self.worldline_bias_persistent):
            self._initialize_worldline_bias(gamma_like)

        bias_field = self._worldline_bias_field
        if bias_field is None:
            return psi_new, {
                "enabled": False,
                "mode": str(self.worldline_bias_mode),
                "reason": "bias_field_none",
            }

        gate = self._worldline_gate(gamma_like, align_real_tmp)
        signed_drive = float(self.worldline_time_ramp_strength) * gate * bias_field

        pos_drive = np.maximum(signed_drive, 0.0)
        neg_drive = np.maximum(-signed_drive, 0.0)

        mode = self.worldline_bias_mode
        if mode == "forced_weaker_branch":
            mode = "both"

        gain_dt = np.zeros_like(gate, dtype=float)
        comp_dt = np.zeros_like(gate, dtype=float)

        if mode in {"gain", "both"} and self.worldline_bias_gain_strength > 0.0:
            gain_dt = np.clip(
                float(self.worldline_bias_gain_strength) * pos_drive * dt,
                0.0,
                self.front_clip,
            )

        if mode in {"competition", "both"} and self.worldline_bias_competition_strength > 0.0:
            comp_dt = np.clip(
                float(self.worldline_bias_competition_strength) * neg_drive * dt,
                0.0,
                self.front_clip,
            )

        psi_out = psi_new * np.exp(gain_dt) * np.exp(-comp_dt)
        _assert_complex_array_2d(psi_out, "psi_out(worldline)")

        aux = {
            "enabled": True,
            "mode": str(self.worldline_bias_mode),
            "persistent": bool(self.worldline_bias_persistent),
            "initialized": bool(self._worldline_bias_initialized),
            "step_counter": int(self._worldline_step_counter),
            "bias_field_mean": float(np.mean(bias_field)),
            "bias_field_max": float(np.max(bias_field)),
            "bias_field_min": float(np.min(bias_field)),
            "worldline_gate_mean": float(np.mean(gate)),
            "worldline_gate_max": float(np.max(gate)),
            "worldline_gain_dt_mean": float(np.mean(gain_dt)),
            "worldline_gain_dt_max": float(np.max(gain_dt)),
            "worldline_comp_dt_mean": float(np.mean(comp_dt)),
            "worldline_comp_dt_max": float(np.max(comp_dt)),
        }

        if self._worldline_selected_peak is not None:
            aux["selected_peak_iy"] = int(self._worldline_selected_peak["iy"])
            aux["selected_peak_ix"] = int(self._worldline_selected_peak["ix"])
            aux["selected_peak_value"] = float(self._worldline_selected_peak["value"])

        return psi_out, aux