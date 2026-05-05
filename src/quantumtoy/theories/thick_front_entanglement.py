# theories/thick_front_entanglement.py

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import maximum_filter, gaussian_filter

from theories.base import TheoryStepResult
from theories.schrodinger import SchrodingerTheory


CHANNELS = ["++", "+-", "-+", "--"]


# ============================================================
# Validation helpers
# ============================================================

def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def _assert_finite_scalar(x, name: str) -> float:
    _assert(np.isscalar(x), f"{name} must be scalar, got {type(x)}")
    xf = float(x)
    _assert(np.isfinite(xf), f"{name} must be finite, got {x}")
    return xf


def _assert_positive_scalar(x, name: str) -> float:
    xf = _assert_finite_scalar(x, name)
    _assert(xf > 0.0, f"{name} must be > 0, got {x}")
    return xf


def _assert_real_array_2d(arr: np.ndarray, name: str):
    _assert(isinstance(arr, np.ndarray), f"{name} must be np.ndarray")
    _assert(arr.ndim == 2, f"{name} must be 2D, got ndim={arr.ndim}")
    _assert(np.all(np.isfinite(arr)), f"{name} contains non-finite values")


def _assert_complex_array_2d(arr: np.ndarray, name: str):
    _assert(isinstance(arr, np.ndarray), f"{name} must be np.ndarray")
    _assert(arr.ndim == 2, f"{name} must be 2D, got ndim={arr.ndim}")
    _assert(np.all(np.isfinite(arr.real)), f"{name}.real contains non-finite values")
    _assert(np.all(np.isfinite(arr.imag)), f"{name}.imag contains non-finite values")


def _assert_complex_spinor_4d(arr: np.ndarray, name: str):
    _assert(isinstance(arr, np.ndarray), f"{name} must be np.ndarray")
    _assert(arr.ndim == 4, f"{name} must be 4D (Ny,Nx,2,2), got ndim={arr.ndim}")
    _assert(arr.shape[-2:] == (2, 2), f"{name} last dims must be (2,2), got {arr.shape}")
    _assert(np.all(np.isfinite(arr.real)), f"{name}.real contains non-finite values")
    _assert(np.all(np.isfinite(arr.imag)), f"{name}.imag contains non-finite values")


def _normalize_unit_spinor_4d(
    psi: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, float]:
    prob = float(np.sum(np.abs(psi) ** 2) * dx * dy)
    if prob <= 0.0:
        return psi, 0.0
    norm = float(np.sqrt(prob))
    return psi / norm, norm


def _identity2() -> np.ndarray:
    return np.eye(2, dtype=np.complex128)


def _spin_basis_from_angle(theta: float) -> np.ndarray:
    """
    Real spin-1/2 measurement basis rotation around y-axis.

    Columns are the + and - basis vectors in computational basis.

    This is enough for quick Bell/EPR-style angle sweeps:
        Ua = _spin_basis_from_angle(theta_a)
        Ub = _spin_basis_from_angle(theta_b)

    If you want arbitrary complex SU(2) bases later, pass custom matrices
    through ent_Ua / ent_Ub after construction.
    """
    c = np.cos(0.5 * float(theta))
    s = np.sin(0.5 * float(theta))
    return np.asarray(
        [
            [c, -s],
            [s,  c],
        ],
        dtype=np.complex128,
    )


def _validate_unitary_2x2(U: np.ndarray, name: str):
    U = np.asarray(U, dtype=np.complex128)
    _assert(U.shape == (2, 2), f"{name} must have shape (2,2), got {U.shape}")
    eye = U.conj().T @ U
    _assert(
        np.allclose(eye, np.eye(2), atol=1e-8, rtol=1e-8),
        f"{name} does not look unitary: U^dagger U={eye}",
    )
    return U


# ============================================================
# Measurement basis helpers
# ============================================================

def rotate_state_to_measurement_basis(
    psi: np.ndarray,
    Ua: np.ndarray,
    Ub: np.ndarray,
) -> np.ndarray:
    """
    psi shape:
        (Ny, Nx, 2, 2)

    Returns:
        psi_m shape:
        (Ny, Nx, 2, 2)

    Uses:
        psi_m[:, :, i, j] =
            sum_ab conj(Ua[a,i]) conj(Ub[b,j]) psi[:, :, a, b]

    Equivalent to applying Ua^dagger and Ub^dagger to the two spin indices.
    """
    _assert_complex_spinor_4d(psi, "psi(rotate)")
    Ua = _validate_unitary_2x2(Ua, "Ua")
    Ub = _validate_unitary_2x2(Ub, "Ub")

    tmp = np.einsum("ia,xyab->xyib", Ua.conj().T, psi)
    out = np.einsum("xyib,jb->xyij", tmp, Ub.conj().T)
    return out.astype(np.complex128)


def rotate_state_from_measurement_basis(
    psi_m: np.ndarray,
    Ua: np.ndarray,
    Ub: np.ndarray,
) -> np.ndarray:
    """
    Inverse of rotate_state_to_measurement_basis.
    """
    _assert_complex_spinor_4d(psi_m, "psi_m(inverse rotate)")
    Ua = _validate_unitary_2x2(Ua, "Ua")
    Ub = _validate_unitary_2x2(Ub, "Ub")

    tmp = np.einsum("ai,xyij->xyaj", Ua, psi_m)
    out = np.einsum("xyaj,bj->xyab", tmp, Ub)
    return out.astype(np.complex128)


def channel_component_densities(
    psi: np.ndarray,
    Ua: np.ndarray,
    Ub: np.ndarray,
) -> dict[str, np.ndarray]:
    psi_m = rotate_state_to_measurement_basis(psi, Ua, Ub)

    return {
        "++": np.abs(psi_m[:, :, 0, 0]) ** 2,
        "+-": np.abs(psi_m[:, :, 0, 1]) ** 2,
        "-+": np.abs(psi_m[:, :, 1, 0]) ** 2,
        "--": np.abs(psi_m[:, :, 1, 1]) ** 2,
    }


def channel_E_from_probs(probs: dict[str, float]) -> float:
    return float(probs["++"] + probs["--"] - probs["+-"] - probs["-+"])


# ============================================================
# Main theory
# ============================================================

@dataclass
class ThickFrontEntanglementTheory(SchrodingerTheory):
    """
    Thick-front theory for an entangled two-spinor field.

    State shape:
        psi[y, x, a, b] with a,b in {0,1}

    Base evolution:
        Schrödinger split-operator step is applied independently to each
        spin channel.

    Thick-front operator:
        scalar front sharpening is computed from total spinor density and
        local spinor phase/coherence.

    Branch competition:
        scalar inhibition is computed from total density/coherence and applied
        to all spin components.

    Entanglement selection:
        state is rotated into measurement basis Ua/Ub, channel densities are
        computed for ++,+-,-+,--, and a persistent or dynamic selected channel
        can be amplified while non-selected channels are damped.

    This gives a practical bridge between:
        - ThickFrontOptimizedTheory-style local sharpening/competition
        - entanglement_posthoc_trf.py channel evidence logic
    """

    # --------------------------------------------------------
    # Thick front parameters
    # --------------------------------------------------------

    front_strength: float = 0.03
    front_misaligned_damp: float = 0.01
    front_diag_weight: float = 0.5
    front_density_weighted: bool = True
    front_eps: float = 1e-12
    front_clip: float = 0.25
    front_gain_blur_sigma: float = 1.0

    # --------------------------------------------------------
    # Branch competition parameters
    # --------------------------------------------------------

    front_branch_competition_strength: float = 0.20
    front_branch_competition_power: float = 1.00
    front_branch_gate_power: float = 1.00
    front_branch_density_power: float = 1.0
    front_branch_align_power: float = 2.0
    front_branch_competition_threshold: float = 0.00
    front_branch_normalize_gamma: bool = True

    front_branch_competition_radius: int = 20
    front_branch_competition_margin: float = 0.90
    front_branch_competition_blur_sigma: float = 0.5

    front_branch_detector_gate_enabled: bool = True
    front_branch_detector_gate_center_x: float = 10.0
    front_branch_detector_gate_width: float = 2.0
    front_branch_detector_gate_boost: float = 10.0

    # --------------------------------------------------------
    # Entanglement measurement basis
    # --------------------------------------------------------

    ent_theta_a: float = 0.0
    ent_theta_b: float = 0.0

    # Optional custom 2x2 unitaries.
    # If None, they are built from ent_theta_a / ent_theta_b.
    ent_Ua: object | None = None
    ent_Ub: object | None = None

    # --------------------------------------------------------
    # Entanglement selection parameters
    # --------------------------------------------------------

    entanglement_enabled: bool = True

    # Modes:
    #   "off"
    #   "max_channel"
    #   "forced_weaker_channel"
    #   "fixed_channel"
    ent_channel_mode: str = "max_channel"
    ent_fixed_channel: str = "++"

    # If true, channel is chosen once and then kept.
    # If false, channel can be re-chosen every step.
    ent_channel_persistent: bool = True

    # Evidence weighting:
    #   False -> evidence = integral channel_density
    #   True  -> evidence = integral channel_density * gamma_like
    ent_channel_evidence_use_gamma: bool = True

    # Spatial selection field:
    #   "density" -> use full selected-channel density as gate
    #   "peak_tube" -> make Gaussian tube around strongest selected-channel peak
    ent_spatial_gate_mode: str = "peak_tube"

    ent_peak_radius_px: int = 18
    ent_peak_rel_threshold: float = 0.03

    ent_tube_sigma_px: float = 10.0
    ent_gate_blur_sigma: float = 1.0
    ent_gate_density_power: float = 1.0
    ent_gate_gamma_power: float = 0.5

    # Channel bias strengths.
    # Applied in measurement basis:
    #   selected channel *= exp(+gain_dt)
    #   other channels    *= exp(-damp_dt)
    ent_selected_gain_strength: float = 1.5
    ent_other_damp_strength: float = 0.25

    # Time ramp for the entanglement bias.
    # Effective gate grows roughly over this many steps.
    ent_time_ramp_steps: int = 50

    # Optional: prefer channels consistent with singlet anti-correlation.
    # This does not force results, it only weights evidence.
    ent_singlet_prior_enabled: bool = False
    ent_singlet_same_weight: float = 0.25
    ent_singlet_opposite_weight: float = 1.0

    ent_print_selection: bool = True

    # --------------------------------------------------------
    # Initial spin state
    # --------------------------------------------------------

    # "singlet", "product", "plus_plus", "plus_minus", "minus_plus", "minus_minus"
#    ent_initial_spin_state: str = "singlet"

    # Used only when ent_initial_spin_state == "product"
#    ent_initial_spin_a: object | None = None
#    ent_initial_spin_b: object | None = None

    ent_initial_spin_state = "product"

    ent_initial_spin_a = [0.70710678118, 0.70710678118]
    ent_initial_spin_b = [1.0, 0.0]

    # --------------------------------------------------------
    # Debug / export
    # --------------------------------------------------------

    front_debug_checks: bool = True
    front_norm_tol: float = 1e-8

    front_export_posthoc_fields: bool = False

    def __post_init__(self):
        super().__post_init__()

        np.seterr(divide="raise", over="raise", invalid="raise")

        self.front_eps = _assert_positive_scalar(self.front_eps, "front_eps")
        self.front_clip = _assert_positive_scalar(self.front_clip, "front_clip")
        self.front_norm_tol = _assert_positive_scalar(self.front_norm_tol, "front_norm_tol")

        _assert(self.front_strength >= 0.0, "front_strength must be >= 0")
        _assert(self.front_misaligned_damp >= 0.0, "front_misaligned_damp must be >= 0")
        _assert(self.front_diag_weight >= 0.0, "front_diag_weight must be >= 0")
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
            self.front_branch_detector_gate_width > 0.0,
            "front_branch_detector_gate_width must be > 0",
        )
        _assert(
            self.front_branch_detector_gate_boost >= 0.0,
            "front_branch_detector_gate_boost must be >= 0",
        )

        # --------------------------------------------------------
        # Initial spin state validation
        # --------------------------------------------------------

        valid_initial_spin_states = {
            "singlet",
            "product",
            "plus_plus",
            "plus_minus",
            "minus_plus",
            "minus_minus",
        }

        _assert(
            self.ent_initial_spin_state in valid_initial_spin_states,
            f"ent_initial_spin_state must be one of "
            f"{sorted(valid_initial_spin_states)}, got {self.ent_initial_spin_state}",
        )

        if self.ent_initial_spin_state == "product":
            if self.ent_initial_spin_a is not None:
                spin_a = np.asarray(self.ent_initial_spin_a, dtype=np.complex128)
                _assert(
                    spin_a.shape == (2,),
                    f"ent_initial_spin_a must have shape (2,), got {spin_a.shape}",
                )
                norm_a = float(np.sqrt(np.sum(np.abs(spin_a) ** 2)))
                _assert(norm_a > 0.0, "ent_initial_spin_a norm must be > 0")
                _assert(np.isfinite(norm_a), "ent_initial_spin_a norm must be finite")

            if self.ent_initial_spin_b is not None:
                spin_b = np.asarray(self.ent_initial_spin_b, dtype=np.complex128)
                _assert(
                    spin_b.shape == (2,),
                    f"ent_initial_spin_b must have shape (2,), got {spin_b.shape}",
                )
                norm_b = float(np.sqrt(np.sum(np.abs(spin_b) ** 2)))
                _assert(norm_b > 0.0, "ent_initial_spin_b norm must be > 0")
                _assert(np.isfinite(norm_b), "ent_initial_spin_b norm must be finite")

        # --------------------------------------------------------
        # Entanglement mode validation
        # --------------------------------------------------------

        valid_modes = {
            "off",
            "max_channel",
            "forced_weaker_channel",
            "fixed_channel",
        }

        _assert(
            self.ent_channel_mode in valid_modes,
            f"ent_channel_mode must be one of {sorted(valid_modes)}, "
            f"got {self.ent_channel_mode}",
        )

        _assert(
            self.ent_fixed_channel in CHANNELS,
            f"ent_fixed_channel must be one of {CHANNELS}, got {self.ent_fixed_channel}",
        )

        _assert(
            self.ent_spatial_gate_mode in {"density", "peak_tube"},
            "ent_spatial_gate_mode must be 'density' or 'peak_tube'",
        )

        _assert(
            isinstance(self.ent_peak_radius_px, int),
            "ent_peak_radius_px must be int",
        )
        _assert(
            self.ent_peak_radius_px >= 1,
            "ent_peak_radius_px must be >= 1",
        )
        _assert(
            self.ent_peak_rel_threshold >= 0.0,
            "ent_peak_rel_threshold must be >= 0",
        )
        _assert(
            self.ent_tube_sigma_px > 0.0,
            "ent_tube_sigma_px must be > 0",
        )
        _assert(
            self.ent_gate_blur_sigma >= 0.0,
            "ent_gate_blur_sigma must be >= 0",
        )
        _assert(
            self.ent_gate_density_power >= 0.0,
            "ent_gate_density_power must be >= 0",
        )
        _assert(
            self.ent_gate_gamma_power >= 0.0,
            "ent_gate_gamma_power must be >= 0",
        )
        _assert(
            self.ent_selected_gain_strength >= 0.0,
            "ent_selected_gain_strength must be >= 0",
        )
        _assert(
            self.ent_other_damp_strength >= 0.0,
            "ent_other_damp_strength must be >= 0",
        )
        _assert(
            isinstance(self.ent_time_ramp_steps, int),
            "ent_time_ramp_steps must be int",
        )
        _assert(
            self.ent_time_ramp_steps >= 1,
            "ent_time_ramp_steps must be >= 1",
        )

        _assert(
            self.ent_singlet_same_weight >= 0.0,
            "ent_singlet_same_weight must be >= 0",
        )
        _assert(
            self.ent_singlet_opposite_weight >= 0.0,
            "ent_singlet_opposite_weight must be >= 0",
        )

        # --------------------------------------------------------
        # Measurement basis setup
        # --------------------------------------------------------

        if self.ent_Ua is None:
            self._ent_Ua = _spin_basis_from_angle(float(self.ent_theta_a))
        else:
            self._ent_Ua = _validate_unitary_2x2(
                np.asarray(self.ent_Ua, dtype=np.complex128),
                "ent_Ua",
            )

        if self.ent_Ub is None:
            self._ent_Ub = _spin_basis_from_angle(float(self.ent_theta_b))
        else:
            self._ent_Ub = _validate_unitary_2x2(
                np.asarray(self.ent_Ub, dtype=np.complex128),
                "ent_Ub",
            )

        # --------------------------------------------------------
        # Runtime state
        # --------------------------------------------------------

        self._detector_gate_cache = None

        self._ent_selected_channel = None
        self._ent_selected_peak = None
        self._ent_initialized = False
        self._ent_step_counter = 0

    def initialize_click_state(
        self,
        x_click: float,
        y_click: float,
        sigma_click: float,
    ) -> np.ndarray:
        """
        Build a 4D entangled/spin-channel click state for backward propagation.

        The spatial part is a scalar Gaussian click packet centered at
        (x_click, y_click). Then it is lifted into the same spin structure
        as initialize_state().

        Returns:
            phi.shape == (Ny, Nx, 2, 2)
        """
        x = self.grid.X
        y = self.grid.Y

        sigma_click = _assert_positive_scalar(sigma_click, "sigma_click")

        spatial = np.exp(
            -0.5 * (
                ((x - float(x_click)) / sigma_click) ** 2
                + ((y - float(y_click)) / sigma_click) ** 2
            )
        ).astype(np.complex128)

        mode = str(self.ent_initial_spin_state)

        if mode == "singlet":
            phi = self.make_singlet_spinor_state(spatial)

        elif mode == "plus_plus":
            phi = np.zeros(spatial.shape + (2, 2), dtype=np.complex128)
            phi[:, :, 0, 0] = spatial

        elif mode == "plus_minus":
            phi = np.zeros(spatial.shape + (2, 2), dtype=np.complex128)
            phi[:, :, 0, 1] = spatial

        elif mode == "minus_plus":
            phi = np.zeros(spatial.shape + (2, 2), dtype=np.complex128)
            phi[:, :, 1, 0] = spatial

        elif mode == "minus_minus":
            phi = np.zeros(spatial.shape + (2, 2), dtype=np.complex128)
            phi[:, :, 1, 1] = spatial

        elif mode == "product":
            if self.ent_initial_spin_a is None:
                spin_a = np.asarray([1.0, 0.0], dtype=np.complex128)
            else:
                spin_a = np.asarray(self.ent_initial_spin_a, dtype=np.complex128)

            if self.ent_initial_spin_b is None:
                spin_b = np.asarray([1.0, 0.0], dtype=np.complex128)
            else:
                spin_b = np.asarray(self.ent_initial_spin_b, dtype=np.complex128)

            phi = self.make_product_spinor_state(
                spatial=spatial,
                spin_a=spin_a,
                spin_b=spin_b,
            )

        else:
            raise ValueError(f"Unknown ent_initial_spin_state={mode!r}")

        phi, norm_factor = _normalize_unit_spinor_4d(phi, self.grid.dx, self.grid.dy)
        phi = phi.astype(np.complex128)

        if self.front_debug_checks:
            prob = self._state_probability(phi)
            _assert(
                np.isclose(prob, 1.0, atol=self.front_norm_tol),
                f"initialize_click_state normalized probability should be 1, got {prob}",
            )
            _assert(
                np.isfinite(float(norm_factor)) and float(norm_factor) > 0.0,
                f"initialize_click_state norm_factor invalid: {norm_factor}",
            )

        return phi
        
    # --------------------------------------------------------
    # Runtime reset
    # --------------------------------------------------------

    def reset_runtime_state(self):
        """
        Call before a fresh run if the same theory instance is reused.
        """
        self._detector_gate_cache = None
        self._ent_selected_channel = None
        self._ent_selected_peak = None
        self._ent_initialized = False
        self._ent_step_counter = 0

    # --------------------------------------------------------
    # Basic spinor utilities
    # --------------------------------------------------------

    def initialize_state(self, psi0: np.ndarray) -> np.ndarray:
        """
        Accept either:

            psi0.shape == (Ny, Nx)
                ordinary scalar packet from the existing runner

            psi0.shape == (Ny, Nx, 2, 2)
                already-built entangled spinor state

        Returns normalized 4D spinor state.
        """
        if not isinstance(psi0, np.ndarray):
            raise TypeError(f"psi0 must be np.ndarray, got {type(psi0)}")

        if psi0.ndim == 4:
            _assert_complex_spinor_4d(psi0, "psi0(initialize_state)")
            out = psi0.astype(np.complex128)

        elif psi0.ndim == 2:
            _assert_complex_array_2d(psi0, "psi0(initialize_state scalar)")
            spatial = psi0.astype(np.complex128)

            mode = str(self.ent_initial_spin_state)

            if mode == "singlet":
                out = self.make_singlet_spinor_state(spatial)

            elif mode == "plus_plus":
                out = np.zeros(spatial.shape + (2, 2), dtype=np.complex128)
                out[:, :, 0, 0] = spatial

            elif mode == "plus_minus":
                out = np.zeros(spatial.shape + (2, 2), dtype=np.complex128)
                out[:, :, 0, 1] = spatial

            elif mode == "minus_plus":
                out = np.zeros(spatial.shape + (2, 2), dtype=np.complex128)
                out[:, :, 1, 0] = spatial

            elif mode == "minus_minus":
                out = np.zeros(spatial.shape + (2, 2), dtype=np.complex128)
                out[:, :, 1, 1] = spatial

            elif mode == "product":
                if self.ent_initial_spin_a is None:
                    spin_a = np.asarray([1.0, 0.0], dtype=np.complex128)
                else:
                    spin_a = np.asarray(self.ent_initial_spin_a, dtype=np.complex128)

                if self.ent_initial_spin_b is None:
                    spin_b = np.asarray([1.0, 0.0], dtype=np.complex128)
                else:
                    spin_b = np.asarray(self.ent_initial_spin_b, dtype=np.complex128)

                out = self.make_product_spinor_state(
                    spatial=spatial,
                    spin_a=spin_a,
                    spin_b=spin_b,
                )

            else:
                raise ValueError(f"Unknown ent_initial_spin_state={mode!r}")

        else:
            raise AssertionError(
                f"psi0 must be either 2D scalar packet or 4D spinor, got shape={psi0.shape}"
            )

        out, norm_factor = _normalize_unit_spinor_4d(out, self.grid.dx, self.grid.dy)
        out = out.astype(np.complex128)

        prob = self._state_probability(out)
        if self.front_debug_checks:
            _assert(
                np.isclose(prob, 1.0, atol=self.front_norm_tol),
                f"initialize_state normalized probability should be 1, got {prob}",
            )
            _assert(
                np.isfinite(float(norm_factor)) and float(norm_factor) > 0.0,
                f"initialize_state norm_factor invalid: {norm_factor}",
            )

        self.reset_runtime_state()

        return out

    def _state_probability(self, psi: np.ndarray) -> float:
        _assert_complex_spinor_4d(psi, "psi(prob)")
        prob = float(np.sum(np.abs(psi) ** 2) * self.grid.dx * self.grid.dy)
        _assert(np.isfinite(prob), "state probability is non-finite")
        _assert(prob >= 0.0, f"state probability must be >= 0, got {prob}")
        return prob

    def _total_density(self, psi: np.ndarray) -> np.ndarray:
        _assert_complex_spinor_4d(psi, "psi(total_density)")
        rho = np.sum(np.abs(psi) ** 2, axis=(-2, -1)).astype(float)
        _assert_real_array_2d(rho, "rho(total_density)")
        return rho

    def _neighbor_average_complex_spinor(self, z: np.ndarray) -> np.ndarray:
        """
        Neighborhood average for spinor-valued unit field.
        z shape: (Ny,Nx,2,2)
        """
        _assert_complex_spinor_4d(z, "z(neighbor_spinor)")

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
        _assert_complex_spinor_4d(out, "neighbor_average_spinor(out)")
        return out.astype(np.complex128)

    def _neighbor_average_real_2d(self, arr: np.ndarray) -> np.ndarray:
        _assert_real_array_2d(arr, "arr(neighbor_real)")

        z = arr.astype(np.complex128)
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

        out = ((w_axis * axis_sum + w_diag * diag_sum) / max(denom, self.front_eps)).real
        _assert_real_array_2d(out, "neighbor_average_real(out)")
        return out.astype(float)

    # --------------------------------------------------------
    # Coherence / current
    # --------------------------------------------------------

    def _coherence_alignment_score(self, psi: np.ndarray):
        """
        Spinor generalization of the scalar phase-alignment score.

        rho:
            total spinor density

        u:
            normalized local spinor direction:
                psi / sqrt(total_density)

        u_local:
            normalized neighboring spinor suggestion

        align_real:
            Re <u | u_local>, roughly [-1, 1]
        """
        _assert_complex_spinor_4d(psi, "psi(coherence)")

        rho = self._total_density(psi)
        amp = np.sqrt(np.maximum(rho, 0.0))

        u = psi / np.maximum(amp[:, :, None, None], self.front_eps)
        _assert_complex_spinor_4d(u, "u(coherence)")

        u_nei = self._neighbor_average_complex_spinor(u)

        if self.front_density_weighted:
            amp_nei = self._neighbor_average_real_2d(amp)
            amp_nei = np.maximum(amp_nei, 0.0)
            u_nei = u_nei * (1.0 + amp_nei[:, :, None, None])

        u_nei_norm = np.sqrt(np.sum(np.abs(u_nei) ** 2, axis=(-2, -1)))
        u_local = u_nei / np.maximum(u_nei_norm[:, :, None, None], self.front_eps)

        overlap = np.sum(np.conjugate(u) * u_local, axis=(-2, -1))
        align_real = np.real(overlap).astype(float)

        _assert_real_array_2d(align_real, "align_real")
        _assert_complex_spinor_4d(u_local, "u_local")

        return align_real, rho, u_local

    def _flow_current_total(self, psi: np.ndarray):
        """
        Spinor total current:
            j = (hbar/m) Im sum_ab conj(psi_ab) grad psi_ab
        """
        _assert_complex_spinor_4d(psi, "psi(flow)")

        dpsi_dx = (
            np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)
        ) / (2.0 * self.grid.dx)

        dpsi_dy = (
            np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)
        ) / (2.0 * self.grid.dy)

        rho = self._total_density(psi)

        jx = (
            (self.hbar / self.m_mass)
            * np.imag(np.sum(np.conjugate(psi) * dpsi_dx, axis=(-2, -1)))
        ).astype(float)

        jy = (
            (self.hbar / self.m_mass)
            * np.imag(np.sum(np.conjugate(psi) * dpsi_dy, axis=(-2, -1)))
        ).astype(float)

        _assert_real_array_2d(jx, "jx(flow)")
        _assert_real_array_2d(jy, "jy(flow)")

        return jx, jy, rho

    # --------------------------------------------------------
    # Branch competition
    # --------------------------------------------------------

    def _make_gamma_like(self, rho: np.ndarray, align_real: np.ndarray) -> np.ndarray:
        _assert_real_array_2d(rho, "rho(gamma)")
        _assert_real_array_2d(align_real, "align_real(gamma)")
        _assert(rho.shape == align_real.shape, "rho and align_real shape mismatch")

        align_pos = np.maximum(align_real, 0.0)

        gamma_like = (
            np.power(np.maximum(rho, 0.0), float(self.front_branch_density_power))
            * np.power(align_pos, float(self.front_branch_align_power))
        ).astype(float)

        if self.front_branch_normalize_gamma:
            gmax = float(np.max(gamma_like))
            if gmax > self.front_eps:
                gamma_like = gamma_like / gmax

        _assert_real_array_2d(gamma_like, "gamma_like")
        return gamma_like

    def _detector_competition_gate(self) -> np.ndarray:
        if self._detector_gate_cache is not None:
            return self._detector_gate_cache

        X = self.grid.X
        xc = float(self.front_branch_detector_gate_center_x)
        sigma = float(self.front_branch_detector_gate_width)

        gate = np.exp(-((X - xc) ** 2) / (2.0 * sigma ** 2)).astype(float)
        _assert_real_array_2d(gate, "detector_gate")

        self._detector_gate_cache = gate
        return gate

    def _branch_competition_field(
        self,
        rho: np.ndarray,
        align_real: np.ndarray,
    ):
        """
        Simple maximum-filter branch competition.

        This is intentionally the robust scalar version first. The flow-aware
        version can be ported later, but this keeps entanglement integration
        easier to validate.
        """
        _assert_real_array_2d(rho, "rho(branch_comp)")
        _assert_real_array_2d(align_real, "align_real(branch_comp)")

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

        align_pos = np.maximum(align_real, 0.0)
        gpow = float(self.front_branch_gate_power)
        if gpow > 0.0:
            competition_gate = np.power(align_pos, gpow)
            competition_raw = competition_raw * competition_gate
        else:
            competition_gate = np.ones_like(competition_raw, dtype=float)

        if self.front_branch_competition_blur_sigma > 0.0:
            competition_raw = gaussian_filter(
                competition_raw,
                sigma=float(self.front_branch_competition_blur_sigma),
                mode="wrap",
            )

        _assert_real_array_2d(gamma_like, "gamma_like(branch_comp)")
        _assert_real_array_2d(neighbor_max, "neighbor_max(branch_comp)")
        _assert_real_array_2d(competition_raw, "competition_raw(branch_comp)")
        _assert_real_array_2d(competition_gate, "competition_gate(branch_comp)")

        return gamma_like, neighbor_max, competition_raw, competition_gate

    # --------------------------------------------------------
    # Entanglement channel selection
    # --------------------------------------------------------

    def _channel_densities(self, psi: np.ndarray) -> dict[str, np.ndarray]:
        return channel_component_densities(psi, self._ent_Ua, self._ent_Ub)

    def _channel_evidence(
        self,
        channel_dens: dict[str, np.ndarray],
        gamma_like: np.ndarray,
    ) -> dict[str, float]:
        dxdy = float(self.grid.dx * self.grid.dy)
        ev = {}

        for ch in CHANNELS:
            arr = channel_dens[ch].astype(float)

            if self.ent_channel_evidence_use_gamma:
                arr = arr * np.maximum(gamma_like, 0.0)

            val = float(np.sum(arr) * dxdy)

            if self.ent_singlet_prior_enabled:
                if ch in {"++", "--"}:
                    val *= float(self.ent_singlet_same_weight)
                else:
                    val *= float(self.ent_singlet_opposite_weight)

            ev[ch] = val

        return ev

    def _channel_probs(self, ev: dict[str, float]) -> dict[str, float]:
        total = float(sum(ev.values()))
        if total <= 0.0:
            return {ch: 0.0 for ch in CHANNELS}
        return {ch: float(ev[ch] / total) for ch in CHANNELS}

    def _choose_entanglement_channel(
        self,
        channel_dens: dict[str, np.ndarray],
        gamma_like: np.ndarray,
    ):
        ev = self._channel_evidence(channel_dens, gamma_like)
        probs = self._channel_probs(ev)

        if self.ent_channel_mode == "off":
            chosen = None
        elif self.ent_channel_mode == "fixed_channel":
            chosen = str(self.ent_fixed_channel)
        else:
            ordered = sorted(CHANNELS, key=lambda ch: ev[ch], reverse=True)
            if self.ent_channel_mode == "forced_weaker_channel" and len(ordered) >= 2:
                chosen = ordered[1]
            else:
                chosen = ordered[0]

        total = float(sum(ev.values()))
        max_ev = max(ev.values()) if ev else 0.0
        min_nonzero = min([v for v in ev.values() if v > 0.0], default=1e-30)

        info = {
            "chosen_channel": chosen,
            "channel_evidence": ev,
            "channel_probs": probs,
            "E": channel_E_from_probs(probs),
            "total_evidence": total,
            "dominance": float(max_ev / max(total, 1e-30)),
            "ratio": float(max_ev / max(min_nonzero, 1e-30)),
        }

        return info

    def _find_peak_in_channel(self, arr: np.ndarray):
        _assert_real_array_2d(arr, "arr(find_peak)")

        peak_radius = int(self.ent_peak_radius_px)
        filt_size = 2 * peak_radius + 1

        local_max = maximum_filter(arr, size=(filt_size, filt_size), mode="wrap")
        is_peak = arr >= local_max - 1e-15

        amax = float(np.max(arr))
        thr = float(self.ent_peak_rel_threshold) * max(amax, self.front_eps)
        is_peak &= arr >= thr

        ys, xs = np.where(is_peak)
        if ys.size == 0:
            iy, ix = np.unravel_index(int(np.argmax(arr)), arr.shape)
            return {
                "iy": int(iy),
                "ix": int(ix),
                "value": float(arr[iy, ix]),
                "fallback_argmax": True,
            }

        peaks = []
        for iy, ix in zip(ys, xs):
            peaks.append({
                "iy": int(iy),
                "ix": int(ix),
                "value": float(arr[iy, ix]),
                "fallback_argmax": False,
            })

        peaks.sort(key=lambda r: r["value"], reverse=True)
        return peaks[0]

    def _build_gaussian_mask_px(
        self,
        iy_center: int,
        ix_center: int,
        shape: tuple[int, int],
        sigma_px: float,
    ) -> np.ndarray:
        ny, nx = shape
        yy = np.arange(ny)[:, None]
        xx = np.arange(nx)[None, :]

        inv2s2 = 1.0 / max(2.0 * sigma_px * sigma_px, 1e-12)
        mask = np.exp(
            -((yy - int(iy_center)) ** 2 + (xx - int(ix_center)) ** 2) * inv2s2
        ).astype(float)

        _assert_real_array_2d(mask, "gaussian_mask")
        return mask

    def _build_entanglement_gate(
        self,
        selected_density: np.ndarray,
        gamma_like: np.ndarray,
    ) -> np.ndarray:
        _assert_real_array_2d(selected_density, "selected_density(ent_gate)")
        _assert_real_array_2d(gamma_like, "gamma_like(ent_gate)")

        if self.ent_spatial_gate_mode == "peak_tube":
            if self._ent_selected_peak is None or not self.ent_channel_persistent:
                self._ent_selected_peak = self._find_peak_in_channel(selected_density)

            iy = int(self._ent_selected_peak["iy"])
            ix = int(self._ent_selected_peak["ix"])

            gate = self._build_gaussian_mask_px(
                iy_center=iy,
                ix_center=ix,
                shape=selected_density.shape,
                sigma_px=float(self.ent_tube_sigma_px),
            )
        else:
            gate = np.maximum(selected_density, 0.0).astype(float)
            gmax = float(np.max(gate))
            if gmax > self.front_eps:
                gate = gate / gmax

        if self.ent_gate_density_power != 1.0:
            gate = np.power(np.maximum(gate, 0.0), float(self.ent_gate_density_power))

        if self.ent_gate_gamma_power > 0.0:
            g = np.maximum(gamma_like, 0.0)
            gmax = float(np.max(g))
            if gmax > self.front_eps:
                g = g / gmax
            gate = gate * np.power(g, float(self.ent_gate_gamma_power))

        if self.ent_gate_blur_sigma > 0.0:
            gate = gaussian_filter(
                gate,
                sigma=float(self.ent_gate_blur_sigma),
                mode="wrap",
            )

        gmax = float(np.max(gate))
        if gmax > self.front_eps:
            gate = gate / gmax

        _assert_real_array_2d(gate, "entanglement_gate")
        return gate.astype(float)

    def _entanglement_time_ramp(self) -> float:
        n = max(1, int(self.ent_time_ramp_steps))
        u = min(1.0, float(self._ent_step_counter) / float(n))
        return float(0.5 - 0.5 * np.cos(np.pi * u))

    def _apply_entanglement_channel_bias(
        self,
        psi: np.ndarray,
        gamma_like: np.ndarray,
        dt: float,
    ):
        _assert_complex_spinor_4d(psi, "psi(ent_bias)")
        _assert_real_array_2d(gamma_like, "gamma_like(ent_bias)")
        _assert_finite_scalar(dt, "dt(ent_bias)")

        if (not self.entanglement_enabled) or self.ent_channel_mode == "off":
            return psi, {
                "enabled": False,
                "reason": "disabled_or_off",
            }

        channel_dens = self._channel_densities(psi)

        if (not self._ent_initialized) or (not self.ent_channel_persistent):
            info = self._choose_entanglement_channel(channel_dens, gamma_like)
            self._ent_selected_channel = info["chosen_channel"]
            self._ent_initialized = True

            if self.ent_print_selection:
                print(
                    "[ENT] selected channel:",
                    self._ent_selected_channel,
                    "| probs:",
                    {ch: f"{info['channel_probs'][ch]:.4f}" for ch in CHANNELS},
                    "| E:",
                    f"{info['E']:.4f}",
                    flush=True,
                )
        else:
            info = self._choose_entanglement_channel(channel_dens, gamma_like)
            info["chosen_channel"] = self._ent_selected_channel

        ch = self._ent_selected_channel
        if ch not in CHANNELS:
            return psi, {
                "enabled": False,
                "reason": "no_valid_selected_channel",
                **info,
            }

        selected_density = channel_dens[ch]
        gate = self._build_entanglement_gate(selected_density, gamma_like)

        ramp = self._entanglement_time_ramp()

        gain_dt = np.clip(
            float(self.ent_selected_gain_strength) * ramp * gate * dt,
            0.0,
            self.front_clip,
        )

        damp_dt = np.clip(
            float(self.ent_other_damp_strength) * ramp * gate * dt,
            0.0,
            self.front_clip,
        )

        psi_m = rotate_state_to_measurement_basis(psi, self._ent_Ua, self._ent_Ub)

        idx = {
            "++": (0, 0),
            "+-": (0, 1),
            "-+": (1, 0),
            "--": (1, 1),
        }

        for ch2 in CHANNELS:
            a, b = idx[ch2]
            if ch2 == ch:
                psi_m[:, :, a, b] *= np.exp(gain_dt)
            else:
                psi_m[:, :, a, b] *= np.exp(-damp_dt)

        psi_out = rotate_state_from_measurement_basis(psi_m, self._ent_Ua, self._ent_Ub)
        _assert_complex_spinor_4d(psi_out, "psi_out(ent_bias)")

        aux = {
            "enabled": True,
            "mode": str(self.ent_channel_mode),
            "persistent": bool(self.ent_channel_persistent),
            "selected_channel": str(ch),
            "step_counter": int(self._ent_step_counter),
            "time_ramp": float(ramp),
            "gate_mean": float(np.mean(gate)),
            "gate_max": float(np.max(gate)),
            "gain_dt_mean": float(np.mean(gain_dt)),
            "gain_dt_max": float(np.max(gain_dt)),
            "damp_dt_mean": float(np.mean(damp_dt)),
            "damp_dt_max": float(np.max(damp_dt)),
            **info,
        }

        if self._ent_selected_peak is not None:
            aux["selected_peak_iy"] = int(self._ent_selected_peak["iy"])
            aux["selected_peak_ix"] = int(self._ent_selected_peak["ix"])
            aux["selected_peak_value"] = float(self._ent_selected_peak["value"])

        return psi_out, aux

    # --------------------------------------------------------
    # Front operator
    # --------------------------------------------------------

    def _front_sharpen_spinor(self, psi: np.ndarray, dt: float):
        _assert_complex_spinor_4d(psi, "psi(front)")
        _assert_finite_scalar(dt, "dt(front)")

        prob_in = self._state_probability(psi)

        align_real, rho, _u_local = self._coherence_alignment_score(psi)

        gain = (
            float(self.front_strength) * np.maximum(align_real, 0.0)
            - float(self.front_misaligned_damp) * np.maximum(-align_real, 0.0)
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

        gain_dt = np.clip(gain * dt, -self.front_clip, self.front_clip)
        _assert_real_array_2d(gain_dt, "gain_dt")

        psi_tmp = psi * np.exp(gain_dt)[:, :, None, None]
        _assert_complex_spinor_4d(psi_tmp, "psi_tmp")

        prob_after_gain = self._state_probability(psi_tmp)

        align_real_tmp, rho_tmp, _u_local_tmp = self._coherence_alignment_score(psi_tmp)

        gamma_like = None
        neighbor_max = None
        competition_raw = None
        competition_gate = None
        comp_dt = None
        detector_gate = None
        local_strength = None

        if self.front_branch_competition_strength > 0.0:
            (
                gamma_like,
                neighbor_max,
                competition_raw,
                competition_gate,
            ) = self._branch_competition_field(rho_tmp, align_real_tmp)

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

            comp_dt = np.clip(
                local_strength * competition_raw * dt,
                0.0,
                self.front_clip,
            )

            psi_new = psi_tmp * np.exp(-comp_dt)[:, :, None, None]
        else:
            psi_new = psi_tmp
            gamma_like = self._make_gamma_like(rho_tmp, align_real_tmp)

        _assert_complex_spinor_4d(psi_new, "psi_new(after competition)")
        prob_after_comp = self._state_probability(psi_new)

        psi_new, aux_ent = self._apply_entanglement_channel_bias(
            psi=psi_new,
            gamma_like=gamma_like,
            dt=dt,
        )

        _assert_complex_spinor_4d(psi_new, "psi_new(after entanglement)")
        prob_after_ent = self._state_probability(psi_new)

        if self.front_debug_checks:
            _assert(prob_in > 0.0, f"prob_in must be > 0, got {prob_in}")
            _assert(prob_after_gain > 0.0, f"prob_after_gain must be > 0, got {prob_after_gain}")
            _assert(prob_after_comp > 0.0, f"prob_after_comp must be > 0, got {prob_after_comp}")
            _assert(prob_after_ent > 0.0, f"prob_after_ent must be > 0, got {prob_after_ent}")

            if self.front_branch_competition_strength > 0.0:
                _assert(
                    prob_after_comp <= prob_after_gain + 1e-12,
                    f"competition should not increase norm: after_comp={prob_after_comp}, after_gain={prob_after_gain}",
                )

        jx, jy, rho_current = self._flow_current_total(psi_tmp)
        speed = np.sqrt(jx * jx + jy * jy) / np.maximum(rho_current, self.front_eps)

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
            "prob_after_entanglement": float(prob_after_ent),

            "gain_dt_mean": float(np.mean(gain_dt)),
            "gain_dt_max": float(np.max(gain_dt)),

            "flow_speed_mean": float(np.mean(speed)),
            "flow_speed_max": float(np.max(speed)),

            "entanglement": aux_ent,
        }

        if competition_raw is not None:
            aux_front.update({
                "branch_gamma_like_mean": float(np.mean(gamma_like)),
                "branch_gamma_like_max": float(np.max(gamma_like)),
                "branch_neighbor_max_mean": float(np.mean(neighbor_max)),
                "branch_neighbor_max_max": float(np.max(neighbor_max)),
                "branch_competition_mean": float(np.mean(competition_raw)),
                "branch_competition_max": float(np.max(competition_raw)),
                "branch_gate_mean": float(np.mean(competition_gate)),
                "branch_gate_max": float(np.max(competition_gate)),
                "branch_comp_dt_mean": float(np.mean(comp_dt)),
                "branch_comp_dt_max": float(np.max(comp_dt)),
                "branch_local_strength_mean": float(np.mean(local_strength)),
                "branch_local_strength_max": float(np.max(local_strength)),
            })

            if detector_gate is not None:
                aux_front["branch_detector_gate_mean"] = float(np.mean(detector_gate))
                aux_front["branch_detector_gate_max"] = float(np.max(detector_gate))

        if self.front_export_posthoc_fields:
            aux_front["posthoc_fields"] = {
                "rho_tmp": rho_tmp.astype(np.float32),
                "align_real_tmp": align_real_tmp.astype(np.float32),
                "gamma_like": gamma_like.astype(np.float32),
            }

            ch_dens = self._channel_densities(psi_tmp)
            aux_front["posthoc_fields"]["channel_density"] = {
                ch: ch_dens[ch].astype(np.float32) for ch in CHANNELS
            }

        return psi_new.astype(np.complex128), aux_front

    # --------------------------------------------------------
    # Base Schrödinger stepping for each spin channel
    # --------------------------------------------------------

    def _step_forward_componentwise(self, state: np.ndarray, dt: float):
        _assert_complex_spinor_4d(state, "state(componentwise)")
        _assert_finite_scalar(dt, "dt(componentwise)")

        out = np.empty_like(state, dtype=np.complex128)
        aux_components = {}

        for a in range(2):
            for b in range(2):
                comp = state[:, :, a, b].astype(np.complex128)
                _assert_complex_array_2d(comp, f"component[{a},{b}]")

                res = super().step_forward(comp, dt)
                out[:, :, a, b] = res.state.astype(np.complex128)

                if res.aux:
                    aux_components[f"{a}{b}"] = res.aux

        return out, aux_components

    def _step_backward_componentwise(self, state: np.ndarray, dt: float):
        _assert_complex_spinor_4d(state, "state(backward_componentwise)")
        _assert_finite_scalar(dt, "dt(backward_componentwise)")

        out = np.empty_like(state, dtype=np.complex128)
        aux_components = {}

        for a in range(2):
            for b in range(2):
                comp = state[:, :, a, b].astype(np.complex128)
                _assert_complex_array_2d(comp, f"backward_component[{a},{b}]")

                res = super().step_backward_adjoint(comp, dt)
                out[:, :, a, b] = res.state.astype(np.complex128)

                if res.aux:
                    aux_components[f"{a}{b}"] = res.aux

        return out, aux_components

    # --------------------------------------------------------
    # Public stepping API
    # --------------------------------------------------------

    def step_forward(self, state: np.ndarray, dt: float) -> TheoryStepResult:
        """
        One forward step:
            componentwise Schrödinger step
            + spinor thick-front sharpening
            + scalar branch competition
            + measurement-basis channel selection
            + final full-spinor normalization
        """
        _assert_complex_spinor_4d(state, "state")
        _assert_finite_scalar(dt, "dt")

        self._ent_step_counter += 1

        psi, aux_base_components = self._step_forward_componentwise(state, dt)
        prob_after_base = self._state_probability(psi)

        psi, aux_front = self._front_sharpen_spinor(psi, dt)
        prob_before_norm = self._state_probability(psi)

        psi, norm_factor = _normalize_unit_spinor_4d(psi, self.grid.dx, self.grid.dy)
        psi = psi.astype(np.complex128)

        prob_after_norm = self._state_probability(psi)

        if self.front_debug_checks:
            _assert(np.isfinite(float(norm_factor)), f"norm_factor non-finite: {norm_factor}")
            _assert(prob_before_norm > 0.0, f"prob_before_norm must be > 0, got {prob_before_norm}")
            _assert(
                np.isclose(prob_after_norm, 1.0, atol=self.front_norm_tol),
                f"final normalized probability must be 1, got {prob_after_norm}",
            )

        aux = {
            "thick_front_entanglement": {
                "front_strength": float(self.front_strength),
                "front_misaligned_damp": float(self.front_misaligned_damp),
                "front_diag_weight": float(self.front_diag_weight),
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

                "entanglement_enabled": bool(self.entanglement_enabled),
                "ent_theta_a": float(self.ent_theta_a),
                "ent_theta_b": float(self.ent_theta_b),
                "ent_channel_mode": str(self.ent_channel_mode),
                "ent_fixed_channel": str(self.ent_fixed_channel),
                "ent_channel_persistent": bool(self.ent_channel_persistent),
                "ent_channel_evidence_use_gamma": bool(self.ent_channel_evidence_use_gamma),
                "ent_spatial_gate_mode": str(self.ent_spatial_gate_mode),
                "ent_selected_gain_strength": float(self.ent_selected_gain_strength),
                "ent_other_damp_strength": float(self.ent_other_damp_strength),
                "ent_time_ramp_steps": int(self.ent_time_ramp_steps),
                "ent_singlet_prior_enabled": bool(self.ent_singlet_prior_enabled),

                "prob_after_base": float(prob_after_base),
                "prob_before_norm": float(prob_before_norm),
                "prob_after_norm": float(prob_after_norm),
                "normalize_unit_returned_norm": float(norm_factor),

                **aux_front,
            }
        }

        if aux_base_components:
            aux["base_components"] = aux_base_components

        return TheoryStepResult(state=psi, aux=aux)

    def step_backward_adjoint(self, state: np.ndarray, dt: float) -> TheoryStepResult:
        """
        Backward evolution for entangled spinor.

        Kept as plain componentwise adjoint Schrödinger evolution. This matches
        the safer post-hoc/TRF library style: no nonlinear front backwards.
        """
        _assert_complex_spinor_4d(state, "state(backward)")
        _assert_finite_scalar(dt, "dt(backward)")

        psi, aux_components = self._step_backward_componentwise(state, dt)

        return TheoryStepResult(
            state=psi.astype(np.complex128),
            aux={
                "thick_front_entanglement_backward": {
                    "mode": "componentwise_plain_schrodinger_adjoint",
                    "base_components": aux_components,
                }
            },
        )

    # --------------------------------------------------------
    # Observables
    # --------------------------------------------------------

    def density(self, state: np.ndarray) -> np.ndarray:
        """
        Total density over both spin indices.
        """
        rho = self._total_density(state)
        _assert(np.all(rho >= -1e-14), "rho contains significantly negative values")
        return rho

    def channel_densities(self, state: np.ndarray) -> dict[str, np.ndarray]:
        """
        Measurement-basis channel densities.
        Useful for plotting ++,+-,-+,--.
        """
        _assert_complex_spinor_4d(state, "state(channel_densities)")
        return self._channel_densities(state)

    def channel_probabilities(self, state: np.ndarray) -> dict[str, float]:
        """
        Integrated channel probabilities in current measurement basis.
        """
        dens = self.channel_densities(state)
        dxdy = float(self.grid.dx * self.grid.dy)
        ev = {ch: float(np.sum(dens[ch]) * dxdy) for ch in CHANNELS}
        total = float(sum(ev.values()))
        if total <= 0.0:
            return {ch: 0.0 for ch in CHANNELS}
        return {ch: float(ev[ch] / total) for ch in CHANNELS}

    def entanglement_E(self, state: np.ndarray) -> float:
        """
        E = P(++ or --) - P(+- or -+)
        """
        probs = self.channel_probabilities(state)
        return channel_E_from_probs(probs)

    def current(self, state_vis: np.ndarray):
        """
        Total spinor current.

        Returns:
            jx, jy, rho_total
        """
        _assert_complex_spinor_4d(state_vis, "state_vis(current)")
        return self._flow_current_total(state_vis)

    # --------------------------------------------------------
    # Initial-state convenience helpers
    # --------------------------------------------------------

    @staticmethod
    def make_product_spinor_state(
        spatial: np.ndarray,
        spin_a: np.ndarray,
        spin_b: np.ndarray,
    ) -> np.ndarray:
        """
        Build:
            psi[y,x,a,b] = spatial[y,x] * spin_a[a] * spin_b[b]
        """
        _assert_complex_array_2d(spatial, "spatial")
        spin_a = np.asarray(spin_a, dtype=np.complex128)
        spin_b = np.asarray(spin_b, dtype=np.complex128)

        _assert(spin_a.shape == (2,), f"spin_a must have shape (2,), got {spin_a.shape}")
        _assert(spin_b.shape == (2,), f"spin_b must have shape (2,), got {spin_b.shape}")

        na = float(np.sqrt(np.sum(np.abs(spin_a) ** 2)))
        nb = float(np.sqrt(np.sum(np.abs(spin_b) ** 2)))
        _assert(na > 0.0, "spin_a norm must be > 0")
        _assert(nb > 0.0, "spin_b norm must be > 0")

        spin_a = spin_a / na
        spin_b = spin_b / nb

        psi = spatial[:, :, None, None] * spin_a[None, None, :, None] * spin_b[None, None, None, :]
        return psi.astype(np.complex128)

    @staticmethod
    def make_singlet_spinor_state(spatial: np.ndarray) -> np.ndarray:
        """
        Build singlet-like spin state:
            (|+-> - |-+>) / sqrt(2)
        with shared spatial packet.
        """
        _assert_complex_array_2d(spatial, "spatial")

        psi = np.zeros(spatial.shape + (2, 2), dtype=np.complex128)
        psi[:, :, 0, 1] = spatial / np.sqrt(2.0)
        psi[:, :, 1, 0] = -spatial / np.sqrt(2.0)
        return psi

    @staticmethod
    def spin_basis_from_angle(theta: float) -> np.ndarray:
        return _spin_basis_from_angle(theta)