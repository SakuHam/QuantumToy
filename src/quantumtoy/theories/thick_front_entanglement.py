# theories/thick_front_entanglement.py

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import maximum_filter, gaussian_filter

from theories.base import TheoryStepResult
from theories.schrodinger import SchrodingerTheory


CHANNELS = ["++", "+-", "-+", "--"]
C_M_PER_S = 299_792_458.0
MM_TO_M = 1_000_000.0


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


def _spin_basis_from_angle(theta: float) -> np.ndarray:
    """
    Real spin-1/2 measurement basis rotation around y-axis.

    Columns are the + and - basis vectors in computational basis.
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


def singlet_spin_state() -> np.ndarray:
    """
    Return the spin-only singlet amplitudes in channel order [Alice, Bob].
    """
    spin = np.zeros((2, 2), dtype=np.complex128)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    spin[0, 1] = inv_sqrt2
    spin[1, 0] = -inv_sqrt2
    return spin


def joint_spin_probabilities(
    spin_state: np.ndarray,
    theta_a: float,
    theta_b: float,
) -> dict[str, float]:
    """
    Born joint spin probabilities for a spin-only two-qubit state.

    For a normalized singlet, Bob's local marginal is independent of
    theta_a even though the joint Alice-Bob correlations change.
    """
    spin_state = np.asarray(spin_state, dtype=np.complex128)
    _assert(spin_state.shape == (2, 2), f"spin_state must have shape (2,2), got {spin_state.shape}")
    _assert(np.all(np.isfinite(spin_state.real)), "spin_state.real contains non-finite values")
    _assert(np.all(np.isfinite(spin_state.imag)), "spin_state.imag contains non-finite values")

    norm = float(np.sum(np.abs(spin_state) ** 2))
    _assert(norm > 0.0, "spin_state norm must be > 0")
    spin_state = spin_state / np.sqrt(norm)

    Ua = _spin_basis_from_angle(theta_a)
    Ub = _spin_basis_from_angle(theta_b)
    spin_m = np.einsum("ia,ab,jb->ij", Ua.conj().T, spin_state, Ub.conj().T)

    probs = {
        "++": float(np.abs(spin_m[0, 0]) ** 2),
        "+-": float(np.abs(spin_m[0, 1]) ** 2),
        "-+": float(np.abs(spin_m[1, 0]) ** 2),
        "--": float(np.abs(spin_m[1, 1]) ** 2),
    }
    total = float(sum(probs.values()))
    _assert(total > 0.0, "joint probability total must be > 0")
    return {ch: float(probs[ch] / total) for ch in CHANNELS}


def bob_plus_probability(probs: dict[str, float]) -> float:
    """
    P(B=+) from joint probabilities in channel order ++,+-,-+,--.
    """
    return float(probs["++"] + probs["-+"])


def history_velocity_profile(
    history_velocity_mode: str = "constant",
    history_v_future_factor: float = 0.5,
    history_v_present_factor: float = 1.0,
    history_v_past_factor: float = 2.0,
    n_frames: int = 101,
    front_fraction: float = 0.5,
    present_width_fraction: float = 0.12,
) -> dict[str, object]:
    """
    Minimal variable history-locking velocity profile.

    This is not physical FTL signalling.  It is a simulator-side distinction
    between observable signal velocity and effective history-consistency
    spread after a history has been selected.

    Normalized time convention for this diagnostic:
      - u < front: future / under-constrained side, weaker consistency spread
      - near front: present / reality-front side, approximately normal spread
      - u > front: locked-past side, optionally wider consistency spread
    """
    mode = str(history_velocity_mode).strip().lower()
    valid_modes = {"constant", "past_superluminal", "symmetric"}
    _assert(mode in valid_modes, f"history_velocity_mode must be one of {sorted(valid_modes)}, got {mode!r}")

    _assert(isinstance(n_frames, int), f"n_frames must be int, got {type(n_frames)}")
    _assert(n_frames >= 3, f"n_frames must be >= 3, got {n_frames}")

    future = _assert_finite_scalar(history_v_future_factor, "history_v_future_factor")
    present = _assert_finite_scalar(history_v_present_factor, "history_v_present_factor")
    past = _assert_finite_scalar(history_v_past_factor, "history_v_past_factor")
    front = _assert_finite_scalar(front_fraction, "front_fraction")
    width = _assert_finite_scalar(present_width_fraction, "present_width_fraction")

    _assert(future >= 0.0, f"history_v_future_factor must be >= 0, got {future}")
    _assert(present >= 0.0, f"history_v_present_factor must be >= 0, got {present}")
    _assert(past >= 0.0, f"history_v_past_factor must be >= 0, got {past}")
    _assert(0.0 <= front <= 1.0, f"front_fraction must be in [0,1], got {front}")
    _assert(width >= 0.0, f"present_width_fraction must be >= 0, got {width}")

    u = np.linspace(0.0, 1.0, int(n_frames), dtype=float)
    half_width = 0.5 * float(width)
    present_mask = np.abs(u - front) <= half_width
    future_mask = u < (front - half_width)
    past_mask = u > (front + half_width)

    if mode == "constant":
        profile = np.full_like(u, present, dtype=float)
        effective_future = present
        effective_present = present
        effective_past = present
    elif mode == "symmetric":
        side = 0.5 * (future + past)
        profile = np.full_like(u, side, dtype=float)
        profile[present_mask] = present
        effective_future = side
        effective_present = present
        effective_past = side
    else:
        profile = np.full_like(u, present, dtype=float)
        profile[future_mask] = future
        profile[present_mask] = present
        profile[past_mask] = past
        effective_future = future
        effective_present = present
        effective_past = past

    def masked_mean(mask: np.ndarray, fallback: float) -> float:
        if np.any(mask):
            return float(np.mean(profile[mask]))
        return float(fallback)

    future_mean = masked_mean(future_mask, effective_future)
    present_mean = masked_mean(present_mask, effective_present)
    past_mean = masked_mean(past_mask, effective_past)
    average = float(np.mean(profile))

    return {
        "history_velocity_mode": mode,
        "history_v_future_factor": float(future),
        "history_v_present_factor": float(present),
        "history_v_past_factor": float(past),
        "n_frames": int(n_frames),
        "front_fraction": float(front),
        "present_width_fraction": float(width),
        "average_trf_spread_radius": average,
        "future_spread_mean": future_mean,
        "present_spread_mean": present_mean,
        "past_spread_mean": past_mean,
        "past_future_bias_ratio": float(past_mean / max(future_mean, 1e-12)),
        "past_present_bias_ratio": float(past_mean / max(present_mean, 1e-12)),
        "future_present_bias_ratio": float(future_mean / max(present_mean, 1e-12)),
        "profile": profile,
    }


def history_locking_amplification(history_diag: dict[str, object]) -> float:
    """
    Convert a history-velocity profile into a small scalar stress factor.

    Safe modes still leave lambda_signal=0 untouched.  For lambda_signal>0,
    this factor amplifies only the deliberately forbidden bias so the
    diagnostic can test whether stronger locked-past consistency would make
    Bob marginal drift easier to catch.
    """
    mode = str(history_diag.get("history_velocity_mode", "constant"))
    if mode == "constant":
        return 1.0
    return float(history_diag.get("past_present_bias_ratio", 1.0))


def compute_history_lock_length(
    history_lock_length_m: float = np.inf,
    history_lock_tau_s: float | None = None,
) -> float:
    """
    Base joint-history coherence length in SI units.

    If history_lock_tau_s is provided, L_lock = c * tau_lock.  Otherwise the
    explicit length is used.  Infinite length is the backward-compatible
    default and means no range attenuation.
    """
    if history_lock_tau_s is not None:
        tau = _assert_finite_scalar(history_lock_tau_s, "history_lock_tau_s")
        _assert(tau >= 0.0, f"history_lock_tau_s must be >= 0, got {tau}")
        return float(C_M_PER_S * tau)

    length = float(history_lock_length_m)
    _assert(length >= 0.0, f"history_lock_length_m must be >= 0, got {length}")
    return length


def compute_relativistic_lock_length(
    base_lock_length_m: float,
    lock_length_mode: str = "constant",
    relative_velocity_fraction_c: float = 0.0,
    lock_anisotropy_eta: float = 1.0,
    lock_direction_cos_theta: float = 1.0,
) -> dict[str, float | str]:
    """
    Optional speculative effective L_lock model under relative motion.

    L_lock is treated here as an effective joint-history coherence length, not
    as an observable signal speed.  The modes are diagnostic hypotheses only.
    """
    L0 = float(base_lock_length_m)
    _assert(L0 >= 0.0, f"base_lock_length_m must be >= 0, got {L0}")

    mode = str(lock_length_mode).strip().lower()
    valid_modes = {"constant", "proper_time_dilated", "lorentz_contracted", "anisotropic"}
    _assert(mode in valid_modes, f"lock_length_mode must be one of {sorted(valid_modes)}, got {mode!r}")

    beta_raw = _assert_finite_scalar(relative_velocity_fraction_c, "relative_velocity_fraction_c")
    beta = float(np.clip(abs(beta_raw), 0.0, 1.0 - 1e-12))
    gamma = float(1.0 / np.sqrt(max(1.0 - beta * beta, 1e-12)))

    eta = _assert_finite_scalar(lock_anisotropy_eta, "lock_anisotropy_eta")
    cos_theta = _assert_finite_scalar(lock_direction_cos_theta, "lock_direction_cos_theta")
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))

    if np.isinf(L0) or mode == "constant":
        L_eff = L0
    elif mode == "proper_time_dilated":
        L_eff = gamma * L0
    elif mode == "lorentz_contracted":
        L_eff = L0 / max(gamma, 1e-12)
    else:
        denom = 1.0 + float(eta) * (gamma - 1.0) * cos_theta * cos_theta
        L_eff = L0 / max(denom, 1e-12)

    return {
        "lock_length_mode": mode,
        "base_lock_length_m": float(L0),
        "effective_lock_length_m": float(L_eff),
        "relative_velocity_fraction_c": float(beta_raw),
        "clamped_beta": float(beta),
        "gamma": float(gamma),
        "lock_anisotropy_eta": float(eta),
        "lock_direction_cos_theta": float(cos_theta),
    }


def compute_lock_attenuation(
    distance_m: float,
    effective_lock_length_m: float,
    history_lock_decay_mode: str = "none",
    history_lock_decay_power: float = 2.0,
) -> float:
    """
    Distance attenuation for joint-history correlation visibility.

    This attenuation never creates signalling by itself.  In the diagnostic it
    reduces Alice-Bob correlation visibility by mixing toward uncorrelated
    local marginals, and it attenuates the deliberately forbidden
    lambda_signal path when that path is explicitly enabled.
    """
    distance = _assert_finite_scalar(distance_m, "distance_m")
    _assert(distance >= 0.0, f"distance_m must be >= 0, got {distance}")

    mode = str(history_lock_decay_mode).strip().lower()
    valid_modes = {"none", "exp", "soft_power"}
    _assert(mode in valid_modes, f"history_lock_decay_mode must be one of {sorted(valid_modes)}, got {mode!r}")

    if mode == "none":
        return 1.0

    L = float(effective_lock_length_m)
    if np.isinf(L):
        return 1.0
    _assert(L >= 0.0, f"effective_lock_length_m must be >= 0, got {L}")
    if L <= 0.0:
        return 1.0 if distance <= 0.0 else 0.0

    x = distance / L
    if mode == "exp":
        return float(np.exp(-x))

    p = _assert_finite_scalar(history_lock_decay_power, "history_lock_decay_power")
    _assert(p > 0.0, f"history_lock_decay_power must be > 0, got {p}")
    return float(1.0 / (1.0 + np.power(x, p)))


def lock_attenuation_diagnostic(
    distance_m: float = 0.0,
    history_lock_length_m: float = np.inf,
    history_lock_tau_s: float | None = None,
    history_lock_decay_mode: str = "none",
    history_lock_decay_power: float = 2.0,
    lock_length_mode: str = "constant",
    relative_velocity_fraction_c: float = 0.0,
    lock_anisotropy_eta: float = 1.0,
    lock_direction_cos_theta: float = 1.0,
) -> dict[str, float | str | None]:
    base_L = compute_history_lock_length(
        history_lock_length_m=history_lock_length_m,
        history_lock_tau_s=history_lock_tau_s,
    )
    rel = compute_relativistic_lock_length(
        base_lock_length_m=base_L,
        lock_length_mode=lock_length_mode,
        relative_velocity_fraction_c=relative_velocity_fraction_c,
        lock_anisotropy_eta=lock_anisotropy_eta,
        lock_direction_cos_theta=lock_direction_cos_theta,
    )
    L_eff = float(rel["effective_lock_length_m"])
    attenuation = compute_lock_attenuation(
        distance_m=distance_m,
        effective_lock_length_m=L_eff,
        history_lock_decay_mode=history_lock_decay_mode,
        history_lock_decay_power=history_lock_decay_power,
    )

    return {
        **rel,
        "history_lock_tau_s": None if history_lock_tau_s is None else float(history_lock_tau_s),
        "history_lock_decay_mode": str(history_lock_decay_mode).strip().lower(),
        "history_lock_decay_power": float(history_lock_decay_power),
        "distance_m": float(distance_m),
        "distance_Mm": float(distance_m / MM_TO_M),
        "lock_attenuation": float(attenuation),
    }


def attenuate_joint_correlation_visibility(
    probs: dict[str, float],
    attenuation: float,
) -> dict[str, float]:
    """
    Reduce joint correlation visibility without changing local marginals.

    This mixes the joint distribution with the product of its Alice and Bob
    marginals.  It therefore can weaken correlations over distance while
    preserving no-signalling local marginals.
    """
    A = float(np.clip(_assert_finite_scalar(attenuation, "attenuation"), 0.0, 1.0))

    p_a_plus = float(probs["++"] + probs["+-"])
    p_a_minus = float(probs["-+"] + probs["--"])
    p_b_plus = float(probs["++"] + probs["-+"])
    p_b_minus = float(probs["+-"] + probs["--"])

    product = {
        "++": p_a_plus * p_b_plus,
        "+-": p_a_plus * p_b_minus,
        "-+": p_a_minus * p_b_plus,
        "--": p_a_minus * p_b_minus,
    }
    mixed = {ch: float(A * probs[ch] + (1.0 - A) * product[ch]) for ch in CHANNELS}
    total = float(sum(mixed.values()))
    _assert(total > 0.0, "attenuated joint probability total must be > 0")
    return {ch: float(mixed[ch] / total) for ch in CHANNELS}


def apply_forbidden_signal_bias(
    probs: dict[str, float],
    alice_setting: float,
    lambda_signal: float,
) -> dict[str, float]:
    """
    Deliberately unphysical TRF stress-test weighting.

    lambda_signal=0 preserves the Born probabilities.  lambda_signal>0 applies
    an Alice-setting-dependent non-unitary weight to Bob's local outcomes, so
    Bob's marginal can drift.  This is intentionally forbidden physics for
    diagnostics, not a physical model.
    """
    lam = _assert_finite_scalar(lambda_signal, "lambda_signal")
    _assert(lam >= 0.0, f"lambda_signal must be >= 0, got {lambda_signal}")

    if lam == 0.0:
        return {ch: float(probs[ch]) for ch in CHANNELS}

    setting_bias = float(lam * np.sin(float(alice_setting)))
    weights = {
        "++": np.exp(setting_bias),
        "-+": np.exp(setting_bias),
        "+-": np.exp(-setting_bias),
        "--": np.exp(-setting_bias),
    }
    biased = {ch: float(probs[ch] * weights[ch]) for ch in CHANNELS}
    total = float(sum(biased.values()))
    _assert(total > 0.0, "biased probability total must be > 0")
    return {ch: float(biased[ch] / total) for ch in CHANNELS}


def trf_joint_probabilities(
    spin_state: np.ndarray,
    alice_setting: float,
    bob_setting: float,
    lambda_signal: float = 0.0,
    history_velocity_mode: str = "constant",
    history_v_future_factor: float = 0.5,
    history_v_present_factor: float = 1.0,
    history_v_past_factor: float = 2.0,
    distance_m: float = 0.0,
    history_lock_length_m: float = np.inf,
    history_lock_tau_s: float | None = None,
    history_lock_decay_mode: str = "none",
    history_lock_decay_power: float = 2.0,
    lock_length_mode: str = "constant",
    relative_velocity_fraction_c: float = 0.0,
    lock_anisotropy_eta: float = 1.0,
    lock_direction_cos_theta: float = 1.0,
) -> dict[str, float]:
    """
    Joint probabilities for the minimal TRF no-signalling diagnostic.
    """
    history_diag = history_velocity_profile(
        history_velocity_mode=history_velocity_mode,
        history_v_future_factor=history_v_future_factor,
        history_v_present_factor=history_v_present_factor,
        history_v_past_factor=history_v_past_factor,
    )
    lock_diag = lock_attenuation_diagnostic(
        distance_m=distance_m,
        history_lock_length_m=history_lock_length_m,
        history_lock_tau_s=history_lock_tau_s,
        history_lock_decay_mode=history_lock_decay_mode,
        history_lock_decay_power=history_lock_decay_power,
        lock_length_mode=lock_length_mode,
        relative_velocity_fraction_c=relative_velocity_fraction_c,
        lock_anisotropy_eta=lock_anisotropy_eta,
        lock_direction_cos_theta=lock_direction_cos_theta,
    )
    attenuation = float(lock_diag["lock_attenuation"])
    effective_lambda = float(lambda_signal) * history_locking_amplification(history_diag) * attenuation
    born = joint_spin_probabilities(spin_state, alice_setting, bob_setting)
    attenuated = attenuate_joint_correlation_visibility(born, attenuation)
    return apply_forbidden_signal_bias(attenuated, alice_setting, effective_lambda)


def run_trf_no_signalling_diagnostic(
    lambda_signal: float = 0.0,
    history_velocity_mode: str = "constant",
    history_v_future_factor: float = 0.5,
    history_v_present_factor: float = 1.0,
    history_v_past_factor: float = 2.0,
    distance_m: float = 0.0,
    history_lock_length_m: float = np.inf,
    history_lock_tau_s: float | None = None,
    history_lock_decay_mode: str = "none",
    history_lock_decay_power: float = 2.0,
    lock_length_mode: str = "constant",
    relative_velocity_fraction_c: float = 0.0,
    lock_anisotropy_eta: float = 1.0,
    lock_direction_cos_theta: float = 1.0,
    alice_setting_a0: float = 0.0,
    alice_setting_a1: float = float(np.pi / 3.0),
    bob_setting: float = float(np.pi / 5.0),
    n_trials: int = 20000,
    rng_seed: int = 12345,
) -> dict[str, object]:
    """
    Run two ensembles with Bob's setting fixed and Alice's setting changed.

    Returns sampled ensemble estimates plus exact probabilities.  The
    history_velocity_mode changes only the simulator's effective history
    consistency spread.  lambda_signal=0 stays no-signalling; lambda_signal>0
    intentionally injects forbidden Alice-setting-dependent bias, optionally
    amplified by stronger locked-past consistency spread.
    """
    _assert(isinstance(n_trials, int), f"n_trials must be int, got {type(n_trials)}")
    _assert(n_trials > 0, f"n_trials must be > 0, got {n_trials}")

    rng = np.random.default_rng(int(rng_seed))
    spin = singlet_spin_state()
    history_diag = history_velocity_profile(
        history_velocity_mode=history_velocity_mode,
        history_v_future_factor=history_v_future_factor,
        history_v_present_factor=history_v_present_factor,
        history_v_past_factor=history_v_past_factor,
    )
    amplification = history_locking_amplification(history_diag)
    lock_diag = lock_attenuation_diagnostic(
        distance_m=distance_m,
        history_lock_length_m=history_lock_length_m,
        history_lock_tau_s=history_lock_tau_s,
        history_lock_decay_mode=history_lock_decay_mode,
        history_lock_decay_power=history_lock_decay_power,
        lock_length_mode=lock_length_mode,
        relative_velocity_fraction_c=relative_velocity_fraction_c,
        lock_anisotropy_eta=lock_anisotropy_eta,
        lock_direction_cos_theta=lock_direction_cos_theta,
    )
    attenuation = float(lock_diag["lock_attenuation"])
    effective_lambda = float(lambda_signal) * amplification * attenuation

    def ensemble(theta_a: float) -> dict[str, object]:
        probs = trf_joint_probabilities(
            spin_state=spin,
            alice_setting=float(theta_a),
            bob_setting=float(bob_setting),
            lambda_signal=float(lambda_signal),
            history_velocity_mode=str(history_velocity_mode),
            history_v_future_factor=float(history_v_future_factor),
            history_v_present_factor=float(history_v_present_factor),
            history_v_past_factor=float(history_v_past_factor),
            distance_m=float(distance_m),
            history_lock_length_m=float(history_lock_length_m),
            history_lock_tau_s=history_lock_tau_s,
            history_lock_decay_mode=str(history_lock_decay_mode),
            history_lock_decay_power=float(history_lock_decay_power),
            lock_length_mode=str(lock_length_mode),
            relative_velocity_fraction_c=float(relative_velocity_fraction_c),
            lock_anisotropy_eta=float(lock_anisotropy_eta),
            lock_direction_cos_theta=float(lock_direction_cos_theta),
        )
        pvec = np.asarray([probs[ch] for ch in CHANNELS], dtype=float)
        counts = rng.multinomial(int(n_trials), pvec)
        sampled = {ch: float(counts[i] / n_trials) for i, ch in enumerate(CHANNELS)}
        return {
            "joint_probabilities": probs,
            "sampled_joint_probabilities": sampled,
            "P_B_plus_exact": bob_plus_probability(probs),
            "P_B_plus_sampled": bob_plus_probability(sampled),
            "E_exact": channel_E_from_probs(probs),
            "E_sampled": channel_E_from_probs(sampled),
            "counts": {ch: int(counts[i]) for i, ch in enumerate(CHANNELS)},
        }

    e0 = ensemble(float(alice_setting_a0))
    e1 = ensemble(float(alice_setting_a1))

    p0 = float(e0["P_B_plus_sampled"])
    p1 = float(e1["P_B_plus_sampled"])
    p0_exact = float(e0["P_B_plus_exact"])
    p1_exact = float(e1["P_B_plus_exact"])

    return {
        "lambda_signal": float(lambda_signal),
        "effective_lambda_signal": float(effective_lambda),
        "history_locking_amplification": float(amplification),
        "apparent_nonlocal_consistency_gain": float(amplification),
        "history_locking": {
            k: v for k, v in history_diag.items() if k != "profile"
        },
        "finite_history_lock": lock_diag,
        "alice_setting_a0": float(alice_setting_a0),
        "alice_setting_a1": float(alice_setting_a1),
        "bob_setting": float(bob_setting),
        "n_trials": int(n_trials),
        "rng_seed": int(rng_seed),
        "a0_ensemble": e0,
        "a1_ensemble": e1,
        "P_B_plus_given_a0": p0,
        "P_B_plus_given_a1": p1,
        "Delta_signal": float(abs(p0 - p1)),
        "P_B_plus_given_a0_exact": p0_exact,
        "P_B_plus_given_a1_exact": p1_exact,
        "Delta_signal_exact": float(abs(p0_exact - p1_exact)),
        "E_a0_exact": float(e0["E_exact"]),
        "E_a1_exact": float(e1["E_exact"]),
        "Delta_correlation_exact": float(abs(float(e0["E_exact"]) - float(e1["E_exact"]))),
    }


def run_trf_history_velocity_matrix(
    lambda_signal_safe: float = 0.0,
    lambda_signal_forbidden: float = 0.4,
    history_v_future_factor: float = 0.5,
    history_v_present_factor: float = 1.0,
    history_v_past_factor: float = 2.0,
    distance_m: float = 0.0,
    history_lock_length_m: float = np.inf,
    history_lock_tau_s: float | None = None,
    history_lock_decay_mode: str = "none",
    history_lock_decay_power: float = 2.0,
    lock_length_mode: str = "constant",
    relative_velocity_fraction_c: float = 0.0,
    lock_anisotropy_eta: float = 1.0,
    lock_direction_cos_theta: float = 1.0,
    alice_setting_a0: float = 0.0,
    alice_setting_a1: float = float(np.pi / 3.0),
    bob_setting: float = float(np.pi / 5.0),
    n_trials: int = 20000,
    rng_seed: int = 12345,
) -> dict[str, object]:
    """
    Compare constant and past-superluminal history-locking in safe and
    deliberately forbidden signalling modes.
    """
    cases = [
        ("A_constant_safe", "constant", float(lambda_signal_safe)),
        ("B_past_superluminal_safe", "past_superluminal", float(lambda_signal_safe)),
        ("C_constant_forbidden", "constant", float(lambda_signal_forbidden)),
        ("D_past_superluminal_forbidden", "past_superluminal", float(lambda_signal_forbidden)),
    ]

    results = {}
    for offset, (name, mode, lam) in enumerate(cases):
        results[name] = run_trf_no_signalling_diagnostic(
            lambda_signal=lam,
            history_velocity_mode=mode,
            history_v_future_factor=float(history_v_future_factor),
            history_v_present_factor=float(history_v_present_factor),
            history_v_past_factor=float(history_v_past_factor),
            distance_m=float(distance_m),
            history_lock_length_m=float(history_lock_length_m),
            history_lock_tau_s=history_lock_tau_s,
            history_lock_decay_mode=str(history_lock_decay_mode),
            history_lock_decay_power=float(history_lock_decay_power),
            lock_length_mode=str(lock_length_mode),
            relative_velocity_fraction_c=float(relative_velocity_fraction_c),
            lock_anisotropy_eta=float(lock_anisotropy_eta),
            lock_direction_cos_theta=float(lock_direction_cos_theta),
            alice_setting_a0=float(alice_setting_a0),
            alice_setting_a1=float(alice_setting_a1),
            bob_setting=float(bob_setting),
            n_trials=int(n_trials),
            rng_seed=int(rng_seed) + offset,
        )

    return {"cases": results}


def run_trf_lock_distance_sweep(
    distances_m: list[float] | tuple[float, ...] | None = None,
    lambda_signal_safe: float = 0.0,
    lambda_signal_forbidden: float = 0.4,
    history_velocity_mode: str = "constant",
    history_lock_tau_s: float | None = 1.0,
    history_lock_length_m: float = np.inf,
    history_lock_decay_power: float = 2.0,
    n_trials: int = 20000,
    rng_seed: int = 12345,
) -> dict[str, object]:
    """
    Distance sweep for finite joint-history lock range.

    Cases:
      1) safe + no decay
      2) safe + soft_power lock attenuation
      3) forbidden + no decay
      4) forbidden + soft_power lock attenuation
    """
    if distances_m is None:
        distances_m = [0.0, 40.0 * MM_TO_M, 300.0 * MM_TO_M, 384.0 * MM_TO_M, 1000.0 * MM_TO_M]

    cases = [
        ("safe_none", float(lambda_signal_safe), "none"),
        ("safe_soft_power", float(lambda_signal_safe), "soft_power"),
        ("forbidden_none", float(lambda_signal_forbidden), "none"),
        ("forbidden_soft_power", float(lambda_signal_forbidden), "soft_power"),
    ]

    rows = []
    for i, distance in enumerate(distances_m):
        for j, (case_name, lam, decay_mode) in enumerate(cases):
            res = run_trf_no_signalling_diagnostic(
                lambda_signal=lam,
                history_velocity_mode=history_velocity_mode,
                distance_m=float(distance),
                history_lock_length_m=float(history_lock_length_m),
                history_lock_tau_s=history_lock_tau_s,
                history_lock_decay_mode=decay_mode,
                history_lock_decay_power=float(history_lock_decay_power),
                n_trials=int(n_trials),
                rng_seed=int(rng_seed) + 100 * i + j,
            )
            lock = res["finite_history_lock"]
            rows.append(
                {
                    "case": case_name,
                    "distance_m": float(distance),
                    "distance_Mm": float(distance / MM_TO_M),
                    "lock_attenuation": float(lock["lock_attenuation"]),
                    "correlation_visibility": float(lock["lock_attenuation"]),
                    "effective_lock_length_m": float(lock["effective_lock_length_m"]),
                    "effective_lock_length_Mm": float(lock["effective_lock_length_m"] / MM_TO_M),
                    "history_lock_decay_mode": str(decay_mode),
                    "lambda_signal": float(lam),
                    "effective_lambda_signal": float(res["effective_lambda_signal"]),
                    "P_B_plus_given_a0": float(res["P_B_plus_given_a0"]),
                    "P_B_plus_given_a1": float(res["P_B_plus_given_a1"]),
                    "Delta_signal": float(res["Delta_signal"]),
                    "P_B_plus_given_a0_exact": float(res["P_B_plus_given_a0_exact"]),
                    "P_B_plus_given_a1_exact": float(res["P_B_plus_given_a1_exact"]),
                    "Delta_signal_exact": float(res["Delta_signal_exact"]),
                    "E_a0_exact": float(res["E_a0_exact"]),
                    "E_a1_exact": float(res["E_a1_exact"]),
                    "Delta_correlation_exact": float(res["Delta_correlation_exact"]),
                }
            )

    return {
        "distance_sweep": rows,
        "history_lock_tau_s": None if history_lock_tau_s is None else float(history_lock_tau_s),
        "history_lock_length_m": float(history_lock_length_m),
        "history_lock_decay_power": float(history_lock_decay_power),
    }


def run_trf_lock_velocity_sweep(
    base_lock_length_m: float = 300.0 * MM_TO_M,
    betas: list[float] | tuple[float, ...] | None = None,
    lock_anisotropy_eta: float = 1.0,
) -> dict[str, object]:
    """
    Compare effective L_lock under simple relative-motion hypotheses.
    """
    if betas is None:
        betas = [0.0, 0.5, 0.8]

    rows = []
    for beta in betas:
        for mode in ("constant", "proper_time_dilated", "lorentz_contracted", "anisotropic"):
            cos_values = [1.0]
            if mode == "anisotropic":
                cos_values = [1.0, 0.0]
            for cos_theta in cos_values:
                diag = compute_relativistic_lock_length(
                    base_lock_length_m=float(base_lock_length_m),
                    lock_length_mode=mode,
                    relative_velocity_fraction_c=float(beta),
                    lock_anisotropy_eta=float(lock_anisotropy_eta),
                    lock_direction_cos_theta=float(cos_theta),
                )
                rows.append(
                    {
                        **diag,
                        "base_lock_length_Mm": float(base_lock_length_m / MM_TO_M),
                        "effective_lock_length_Mm": float(diag["effective_lock_length_m"] / MM_TO_M),
                    }
                )

    return {"velocity_sweep": rows}


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

    Stern-Gerlach effective analyzer:
        Optional channel-dependent phase gradient:
            V_ab(x,y,t) = - ent_sg_strength * sign_ab
                         * (Y - y0) * window_x * drive(x,t)

        Forward phase:
            exp(-i V_ab dt / hbar)

        Since F_y = -dV/dy, positive sign_ab receives positive-y force.

        The default drive is the phase-flip wave from
        experimental/entanglement_phase_flip_poc.py: the analyzer basis stays
        fixed and only the local SG coupling flips sign/strength as cos(kx-wt).
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

    ent_Ua: object | None = None
    ent_Ub: object | None = None

    # --------------------------------------------------------
    # Effective Stern-Gerlach analyzer
    # --------------------------------------------------------

    ent_sg_enabled: bool = True

    # Modes:
    #   "off"
    #   "A"
    #   "B"
    #   "sum"
    #   "difference"
    #
    # A:
    #   ++,+- receive + force; -+,-- receive - force
    #
    # B:
    #   ++,-+ receive + force; +-,-- receive - force
    #
    # sum:
    #   ++ -> +2, +- -> 0, -+ -> 0, -- -> -2
    #
    # difference:
    #   ++ -> 0, +- -> +2, -+ -> -2, -- -> 0
    ent_sg_mode: str = "A"

    # Strength of y-force. Positive strength means positive channel sign
    # accelerates toward +y.
    ent_sg_strength: float = 1.0 #0.10

    # Analyzer x-window.
    ent_sg_center_x: float = -6.0 #2.0
    ent_sg_width_x: float = 4.0

    # Linear potential is proportional to (Y - ent_sg_y_center).
    ent_sg_y_center: float = 0.0

    # If true, SG signs are applied after rotating into measurement basis Ua/Ub.
    # This is usually the natural analyzer interpretation.
    ent_sg_use_measurement_basis: bool = True

    # Optional absolute potential clipping. 0 means disabled.
    ent_sg_clip_abs: float = 0.0

    # Coupling drive:
    #   "static"          -> old fixed SG coupling
    #   "phase_flip_wave" -> fixed analyzer axis, cosine sign/strength flip
    ent_sg_drive_mode: str = "phase_flip_wave"
    ent_sg_wave_k: float = 0.45
    ent_sg_wave_omega: float = 1.2
    ent_sg_wave_phase: float = 0.0

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

    ent_channel_persistent: bool = True

    ent_channel_evidence_use_gamma: bool = True

    # Spatial selection field:
    #   "density"
    #   "peak_tube"
    ent_spatial_gate_mode: str = "peak_tube"

    ent_peak_radius_px: int = 18
    ent_peak_rel_threshold: float = 0.03

    ent_tube_sigma_px: float = 10.0
    ent_gate_blur_sigma: float = 1.0
    ent_gate_density_power: float = 1.0
    ent_gate_gamma_power: float = 0.5

    ent_selected_gain_strength: float = 1.5
    ent_other_damp_strength: float = 0.25

    ent_time_ramp_steps: int = 50

    ent_singlet_prior_enabled: bool = False
    ent_singlet_same_weight: float = 0.25
    ent_singlet_opposite_weight: float = 1.0

    ent_print_selection: bool = True

    # --------------------------------------------------------
    # Initial spin state
    # --------------------------------------------------------

    # Allowed:
    #   "singlet"
    #   "product"
    #   "plus_plus"
    #   "plus_minus"
    #   "minus_plus"
    #   "minus_minus"
    #
    # Recommended SG tests:
    #   singlet + ent_sg_mode="difference"
    #   product superposition + ent_sg_mode="A"
#    ent_initial_spin_state: str = "product"

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

        _assert(self.front_branch_competition_strength >= 0.0, "front_branch_competition_strength must be >= 0")
        _assert(self.front_branch_competition_power >= 0.0, "front_branch_competition_power must be >= 0")
        _assert(self.front_branch_gate_power >= 0.0, "front_branch_gate_power must be >= 0")
        _assert(self.front_branch_density_power >= 0.0, "front_branch_density_power must be >= 0")
        _assert(self.front_branch_align_power >= 0.0, "front_branch_align_power must be >= 0")
        _assert(self.front_branch_competition_threshold >= 0.0, "front_branch_competition_threshold must be >= 0")

        _assert(isinstance(self.front_branch_competition_radius, int), "front_branch_competition_radius must be int")
        _assert(self.front_branch_competition_radius >= 1, "front_branch_competition_radius must be >= 1")
        _assert(self.front_branch_competition_margin > 0.0, "front_branch_competition_margin must be > 0")
        _assert(self.front_branch_competition_blur_sigma >= 0.0, "front_branch_competition_blur_sigma must be >= 0")
        _assert(self.front_branch_detector_gate_width > 0.0, "front_branch_detector_gate_width must be > 0")
        _assert(self.front_branch_detector_gate_boost >= 0.0, "front_branch_detector_gate_boost must be >= 0")

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
                _assert(spin_a.shape == (2,), f"ent_initial_spin_a must have shape (2,), got {spin_a.shape}")
                norm_a = float(np.sqrt(np.sum(np.abs(spin_a) ** 2)))
                _assert(norm_a > 0.0, "ent_initial_spin_a norm must be > 0")
                _assert(np.isfinite(norm_a), "ent_initial_spin_a norm must be finite")

            if self.ent_initial_spin_b is not None:
                spin_b = np.asarray(self.ent_initial_spin_b, dtype=np.complex128)
                _assert(spin_b.shape == (2,), f"ent_initial_spin_b must have shape (2,), got {spin_b.shape}")
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

        _assert(isinstance(self.ent_peak_radius_px, int), "ent_peak_radius_px must be int")
        _assert(self.ent_peak_radius_px >= 1, "ent_peak_radius_px must be >= 1")
        _assert(self.ent_peak_rel_threshold >= 0.0, "ent_peak_rel_threshold must be >= 0")
        _assert(self.ent_tube_sigma_px > 0.0, "ent_tube_sigma_px must be > 0")
        _assert(self.ent_gate_blur_sigma >= 0.0, "ent_gate_blur_sigma must be >= 0")
        _assert(self.ent_gate_density_power >= 0.0, "ent_gate_density_power must be >= 0")
        _assert(self.ent_gate_gamma_power >= 0.0, "ent_gate_gamma_power must be >= 0")
        _assert(self.ent_selected_gain_strength >= 0.0, "ent_selected_gain_strength must be >= 0")
        _assert(self.ent_other_damp_strength >= 0.0, "ent_other_damp_strength must be >= 0")
        _assert(isinstance(self.ent_time_ramp_steps, int), "ent_time_ramp_steps must be int")
        _assert(self.ent_time_ramp_steps >= 1, "ent_time_ramp_steps must be >= 1")
        _assert(self.ent_singlet_same_weight >= 0.0, "ent_singlet_same_weight must be >= 0")
        _assert(self.ent_singlet_opposite_weight >= 0.0, "ent_singlet_opposite_weight must be >= 0")

        # --------------------------------------------------------
        # Stern-Gerlach validation
        # --------------------------------------------------------

        self.ent_sg_mode = str(self.ent_sg_mode).strip().lower()

        valid_sg_modes = {"off", "a", "b", "sum", "difference"}
        _assert(
            self.ent_sg_mode in valid_sg_modes,
            f"ent_sg_mode must be one of {sorted(valid_sg_modes)}, got {self.ent_sg_mode}",
        )

        _assert(self.ent_sg_strength >= 0.0, "ent_sg_strength must be >= 0")
        _assert(self.ent_sg_width_x > 0.0, "ent_sg_width_x must be > 0")
        _assert(self.ent_sg_clip_abs >= 0.0, "ent_sg_clip_abs must be >= 0")

        self.ent_sg_drive_mode = str(self.ent_sg_drive_mode).strip().lower()
        valid_sg_drive_modes = {"static", "phase_flip_wave"}
        _assert(
            self.ent_sg_drive_mode in valid_sg_drive_modes,
            f"ent_sg_drive_mode must be one of {sorted(valid_sg_drive_modes)}, "
            f"got {self.ent_sg_drive_mode}",
        )

        _assert_finite_scalar(self.ent_sg_center_x, "ent_sg_center_x")
        _assert_finite_scalar(self.ent_sg_y_center, "ent_sg_y_center")
        _assert_finite_scalar(self.ent_sg_wave_k, "ent_sg_wave_k")
        _assert_finite_scalar(self.ent_sg_wave_omega, "ent_sg_wave_omega")
        _assert_finite_scalar(self.ent_sg_wave_phase, "ent_sg_wave_phase")

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
        self._ent_sg_profile_cache = None

        self._ent_selected_channel = None
        self._ent_selected_peak = None
        self._ent_initialized = False
        self._ent_step_counter = 0

    # --------------------------------------------------------
    # Runtime reset
    # --------------------------------------------------------

    def reset_runtime_state(self):
        """
        Call before a fresh run if the same theory instance is reused.
        """
        self._detector_gate_cache = None
        self._ent_sg_profile_cache = None

        self._ent_selected_channel = None
        self._ent_selected_peak = None
        self._ent_initialized = False
        self._ent_step_counter = 0

    # --------------------------------------------------------
    # Initial state construction
    # --------------------------------------------------------

    def initialize_click_state(
        self,
        x_click: float,
        y_click: float,
        sigma_click: float,
    ) -> np.ndarray:
        """
        Build a 4D entangled/spin-channel click state for backward propagation.
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

        phi = self._lift_scalar_to_initial_spinor(spatial)

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

    def initialize_channel_click_state(
        self,
        x_click: float,
        y_click: float,
        sigma_click: float,
        channel: str,
    ) -> np.ndarray:
        """
        Build a localized coincidence click in one measurement-basis channel.

        The spatial click remains the usual 2D detector packet, while the spin
        part is projected into one joint outcome: ++, +-, -+, or --.
        """
        _assert(channel in CHANNELS, f"channel must be one of {CHANNELS}, got {channel!r}")

        x = self.grid.X
        y = self.grid.Y

        sigma_click = _assert_positive_scalar(sigma_click, "sigma_click")

        spatial = np.exp(
            -0.5 * (
                ((x - float(x_click)) / sigma_click) ** 2
                + ((y - float(y_click)) / sigma_click) ** 2
            )
        ).astype(np.complex128)

        idx = {
            "++": (0, 0),
            "+-": (0, 1),
            "-+": (1, 0),
            "--": (1, 1),
        }

        psi_m = np.zeros(spatial.shape + (2, 2), dtype=np.complex128)
        a, b = idx[channel]
        psi_m[:, :, a, b] = spatial

        phi = rotate_state_from_measurement_basis(psi_m, self._ent_Ua, self._ent_Ub)
        phi, norm_factor = _normalize_unit_spinor_4d(phi, self.grid.dx, self.grid.dy)
        phi = phi.astype(np.complex128)

        if self.front_debug_checks:
            prob = self._state_probability(phi)
            _assert(
                np.isclose(prob, 1.0, atol=self.front_norm_tol),
                f"initialize_channel_click_state normalized probability should be 1, got {prob}",
            )
            _assert(
                np.isfinite(float(norm_factor)) and float(norm_factor) > 0.0,
                f"initialize_channel_click_state norm_factor invalid: {norm_factor}",
            )

        return phi

    def initialize_two_position_channel_click_state(
        self,
        x_click_a: float,
        y_click_a: float,
        x_click_b: float,
        y_click_b: float,
        sigma_click: float,
        channel: str,
    ) -> np.ndarray:
        """
        Build a minimal two-readout coincidence click in one joint channel.

        This theory still has one 2D spatial coordinate, so the two detector
        readouts are represented as a coherent two-lobe spatial seed inside the
        selected measurement-basis channel.
        """
        _assert(channel in CHANNELS, f"channel must be one of {CHANNELS}, got {channel!r}")

        x = self.grid.X
        y = self.grid.Y

        sigma_click = _assert_positive_scalar(sigma_click, "sigma_click")

        spatial_a = np.exp(
            -0.5 * (
                ((x - float(x_click_a)) / sigma_click) ** 2
                + ((y - float(y_click_a)) / sigma_click) ** 2
            )
        ).astype(np.complex128)

        spatial_b = np.exp(
            -0.5 * (
                ((x - float(x_click_b)) / sigma_click) ** 2
                + ((y - float(y_click_b)) / sigma_click) ** 2
            )
        ).astype(np.complex128)

        spatial = (spatial_a + spatial_b).astype(np.complex128)

        idx = {
            "++": (0, 0),
            "+-": (0, 1),
            "-+": (1, 0),
            "--": (1, 1),
        }

        psi_m = np.zeros(spatial.shape + (2, 2), dtype=np.complex128)
        a, b = idx[channel]
        psi_m[:, :, a, b] = spatial

        phi = rotate_state_from_measurement_basis(psi_m, self._ent_Ua, self._ent_Ub)
        phi, norm_factor = _normalize_unit_spinor_4d(phi, self.grid.dx, self.grid.dy)
        phi = phi.astype(np.complex128)

        if self.front_debug_checks:
            prob = self._state_probability(phi)
            _assert(
                np.isclose(prob, 1.0, atol=self.front_norm_tol),
                f"initialize_two_position_channel_click_state normalized probability should be 1, got {prob}",
            )
            _assert(
                np.isfinite(float(norm_factor)) and float(norm_factor) > 0.0,
                f"initialize_two_position_channel_click_state norm_factor invalid: {norm_factor}",
            )

        return phi

    def initialize_state(self, psi0: np.ndarray) -> np.ndarray:
        """
        Accept either:
            psi0.shape == (Ny, Nx)
            psi0.shape == (Ny, Nx, 2, 2)

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
            out = self._lift_scalar_to_initial_spinor(spatial)

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

    def _lift_scalar_to_initial_spinor(self, spatial: np.ndarray) -> np.ndarray:
        _assert_complex_array_2d(spatial, "spatial(lift_initial)")

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

        return out.astype(np.complex128)

    # --------------------------------------------------------
    # Basic spinor utilities
    # --------------------------------------------------------

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
    # Stern-Gerlach effective analyzer
    # --------------------------------------------------------

    def _sg_is_active(self) -> bool:
        return (
            bool(self.ent_sg_enabled)
            and str(self.ent_sg_mode).lower() != "off"
            and float(self.ent_sg_strength) > 0.0
        )

    def _sg_sign_matrix(self) -> np.ndarray:
        """
        Return signs in measurement-basis channel order:
            [[++, +-],
             [-+, --]]
        """
        mode = str(self.ent_sg_mode).lower()

        sA = np.asarray(
            [
                [1.0, 1.0],
                [-1.0, -1.0],
            ],
            dtype=float,
        )

        sB = np.asarray(
            [
                [1.0, -1.0],
                [1.0, -1.0],
            ],
            dtype=float,
        )

        if mode == "a":
            signs = sA
        elif mode == "b":
            signs = sB
        elif mode == "sum":
            signs = sA + sB
        elif mode == "difference":
            signs = sA - sB
        elif mode == "off":
            signs = np.zeros((2, 2), dtype=float)
        else:
            raise ValueError(f"Unknown ent_sg_mode={self.ent_sg_mode!r}")

        return signs.astype(float)

    def _sg_base_spatial_profile(self) -> np.ndarray:
        """
        Time-independent spatial part of SG potential.

            V_ab(x,y) = sign_ab * profile(x,y)

        profile:
            -strength * (Y - y0) * exp(-(X-xc)^2/(2 width^2))

        Positive sign receives positive-y force:
            F_y = -dV/dy = strength * sign
        """
        if self._ent_sg_profile_cache is not None:
            return self._ent_sg_profile_cache

        X = self.grid.X
        Y = self.grid.Y

        xc = float(self.ent_sg_center_x)
        wx = float(self.ent_sg_width_x)
        y0 = float(self.ent_sg_y_center)
        strength = float(self.ent_sg_strength)

        window_x = np.exp(-0.5 * ((X - xc) / wx) ** 2).astype(float)

        profile = (-strength * (Y - y0) * window_x).astype(float)

        clip_abs = float(self.ent_sg_clip_abs)
        if clip_abs > 0.0:
            profile = np.clip(profile, -clip_abs, clip_abs).astype(float)

        _assert_real_array_2d(profile, "sg_spatial_profile")

        self._ent_sg_profile_cache = profile
        return profile

    def _sg_drive_field(self, t: float) -> np.ndarray:
        """
        Local coupling drive for the SG analyzer.

        In phase_flip_wave mode this mirrors the PoC's key move: do not rotate
        the spin/analyzer axis in time, only multiply the diagonal SG coupling
        by a cosine so the force alternates sign locally.
        """
        _assert_finite_scalar(t, "t(sg_drive)")

        if self.ent_sg_drive_mode == "static":
            drive = np.ones_like(self.grid.X, dtype=float)
        elif self.ent_sg_drive_mode == "phase_flip_wave":
            phase = (
                float(self.ent_sg_wave_k) * (self.grid.X - float(self.ent_sg_center_x))
                - float(self.ent_sg_wave_omega) * float(t)
                + float(self.ent_sg_wave_phase)
            )
            drive = np.cos(phase).astype(float)
        else:
            raise ValueError(f"Unknown ent_sg_drive_mode={self.ent_sg_drive_mode!r}")

        _assert_real_array_2d(drive, "sg_drive")
        return drive

    def _sg_spatial_profile(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        base_profile = self._sg_base_spatial_profile()
        drive = self._sg_drive_field(t)
        profile = (base_profile * drive).astype(float)

        _assert_real_array_2d(profile, "sg_spatial_profile_driven")
        return profile, drive

    def _apply_sg_phase(
        self,
        psi: np.ndarray,
        dt: float,
        t: float,
        adjoint: bool,
    ) -> tuple[np.ndarray, dict]:
        """
        Apply diagonal SG phase in either measurement or computational basis.

        Forward:
            exp(-i V dt / hbar)

        Adjoint/backward:
            exp(+i V dt / hbar)
        """
        _assert_complex_spinor_4d(psi, "psi(sg)")
        _assert_finite_scalar(dt, "dt(sg)")
        _assert_finite_scalar(t, "t(sg)")

        if not self._sg_is_active():
            return psi, {
                "enabled": False,
                "mode": str(self.ent_sg_mode),
                "drive_mode": str(self.ent_sg_drive_mode),
            }

        signs = self._sg_sign_matrix()
        profile, drive = self._sg_spatial_profile(t)

        if self.ent_sg_use_measurement_basis:
            work = rotate_state_to_measurement_basis(psi, self._ent_Ua, self._ent_Ub)
        else:
            work = psi.copy()

        phase_sign = 1.0 if adjoint else -1.0

        hbar = max(float(self.hbar), self.front_eps)

        for a in range(2):
            for b in range(2):
                s = float(signs[a, b])
                if abs(s) <= 0.0:
                    continue

                V = s * profile
                phase = np.exp(1j * phase_sign * V * float(dt) / hbar)
                work[:, :, a, b] *= phase

        if self.ent_sg_use_measurement_basis:
            out = rotate_state_from_measurement_basis(work, self._ent_Ua, self._ent_Ub)
        else:
            out = work

        _assert_complex_spinor_4d(out, "psi_out(sg)")

        aux = {
            "enabled": True,
            "mode": str(self.ent_sg_mode),
            "drive_mode": str(self.ent_sg_drive_mode),
            "adjoint": bool(adjoint),
            "time": float(t),
            "strength": float(self.ent_sg_strength),
            "center_x": float(self.ent_sg_center_x),
            "width_x": float(self.ent_sg_width_x),
            "y_center": float(self.ent_sg_y_center),
            "wave_k": float(self.ent_sg_wave_k),
            "wave_omega": float(self.ent_sg_wave_omega),
            "wave_phase": float(self.ent_sg_wave_phase),
            "use_measurement_basis": bool(self.ent_sg_use_measurement_basis),
            "signs": {
                "++": float(signs[0, 0]),
                "+-": float(signs[0, 1]),
                "-+": float(signs[1, 0]),
                "--": float(signs[1, 1]),
            },
            "profile_min": float(np.min(profile)),
            "profile_max": float(np.max(profile)),
            "profile_absmax": float(np.max(np.abs(profile))),
            "drive_min": float(np.min(drive)),
            "drive_max": float(np.max(drive)),
            "drive_mean": float(np.mean(drive)),
        }

        return out.astype(np.complex128), aux

    def _apply_sg_phase_forward(self, psi: np.ndarray, dt: float, t: float) -> tuple[np.ndarray, dict]:
        return self._apply_sg_phase(psi=psi, dt=dt, t=t, adjoint=False)

    def _apply_sg_phase_adjoint(self, psi: np.ndarray, dt: float, t: float) -> tuple[np.ndarray, dict]:
        return self._apply_sg_phase(psi=psi, dt=dt, t=t, adjoint=True)

    # --------------------------------------------------------
    # Coherence / current
    # --------------------------------------------------------

    def _coherence_alignment_score(self, psi: np.ndarray):
        """
        Spinor generalization of scalar phase-alignment score.
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
            peaks.append(
                {
                    "iy": int(iy),
                    "ix": int(ix),
                    "value": float(arr[iy, ix]),
                    "fallback_argmax": False,
                }
            )

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
            aux_front.update(
                {
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
                }
            )

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
        _assert_finite_scalar(dt, "dt(componentwise backward)")

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
            + optional Stern-Gerlach phase gradient
            + spinor thick-front sharpening
            + scalar branch competition
            + measurement-basis channel selection
            + final full-spinor normalization
        """
        _assert_complex_spinor_4d(state, "state")
        _assert_finite_scalar(dt, "dt")

        self._ent_step_counter += 1
        t_sg = float(self._ent_step_counter - 1) * float(dt)

        psi, aux_base_components = self._step_forward_componentwise(state, dt)
        prob_after_base = self._state_probability(psi)

        psi, aux_sg = self._apply_sg_phase_forward(psi, dt, t_sg)
        prob_after_sg = self._state_probability(psi)

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

                "stern_gerlach": aux_sg,

                "prob_after_base": float(prob_after_base),
                "prob_after_sg": float(prob_after_sg),
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

        If SG is enabled, apply the adjoint SG phase before componentwise
        adjoint Schrödinger stepping. This approximates the inverse order of:
            forward base step -> forward SG phase
        """
        _assert_complex_spinor_4d(state, "state(backward)")
        _assert_finite_scalar(dt, "dt(backward)")

        t_sg = max(0.0, float(self._ent_step_counter - 1) * float(dt))
        psi, aux_sg = self._apply_sg_phase_adjoint(state, dt, t_sg)
        psi, aux_components = self._step_backward_componentwise(psi, dt)

        return TheoryStepResult(
            state=psi.astype(np.complex128),
            aux={
                "thick_front_entanglement_backward": {
                    "mode": "componentwise_schrodinger_adjoint_with_optional_sg_adjoint",
                    "stern_gerlach": aux_sg,
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

        psi = (
            spatial[:, :, None, None]
            * spin_a[None, None, :, None]
            * spin_b[None, None, None, :]
        )

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


@dataclass
class SignallingTRFEntanglementTheory(ThickFrontEntanglementTheory):
    """
    Minimal experimental TRF entanglement stress-test theory.

    lambda_signal=0.0 is the no-signalling-safe baseline: joint correlations
    depend on Alice's setting, but Bob's local Born marginal does not.

    lambda_signal>0.0 deliberately applies an Alice-setting-dependent
    non-unitary TRF weight to Bob's outcome channels.  This is intentional
    forbidden physics for diagnostics only, useful for checking that a
    no-signalling test can catch local marginal drift.
    """

    lambda_signal: float = 0.0
    history_velocity_mode: str = "constant"
    history_v_future_factor: float = 0.5
    history_v_present_factor: float = 1.0
    history_v_past_factor: float = 2.0
    distance_m: float = 0.0
    history_lock_length_m: float = np.inf
    history_lock_tau_s: float | None = None
    history_lock_decay_mode: str = "none"
    history_lock_decay_power: float = 2.0
    lock_length_mode: str = "constant"
    relative_velocity_fraction_c: float = 0.0
    lock_anisotropy_eta: float = 1.0
    lock_direction_cos_theta: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.lambda_signal = _assert_finite_scalar(self.lambda_signal, "lambda_signal")
        _assert(self.lambda_signal >= 0.0, f"lambda_signal must be >= 0, got {self.lambda_signal}")
        self.history_velocity_mode = str(self.history_velocity_mode).strip().lower()
        history_velocity_profile(
            history_velocity_mode=self.history_velocity_mode,
            history_v_future_factor=float(self.history_v_future_factor),
            history_v_present_factor=float(self.history_v_present_factor),
            history_v_past_factor=float(self.history_v_past_factor),
        )
        lock_attenuation_diagnostic(
            distance_m=float(self.distance_m),
            history_lock_length_m=float(self.history_lock_length_m),
            history_lock_tau_s=self.history_lock_tau_s,
            history_lock_decay_mode=str(self.history_lock_decay_mode),
            history_lock_decay_power=float(self.history_lock_decay_power),
            lock_length_mode=str(self.lock_length_mode),
            relative_velocity_fraction_c=float(self.relative_velocity_fraction_c),
            lock_anisotropy_eta=float(self.lock_anisotropy_eta),
            lock_direction_cos_theta=float(self.lock_direction_cos_theta),
        )

    def joint_probabilities_for_settings(
        self,
        state: np.ndarray,
        alice_setting: float,
        bob_setting: float,
    ) -> dict[str, float]:
        """
        Measurement-basis joint probabilities with optional forbidden TRF bias.
        """
        Ua = _spin_basis_from_angle(float(alice_setting))
        Ub = _spin_basis_from_angle(float(bob_setting))

        if state.shape == (2, 2):
            probs = joint_spin_probabilities(state, alice_setting, bob_setting)
        else:
            _assert_complex_spinor_4d(state, "state(joint_probabilities_for_settings)")
            psi_m = rotate_state_to_measurement_basis(state, Ua, Ub)
            dxdy = float(self.grid.dx * self.grid.dy)
            probs = {
                "++": float(np.sum(np.abs(psi_m[:, :, 0, 0]) ** 2) * dxdy),
                "+-": float(np.sum(np.abs(psi_m[:, :, 0, 1]) ** 2) * dxdy),
                "-+": float(np.sum(np.abs(psi_m[:, :, 1, 0]) ** 2) * dxdy),
                "--": float(np.sum(np.abs(psi_m[:, :, 1, 1]) ** 2) * dxdy),
            }
            total = float(sum(probs.values()))
            _assert(total > 0.0, "joint probability total must be > 0")
            probs = {ch: float(probs[ch] / total) for ch in CHANNELS}

        history_diag = history_velocity_profile(
            history_velocity_mode=self.history_velocity_mode,
            history_v_future_factor=float(self.history_v_future_factor),
            history_v_present_factor=float(self.history_v_present_factor),
            history_v_past_factor=float(self.history_v_past_factor),
        )
        lock_diag = lock_attenuation_diagnostic(
            distance_m=float(self.distance_m),
            history_lock_length_m=float(self.history_lock_length_m),
            history_lock_tau_s=self.history_lock_tau_s,
            history_lock_decay_mode=str(self.history_lock_decay_mode),
            history_lock_decay_power=float(self.history_lock_decay_power),
            lock_length_mode=str(self.lock_length_mode),
            relative_velocity_fraction_c=float(self.relative_velocity_fraction_c),
            lock_anisotropy_eta=float(self.lock_anisotropy_eta),
            lock_direction_cos_theta=float(self.lock_direction_cos_theta),
        )
        attenuation = float(lock_diag["lock_attenuation"])
        effective_lambda = float(self.lambda_signal) * history_locking_amplification(history_diag) * attenuation
        probs = attenuate_joint_correlation_visibility(probs, attenuation)

        return apply_forbidden_signal_bias(
            probs=probs,
            alice_setting=float(alice_setting),
            lambda_signal=effective_lambda,
        )

    def bob_plus_probability_for_settings(
        self,
        state: np.ndarray,
        alice_setting: float,
        bob_setting: float,
    ) -> float:
        probs = self.joint_probabilities_for_settings(
            state=state,
            alice_setting=alice_setting,
            bob_setting=bob_setting,
        )
        return bob_plus_probability(probs)

    def no_signalling_diagnostic(
        self,
        state: np.ndarray,
        alice_setting_a0: float = 0.0,
        alice_setting_a1: float = float(np.pi / 3.0),
        bob_setting: float = float(np.pi / 5.0),
    ) -> dict[str, object]:
        """
        Compare P(B=+ | Alice setting a0) and P(B=+ | Alice setting a1).
        """
        probs_a0 = self.joint_probabilities_for_settings(state, alice_setting_a0, bob_setting)
        probs_a1 = self.joint_probabilities_for_settings(state, alice_setting_a1, bob_setting)
        p0 = bob_plus_probability(probs_a0)
        p1 = bob_plus_probability(probs_a1)
        history_diag = history_velocity_profile(
            history_velocity_mode=self.history_velocity_mode,
            history_v_future_factor=float(self.history_v_future_factor),
            history_v_present_factor=float(self.history_v_present_factor),
            history_v_past_factor=float(self.history_v_past_factor),
        )
        lock_diag = lock_attenuation_diagnostic(
            distance_m=float(self.distance_m),
            history_lock_length_m=float(self.history_lock_length_m),
            history_lock_tau_s=self.history_lock_tau_s,
            history_lock_decay_mode=str(self.history_lock_decay_mode),
            history_lock_decay_power=float(self.history_lock_decay_power),
            lock_length_mode=str(self.lock_length_mode),
            relative_velocity_fraction_c=float(self.relative_velocity_fraction_c),
            lock_anisotropy_eta=float(self.lock_anisotropy_eta),
            lock_direction_cos_theta=float(self.lock_direction_cos_theta),
        )

        return {
            "lambda_signal": float(self.lambda_signal),
            "effective_lambda_signal": float(
                self.lambda_signal
                * history_locking_amplification(history_diag)
                * float(lock_diag["lock_attenuation"])
            ),
            "history_velocity_mode": str(self.history_velocity_mode),
            "finite_history_lock": lock_diag,
            "alice_setting_a0": float(alice_setting_a0),
            "alice_setting_a1": float(alice_setting_a1),
            "bob_setting": float(bob_setting),
            "joint_probabilities_a0": probs_a0,
            "joint_probabilities_a1": probs_a1,
            "P_B_plus_given_a0": float(p0),
            "P_B_plus_given_a1": float(p1),
            "Delta_signal": float(abs(p0 - p1)),
        }


ForbiddenSignalTRFTheory = SignallingTRFEntanglementTheory
VariableHistoryVelocityTRFTheory = SignallingTRFEntanglementTheory
CausalLockingTRFTheory = SignallingTRFEntanglementTheory
HistoryConsistencyVelocityTRF = SignallingTRFEntanglementTheory
