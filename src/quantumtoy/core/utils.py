from __future__ import annotations
import numpy as np


def norm_L2(field: np.ndarray, dx: float, dy: float) -> float:
    dx = float(dx)
    dy = float(dy)
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive")
    val = np.sum(np.abs(field) ** 2) * dx * dy
    return float(np.sqrt(val))


def normalize_unit(field: np.ndarray, dx: float, dy: float):
    n = norm_L2(field, dx, dy)
    if not np.isfinite(n) or n <= 0.0:
        return field.copy(), 0.0
    return field / n, n


def norm_prob(rho: np.ndarray, dx: float, dy: float) -> float:
    dx = float(dx)
    dy = float(dy)
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive")
    total = np.sum(np.real(rho)) * dx * dy
    return float(total)


def make_packet(X, Y, x0, y0, sigma0, k0x, k0y):
    sigma = float(sigma0)
    if sigma <= 0.0:
        raise ValueError("sigma0 must be positive")

    XR = X - x0
    YR = Y - y0

    amp = np.exp(-(XR**2 + YR**2) / (2.0 * sigma**2))
    phase = np.exp(1j * (k0x * X + k0y * Y))

    return (amp * phase).astype(np.complex128)

def make_packet_scout_main_scalar_seed(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    # Main packet
    main_amp: float = 1.0,
    main_x0: float = -15.0,
    main_y0: float = 0.0,
    main_sigma_x: float = 1.8,
    main_sigma_y: float = 1.8,
    main_kx: float = 4.0,
    main_ky: float = 0.0,
    # Scout packet
    scout_amp: float = 0.15,
    scout_x0: float = -6.0,
    scout_y0: float = 0.0,
    scout_sigma_x: float = 2.8,
    scout_sigma_y: float = 2.0,
    scout_kx: float = 4.0,
    scout_ky: float = 0.0,
    # Optional chirp / extra phase curvature
    main_chirp_x: float = 0.0,
    main_chirp_y: float = 0.0,
    scout_chirp_x: float = 0.0,
    scout_chirp_y: float = 0.0,
    # Global options
    normalize_l2: bool = True,
    dx: float | None = None,
    dy: float | None = None,
    global_phase: float = 0.0,
    relative_phase: float = 0.0,
) -> np.ndarray:
    """
    Build a scalar complex seed field consisting of two Gaussian packets:

      psi = main_packet + scout_packet

    Intended use:
      - main packet = strong packet farther from the screen
      - scout packet = weak packet closer to the screen

    Parameters
    ----------
    X, Y
        2D meshgrid arrays with shape (Ny, Nx).
    main_amp, scout_amp
        Relative amplitudes of the main and scout packets.
    main_x0, main_y0, scout_x0, scout_y0
        Packet centers.
    main_sigma_x, main_sigma_y, scout_sigma_x, scout_sigma_y
        Gaussian widths.
    main_kx, main_ky, scout_kx, scout_ky
        Carrier wavevectors.
    main_chirp_x, main_chirp_y, scout_chirp_x, scout_chirp_y
        Optional quadratic phase terms. Useful for making arrival behavior
        more interesting without changing the basic packet layout.
    normalize_l2
        If True, normalize using dx*dy weighted L2 norm.
    dx, dy
        Required if normalize_l2=True.
    global_phase
        Overall phase applied to the final field.
    relative_phase
        Extra phase applied only to the scout packet.

    Returns
    -------
    psi : np.ndarray
        Complex scalar field with shape (Ny, Nx).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have same shape, got {X.shape} vs {Y.shape}")

    if main_sigma_x <= 0.0 or main_sigma_y <= 0.0:
        raise ValueError("main sigmas must be > 0")
    if scout_sigma_x <= 0.0 or scout_sigma_y <= 0.0:
        raise ValueError("scout sigmas must be > 0")

    # Main packet envelope
    main_env = np.exp(
        -((X - main_x0) ** 2) / (2.0 * main_sigma_x ** 2)
        -((Y - main_y0) ** 2) / (2.0 * main_sigma_y ** 2)
    )

    # Scout packet envelope
    scout_env = np.exp(
        -((X - scout_x0) ** 2) / (2.0 * scout_sigma_x ** 2)
        -((Y - scout_y0) ** 2) / (2.0 * scout_sigma_y ** 2)
    )

    # Linear carrier phases
    main_phase = main_kx * X + main_ky * Y
    scout_phase = scout_kx * X + scout_ky * Y + relative_phase

    # Optional quadratic chirp phases
    if main_chirp_x != 0.0:
        main_phase += main_chirp_x * (X - main_x0) ** 2
    if main_chirp_y != 0.0:
        main_phase += main_chirp_y * (Y - main_y0) ** 2

    if scout_chirp_x != 0.0:
        scout_phase += scout_chirp_x * (X - scout_x0) ** 2
    if scout_chirp_y != 0.0:
        scout_phase += scout_chirp_y * (Y - scout_y0) ** 2

    main_packet = main_amp * main_env * np.exp(1j * main_phase)
    scout_packet = scout_amp * scout_env * np.exp(1j * scout_phase)

    psi = main_packet + scout_packet

    if global_phase != 0.0:
        psi = psi * np.exp(1j * global_phase)

    psi = np.asarray(psi, dtype=np.complex128)

    if normalize_l2:
        if dx is None or dy is None:
            raise ValueError("dx and dy must be provided when normalize_l2=True")

        norm2 = float(np.sum(np.abs(psi) ** 2) * dx * dy)
        if not np.isfinite(norm2) or norm2 <= 0.0:
            raise ValueError(f"Seed norm is invalid: {norm2}")

        psi = psi / np.sqrt(norm2)

    return psi