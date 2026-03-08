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