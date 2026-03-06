from __future__ import annotations
import numpy as np


def norm_L2(field: np.ndarray, dx: float, dy: float) -> float:
    return float(np.sqrt(np.sum(np.abs(field) ** 2) * dx * dy))


def normalize_unit(field: np.ndarray, dx: float, dy: float):
    n = norm_L2(field, dx, dy)
    if n <= 0:
        return field, 0.0
    return field / n, n


def norm_prob(rho: np.ndarray, dx: float, dy: float) -> float:
    return float(np.sum(rho) * dx * dy)


def make_packet(X, Y, x0, y0, sigma0, k0x, k0y):
    XR = X - x0
    YR = Y - y0
    amp = np.exp(-(XR**2 + YR**2) / (2 * sigma0**2))
    phase = np.exp(1j * (k0x * X + k0y * Y))
    return (amp * phase).astype(np.complex128)