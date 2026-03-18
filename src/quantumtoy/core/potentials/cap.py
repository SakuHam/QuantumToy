from __future__ import annotations

import numpy as np

from .validation import (
    _assert,
    _assert_array_shape,
    _assert_finite_array,
    _assert_nonnegative_scalar,
    _assert_positive_scalar,
)


def smooth_cap_edge(X, Y, Lx, Ly, cap_width=8.0, strength=2.0, power=4):
    _assert(isinstance(X, np.ndarray), "X must be np.ndarray")
    _assert(isinstance(Y, np.ndarray), "Y must be np.ndarray")
    _assert(X.ndim == 2, f"X must be 2D, got ndim={X.ndim}")
    _assert(Y.ndim == 2, f"Y must be 2D, got ndim={Y.ndim}")
    _assert(X.shape == Y.shape, f"X.shape {X.shape} != Y.shape {Y.shape}")
    _assert_finite_array(X, "X")
    _assert_finite_array(Y, "Y")

    Lx = _assert_positive_scalar(Lx, "Lx")
    Ly = _assert_positive_scalar(Ly, "Ly")
    cap_width = _assert_nonnegative_scalar(cap_width, "cap_width")
    strength = _assert_nonnegative_scalar(strength, "strength")
    power = _assert_positive_scalar(power, "power")

    dist_to_x = (Lx / 2.0) - np.abs(X)
    dist_to_y = (Ly / 2.0) - np.abs(Y)
    dist_to_edge = np.minimum(dist_to_x, dist_to_y)

    _assert_array_shape(dist_to_edge, X.shape, "dist_to_edge")
    _assert_finite_array(dist_to_edge, "dist_to_edge")

    W = np.zeros_like(X, dtype=float)

    if cap_width == 0.0 or strength == 0.0:
        return W

    mask = dist_to_edge < cap_width
    s = (cap_width - dist_to_edge[mask]) / cap_width
    W[mask] = strength * (s ** power)

    _assert_array_shape(W, X.shape, "W")
    _assert_finite_array(W, "W")
    _assert(np.all(W >= -1e-14), "W contains significantly negative values")

    return W