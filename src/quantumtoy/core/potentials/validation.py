from __future__ import annotations

import numpy as np


def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def _require_attr(obj, name: str):
    _assert(hasattr(obj, name), f"Missing required attribute: {name}")
    return getattr(obj, name)


def _assert_finite_scalar(x, name: str):
    _assert(np.isscalar(x), f"{name} must be a scalar, got type={type(x)}")
    xf = float(x)
    _assert(np.isfinite(xf), f"{name} must be finite, got {x}")
    return xf


def _assert_positive_scalar(x, name: str):
    xf = _assert_finite_scalar(x, name)
    _assert(xf > 0.0, f"{name} must be > 0, got {x}")
    return xf


def _assert_nonnegative_scalar(x, name: str):
    xf = _assert_finite_scalar(x, name)
    _assert(xf >= 0.0, f"{name} must be >= 0, got {x}")
    return xf


def _assert_array_shape(arr: np.ndarray, shape: tuple[int, ...], name: str):
    _assert(isinstance(arr, np.ndarray), f"{name} must be np.ndarray")
    _assert(arr.shape == shape, f"{name}.shape {arr.shape} != expected {shape}")


def _assert_finite_array(arr: np.ndarray, name: str):
    _assert(np.all(np.isfinite(arr)), f"{name} contains non-finite values")