from __future__ import annotations

import numpy as np
from core.simulation_types import GridSpec


def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def _assert_finite_positive_scalar(x, name: str):
    _assert(np.isscalar(x), f"{name} must be a scalar, got type={type(x)}")
    xf = float(x)
    _assert(np.isfinite(xf), f"{name} must be finite, got {x}")
    _assert(xf > 0.0, f"{name} must be > 0, got {x}")


def _assert_positive_int(x, name: str):
    _assert(isinstance(x, int), f"{name} must be int, got type={type(x)}")
    _assert(x > 0, f"{name} must be > 0, got {x}")


def build_grid(
    visible_lx: float,
    visible_ly: float,
    n_visible_x: int,
    n_visible_y: int,
    pad_factor: int,
) -> GridSpec:
    _assert_finite_positive_scalar(visible_lx, "visible_lx")
    _assert_finite_positive_scalar(visible_ly, "visible_ly")
    _assert_positive_int(n_visible_x, "n_visible_x")
    _assert_positive_int(n_visible_y, "n_visible_y")
    _assert_positive_int(pad_factor, "pad_factor")

    _assert(
        n_visible_x % 2 == 0,
        f"n_visible_x must be even for centered crop, got {n_visible_x}",
    )
    _assert(
        n_visible_y % 2 == 0,
        f"n_visible_y must be even for centered crop, got {n_visible_y}",
    )

    Lx = float(visible_lx * pad_factor)
    Ly = float(visible_ly * pad_factor)
    Nx = int(n_visible_x * pad_factor)
    Ny = int(n_visible_y * pad_factor)

    _assert(Lx > 0.0 and np.isfinite(Lx), f"Lx invalid: {Lx}")
    _assert(Ly > 0.0 and np.isfinite(Ly), f"Ly invalid: {Ly}")
    _assert(Nx > 0, f"Nx invalid: {Nx}")
    _assert(Ny > 0, f"Ny invalid: {Ny}")

    dx = Lx / Nx
    dy = Ly / Ny

    _assert(np.isfinite(dx) and dx > 0.0, f"dx invalid: {dx}")
    _assert(np.isfinite(dy) and dy > 0.0, f"dy invalid: {dy}")

    x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)

    _assert(x.shape == (Nx,), f"x shape {x.shape} != {(Nx,)}")
    _assert(y.shape == (Ny,), f"y shape {y.shape} != {(Ny,)}")
    _assert(X.shape == (Ny, Nx), f"X shape {X.shape} != {(Ny, Nx)}")
    _assert(Y.shape == (Ny, Nx), f"Y shape {Y.shape} != {(Ny, Nx)}")

    _assert(np.all(np.isfinite(x)), "x contains non-finite values")
    _assert(np.all(np.isfinite(y)), "y contains non-finite values")
    _assert(np.all(np.isfinite(X)), "X contains non-finite values")
    _assert(np.all(np.isfinite(Y)), "Y contains non-finite values")

    if Nx >= 2:
        _assert(
            np.allclose(np.diff(x), dx, rtol=1e-12, atol=1e-12),
            f"x spacing inconsistent with dx={dx}",
        )
    if Ny >= 2:
        _assert(
            np.allclose(np.diff(y), dy, rtol=1e-12, atol=1e-12),
            f"y spacing inconsistent with dy={dy}",
        )

    _assert(np.allclose(X[0, :], x), "X first row must equal x")
    _assert(np.allclose(Y[:, 0], y), "Y first column must equal y")

    cx = Nx // 2
    cy = Ny // 2
    hx = n_visible_x // 2
    hy = n_visible_y // 2

    x0 = cx - hx
    x1 = cx + hx
    y0 = cy - hy
    y1 = cy + hy

    _assert(0 <= x0 < x1 <= Nx, f"x crop out of bounds: [{x0}, {x1}) for Nx={Nx}")
    _assert(0 <= y0 < y1 <= Ny, f"y crop out of bounds: [{y0}, {y1}) for Ny={Ny}")

    _assert((x1 - x0) == n_visible_x, f"x crop width {(x1 - x0)} != n_visible_x={n_visible_x}")
    _assert((y1 - y0) == n_visible_y, f"y crop height {(y1 - y0)} != n_visible_y={n_visible_y}")

    xs = slice(x0, x1)
    ys = slice(y0, y1)

    x_vis_1d = x[xs]
    y_vis_1d = y[ys]
    X_vis = X[ys, xs]
    Y_vis = Y[ys, xs]

    _assert(x_vis_1d.shape == (n_visible_x,), f"x_vis_1d shape {x_vis_1d.shape} != {(n_visible_x,)}")
    _assert(y_vis_1d.shape == (n_visible_y,), f"y_vis_1d shape {y_vis_1d.shape} != {(n_visible_y,)}")
    _assert(X_vis.shape == (n_visible_y, n_visible_x), f"X_vis shape {X_vis.shape} incorrect")
    _assert(Y_vis.shape == (n_visible_y, n_visible_x), f"Y_vis shape {Y_vis.shape} incorrect")

    _assert(np.all(np.isfinite(x_vis_1d)), "x_vis_1d contains non-finite values")
    _assert(np.all(np.isfinite(y_vis_1d)), "y_vis_1d contains non-finite values")
    _assert(np.all(np.isfinite(X_vis)), "X_vis contains non-finite values")
    _assert(np.all(np.isfinite(Y_vis)), "Y_vis contains non-finite values")

    _assert(np.allclose(X_vis[0, :], x_vis_1d), "X_vis first row must equal x_vis_1d")
    _assert(np.allclose(Y_vis[:, 0], y_vis_1d), "Y_vis first column must equal y_vis_1d")

    x_vis_min = float(x_vis_1d[0])
    x_vis_max = float(x_vis_1d[-1] + dx)
    y_vis_min = float(y_vis_1d[0])
    y_vis_max = float(y_vis_1d[-1] + dy)

    _assert(np.isclose(x_vis_max - x_vis_min, visible_lx, rtol=1e-12, atol=1e-12),
            f"x visible extent width {x_vis_max - x_vis_min} != visible_lx={visible_lx}")
    _assert(np.isclose(y_vis_max - y_vis_min, visible_ly, rtol=1e-12, atol=1e-12),
            f"y visible extent height {y_vis_max - y_vis_min} != visible_ly={visible_ly}")

    mask_visible = np.zeros_like(X, dtype=bool)
    mask_visible[ys, xs] = True

    _assert(mask_visible.shape == (Ny, Nx), f"mask_visible shape {mask_visible.shape} != {(Ny, Nx)}")
    _assert(np.count_nonzero(mask_visible) == n_visible_x * n_visible_y,
            f"mask_visible count {np.count_nonzero(mask_visible)} != {n_visible_x * n_visible_y}")

    _assert(np.all(mask_visible[ys, xs]), "mask_visible must be True on visible crop")
    if y0 > 0:
        _assert(not np.any(mask_visible[:y0, :]), "mask_visible leaks above visible crop")
    if y1 < Ny:
        _assert(not np.any(mask_visible[y1:, :]), "mask_visible leaks below visible crop")
    if x0 > 0:
        _assert(not np.any(mask_visible[:, :x0]), "mask_visible leaks left of visible crop")
    if x1 < Nx:
        _assert(not np.any(mask_visible[:, x1:]), "mask_visible leaks right of visible crop")

    mask_expected = np.zeros((Ny, Nx), dtype=bool)
    mask_expected[ys, xs] = True
    _assert(np.array_equal(mask_visible, mask_expected), "mask_visible must equal exact crop mask")

    _assert(
        x0 == (Nx - n_visible_x) // 2,
        f"x crop is not centered: x0={x0}, expected {(Nx - n_visible_x) // 2}",
    )
    _assert(
        y0 == (Ny - n_visible_y) // 2,
        f"y crop is not centered: y0={y0}, expected {(Ny - n_visible_y) // 2}",
    )

    _assert(abs(float(np.mean(x_vis_1d))) <= max(dx, 1e-15),
            f"x_vis_1d not centered near zero, mean={np.mean(x_vis_1d)}")
    _assert(abs(float(np.mean(y_vis_1d))) <= max(dy, 1e-15),
            f"y_vis_1d not centered near zero, mean={np.mean(y_vis_1d)}")

    return GridSpec(
        visible_lx=visible_lx,
        visible_ly=visible_ly,
        n_visible_x=n_visible_x,
        n_visible_y=n_visible_y,
        pad_factor=pad_factor,
        Lx=Lx,
        Ly=Ly,
        Nx=Nx,
        Ny=Ny,
        dx=dx,
        dy=dy,
        x=x,
        y=y,
        X=X,
        Y=Y,
        cx=cx,
        cy=cy,
        hx=hx,
        hy=hy,
        xs=xs,
        ys=ys,
        x_vis_1d=x_vis_1d,
        y_vis_1d=y_vis_1d,
        X_vis=X_vis,
        Y_vis=Y_vis,
        x_vis_min=x_vis_min,
        x_vis_max=x_vis_max,
        y_vis_min=y_vis_min,
        y_vis_max=y_vis_max,
        mask_visible=mask_visible,
    )