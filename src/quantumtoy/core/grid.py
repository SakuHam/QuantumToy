from __future__ import annotations
import numpy as np
from core.simulation_types import GridSpec


def build_grid(visible_lx: float, visible_ly: float, n_visible_x: int, n_visible_y: int, pad_factor: int) -> GridSpec:
    Lx = visible_lx * pad_factor
    Ly = visible_ly * pad_factor
    Nx = n_visible_x * pad_factor
    Ny = n_visible_y * pad_factor

    dx = Lx / Nx
    dy = Ly / Ny

    x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)

    cx = Nx // 2
    cy = Ny // 2
    hx = n_visible_x // 2
    hy = n_visible_y // 2

    xs = slice(cx - hx, cx + hx)
    ys = slice(cy - hy, cy + hy)

    x_vis_1d = x[xs]
    y_vis_1d = y[ys]
    X_vis = X[ys, xs]
    Y_vis = Y[ys, xs]

    x_vis_min = float(x_vis_1d[0])
    x_vis_max = float(x_vis_1d[-1] + dx)
    y_vis_min = float(y_vis_1d[0])
    y_vis_max = float(y_vis_1d[-1] + dy)

    mask_visible = np.zeros_like(X, dtype=bool)
    mask_visible[ys, xs] = True

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