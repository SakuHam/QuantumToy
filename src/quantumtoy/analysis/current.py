from __future__ import annotations

import numpy as np


# ============================================================
# Ridge tangent
# ============================================================

def ridge_tangent_unit(ridge_x: np.ndarray, ridge_y: np.ndarray):
    """
    Compute unit tangent vector along ridge trajectory.
    Uses central difference.
    """
    Nt_ = len(ridge_x)

    tx = np.zeros(Nt_, dtype=float)
    ty = np.zeros(Nt_, dtype=float)

    for i in range(1, Nt_ - 1):
        dxp = ridge_x[i + 1] - ridge_x[i - 1]
        dyp = ridge_y[i + 1] - ridge_y[i - 1]

        n = np.hypot(dxp, dyp)
        if n > 0:
            tx[i] = dxp / n
            ty[i] = dyp / n

    if Nt_ >= 2:
        tx[0], ty[0] = tx[1], ty[1]
        tx[-1], ty[-1] = tx[-2], ty[-2]

    return tx, ty


# ============================================================
# Gaussian kernel
# ============================================================

def gaussian_kernel_2d(radius: int, sigma: float) -> np.ndarray:
    r = int(radius)

    if r <= 0:
        return np.array([[1.0]], dtype=float)

    ax = np.arange(-r, r + 1, dtype=float)
    xx, yy = np.meshgrid(ax, ax)

    k = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))

    s = float(np.sum(k))
    return k / s if s > 0 else k


# ============================================================
# Local weighted velocity average
# ============================================================

def local_weighted_mean(
    vx: np.ndarray,
    vy: np.ndarray,
    speed: np.ndarray,
    ix: int,
    iy: int,
    kernel: np.ndarray,
):
    """
    Compute local averaged velocity direction weighted by speed.
    """

    H, W_ = vx.shape
    r = kernel.shape[0] // 2

    x1 = max(0, ix - r)
    x2 = min(W_, ix + r + 1)
    y1 = max(0, iy - r)
    y2 = min(H, iy + r + 1)

    sub_vx = vx[y1:y2, x1:x2]
    sub_vy = vy[y1:y2, x1:x2]
    sub_sp = speed[y1:y2, x1:x2]

    ky1 = y1 - (iy - r)
    ky2 = ky1 + (y2 - y1)
    kx1 = x1 - (ix - r)
    kx2 = kx1 + (x2 - x1)

    k = kernel[ky1:ky2, kx1:kx2]

    w = k * sub_sp
    wsum = float(np.sum(w))

    if wsum <= 0:
        return np.nan, np.nan, np.nan

    vxm = float(np.sum(w * sub_vx) / wsum)
    vym = float(np.sum(w * sub_vy) / wsum)

    spd = float(np.hypot(vxm, vym))

    if spd <= 0:
        return np.nan, np.nan, np.nan

    return vxm / spd, vym / spd, spd


# ============================================================
# Divergence diagnostic
# ============================================================

def divergence_of_velocity(vx: np.ndarray, vy: np.ndarray, dx: float, dy: float):
    dvx_dx = (np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1)) / (2.0 * dx)
    dvy_dy = (np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0)) / (2.0 * dy)

    return dvx_dx + dvy_dy


# ============================================================
# Alignment diagnostics
# ============================================================

def alignment_and_diagnostics_from_state_frames(
    theory,
    state_vis_frames: np.ndarray,
    ridge_x: np.ndarray,
    ridge_y: np.ndarray,
    x_vis_1d: np.ndarray,
    y_vis_1d: np.ndarray,
    dx: float,
    dy: float,
    enable_divergence: bool = True,
    arrow_spatial_avg: bool = True,
    arrow_avg_radius: int = 3,
    arrow_avg_gauss_sigma: float = 1.5,
    arrow_temporal_smooth: bool = True,
    arrow_smooth_alpha: float = 0.20,
    align_eps_rho: float = 1e-10,
    align_eps_speed: float = 1e-12,
):
    """
    Compute velocity alignment diagnostics along ridge.
    """

    Nt_ = len(ridge_x)

    cos_th = np.full(Nt_, np.nan, dtype=float)
    speed = np.full(Nt_, np.nan, dtype=float)
    ux = np.full(Nt_, np.nan, dtype=float)
    uy = np.full(Nt_, np.nan, dtype=float)

    div_v_at_ridge = np.full(Nt_, np.nan, dtype=float)

    tx, ty = ridge_tangent_unit(ridge_x, ridge_y)

    kernel = gaussian_kernel_2d(arrow_avg_radius, arrow_avg_gauss_sigma)

    for i in range(Nt_):

        ix = int(np.argmin(np.abs(x_vis_1d - ridge_x[i])))
        iy = int(np.argmin(np.abs(y_vis_1d - ridge_y[i])))

        vx, vy, sp = theory.velocity(
            state_vis_frames[i],
            eps_rho=align_eps_rho
        )

        if arrow_spatial_avg:

            uxi, uyi, spd = local_weighted_mean(
                vx, vy, sp, ix, iy, kernel
            )

        else:

            spd = float(sp[iy, ix])

            if spd > 0:
                uxi = float(vx[iy, ix] / spd)
                uyi = float(vy[iy, ix] / spd)
            else:
                uxi, uyi = np.nan, np.nan

        if (
            np.isfinite(uxi)
            and np.isfinite(uyi)
            and np.isfinite(spd)
            and (spd > align_eps_speed)
        ):
            ux[i] = uxi
            uy[i] = uyi
            speed[i] = spd

            cos_th[i] = float(
                np.clip(
                    uxi * tx[i] + uyi * ty[i],
                    -1.0,
                    1.0,
                )
            )

        if enable_divergence:

            div_v = divergence_of_velocity(vx, vy, dx, dy)

            div_v_at_ridge[i] = float(div_v[iy, ix])

    # ============================================================
    # Temporal smoothing
    # ============================================================

    if arrow_temporal_smooth:

        a = float(np.clip(arrow_smooth_alpha, 0.0, 1.0))

        ux_f = ux.copy()
        uy_f = uy.copy()
        sp_f = speed.copy()

        last_u = None
        last_v = None
        last_s = None

        for i in range(Nt_):

            if np.isfinite(ux_f[i]) and np.isfinite(uy_f[i]):

                if last_u is not None:

                    uu = (1 - a) * ux_f[i] + a * last_u
                    vv = (1 - a) * uy_f[i] + a * last_v

                    nn = float(np.hypot(uu, vv))

                    if nn > 0:
                        uu /= nn
                        vv /= nn

                    ux_f[i] = uu
                    uy_f[i] = vv

                    if np.isfinite(sp_f[i]) and last_s is not None:
                        sp_f[i] = (1 - a) * sp_f[i] + a * last_s

                last_u = ux_f[i]
                last_v = uy_f[i]

                if np.isfinite(sp_f[i]):
                    last_s = sp_f[i]

        ux, uy, speed = ux_f, uy_f, sp_f

        tx, ty = ridge_tangent_unit(ridge_x, ridge_y)

        for i in range(Nt_):
            if np.isfinite(ux[i]) and np.isfinite(uy[i]):
                cos_th[i] = float(
                    np.clip(
                        ux[i] * tx[i] + uy[i] * ty[i],
                        -1.0,
                        1.0,
                    )
                )

    return cos_th, speed, ux, uy, div_v_at_ridge