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

    sigma = float(sigma)
    if sigma <= 0.0:
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
# Shape helpers
# ============================================================

def _center_crop_slices_from_shapes(
    full_hw: tuple[int, int],
    vis_hw: tuple[int, int],
):
    """
    Return centered crop slices that map a full-grid field to the visible grid.
    """
    full_h, full_w = int(full_hw[0]), int(full_hw[1])
    vis_h, vis_w = int(vis_hw[0]), int(vis_hw[1])

    if vis_h > full_h or vis_w > full_w:
        raise ValueError(
            f"Visible shape {vis_hw} cannot be larger than full shape {full_hw}"
        )

    y0 = (full_h - vis_h) // 2
    x0 = (full_w - vis_w) // 2
    y1 = y0 + vis_h
    x1 = x0 + vis_w

    return slice(y0, y1), slice(x0, x1)


def _theory_hbar_mass(theory) -> tuple[float, float]:
    """
    Extract hbar and mass from theory with robust fallbacks.
    """
    hbar = float(getattr(theory, "hbar", 1.0))
    mass = float(
        getattr(
            theory,
            "m_mass",
            getattr(theory, "mass", getattr(theory, "m", 1.0)),
        )
    )

    if not np.isfinite(hbar):
        hbar = 1.0

    if (not np.isfinite(mass)) or abs(mass) <= 1e-30:
        mass = 1.0

    return hbar, mass


# ============================================================
# Local Dirac velocity from spinor
# ============================================================

def _dirac_velocity_from_spinor_frame(
    theory,
    spinor_frame: np.ndarray,
    eps_rho: float,
):
    """
    Compute Dirac velocity directly from a local/cropped spinor frame.

    This avoids requiring the frame to match theory.grid full shape.
    Uses the same formulas as DiracTheory.current()/velocity():

        rho = |psi1|^2 + |psi2|^2
        jx  = 2 c Re(conj(psi1) * psi2)
        jy  = 2 c Im(conj(psi1) * psi2)
        v   = j / rho, clamped to |v| <= c
    """
    frame = np.asarray(spinor_frame, dtype=np.complex128)

    if frame.ndim != 3 or frame.shape[0] != 2:
        raise ValueError(
            f"spinor_frame must have shape (2, H, W), got {frame.shape}"
        )

    psi1 = frame[0]
    psi2 = frame[1]

    rho = (np.abs(psi1) ** 2 + np.abs(psi2) ** 2).astype(float)
    overlap = np.conjugate(psi1) * psi2

    c = float(getattr(theory, "c_light", 1.0))

    jx = (2.0 * c * np.real(overlap)).astype(float)
    jy = (2.0 * c * np.imag(overlap)).astype(float)

    denom = np.maximum(rho, float(eps_rho))

    vx = jx / denom
    vy = jy / denom
    sp = np.hypot(vx, vy)

    mask = sp > c
    if np.any(mask):
        scale = c / np.maximum(sp[mask], float(eps_rho))
        vx = vx.copy()
        vy = vy.copy()
        sp = sp.copy()
        vx[mask] *= scale
        vy[mask] *= scale
        sp[mask] = c

    return vx.astype(float), vy.astype(float), sp.astype(float)


# ============================================================
# Local Schrödinger-like velocity from scalar frame
# ============================================================

def _schrodinger_velocity_from_scalar_frame(
    theory,
    scalar_frame: np.ndarray,
    dx: float,
    dy: float,
    eps_rho: float,
):
    """
    Compute Schrödinger/Bohm-like velocity directly from a scalar complex frame.

        rho = |psi|^2
        j   = (hbar/m) Im(conj(psi) grad psi)
        v   = j / rho

    This is useful for visible/cropped scalar frames when theory.velocity()
    expects a full-grid state.
    """
    psi = np.asarray(scalar_frame, dtype=np.complex128)

    if psi.ndim != 2:
        raise ValueError(f"scalar_frame must be 2D, got shape={psi.shape}")

    hbar, mass = _theory_hbar_mass(theory)

    rho = (np.abs(psi) ** 2).astype(float)

    dpsi_dy = np.gradient(psi, float(dy), axis=0)
    dpsi_dx = np.gradient(psi, float(dx), axis=1)

    pref = hbar / mass

    jx = (pref * np.imag(np.conjugate(psi) * dpsi_dx)).astype(float)
    jy = (pref * np.imag(np.conjugate(psi) * dpsi_dy)).astype(float)

    denom = np.maximum(rho, float(eps_rho))

    vx = jx / denom
    vy = jy / denom
    sp = np.hypot(vx, vy)

    return vx.astype(float), vy.astype(float), sp.astype(float)


# ============================================================
# Local entangled velocity from 4D spinor frame
# ============================================================

def _entangled_velocity_from_spinor_frame(
    theory,
    ent_frame: np.ndarray,
    dx: float,
    dy: float,
    eps_rho: float,
):
    """
    Compute Schrödinger-like velocity for an entangled spinor frame.

    Input:
        ent_frame.shape == (H, W, 2, 2)

    Formula:
        rho[y,x] = sum_ab |psi[y,x,a,b]|^2

        jx[y,x] = (hbar/m) Im sum_ab conj(psi[y,x,a,b]) dpsi[y,x,a,b]/dx
        jy[y,x] = (hbar/m) Im sum_ab conj(psi[y,x,a,b]) dpsi[y,x,a,b]/dy

        v = j / rho

    This treats the spin labels as internal components and sums the current
    over those components.
    """
    psi = np.asarray(ent_frame, dtype=np.complex128)

    if psi.ndim != 4 or psi.shape[-2:] != (2, 2):
        raise ValueError(
            f"entangled frame must have shape (H, W, 2, 2), got {psi.shape}"
        )

    hbar, mass = _theory_hbar_mass(theory)

    rho = np.sum(np.abs(psi) ** 2, axis=(-2, -1)).astype(float)

    dpsi_dy = np.gradient(psi, float(dy), axis=0)
    dpsi_dx = np.gradient(psi, float(dx), axis=1)

    pref = hbar / mass

    jx = (
        pref
        * np.imag(
            np.sum(np.conjugate(psi) * dpsi_dx, axis=(-2, -1))
        )
    ).astype(float)

    jy = (
        pref
        * np.imag(
            np.sum(np.conjugate(psi) * dpsi_dy, axis=(-2, -1))
        )
    ).astype(float)

    denom = np.maximum(rho, float(eps_rho))

    vx = jx / denom
    vy = jy / denom
    sp = np.hypot(vx, vy)

    return vx.astype(float), vy.astype(float), sp.astype(float)


# ============================================================
# Velocity field extraction
# ============================================================

def _extract_visible_velocity_fields(
    theory,
    state_frame: np.ndarray,
    vis_hw: tuple[int, int],
    eps_rho: float,
    dx: float,
    dy: float,
):
    """
    Compute velocity fields for one frame, then crop to visible shape if needed.

    Supported input frame shapes:
      - scalar visible/full field:
          (H, W)

      - Dirac visible/full state:
          (2, H, W)

      - entangled visible/full state:
          (H, W, 2, 2)

    Important:
      - visible/cropped scalar and entangled frames are computed locally
        from current formulas.
      - visible/cropped Dirac spinors are computed locally from the Dirac
        current formula.
      - full-grid frames may use theory.velocity(...) when safe, then crop.
    """
    frame = np.asarray(state_frame)

    vis_h, vis_w = int(vis_hw[0]), int(vis_hw[1])

    expected_full = None
    if hasattr(theory, "grid") and hasattr(theory.grid, "Ny") and hasattr(theory.grid, "Nx"):
        expected_full = (int(theory.grid.Ny), int(theory.grid.Nx))

    # --------------------------------------------------------
    # Entangled spinor frame: (H, W, 2, 2)
    # --------------------------------------------------------
    if frame.ndim == 4 and frame.shape[-2:] == (2, 2):
        h, w = frame.shape[0], frame.shape[1]

        # Visible-size cropped entangled frame -> compute locally.
        if (h, w) == (vis_h, vis_w):
            return _entangled_velocity_from_spinor_frame(
                theory=theory,
                ent_frame=frame,
                dx=dx,
                dy=dy,
                eps_rho=eps_rho,
            )

        # Full-size entangled frame.
        # Prefer theory.velocity if it supports this shape; otherwise compute
        # locally and center-crop.
        if expected_full is not None and (h, w) == expected_full:
            try:
                vx_full, vy_full, sp_full = theory.velocity(frame, eps_rho=eps_rho)
            except Exception:
                vx_full, vy_full, sp_full = _entangled_velocity_from_spinor_frame(
                    theory=theory,
                    ent_frame=frame,
                    dx=dx,
                    dy=dy,
                    eps_rho=eps_rho,
                )

            sy, sx = _center_crop_slices_from_shapes((h, w), (vis_h, vis_w))
            return vx_full[sy, sx], vy_full[sy, sx], sp_full[sy, sx]

        raise ValueError(
            f"Unsupported entangled frame shape {frame.shape}. "
            f"Expected either ({vis_h}, {vis_w}, 2, 2) visible or "
            f"({expected_full[0]}, {expected_full[1]}, 2, 2) full."
            if expected_full is not None
            else f"Unsupported entangled frame shape {frame.shape}."
        )

    # --------------------------------------------------------
    # Dirac / component-first spinor frame: (2, H, W)
    # --------------------------------------------------------
    if frame.ndim == 3 and frame.shape[0] == 2:
        h, w = frame.shape[1], frame.shape[2]

        # Visible-size cropped spinor -> compute locally.
        if (h, w) == (vis_h, vis_w):
            return _dirac_velocity_from_spinor_frame(
                theory=theory,
                spinor_frame=frame,
                eps_rho=eps_rho,
            )

        # Full-size spinor -> compute on full grid, then crop.
        if expected_full is not None and (h, w) == expected_full:
            try:
                vx_full, vy_full, sp_full = theory.velocity(frame, eps_rho=eps_rho)
            except Exception:
                vx_full, vy_full, sp_full = _dirac_velocity_from_spinor_frame(
                    theory=theory,
                    spinor_frame=frame,
                    eps_rho=eps_rho,
                )

            sy, sx = _center_crop_slices_from_shapes((h, w), (vis_h, vis_w))
            return vx_full[sy, sx], vy_full[sy, sx], sp_full[sy, sx]

        raise ValueError(
            f"Unsupported spinor frame shape {frame.shape}. "
            f"Expected either (2, {vis_h}, {vis_w}) visible or "
            f"(2, {expected_full[0]}, {expected_full[1]}) full."
            if expected_full is not None
            else f"Unsupported spinor frame shape {frame.shape}."
        )

    # --------------------------------------------------------
    # Scalar frame: (H, W)
    # --------------------------------------------------------
    if frame.ndim == 2:
        h, w = frame.shape

        # Visible scalar field.
        # Prefer direct local current formula; theory.velocity may expect full grid.
        if (h, w) == (vis_h, vis_w):
            try:
                vx, vy, sp = theory.velocity(frame, eps_rho=eps_rho)
                if vx.shape == (vis_h, vis_w) and vy.shape == (vis_h, vis_w):
                    return vx, vy, sp
            except Exception:
                pass

            return _schrodinger_velocity_from_scalar_frame(
                theory=theory,
                scalar_frame=frame,
                dx=dx,
                dy=dy,
                eps_rho=eps_rho,
            )

        # Full scalar field -> compute then crop.
        if expected_full is not None and (h, w) == expected_full:
            try:
                vx_full, vy_full, sp_full = theory.velocity(frame, eps_rho=eps_rho)
            except Exception:
                vx_full, vy_full, sp_full = _schrodinger_velocity_from_scalar_frame(
                    theory=theory,
                    scalar_frame=frame,
                    dx=dx,
                    dy=dy,
                    eps_rho=eps_rho,
                )

            sy, sx = _center_crop_slices_from_shapes((h, w), (vis_h, vis_w))
            return vx_full[sy, sx], vy_full[sy, sx], sp_full[sy, sx]

        raise ValueError(
            f"Unsupported scalar frame shape {frame.shape}. "
            f"Expected either ({vis_h}, {vis_w}) visible or {expected_full} full."
        )

    raise ValueError(
        f"Unsupported state frame ndim/shape: ndim={frame.ndim}, shape={frame.shape}"
    )


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

    Supports:
      - scalar frame stacks:
          (T, H, W)

      - Dirac/spinor frame stacks:
          (T, 2, H, W)

      - entangled spinor frame stacks:
          (T, H, W, 2, 2)

    If frames are full-grid, velocity is computed first and then center-cropped.
    If frames are already visible/cropped, velocity is computed directly from
    local current formulas.
    """
    state_vis_frames = np.asarray(state_vis_frames)

    Nt_ = len(ridge_x)

    if len(ridge_y) != Nt_:
        raise ValueError("ridge_x and ridge_y must have the same length")

    if state_vis_frames.shape[0] != Nt_:
        raise ValueError(
            f"Number of frames ({state_vis_frames.shape[0]}) does not match "
            f"ridge length ({Nt_})"
        )

    if state_vis_frames.ndim not in (3, 4, 5):
        raise ValueError(
            f"state_vis_frames must have ndim 3, 4, or 5, got "
            f"shape={state_vis_frames.shape}"
        )

    vis_h = len(y_vis_1d)
    vis_w = len(x_vis_1d)

    if not np.isfinite(float(dx)) or float(dx) <= 0.0:
        raise ValueError(f"dx must be positive finite, got {dx}")

    if not np.isfinite(float(dy)) or float(dy) <= 0.0:
        raise ValueError(f"dy must be positive finite, got {dy}")

    cos_th = np.full(Nt_, np.nan, dtype=float)
    speed = np.full(Nt_, np.nan, dtype=float)
    ux = np.full(Nt_, np.nan, dtype=float)
    uy = np.full(Nt_, np.nan, dtype=float)
    div_v_at_ridge = np.full(Nt_, np.nan, dtype=float)

    tx, ty = ridge_tangent_unit(ridge_x, ridge_y)
    kernel = gaussian_kernel_2d(arrow_avg_radius, arrow_avg_gauss_sigma)

    for i in range(Nt_):
        state_frame = state_vis_frames[i]

        vx, vy, sp = _extract_visible_velocity_fields(
            theory=theory,
            state_frame=state_frame,
            vis_hw=(vis_h, vis_w),
            eps_rho=align_eps_rho,
            dx=dx,
            dy=dy,
        )

        vx = np.asarray(vx, dtype=float)
        vy = np.asarray(vy, dtype=float)
        sp = np.asarray(sp, dtype=float)

        if vx.shape != (vis_h, vis_w) or vy.shape != (vis_h, vis_w) or sp.shape != (vis_h, vis_w):
            raise ValueError(
                f"Velocity field shape mismatch at frame {i}: "
                f"vx={vx.shape}, vy={vy.shape}, sp={sp.shape}, "
                f"expected {(vis_h, vis_w)}"
            )

        if not np.all(np.isfinite(vx)):
            raise ValueError(f"vx contains non-finite values at frame {i}")

        if not np.all(np.isfinite(vy)):
            raise ValueError(f"vy contains non-finite values at frame {i}")

        if not np.all(np.isfinite(sp)):
            raise ValueError(f"speed contains non-finite values at frame {i}")

        ix = int(np.argmin(np.abs(x_vis_1d - ridge_x[i])))
        iy = int(np.argmin(np.abs(y_vis_1d - ridge_y[i])))

        ix = int(np.clip(ix, 0, vis_w - 1))
        iy = int(np.clip(iy, 0, vis_h - 1))

        if arrow_spatial_avg:
            uxi, uyi, spd = local_weighted_mean(vx, vy, sp, ix, iy, kernel)
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
            if np.isfinite(div_v[iy, ix]):
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