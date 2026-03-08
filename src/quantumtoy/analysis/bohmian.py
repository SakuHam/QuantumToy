from __future__ import annotations

import numpy as np


# ============================================================
# Bilinear interpolation
# ============================================================

def bilinear_interpolate_scalar(
    field: np.ndarray,
    xq: float,
    yq: float,
    x0_arr: np.ndarray,
    y0_arr: np.ndarray,
    dx_: float,
    dy_: float,
):
    """
    Bilinear interpolation of a scalar field sampled on a regular grid.
    Returns NaN if query point is outside interpolation range.
    """
    nx_ = len(x0_arr)
    ny_ = len(y0_arr)

    fx = (xq - x0_arr[0]) / dx_
    fy = (yq - y0_arr[0]) / dy_

    if not (0.0 <= fx < nx_ - 1 and 0.0 <= fy < ny_ - 1):
        return np.nan

    ix0 = int(np.floor(fx))
    iy0 = int(np.floor(fy))

    txi = fx - ix0
    tyi = fy - iy0

    f00 = field[iy0, ix0]
    f10 = field[iy0, ix0 + 1]
    f01 = field[iy0 + 1, ix0]
    f11 = field[iy0 + 1, ix0 + 1]

    return (
        (1.0 - txi) * (1.0 - tyi) * f00
        + txi * (1.0 - tyi) * f10
        + (1.0 - txi) * tyi * f01
        + txi * tyi * f11
    )


def bilinear_interpolate_vector(
    vx: np.ndarray,
    vy: np.ndarray,
    xq: float,
    yq: float,
    x0_arr: np.ndarray,
    y0_arr: np.ndarray,
    dx_: float,
    dy_: float,
):
    vxq = bilinear_interpolate_scalar(vx, xq, yq, x0_arr, y0_arr, dx_, dy_)
    vyq = bilinear_interpolate_scalar(vy, xq, yq, x0_arr, y0_arr, dx_, dy_)
    return vxq, vyq


# ============================================================
# Visible region helpers
# ============================================================

def is_inside_visible(
    xp: float,
    yp: float,
    x_vis_min: float,
    x_vis_max: float,
    y_vis_min: float,
    y_vis_max: float,
) -> bool:
    return (x_vis_min <= xp < x_vis_max) and (y_vis_min <= yp < y_vis_max)


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


# ============================================================
# Local Dirac rho/current/velocity from spinor
# ============================================================

def _dirac_rho_v_from_spinor_frame(
    theory,
    frame: np.ndarray,
    eps_rho: float = 1e-10,
):
    """
    Compute rho, vx, vy directly from a local/cropped Dirac spinor frame.

    Uses:
        rho = |psi1|^2 + |psi2|^2
        jx  = 2 c Re(conj(psi1) * psi2)
        jy  = 2 c Im(conj(psi1) * psi2)
        v   = j / rho, clamped to |v| <= c
    """
    frame = np.asarray(frame, dtype=np.complex128)

    if frame.ndim != 3 or frame.shape[0] != 2:
        raise ValueError(
            f"Expected Dirac spinor frame shape (2, H, W), got {frame.shape}"
        )

    psi1 = frame[0]
    psi2 = frame[1]

    rho = (np.abs(psi1) ** 2 + np.abs(psi2) ** 2).astype(float)
    overlap = np.conjugate(psi1) * psi2

    c = float(theory.c_light)

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

    return vx.astype(float), vy.astype(float), rho.astype(float)


def _rho_v_from_frame(
    theory,
    frame: np.ndarray,
    eps_rho: float = 1e-10,
    visible_hw: tuple[int, int] | None = None,
):
    """
    Unified rho/vx/vy extraction from one frame.

    Supported:
      - scalar visible/full frame: (H, W)
      - Dirac visible/full frame:  (2, H, W)

    Rules:
      - full Dirac frame matching theory.grid -> use theory.velocity/theory.density
      - cropped Dirac frame -> compute locally from spinor
      - full scalar frame -> theory.velocity/theory.density, optionally center-crop
      - visible scalar frame -> theory.velocity/theory.density directly
    """
    frame = np.asarray(frame)

    expected_full = None
    if hasattr(theory, "grid") and hasattr(theory.grid, "Ny") and hasattr(theory.grid, "Nx"):
        expected_full = (int(theory.grid.Ny), int(theory.grid.Nx))

    # --------------------------------------------------------
    # Dirac / spinor frame
    # --------------------------------------------------------
    if frame.ndim == 3 and frame.shape[0] == 2:
        h, w = frame.shape[1], frame.shape[2]

        # Full Dirac frame
        if expected_full is not None and (h, w) == expected_full:
            rho_full = theory.density(frame)
            vx_full, vy_full, _ = theory.velocity(frame, eps_rho=eps_rho)

            if visible_hw is None or visible_hw == (h, w):
                return vx_full, vy_full, rho_full

            sy, sx = _center_crop_slices_from_shapes((h, w), visible_hw)
            return vx_full[sy, sx], vy_full[sy, sx], rho_full[sy, sx]

        # Visible/cropped Dirac frame
        return _dirac_rho_v_from_spinor_frame(
            theory=theory,
            frame=frame,
            eps_rho=eps_rho,
        )

    # --------------------------------------------------------
    # Scalar frame
    # --------------------------------------------------------
    if frame.ndim == 2:
        h, w = frame.shape

        rho = theory.density(frame)
        vx, vy, _ = theory.velocity(frame, eps_rho=eps_rho)

        if visible_hw is None or (h, w) == visible_hw:
            return vx, vy, rho

        if expected_full is not None and (h, w) == expected_full:
            sy, sx = _center_crop_slices_from_shapes((h, w), visible_hw)
            return vx[sy, sx], vy[sy, sx], rho[sy, sx]

        return vx, vy, rho

    raise ValueError(f"Unsupported frame shape: {frame.shape}")


# ============================================================
# Velocity frame construction
# ============================================================

def build_velocity_frames_from_state(
    theory,
    state_vis_frames: np.ndarray,
    eps_rho: float = 1e-10,
):
    """
    Build visible-grid velocity and density frames.

    Supports:
        scalar state frames: (Nt, H, W)
        spinor state frames: (Nt, C, H, W)

    For Dirac/spinor frames:
      - full-grid spinors are handled via theory.velocity/theory.density
      - cropped visible spinors are handled directly from local spinor formulas

    Returns:
        vx_frames, vy_frames, rho_frames
    """
    state_vis_frames = np.asarray(state_vis_frames)

    if state_vis_frames.ndim not in (3, 4):
        raise ValueError(
            f"Unsupported state_vis_frames ndim={state_vis_frames.ndim}, "
            f"shape={state_vis_frames.shape}"
        )

    Nt_ = state_vis_frames.shape[0]

    # Determine visible output shape from first frame.
    frame0 = np.asarray(state_vis_frames[0])

    if frame0.ndim == 2:
        visible_hw = frame0.shape
    elif frame0.ndim == 3 and frame0.shape[0] == 2:
        visible_hw = (frame0.shape[1], frame0.shape[2])
    else:
        raise ValueError(f"Unsupported first frame shape: {frame0.shape}")

    vx0, vy0, rho0 = _rho_v_from_frame(
        theory=theory,
        frame=frame0,
        eps_rho=eps_rho,
        visible_hw=visible_hw,
    )

    H, W = rho0.shape

    vx_frames = np.zeros((Nt_, H, W), dtype=float)
    vy_frames = np.zeros((Nt_, H, W), dtype=float)
    rho_frames = np.zeros((Nt_, H, W), dtype=float)

    vx_frames[0] = vx0
    vy_frames[0] = vy0
    rho_frames[0] = rho0

    for i in range(1, Nt_):
        vxi, vyi, rhoi = _rho_v_from_frame(
            theory=theory,
            frame=state_vis_frames[i],
            eps_rho=eps_rho,
            visible_hw=visible_hw,
        )

        if vxi.shape != (H, W) or vyi.shape != (H, W) or rhoi.shape != (H, W):
            raise ValueError(
                f"Frame {i} produced inconsistent shapes: "
                f"vx={vxi.shape}, vy={vyi.shape}, rho={rhoi.shape}, expected {(H, W)}"
            )

        vx_frames[i] = vxi
        vy_frames[i] = vyi
        rho_frames[i] = rhoi

    return vx_frames, vy_frames, rho_frames


# ============================================================
# Time interpolation
# ============================================================

def velocity_rho_at_time(
    vx_frames: np.ndarray,
    vy_frames: np.ndarray,
    rho_frames: np.ndarray,
    times: np.ndarray,
    tau_step: float,
    t_query: float,
):
    """
    Linear interpolation in time between saved frames.
    Returns full 2D fields on visible grid.
    """
    if t_query <= times[0]:
        return vx_frames[0], vy_frames[0], rho_frames[0]

    if t_query >= times[-1]:
        return vx_frames[-1], vy_frames[-1], rho_frames[-1]

    s = (t_query - times[0]) / tau_step
    i0 = int(np.floor(s))
    i1 = min(i0 + 1, len(times) - 1)
    a = float(s - i0)

    vx_t = (1.0 - a) * vx_frames[i0] + a * vx_frames[i1]
    vy_t = (1.0 - a) * vy_frames[i0] + a * vy_frames[i1]
    rho_t = (1.0 - a) * rho_frames[i0] + a * rho_frames[i1]

    return vx_t, vy_t, rho_t


def velocity_sample_time_space(
    vx_frames: np.ndarray,
    vy_frames: np.ndarray,
    rho_frames: np.ndarray,
    times: np.ndarray,
    tau_step: float,
    t_query: float,
    xq: float,
    yq: float,
    x_vis_1d: np.ndarray,
    y_vis_1d: np.ndarray,
    dx: float,
    dy: float,
):
    """
    Sample velocity and density at arbitrary time and position
    using linear interpolation in time and bilinear interpolation in space.
    """
    vx_t, vy_t, rho_t = velocity_rho_at_time(
        vx_frames=vx_frames,
        vy_frames=vy_frames,
        rho_frames=rho_frames,
        times=times,
        tau_step=tau_step,
        t_query=t_query,
    )

    rho_q = bilinear_interpolate_scalar(
        rho_t, xq, yq, x_vis_1d, y_vis_1d, dx, dy
    )

    vx_q, vy_q = bilinear_interpolate_vector(
        vx_t, vy_t, xq, yq, x_vis_1d, y_vis_1d, dx, dy
    )

    return vx_q, vy_q, rho_q


# ============================================================
# Initial point selection
# ============================================================

def sample_born_initial_points_from_visible_psi(
    psi0_vis: np.ndarray,
    ntraj: int,
    rng: np.random.Generator,
    x_vis_1d: np.ndarray,
    y_vis_1d: np.ndarray,
    with_replacement: bool = False,
):
    """
    Sample initial Bohmian points from Born density on visible grid.

    Supports:
        scalar wavefunction: (H, W)
        spinor wavefunction: (C, H, W)
    """
    if psi0_vis.ndim == 2:
        rho0 = np.abs(psi0_vis) ** 2
    elif psi0_vis.ndim == 3:
        rho0 = np.sum(np.abs(psi0_vis) ** 2, axis=0)
    else:
        raise ValueError(f"Unsupported psi0_vis ndim={psi0_vis.ndim}")

    w = rho0.ravel().astype(float)
    s = float(np.sum(w))

    if s <= 0:
        return [(float(x_vis_1d[0]), float(y_vis_1d[len(y_vis_1d) // 2]))]

    p = w / s

    nsel = min(ntraj, p.size) if not with_replacement else ntraj
    idxs = rng.choice(
        p.size,
        size=nsel,
        replace=with_replacement,
        p=p,
    )

    pts = []
    for idx in np.atleast_1d(idxs):
        iy0, ix0 = np.unravel_index(int(idx), rho0.shape)
        pts.append((float(x_vis_1d[ix0]), float(y_vis_1d[iy0])))

    return pts


def make_bohmian_initial_points(
    mode: str,
    ntraj: int,
    custom_points,
    ridge_x0: float,
    ridge_y0: float,
    x0_packet: float,
    y0_packet: float,
    psi0_vis: np.ndarray,
    x_vis_1d: np.ndarray,
    y_vis_1d: np.ndarray,
    jitter: float = 0.0,
    rng_seed: int = 20260306,
    with_replacement: bool = False,
):
    """
    Build initial Bohmian points.

    Modes:
        - 'born_initial'
        - 'packet_center'
        - 'ridge_start'
        - 'custom'
    """
    rng_b = np.random.default_rng(rng_seed)

    if mode == "born_initial":
        return sample_born_initial_points_from_visible_psi(
            psi0_vis=psi0_vis,
            ntraj=ntraj,
            rng=rng_b,
            x_vis_1d=x_vis_1d,
            y_vis_1d=y_vis_1d,
            with_replacement=with_replacement,
        )

    if mode == "packet_center":
        if ntraj <= 1:
            return [(x0_packet, y0_packet)]

        offsets = (
            np.linspace(-(ntraj - 1) / 2.0, (ntraj - 1) / 2.0, ntraj)
            * max(jitter, 0.15)
        )
        return [(x0_packet, y0_packet + off) for off in offsets]

    if mode == "ridge_start":
        if ntraj <= 1:
            return [(ridge_x0, ridge_y0)]

        offsets = (
            np.linspace(-(ntraj - 1) / 2.0, (ntraj - 1) / 2.0, ntraj)
            * max(jitter, 0.15)
        )
        return [(ridge_x0, ridge_y0 + off) for off in offsets]

    if mode == "custom":
        return list(custom_points[:ntraj])

    raise ValueError(f"Unknown BOHMIAN_INIT_MODE: {mode}")


# ============================================================
# Bohmian RHS
# ============================================================

def bohmian_rhs(
    vx_frames: np.ndarray,
    vy_frames: np.ndarray,
    rho_frames: np.ndarray,
    times: np.ndarray,
    tau_step: float,
    t_query: float,
    xq: float,
    yq: float,
    x_vis_1d: np.ndarray,
    y_vis_1d: np.ndarray,
    dx: float,
    dy: float,
    stop_outside_visible: bool,
    x_vis_min: float,
    x_vis_max: float,
    y_vis_min: float,
    y_vis_max: float,
    stop_on_low_rho: bool,
    min_rho: float,
):
    """
    Evaluate Bohmian velocity field at (t_query, xq, yq).
    Returns:
        vx_q, vy_q, rho_q
    """
    if stop_outside_visible and not is_inside_visible(
        xq, yq, x_vis_min, x_vis_max, y_vis_min, y_vis_max
    ):
        return np.nan, np.nan, np.nan

    vx_q, vy_q, rho_q = velocity_sample_time_space(
        vx_frames=vx_frames,
        vy_frames=vy_frames,
        rho_frames=rho_frames,
        times=times,
        tau_step=tau_step,
        t_query=t_query,
        xq=xq,
        yq=yq,
        x_vis_1d=x_vis_1d,
        y_vis_1d=y_vis_1d,
        dx=dx,
        dy=dy,
    )

    if stop_on_low_rho and (not np.isfinite(rho_q) or rho_q < min_rho):
        return np.nan, np.nan, rho_q

    if not (np.isfinite(vx_q) and np.isfinite(vy_q)):
        return np.nan, np.nan, rho_q

    return float(vx_q), float(vy_q), float(rho_q)


# ============================================================
# Integrators
# ============================================================

def rk4_step_bohmian(
    vx_frames: np.ndarray,
    vy_frames: np.ndarray,
    rho_frames: np.ndarray,
    times: np.ndarray,
    tau_step: float,
    t0: float,
    xcur: float,
    ycur: float,
    h: float,
    x_vis_1d: np.ndarray,
    y_vis_1d: np.ndarray,
    dx: float,
    dy: float,
    stop_outside_visible: bool,
    x_vis_min: float,
    x_vis_max: float,
    y_vis_min: float,
    y_vis_max: float,
    stop_on_low_rho: bool,
    min_rho: float,
):
    k1x, k1y, _ = bohmian_rhs(
        vx_frames, vy_frames, rho_frames,
        times, tau_step,
        t0, xcur, ycur,
        x_vis_1d, y_vis_1d, dx, dy,
        stop_outside_visible, x_vis_min, x_vis_max, y_vis_min, y_vis_max,
        stop_on_low_rho, min_rho,
    )
    if not (np.isfinite(k1x) and np.isfinite(k1y)):
        return np.nan, np.nan

    k2x, k2y, _ = bohmian_rhs(
        vx_frames, vy_frames, rho_frames,
        times, tau_step,
        t0 + 0.5 * h,
        xcur + 0.5 * h * k1x,
        ycur + 0.5 * h * k1y,
        x_vis_1d, y_vis_1d, dx, dy,
        stop_outside_visible, x_vis_min, x_vis_max, y_vis_min, y_vis_max,
        stop_on_low_rho, min_rho,
    )
    if not (np.isfinite(k2x) and np.isfinite(k2y)):
        return np.nan, np.nan

    k3x, k3y, _ = bohmian_rhs(
        vx_frames, vy_frames, rho_frames,
        times, tau_step,
        t0 + 0.5 * h,
        xcur + 0.5 * h * k2x,
        ycur + 0.5 * h * k2y,
        x_vis_1d, y_vis_1d, dx, dy,
        stop_outside_visible, x_vis_min, x_vis_max, y_vis_min, y_vis_max,
        stop_on_low_rho, min_rho,
    )
    if not (np.isfinite(k3x) and np.isfinite(k3y)):
        return np.nan, np.nan

    k4x, k4y, _ = bohmian_rhs(
        vx_frames, vy_frames, rho_frames,
        times, tau_step,
        t0 + h,
        xcur + h * k3x,
        ycur + h * k3y,
        x_vis_1d, y_vis_1d, dx, dy,
        stop_outside_visible, x_vis_min, x_vis_max, y_vis_min, y_vis_max,
        stop_on_low_rho, min_rho,
    )
    if not (np.isfinite(k4x) and np.isfinite(k4y)):
        return np.nan, np.nan

    xnext = xcur + (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    ynext = ycur + (h / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)

    return xnext, ynext


def euler_step_bohmian(
    vx_frames: np.ndarray,
    vy_frames: np.ndarray,
    rho_frames: np.ndarray,
    times: np.ndarray,
    tau_step: float,
    t0: float,
    xcur: float,
    ycur: float,
    h: float,
    x_vis_1d: np.ndarray,
    y_vis_1d: np.ndarray,
    dx: float,
    dy: float,
    stop_outside_visible: bool,
    x_vis_min: float,
    x_vis_max: float,
    y_vis_min: float,
    y_vis_max: float,
    stop_on_low_rho: bool,
    min_rho: float,
):
    vx_q, vy_q, _ = bohmian_rhs(
        vx_frames, vy_frames, rho_frames,
        times, tau_step,
        t0, xcur, ycur,
        x_vis_1d, y_vis_1d, dx, dy,
        stop_outside_visible, x_vis_min, x_vis_max, y_vis_min, y_vis_max,
        stop_on_low_rho, min_rho,
    )

    if not (np.isfinite(vx_q) and np.isfinite(vy_q)):
        return np.nan, np.nan

    return xcur + h * vx_q, ycur + h * vy_q


# ============================================================
# Trajectory integration
# ============================================================

def integrate_bohmian_trajectories(
    vx_frames: np.ndarray,
    vy_frames: np.ndarray,
    rho_frames: np.ndarray,
    times: np.ndarray,
    tau_step: float,
    x_vis_1d: np.ndarray,
    y_vis_1d: np.ndarray,
    dx: float,
    dy: float,
    init_points,
    stop_outside_visible: bool,
    x_vis_min: float,
    x_vis_max: float,
    y_vis_min: float,
    y_vis_max: float,
    stop_on_low_rho: bool,
    min_rho: float,
    use_rk4: bool = True,
):
    """
    Integrate Bohmian trajectories across saved visible frames.

    Returns:
        traj_x, traj_y, traj_alive
    """
    Nt_ = len(times)
    ntraj = len(init_points)

    traj_x = np.full((ntraj, Nt_), np.nan, dtype=float)
    traj_y = np.full((ntraj, Nt_), np.nan, dtype=float)
    traj_alive = np.zeros((ntraj, Nt_), dtype=bool)

    stepper = rk4_step_bohmian if use_rk4 else euler_step_bohmian

    for k, (x_init, y_init) in enumerate(init_points):
        xcur = float(x_init)
        ycur = float(y_init)

        if is_inside_visible(
            xcur, ycur, x_vis_min, x_vis_max, y_vis_min, y_vis_max
        ):
            traj_x[k, 0] = xcur
            traj_y[k, 0] = ycur
            traj_alive[k, 0] = True

        for i in range(0, Nt_ - 1):
            if not traj_alive[k, i]:
                break

            xnext, ynext = stepper(
                vx_frames=vx_frames,
                vy_frames=vy_frames,
                rho_frames=rho_frames,
                times=times,
                tau_step=tau_step,
                t0=times[i],
                xcur=xcur,
                ycur=ycur,
                h=tau_step,
                x_vis_1d=x_vis_1d,
                y_vis_1d=y_vis_1d,
                dx=dx,
                dy=dy,
                stop_outside_visible=stop_outside_visible,
                x_vis_min=x_vis_min,
                x_vis_max=x_vis_max,
                y_vis_min=y_vis_min,
                y_vis_max=y_vis_max,
                stop_on_low_rho=stop_on_low_rho,
                min_rho=min_rho,
            )

            if not (np.isfinite(xnext) and np.isfinite(ynext)):
                break

            if stop_outside_visible and not is_inside_visible(
                xnext, ynext, x_vis_min, x_vis_max, y_vis_min, y_vis_max
            ):
                break

            traj_x[k, i + 1] = xnext
            traj_y[k, i + 1] = ynext
            traj_alive[k, i + 1] = True

            xcur, ycur = xnext, ynext

    return traj_x, traj_y, traj_alive