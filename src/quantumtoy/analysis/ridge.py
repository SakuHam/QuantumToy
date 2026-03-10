from __future__ import annotations

import numpy as np


def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def _assert_frame_pair_compatible(frame_psi, frame_emix):
    _assert(isinstance(frame_psi, np.ndarray), "frame_psi must be np.ndarray")
    _assert(isinstance(frame_emix, np.ndarray), "frame_emix must be np.ndarray")
    _assert(frame_psi.shape == frame_emix.shape,
            f"frame_psi.shape {frame_psi.shape} != frame_emix.shape {frame_emix.shape}")
    _assert(frame_psi.ndim in (2, 3),
            f"frame_psi.ndim must be 2 or 3, got {frame_psi.ndim}")
    _assert(np.all(np.isfinite(frame_psi.real)), "frame_psi.real contains non-finite values")
    _assert(np.all(np.isfinite(frame_psi.imag)), "frame_psi.imag contains non-finite values")
    _assert(np.all(np.isfinite(frame_emix.real)), "frame_emix.real contains non-finite values")
    _assert(np.all(np.isfinite(frame_emix.imag)), "frame_emix.imag contains non-finite values")


def _assert_vis_axes(x_vis_1d, y_vis_1d, Gamma_shape):
    _assert(isinstance(x_vis_1d, np.ndarray), "x_vis_1d must be np.ndarray")
    _assert(isinstance(y_vis_1d, np.ndarray), "y_vis_1d must be np.ndarray")
    _assert(x_vis_1d.ndim == 1, f"x_vis_1d must be 1D, got {x_vis_1d.ndim}")
    _assert(y_vis_1d.ndim == 1, f"y_vis_1d must be 1D, got {y_vis_1d.ndim}")
    _assert(Gamma_shape == (len(y_vis_1d), len(x_vis_1d)),
            f"Gamma shape {Gamma_shape} != {(len(y_vis_1d), len(x_vis_1d))}")
    _assert(np.all(np.isfinite(x_vis_1d)), "x_vis_1d contains non-finite values")
    _assert(np.all(np.isfinite(y_vis_1d)), "y_vis_1d contains non-finite values")


def local_gamma_field(frame_psi, frame_emix):
    """
    Build positive scalar ridge field Gamma from local amplitude overlap.

    Schrödinger:
        A = conj(emix) * psi
        Gamma = |A|^2

    Dirac:
        A = sum_a conj(emix_a) * psi_a
        Gamma = |A|^2
    """
    _assert_frame_pair_compatible(frame_psi, frame_emix)

    if frame_psi.ndim == 2 and frame_emix.ndim == 2:
        A = np.conjugate(frame_emix) * frame_psi
        Gamma = (np.abs(A) ** 2).astype(float)

    elif frame_psi.ndim == 3 and frame_emix.ndim == 3:
        A = np.sum(np.conjugate(frame_emix) * frame_psi, axis=0)
        Gamma = (np.abs(A) ** 2).astype(float)

    else:
        raise ValueError(
            f"Incompatible frame shapes: frame_psi.ndim={frame_psi.ndim}, "
            f"frame_emix.ndim={frame_emix.ndim}"
        )

    _assert(np.all(np.isfinite(Gamma)), "Gamma contains non-finite values")
    _assert(np.all(Gamma >= -1e-14), "Gamma contains significantly negative values")
    return Gamma


def ridge_argmax(Gamma, x_vis_1d, y_vis_1d):
    _assert_vis_axes(x_vis_1d, y_vis_1d, Gamma.shape)
    idx = int(np.argmax(Gamma))
    iy, ix = np.unravel_index(idx, Gamma.shape)
    return float(x_vis_1d[ix]), float(y_vis_1d[iy]), float(Gamma[iy, ix])


def ridge_centroid_top(Gamma, x_vis_1d, y_vis_1d, top_q=0.02, eps=1e-30):
    _assert_vis_axes(x_vis_1d, y_vis_1d, Gamma.shape)
    _assert(0.0 < top_q < 1.0, f"top_q must be in (0,1), got {top_q}")

    g = Gamma.astype(float)
    gmax = float(np.max(g))
    if gmax <= 0:
        return ridge_argmax(g, x_vis_1d, y_vis_1d)

    thr = np.quantile(g.ravel(), 1.0 - top_q)
    mask = g >= thr
    w = g[mask] + eps

    if w.size == 0 or float(np.sum(w)) <= 0:
        return ridge_argmax(g, x_vis_1d, y_vis_1d)

    iy_idx, ix_idx = np.where(mask)
    xs_ = x_vis_1d[ix_idx]
    ys_ = y_vis_1d[iy_idx]

    wsum = float(np.sum(w))
    xc = float(np.sum(xs_ * w) / wsum)
    yc = float(np.sum(ys_ * w) / wsum)

    ixn = int(np.argmin(np.abs(x_vis_1d - xc)))
    iyn = int(np.argmin(np.abs(y_vis_1d - yc)))
    score = float(g[iyn, ixn])

    return xc, yc, score


def snap_to_localmax_near_point(Gamma, x_vis_1d, y_vis_1d, xc, yc, radius=12):
    """
    Snap a continuous point (xc, yc) to the strongest local max in a window.
    """
    _assert_vis_axes(x_vis_1d, y_vis_1d, Gamma.shape)
    H, W = Gamma.shape

    ix0 = int(np.argmin(np.abs(x_vis_1d - xc)))
    iy0 = int(np.argmin(np.abs(y_vis_1d - yc)))

    x1 = max(0, ix0 - radius)
    x2 = min(W, ix0 + radius + 1)
    y1 = max(0, iy0 - radius)
    y2 = min(H, iy0 + radius + 1)

    sub = Gamma[y1:y2, x1:x2]
    if sub.size == 0:
        return float(x_vis_1d[ix0]), float(y_vis_1d[iy0]), float(Gamma[iy0, ix0]), ix0, iy0

    idx = int(np.argmax(sub))
    sy, sx = np.unravel_index(idx, sub.shape)
    iy = y1 + sy
    ix = x1 + sx

    return float(x_vis_1d[ix]), float(y_vis_1d[iy]), float(Gamma[iy, ix]), ix, iy


def ridge_centroid_top_snap_localmax(
    Gamma, x_vis_1d, y_vis_1d, top_q=0.02, snap_radius=12
):
    xc, yc, _ = ridge_centroid_top(Gamma, x_vis_1d, y_vis_1d, top_q=top_q)
    xs, ys, ss, _, _ = snap_to_localmax_near_point(
        Gamma, x_vis_1d, y_vis_1d, xc, yc, radius=snap_radius
    )
    return xs, ys, ss


def ridge_localmax_track(Gamma, x_vis_1d, y_vis_1d, prev_ix, prev_iy, radius=20):
    _assert_vis_axes(x_vis_1d, y_vis_1d, Gamma.shape)

    H, W_ = Gamma.shape

    x0i = int(np.clip(prev_ix, 0, W_ - 1))
    y0i = int(np.clip(prev_iy, 0, H - 1))

    x1 = max(0, x0i - radius)
    x2 = min(W_, x0i + radius + 1)
    y1 = max(0, y0i - radius)
    y2 = min(H, y0i + radius + 1)

    sub = Gamma[y1:y2, x1:x2]

    if sub.size == 0:
        xg, yg, sg = ridge_argmax(Gamma, x_vis_1d, y_vis_1d)
        return xg, yg, sg, x0i, y0i

    idx = int(np.argmax(sub))
    sy, sx = np.unravel_index(idx, sub.shape)

    iy = y1 + sy
    ix = x1 + sx

    return float(x_vis_1d[ix]), float(y_vis_1d[iy]), float(Gamma[iy, ix]), ix, iy


def compute_ridge_xy(
    frames_psi,
    Emix,
    x_vis_1d,
    y_vis_1d,
    mode="argmax",
    top_q=0.02,
    radius=20,
    alpha_smooth=0.0,
    snap_radius=12,
):
    """
    Compute ridge trajectory from amplitude-level forward/backward overlap.

    Modes
    -----
    "argmax"
    "centroid_top"
    "centroid_top_snap_localmax"
    "localmax_track"
    """
    _assert(isinstance(frames_psi, np.ndarray), "frames_psi must be np.ndarray")
    _assert(isinstance(Emix, np.ndarray), "Emix must be np.ndarray")
    _assert(frames_psi.shape[0] == Emix.shape[0],
            "frames_psi and Emix must have same number of time frames")
    _assert(alpha_smooth >= 0.0, f"alpha_smooth must be >= 0, got {alpha_smooth}")

    Nt_ = frames_psi.shape[0]

    ridge_x = np.zeros(Nt_, dtype=float)
    ridge_y = np.zeros(Nt_, dtype=float)
    ridge_s = np.zeros(Nt_, dtype=float)

    Gamma0 = local_gamma_field(frames_psi[0], Emix[0])
    x0r, y0r, s0 = ridge_argmax(Gamma0, x_vis_1d, y_vis_1d)

    ridge_x[0], ridge_y[0], ridge_s[0] = x0r, y0r, s0

    prev_ix = int(np.argmin(np.abs(x_vis_1d - x0r)))
    prev_iy = int(np.argmin(np.abs(y_vis_1d - y0r)))

    for i in range(1, Nt_):
        Gamma = local_gamma_field(frames_psi[i], Emix[i])

        if mode == "argmax":
            xi, yi, si = ridge_argmax(Gamma, x_vis_1d, y_vis_1d)

        elif mode == "centroid_top":
            xi, yi, si = ridge_centroid_top(
                Gamma, x_vis_1d, y_vis_1d, top_q=top_q
            )

        elif mode == "centroid_top_snap_localmax":
            xi, yi, si = ridge_centroid_top_snap_localmax(
                Gamma, x_vis_1d, y_vis_1d, top_q=top_q, snap_radius=snap_radius
            )

        elif mode == "localmax_track":
            xi, yi, si, prev_ix, prev_iy = ridge_localmax_track(
                Gamma, x_vis_1d, y_vis_1d, prev_ix, prev_iy, radius=radius
            )

        else:
            raise ValueError(f"Unknown RIDGE_MODE: {mode}")

        if alpha_smooth and alpha_smooth > 0.0:
            xi = (1.0 - alpha_smooth) * xi + alpha_smooth * ridge_x[i - 1]
            yi = (1.0 - alpha_smooth) * yi + alpha_smooth * ridge_y[i - 1]

        ridge_x[i], ridge_y[i], ridge_s[i] = xi, yi, si

        if mode != "localmax_track":
            prev_ix = int(np.argmin(np.abs(x_vis_1d - xi)))
            prev_iy = int(np.argmin(np.abs(y_vis_1d - yi)))

    return ridge_x, ridge_y, ridge_s