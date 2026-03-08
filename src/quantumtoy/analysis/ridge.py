from __future__ import annotations

import numpy as np


def local_gamma_field(frame_psi, frame_emix):
    """
    Build positive scalar ridge field Gamma from local amplitude overlap.

    Supported inputs
    ----------------
    Schrödinger:
        frame_psi  shape (Ny, Nx), complex
        frame_emix shape (Ny, Nx), complex

        A = conj(emix) * psi
        Gamma = |A|^2

    Dirac:
        frame_psi  shape (2, Ny, Nx), complex
        frame_emix shape (2, Ny, Nx), complex

        A = sum_a conj(emix_a) * psi_a
        Gamma = |A|^2

    Returns
    -------
    Gamma : np.ndarray, shape (Ny, Nx), float
    """
    if frame_psi.ndim == 2 and frame_emix.ndim == 2:
        A = np.conjugate(frame_emix) * frame_psi
        return (np.abs(A) ** 2).astype(float)

    if frame_psi.ndim == 3 and frame_emix.ndim == 3:
        A = np.sum(np.conjugate(frame_emix) * frame_psi, axis=0)
        return (np.abs(A) ** 2).astype(float)

    raise ValueError(
        f"Incompatible frame shapes: frame_psi.ndim={frame_psi.ndim}, "
        f"frame_emix.ndim={frame_emix.ndim}"
    )


def ridge_argmax(Gamma, x_vis_1d, y_vis_1d):
    idx = int(np.argmax(Gamma))
    iy, ix = np.unravel_index(idx, Gamma.shape)
    return float(x_vis_1d[ix]), float(y_vis_1d[iy]), float(Gamma[iy, ix])


def ridge_centroid_top(Gamma, x_vis_1d, y_vis_1d, top_q=0.02, eps=1e-30):
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


def ridge_localmax_track(Gamma, x_vis_1d, y_vis_1d, prev_ix, prev_iy, radius=20):
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
):
    """
    Compute ridge trajectory from amplitude-level forward/backward overlap.

    Parameters
    ----------
    frames_psi:
        Schrödinger: shape (Nt, Ny, Nx), complex
        Dirac:       shape (Nt, 2, Ny, Nx), complex

    Emix:
        Same time-grid and visible-grid shape as frames_psi.

    Returns
    -------
    ridge_x, ridge_y, ridge_s
        where ridge_s is the local Gamma = |A|^2 score at the chosen ridge.
    """
    if frames_psi.shape[0] != Emix.shape[0]:
        raise ValueError("frames_psi and Emix must have same number of time frames")

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