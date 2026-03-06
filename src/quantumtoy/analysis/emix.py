from __future__ import annotations

import numpy as np


def gaussian_weights(Tk: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if sigma <= 0:
        w = np.zeros_like(Tk, dtype=float)
        if Tk.size > 0:
            w[int(np.argmin(np.abs(Tk - mu)))] = 1.0
        return w

    z = (Tk - mu) / sigma
    w = np.exp(-0.5 * z * z)
    s = float(np.sum(w))
    return w / s if s > 0 else w


def detect_click_from_screen(
    frames_density: np.ndarray,
    times: np.ndarray,
    screen_mask_vis: np.ndarray,
    dx: float,
    dy: float,
    X_vis: np.ndarray,
    Y_vis: np.ndarray,
    rng_seed: int = 123456,
):
    """
    Returns:
        idx_det, t_det, x_click, y_click, screen_int
    """
    if not np.any(screen_mask_vis):
        raise RuntimeError("screen_mask_vis is empty")

    screen_int = np.array(
        [np.sum(frames_density[i][screen_mask_vis]) * dx * dy for i in range(len(times))],
        dtype=float,
    )

    idx_det = int(np.argmax(screen_int))
    t_det = float(times[idx_det])

    w = frames_density[idx_det].copy()
    w = np.where(screen_mask_vis, w, 0.0)
    wsum = float(np.sum(w))
    if wsum <= 0:
        raise RuntimeError("No intensity on screen at detection time.")

    p = (w / wsum).ravel()
    rng = np.random.default_rng(rng_seed)
    flat_idx = int(rng.choice(p.size, p=p))
    iy_vis, ix_vis = np.unravel_index(flat_idx, w.shape)

    x_click = float(X_vis[iy_vis, ix_vis])
    y_click = float(Y_vis[iy_vis, ix_vis])
    return idx_det, t_det, x_click, y_click, screen_int


def build_backward_library(
    theory,
    grid,
    times: np.ndarray,
    tau_step: float,
    x_click: float,
    y_click: float,
    sigma_click: float,
    save_every: int,
    print_every_frames: int = 20,
) -> np.ndarray:
    """
    Build phi_tau_frames so that frame i corresponds to backward-evolved
    click-state density at tau = i * tau_step on the visible grid.
    """
    Nt = len(times)
    phi_cur = theory.initialize_click_state(x_click, y_click, sigma_click)
    phi_tau_frames = np.zeros((Nt, grid.n_visible_y, grid.n_visible_x), dtype=float)

    for i in range(Nt):
        phi_tau_frames[i] = theory.density(phi_cur)[grid.ys, grid.xs]

        if (i % print_every_frames) == 0 or i == Nt - 1:
            norm_phi = float(np.sum(theory.density(phi_cur)) * grid.dx * grid.dy)
            tau = i * tau_step
            print(f"[BWD] frame {i:4d}/{Nt-1}, tau={tau:7.3f}, norm≈{norm_phi:.6f}")

        if i < Nt - 1:
            dt_small = tau_step / save_every
            for _ in range(save_every):
                phi_cur = theory.step_backward_adjoint(phi_cur, dt_small).state

    return phi_tau_frames

def build_Emix_from_phi_tau(
    phi_tau_frames: np.ndarray,
    times: np.ndarray,
    t_det: float,
    sigmaT: float,
    tau_step: float,
    K_JITTER: int = 13,
) -> np.ndarray:
    Nt_ = len(times)
    halfK = K_JITTER // 2
    idx_det2 = int(np.argmin(np.abs(times - t_det)))

    k_inds = np.arange(idx_det2 - halfK, idx_det2 + halfK + 1)
    k_inds = np.clip(k_inds, 0, Nt_ - 1)
    k_inds = np.unique(k_inds)

    Tk = times[k_inds]
    w = gaussian_weights(Tk, t_det, sigmaT)

    Emix = np.zeros((Nt_, phi_tau_frames.shape[1], phi_tau_frames.shape[2]), dtype=float)

    for i, ti in enumerate(times):
        tau = Tk - ti
        valid = tau >= 0.0
        if not np.any(valid):
            continue

        j_float = tau[valid] / tau_step
        j0 = np.floor(j_float).astype(int)
        j1 = j0 + 1

        j0 = np.clip(j0, 0, Nt_ - 1)
        j1 = np.clip(j1, 0, Nt_ - 1)
        alpha = (j_float - j0).astype(float)

        phi_interp = (
            (1.0 - alpha)[:, None, None] * phi_tau_frames[j0]
            + alpha[:, None, None] * phi_tau_frames[j1]
        )
        Emix[i] = np.sum(w[valid][:, None, None] * phi_interp, axis=0)

    return Emix

def make_rho(frames_psi: np.ndarray, Emix: np.ndarray, dx: float, dy: float) -> np.ndarray:
    out = np.zeros_like(frames_psi, dtype=float)
    for i in range(frames_psi.shape[0]):
        rho = frames_psi[i] * Emix[i]
        s = float(np.sum(rho) * dx * dy)
        if s > 0:
            rho = rho / s
        out[i] = rho
    return out
