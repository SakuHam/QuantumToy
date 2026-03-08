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


def _crop_visible_state(state, grid):
    """
    Crop a full-grid state to visible window.

    Supports:
        - Schrödinger-like scalar state: shape (Ny, Nx)
        - Dirac-like spinor state:       shape (2, Ny, Nx)
    """
    if state.ndim == 2:
        return state[grid.ys, grid.xs].copy()
    if state.ndim == 3:
        return state[:, grid.ys, grid.xs].copy()

    raise ValueError(f"Unsupported state ndim={state.ndim}")


def _state_density(state_vis: np.ndarray) -> np.ndarray:
    """
    Convert one visible state to density.

    Supports:
        - Schrödinger-like scalar state: shape (Ny, Nx)
        - Dirac-like spinor state:       shape (2, Ny, Nx)
    """
    if state_vis.ndim == 2:
        return (np.abs(state_vis) ** 2).astype(float)

    if state_vis.ndim == 3:
        return np.sum(np.abs(state_vis) ** 2, axis=0).astype(float)

    raise ValueError(f"Unsupported state_vis ndim={state_vis.ndim}")


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
):
    """
    Build complex backward-evolved click-state library.

    Output:
        - Schrödinger: array shape (Nt, Ny_vis, Nx_vis), complex128
        - Dirac:       array shape (Nt, 2, Ny_vis, Nx_vis), complex128

    Frame i corresponds to backward-evolved click-state at:
        tau = i * tau_step
    """
    Nt = len(times)
    phi_cur = theory.initialize_click_state(x_click, y_click, sigma_click)

    phi0_vis = _crop_visible_state(phi_cur, grid)

    if phi0_vis.ndim == 2:
        phi_tau_frames = np.zeros(
            (Nt, grid.n_visible_y, grid.n_visible_x),
            dtype=np.complex128,
        )
    elif phi0_vis.ndim == 3:
        ncomp = phi0_vis.shape[0]
        phi_tau_frames = np.zeros(
            (Nt, ncomp, grid.n_visible_y, grid.n_visible_x),
            dtype=np.complex128,
        )
    else:
        raise ValueError(f"Unsupported visible backward state ndim={phi0_vis.ndim}")

    for i in range(Nt):
        phi_tau_frames[i] = _crop_visible_state(phi_cur, grid)

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
):
    """
    Build amplitude-level mixed backward field.

    Returns:
        - Schrödinger: complex array (Nt, Ny, Nx)
        - Dirac:       complex array (Nt, 2, Ny, Nx)
    """
    Nt_ = len(times)
    halfK = K_JITTER // 2
    idx_det2 = int(np.argmin(np.abs(times - t_det)))

    k_inds = np.arange(idx_det2 - halfK, idx_det2 + halfK + 1)
    k_inds = np.clip(k_inds, 0, Nt_ - 1)
    k_inds = np.unique(k_inds)

    Tk = times[k_inds]
    w_full = gaussian_weights(Tk, t_det, sigmaT)

    Emix = np.zeros_like(phi_tau_frames, dtype=np.complex128)

    for i, ti in enumerate(times):
        tau = Tk - ti
        valid = tau >= 0.0
        if not np.any(valid):
            continue

        wv = w_full[valid].astype(float)
        sw = float(np.sum(wv))
        if sw <= 0.0:
            continue
        wv = wv / sw

        j_float = tau[valid] / tau_step
        j0 = np.floor(j_float).astype(int)
        j1 = j0 + 1

        j0 = np.clip(j0, 0, Nt_ - 1)
        j1 = np.clip(j1, 0, Nt_ - 1)
        alpha = (j_float - j0).astype(float)

        if phi_tau_frames.ndim == 3:
            phi_interp = (
                (1.0 - alpha)[:, None, None] * phi_tau_frames[j0]
                + alpha[:, None, None] * phi_tau_frames[j1]
            )
            Emix[i] = np.sum(wv[:, None, None] * phi_interp, axis=0)

        elif phi_tau_frames.ndim == 4:
            phi_interp = (
                (1.0 - alpha)[:, None, None, None] * phi_tau_frames[j0]
                + alpha[:, None, None, None] * phi_tau_frames[j1]
            )
            Emix[i] = np.sum(wv[:, None, None, None] * phi_interp, axis=0)

        else:
            raise ValueError(f"Unsupported phi_tau_frames ndim={phi_tau_frames.ndim}")

    return Emix


def build_Emix_density_from_phi_tau(
    phi_tau_frames: np.ndarray,
    times: np.ndarray,
    t_det: float,
    sigmaT: float,
    tau_step: float,
    K_JITTER: int = 13,
) -> np.ndarray:
    """
    Build OLD-STYLE density-level mixed backward field.

    This matches the old logic much better:

        Emix_density(t) = sum_k w_k * rho_phi(tau_k - t)

    where rho_phi is density of each backward frame BEFORE mixing.

    Returns:
        real array shape (Nt, Ny, Nx)
    """
    Nt_ = len(times)
    halfK = K_JITTER // 2
    idx_det2 = int(np.argmin(np.abs(times - t_det)))

    k_inds = np.arange(idx_det2 - halfK, idx_det2 + halfK + 1)
    k_inds = np.clip(k_inds, 0, Nt_ - 1)
    k_inds = np.unique(k_inds)

    Tk = times[k_inds]
    w_full = gaussian_weights(Tk, t_det, sigmaT)

    # Convert full backward library to density first
    if phi_tau_frames.ndim == 3:
        # Schrödinger
        phi_tau_density = (np.abs(phi_tau_frames) ** 2).astype(float)
    elif phi_tau_frames.ndim == 4:
        # Dirac
        phi_tau_density = np.sum(np.abs(phi_tau_frames) ** 2, axis=1).astype(float)
    else:
        raise ValueError(f"Unsupported phi_tau_frames ndim={phi_tau_frames.ndim}")

    Emix_density = np.zeros_like(phi_tau_density, dtype=float)

    for i, ti in enumerate(times):
        tau = Tk - ti
        valid = tau >= 0.0
        if not np.any(valid):
            continue

        wv = w_full[valid].astype(float)
        sw = float(np.sum(wv))
        if sw <= 0.0:
            continue
        wv = wv / sw

        j_float = tau[valid] / tau_step
        j0 = np.floor(j_float).astype(int)
        j1 = j0 + 1

        j0 = np.clip(j0, 0, Nt_ - 1)
        j1 = np.clip(j1, 0, Nt_ - 1)
        alpha = (j_float - j0).astype(float)

        rho_interp = (
            (1.0 - alpha)[:, None, None] * phi_tau_density[j0]
            + alpha[:, None, None] * phi_tau_density[j1]
        )

        Emix_density[i] = np.sum(wv[:, None, None] * rho_interp, axis=0)

    return Emix_density


def local_amplitude_overlap(frames_psi, Emix: np.ndarray) -> np.ndarray:
    """
    Local amplitude overlap field between forward state and mixed backward state.

    For Schrödinger:
        A(x,y,t) = conj(Emix) * psi

    For Dirac:
        A(x,y,t) = sum_a conj(Emix_a) * psi_a

    Returns:
        complex array shape (Nt, Ny, Nx)
    """
    if frames_psi.shape[0] != Emix.shape[0]:
        raise ValueError("frames_psi and Emix must have same number of time frames")

    if frames_psi.ndim == 3 and Emix.ndim == 3:
        return np.conjugate(Emix) * frames_psi

    if frames_psi.ndim == 4 and Emix.ndim == 4:
        return np.sum(np.conjugate(Emix) * frames_psi, axis=1)

    raise ValueError(
        f"Incompatible shapes: frames_psi.ndim={frames_psi.ndim}, Emix.ndim={Emix.ndim}"
    )


def make_emix_density(Emix: np.ndarray) -> np.ndarray:
    """
    Density of amplitude-level Emix.

    Returns:
        - Schrödinger: |Emix|^2
        - Dirac:       sum_a |Emix_a|^2
    """
    if Emix.ndim == 3:
        return (np.abs(Emix) ** 2).astype(float)

    if Emix.ndim == 4:
        return np.sum(np.abs(Emix) ** 2, axis=1).astype(float)

    raise ValueError(f"Unsupported Emix ndim={Emix.ndim}")


def _forward_density_from_frames(frames_psi) -> np.ndarray:
    """
    Convert forward complex frames to density.

    Supports:
        - Schrödinger: (Nt, Ny, Nx)
        - Dirac:       (Nt, 2, Ny, Nx)
    """
    if frames_psi.ndim == 3:
        return (np.abs(frames_psi) ** 2).astype(float)

    if frames_psi.ndim == 4:
        return np.sum(np.abs(frames_psi) ** 2, axis=1).astype(float)

    raise ValueError(f"Unsupported frames_psi ndim={frames_psi.ndim}")


def _normalize_framewise(rho: np.ndarray, dx: float, dy: float) -> np.ndarray:
    out = np.zeros_like(rho, dtype=float)
    for i in range(rho.shape[0]):
        ri = rho[i].astype(float)
        s = float(np.sum(ri) * dx * dy)
        if s > 0:
            ri = ri / s
        out[i] = ri
    return out


def make_rho_amplitude_overlap(frames_psi, Emix: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    New amplitude-level visualization:
        A(x,y,t) = <phi_mix(x,y,t) | psi(x,y,t)>_local
        rho_out  = |A|^2
    """
    A = local_amplitude_overlap(frames_psi, Emix)
    rho = (np.abs(A) ** 2).astype(float)
    return _normalize_framewise(rho, dx, dy)


def make_rho_density_product(frames_psi, Emix: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Density product using density(amplitude-mix):
        rho_out ~ rho_forward * rho_Emix_amplitude_mix
    """
    rho_fwd = _forward_density_from_frames(frames_psi)
    rho_emix = make_emix_density(Emix)
    rho = (rho_fwd * rho_emix).astype(float)
    return _normalize_framewise(rho, dx, dy)


def make_rho_density_product_oldstyle(
    frames_psi,
    Emix_density: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    OLD-STYLE density product:
        rho_out ~ rho_forward * Emix_density

    where Emix_density is built by mixing backward densities directly,
    not by taking density of mixed amplitudes.
    """
    rho_fwd = _forward_density_from_frames(frames_psi)

    if Emix_density.ndim != 3:
        raise ValueError(f"Emix_density must have ndim=3, got {Emix_density.ndim}")

    rho = (rho_fwd * Emix_density).astype(float)
    return _normalize_framewise(rho, dx, dy)


def make_rho_blended(
    frames_psi,
    Emix: np.ndarray,
    dx: float,
    dy: float,
    blend_alpha: float = 0.5,
) -> np.ndarray:
    """
    Hybrid mode:
        rho = (1-alpha) * density_product + alpha * amplitude_overlap

    alpha=0   -> pure density_product
    alpha=1   -> pure amplitude_overlap
    """
    a = float(np.clip(blend_alpha, 0.0, 1.0))

    rho_dp = make_rho_density_product(frames_psi, Emix, dx, dy)
    rho_ao = make_rho_amplitude_overlap(frames_psi, Emix, dx, dy)

    rho = (1.0 - a) * rho_dp + a * rho_ao
    return _normalize_framewise(rho, dx, dy)


def make_rho(
    frames_psi,
    Emix: np.ndarray | None,
    dx: float,
    dy: float,
    mode: str = "amplitude_overlap",
    blend_alpha: float = 0.5,
    Emix_density: np.ndarray | None = None,
) -> np.ndarray:
    """
    Hybrid visualization builder.

    Modes
    -----
    "amplitude_overlap"
        rho = |<phi_mix|psi>|^2

    "density_product"
        rho = rho_forward * density(Emix_amplitude_mix)

    "density_product_oldstyle"
        rho = rho_forward * Emix_density
        (this matches the old code much better)

    "blended"
        rho = (1-a)*density_product + a*amplitude_overlap
    """
    if mode == "amplitude_overlap":
        if Emix is None:
            raise ValueError("Emix is required for mode='amplitude_overlap'")
        return make_rho_amplitude_overlap(frames_psi, Emix, dx, dy)

    if mode == "density_product":
        if Emix is None:
            raise ValueError("Emix is required for mode='density_product'")
        return make_rho_density_product(frames_psi, Emix, dx, dy)

    if mode == "density_product_oldstyle":
        if Emix_density is None:
            raise ValueError("Emix_density is required for mode='density_product_oldstyle'")
        return make_rho_density_product_oldstyle(frames_psi, Emix_density, dx, dy)

    if mode == "blended":
        if Emix is None:
            raise ValueError("Emix is required for mode='blended'")
        return make_rho_blended(frames_psi, Emix, dx, dy, blend_alpha=blend_alpha)

    raise ValueError(f"Unknown make_rho mode: {mode}")