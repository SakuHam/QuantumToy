from __future__ import annotations

import numpy as np


# ============================================================
# Validation helpers
# ============================================================

def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def _assert_finite_scalar(x, name: str):
    _assert(np.isscalar(x), f"{name} must be a scalar, got type={type(x)}")
    xf = float(x)
    _assert(np.isfinite(xf), f"{name} must be finite, got {x}")


def _assert_positive_scalar(x, name: str):
    _assert_finite_scalar(x, name)
    _assert(float(x) > 0.0, f"{name} must be > 0, got {x}")


def _assert_1d_finite(arr: np.ndarray, name: str):
    _assert(isinstance(arr, np.ndarray), f"{name} must be np.ndarray")
    _assert(arr.ndim == 1, f"{name} must be 1D, got ndim={arr.ndim}")
    _assert(arr.size > 0, f"{name} must be non-empty")
    _assert(np.all(np.isfinite(arr)), f"{name} contains non-finite values")


def _assert_times(times: np.ndarray):
    _assert_1d_finite(times, "times")
    if times.size >= 2:
        dt = np.diff(times)
        _assert(np.all(dt > 0.0), "times must be strictly increasing")
        _assert(np.all(np.isfinite(dt)), "times diffs must be finite")


def _assert_visible_shapes(
    frames_density: np.ndarray,
    screen_mask_vis: np.ndarray,
    X_vis: np.ndarray,
    Y_vis: np.ndarray,
):
    _assert(isinstance(frames_density, np.ndarray), "frames_density must be np.ndarray")
    _assert(frames_density.ndim == 3, f"frames_density must have ndim=3, got {frames_density.ndim}")
    Nt, Ny, Nx = frames_density.shape

    _assert(
        screen_mask_vis.shape == (Ny, Nx),
        f"screen_mask_vis shape {screen_mask_vis.shape} must match visible frame shape {(Ny, Nx)}",
    )
    _assert(
        X_vis.shape == (Ny, Nx),
        f"X_vis shape {X_vis.shape} must match visible frame shape {(Ny, Nx)}",
    )
    _assert(
        Y_vis.shape == (Ny, Nx),
        f"Y_vis shape {Y_vis.shape} must match visible frame shape {(Ny, Nx)}",
    )

    _assert(
        np.issubdtype(screen_mask_vis.dtype, np.bool_),
        f"screen_mask_vis must be boolean, got dtype={screen_mask_vis.dtype}",
    )
    _assert(np.all(np.isfinite(frames_density)), "frames_density contains non-finite values")
    _assert(np.all(frames_density >= -1e-14), "frames_density contains significantly negative values")
    _assert(np.all(np.isfinite(X_vis)), "X_vis contains non-finite values")
    _assert(np.all(np.isfinite(Y_vis)), "Y_vis contains non-finite values")
    _assert(Nt > 0 and Ny > 0 and Nx > 0, "frames_density must be non-empty in all dimensions")


def _grid_shape(grid) -> tuple[int, int]:
    """
    Return full-grid shape as (Ny, Nx).

    Supports both grid.Ny/grid.Nx and grid.y/grid.x style grids.
    """
    if hasattr(grid, "Ny") and hasattr(grid, "Nx"):
        return int(grid.Ny), int(grid.Nx)

    if hasattr(grid, "y") and hasattr(grid, "x"):
        return int(np.asarray(grid.y).size), int(np.asarray(grid.x).size)

    raise AttributeError("grid must expose either Ny/Nx or y/x")


def _visible_shape(grid) -> tuple[int, int]:
    """
    Return visible-grid shape as (Ny_vis, Nx_vis).
    """
    return int(grid.n_visible_y), int(grid.n_visible_x)


def _assert_state_shape_matches_grid(state: np.ndarray, grid, name: str):
    """
    Validate state shape against full grid.

    Supported full-grid states:
      - scalar Schrödinger:      (Ny, Nx)
      - Dirac-like spinor:       (C, Ny, Nx)
      - entangled spinor:        (Ny, Nx, 2, 2)
    """
    _assert(isinstance(state, np.ndarray), f"{name} must be np.ndarray")

    Ny, Nx = _grid_shape(grid)

    if state.ndim == 2:
        _assert(
            state.shape == (Ny, Nx),
            f"{name} scalar state shape {state.shape} != full-grid shape {(Ny, Nx)}",
        )
        return

    if state.ndim == 3:
        _assert(
            state.shape[1:] == (Ny, Nx),
            f"{name} spinor state spatial shape {state.shape[1:]} != full-grid shape {(Ny, Nx)}",
        )
        _assert(state.shape[0] >= 1, f"{name} must have at least 1 component")
        return

    if state.ndim == 4:
        _assert(
            state.shape == (Ny, Nx, 2, 2),
            f"{name} entangled spinor shape {state.shape} != full-grid shape {(Ny, Nx, 2, 2)}",
        )
        return

    raise AssertionError(
        f"{name} ndim must be 2, 3, or 4, got ndim={state.ndim}, shape={state.shape}"
    )


def _assert_phi_tau_shape(phi_tau_frames: np.ndarray, Nt: int):
    """
    Validate backward-library frame shape.

    Supported:
      - scalar:      (Nt, Ny_vis, Nx_vis)
      - Dirac-like:  (Nt, C, Ny_vis, Nx_vis)
      - entangled:   (Nt, Ny_vis, Nx_vis, 2, 2)
    """
    _assert(isinstance(phi_tau_frames, np.ndarray), "phi_tau_frames must be np.ndarray")
    _assert(
        phi_tau_frames.ndim in (3, 4, 5),
        f"phi_tau_frames ndim must be 3, 4, or 5, got {phi_tau_frames.ndim}",
    )
    _assert(
        phi_tau_frames.shape[0] == Nt,
        f"phi_tau_frames first dim {phi_tau_frames.shape[0]} must equal Nt={Nt}",
    )

    if phi_tau_frames.ndim == 5:
        _assert(
            phi_tau_frames.shape[-2:] == (2, 2),
            f"5D phi_tau_frames must end with spin axes (2,2), got {phi_tau_frames.shape}",
        )

    _assert(np.all(np.isfinite(phi_tau_frames.real)), "phi_tau_frames.real contains non-finite values")
    _assert(np.all(np.isfinite(phi_tau_frames.imag)), "phi_tau_frames.imag contains non-finite values")


def _assert_frames_psi_emix_compatible(frames_psi: np.ndarray, Emix: np.ndarray):
    """
    Validate forward frames and Emix compatibility.

    Supported:
      - scalar:      frames_psi, Emix shape (Nt, Ny, Nx)
      - Dirac-like:  frames_psi, Emix shape (Nt, C, Ny, Nx)
      - entangled:   frames_psi, Emix shape (Nt, Ny, Nx, 2, 2)
    """
    _assert(isinstance(frames_psi, np.ndarray), "frames_psi must be np.ndarray")
    _assert(isinstance(Emix, np.ndarray), "Emix must be np.ndarray")
    _assert(
        frames_psi.shape[0] == Emix.shape[0],
        "frames_psi and Emix must have same number of time frames",
    )

    if frames_psi.ndim == 3 and Emix.ndim == 3:
        _assert(
            frames_psi.shape == Emix.shape,
            f"Schrödinger shapes must match, got {frames_psi.shape} vs {Emix.shape}",
        )
        return

    if frames_psi.ndim == 4 and Emix.ndim == 4:
        _assert(
            frames_psi.shape == Emix.shape,
            f"Dirac-like shapes must match, got {frames_psi.shape} vs {Emix.shape}",
        )
        return

    if frames_psi.ndim == 5 and Emix.ndim == 5:
        _assert(
            frames_psi.shape == Emix.shape,
            f"Entangled spinor shapes must match, got {frames_psi.shape} vs {Emix.shape}",
        )
        _assert(
            frames_psi.shape[-2:] == (2, 2),
            f"Entangled frames_psi must end with (2,2), got {frames_psi.shape}",
        )
        return

    raise ValueError(
        f"Incompatible shapes: frames_psi.ndim={frames_psi.ndim}, Emix.ndim={Emix.ndim}; "
        f"frames_psi.shape={frames_psi.shape}, Emix.shape={Emix.shape}"
    )


def _assert_nonnegative_density_cube(arr: np.ndarray, name: str):
    _assert(isinstance(arr, np.ndarray), f"{name} must be np.ndarray")
    _assert(arr.ndim == 3, f"{name} must have ndim=3, got {arr.ndim}")
    _assert(np.all(np.isfinite(arr)), f"{name} contains non-finite values")
    _assert(np.all(arr >= -1e-14), f"{name} contains significantly negative values")


# ============================================================
# Generic state density helpers
# ============================================================

def _state_density(state_vis: np.ndarray) -> np.ndarray:
    """
    Convert a visible state to scalar 2D density.

    Supported visible states:
      - scalar:      (Ny, Nx)
      - Dirac-like:  (C, Ny, Nx)
      - entangled:   (Ny, Nx, 2, 2)
    """
    _assert(isinstance(state_vis, np.ndarray), "state_vis must be np.ndarray")

    if state_vis.ndim == 2:
        rho = (np.abs(state_vis) ** 2).astype(float)

    elif state_vis.ndim == 3:
        rho = np.sum(np.abs(state_vis) ** 2, axis=0).astype(float)

    elif state_vis.ndim == 4 and state_vis.shape[-2:] == (2, 2):
        rho = np.sum(np.abs(state_vis) ** 2, axis=(-2, -1)).astype(float)

    else:
        raise AssertionError(f"Unsupported state_vis shape for density: {state_vis.shape}")

    _assert(np.all(np.isfinite(rho)), "_state_density produced non-finite values")
    _assert(np.all(rho >= -1e-14), "_state_density produced significantly negative density")
    return rho


def _frames_density_from_complex_frames(frames: np.ndarray) -> np.ndarray:
    """
    Convert complex frames to scalar density frames.

    Supported:
      - scalar:      (Nt, Ny, Nx)
      - Dirac-like:  (Nt, C, Ny, Nx)
      - entangled:   (Nt, Ny, Nx, 2, 2)
    """
    _assert(isinstance(frames, np.ndarray), "frames must be np.ndarray")

    if frames.ndim == 3:
        rho = (np.abs(frames) ** 2).astype(float)

    elif frames.ndim == 4:
        rho = np.sum(np.abs(frames) ** 2, axis=1).astype(float)

    elif frames.ndim == 5 and frames.shape[-2:] == (2, 2):
        rho = np.sum(np.abs(frames) ** 2, axis=(-2, -1)).astype(float)

    else:
        raise AssertionError(f"Unsupported complex frame shape for density: {frames.shape}")

    _assert_nonnegative_density_cube(rho, "_frames_density_from_complex_frames output")
    return rho


# ============================================================
# Gaussian weights
# ============================================================

def gaussian_weights(Tk: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    _assert_1d_finite(Tk, "Tk")
    _assert_finite_scalar(mu, "mu")
    _assert_finite_scalar(sigma, "sigma")

    if sigma <= 0:
        w = np.zeros_like(Tk, dtype=float)
        if Tk.size > 0:
            w[int(np.argmin(np.abs(Tk - mu)))] = 1.0
        _assert(np.isclose(np.sum(w), 1.0), "delta-like gaussian_weights must sum to 1")
        return w

    z = (Tk - mu) / sigma
    w = np.exp(-0.5 * z * z)
    s = float(np.sum(w))

    _assert(np.isfinite(s), "gaussian_weights normalization sum is non-finite")
    if s <= 0.0:
        return np.zeros_like(Tk, dtype=float)

    w = w / s
    _assert(np.all(np.isfinite(w)), "gaussian_weights produced non-finite values")
    _assert(np.all(w >= 0.0), "gaussian_weights produced negative weights")
    _assert(
        np.isclose(np.sum(w), 1.0, atol=1e-12),
        f"gaussian_weights must sum to 1, got {np.sum(w)}",
    )
    return w


# ============================================================
# Click detection helpers
# ============================================================

def _nearest_detector_cell_to_point(
    screen_mask_vis: np.ndarray,
    X_vis: np.ndarray,
    Y_vis: np.ndarray,
    x_target: float,
    y_target: float,
):
    """
    Return nearest visible detector cell to a requested forced click point.

    Returns:
        iy_vis, ix_vis, x_click, y_click
    """
    _assert(np.any(screen_mask_vis), "screen_mask_vis is empty")
    _assert_finite_scalar(x_target, "force_click_x")
    _assert_finite_scalar(y_target, "force_click_y")

    iy_idx, ix_idx = np.where(screen_mask_vis)
    _assert(iy_idx.size > 0, "No detector cells found in screen_mask_vis")

    xs = X_vis[iy_idx, ix_idx]
    ys = Y_vis[iy_idx, ix_idx]

    d2 = (xs - float(x_target)) ** 2 + (ys - float(y_target)) ** 2
    _assert(np.all(np.isfinite(d2)), "forced click distance array contains non-finite values")

    j = int(np.argmin(d2))
    iy_vis = int(iy_idx[j])
    ix_vis = int(ix_idx[j])

    x_click = float(X_vis[iy_vis, ix_vis])
    y_click = float(Y_vis[iy_vis, ix_vis])

    _assert(
        screen_mask_vis[iy_vis, ix_vis],
        "nearest forced click cell lies outside screen_mask_vis",
    )

    return iy_vis, ix_vis, x_click, y_click


# ============================================================
# Click detection
# ============================================================

def detect_click_from_screen(
    frames_density: np.ndarray,
    times: np.ndarray,
    screen_mask_vis: np.ndarray,
    dx: float,
    dy: float,
    X_vis: np.ndarray,
    Y_vis: np.ndarray,
    rng_seed: int = 123456,
    click_mode: str = "born",
    force_click_x: float | None = None,
    force_click_y: float | None = None,
):
    """
    Returns:
        idx_det, t_det, x_click, y_click, screen_int

    click_mode
    ----------
    "born"
        Sample click position from detector Born weights at detection time.

    "forced_point"
        Keep detection time from detector argmax, but force click position
        to the nearest detector cell to (force_click_x, force_click_y).
    """
    _assert_times(times)
    _assert_positive_scalar(dx, "dx")
    _assert_positive_scalar(dy, "dy")
    _assert_visible_shapes(frames_density, screen_mask_vis, X_vis, Y_vis)
    _assert(
        frames_density.shape[0] == len(times),
        f"frames_density Nt={frames_density.shape[0]} must equal len(times)={len(times)}",
    )
    _assert(np.any(screen_mask_vis), "screen_mask_vis is empty")
    _assert(
        click_mode in {"born", "forced_point"},
        f"click_mode must be 'born' or 'forced_point', got {click_mode!r}",
    )

    if click_mode == "forced_point":
        _assert(force_click_x is not None, "force_click_x is required for click_mode='forced_point'")
        _assert(force_click_y is not None, "force_click_y is required for click_mode='forced_point'")
        _assert_finite_scalar(force_click_x, "force_click_x")
        _assert_finite_scalar(force_click_y, "force_click_y")

    screen_int = np.array(
        [np.sum(frames_density[i][screen_mask_vis]) * dx * dy for i in range(len(times))],
        dtype=float,
    )

    _assert(np.all(np.isfinite(screen_int)), "screen_int contains non-finite values")
    _assert(np.all(screen_int >= -1e-14), "screen_int contains significantly negative values")

    idx_det = int(np.argmax(screen_int))
    _assert(0 <= idx_det < len(times), f"idx_det out of range: {idx_det}")

    t_det = float(times[idx_det])

    w = frames_density[idx_det].copy()
    _assert(
        w.shape == screen_mask_vis.shape,
        "detection frame and screen_mask_vis shape mismatch",
    )

    w = np.where(screen_mask_vis, w, 0.0)
    w = np.clip(w, 0.0, None)

    wsum = float(np.sum(w))
    _assert(np.isfinite(wsum), "wsum is non-finite")
    _assert(wsum > 0.0, "No positive intensity on screen at detection time")

    if click_mode == "born":
        p = (w / wsum).ravel()
        _assert(np.all(np.isfinite(p)), "click probability vector contains non-finite values")
        _assert(np.all(p >= 0.0), "click probability vector contains negative values")
        _assert(
            np.isclose(np.sum(p), 1.0, atol=1e-12),
            f"click probability vector must sum to 1, got {np.sum(p)}",
        )

        rng = np.random.default_rng(rng_seed)
        flat_idx = int(rng.choice(p.size, p=p))
        iy_vis, ix_vis = np.unravel_index(flat_idx, w.shape)

        _assert(
            screen_mask_vis[iy_vis, ix_vis],
            "chosen click point lies outside screen_mask_vis",
        )

        x_click = float(X_vis[iy_vis, ix_vis])
        y_click = float(Y_vis[iy_vis, ix_vis])

    elif click_mode == "forced_point":
        iy_vis, ix_vis, x_click, y_click = _nearest_detector_cell_to_point(
            screen_mask_vis=screen_mask_vis,
            X_vis=X_vis,
            Y_vis=Y_vis,
            x_target=float(force_click_x),
            y_target=float(force_click_y),
        )

    else:
        raise ValueError(f"Unsupported click_mode: {click_mode}")

    _assert(
        np.isfinite(x_click) and np.isfinite(y_click),
        f"click coordinates must be finite, got ({x_click}, {y_click})",
    )

    return idx_det, t_det, x_click, y_click, screen_int


# ============================================================
# State crop / density helpers
# ============================================================

def _crop_visible_state(state, grid):
    """
    Crop a full-grid state to visible window.

    Supports:
        - Schrödinger-like scalar state: shape (Ny, Nx)
        - Dirac-like spinor state:       shape (C, Ny, Nx)
        - entangled spinor state:        shape (Ny, Nx, 2, 2)
    """
    _assert_state_shape_matches_grid(state, grid, "state")

    Ny_vis, Nx_vis = _visible_shape(grid)

    if state.ndim == 2:
        out = state[grid.ys, grid.xs].copy()
        _assert(
            out.shape == (Ny_vis, Nx_vis),
            f"cropped scalar visible shape mismatch: {out.shape}",
        )
        return out

    if state.ndim == 3:
        out = state[:, grid.ys, grid.xs].copy()
        _assert(
            out.shape[1:] == (Ny_vis, Nx_vis),
            f"cropped spinor visible shape mismatch: {out.shape}",
        )
        return out

    if state.ndim == 4:
        out = state[grid.ys, grid.xs, :, :].copy()
        _assert(
            out.shape == (Ny_vis, Nx_vis, 2, 2),
            f"cropped entangled visible shape mismatch: {out.shape}",
        )
        return out

    raise ValueError(f"Unsupported state ndim={state.ndim}")


# ============================================================
# Backward library
# ============================================================

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
        - Dirac-like:  array shape (Nt, C, Ny_vis, Nx_vis), complex128
        - Entangled:   array shape (Nt, Ny_vis, Nx_vis, 2, 2), complex128

    Frame i corresponds to backward-evolved click-state at:
        tau = i * tau_step
    """
    _assert_times(times)
    _assert_positive_scalar(tau_step, "tau_step")
    _assert_positive_scalar(sigma_click, "sigma_click")
    _assert(
        isinstance(save_every, int) and save_every >= 1,
        f"save_every must be int >= 1, got {save_every}",
    )
    _assert(
        isinstance(print_every_frames, int) and print_every_frames >= 1,
        f"print_every_frames must be int >= 1, got {print_every_frames}",
    )
    _assert_finite_scalar(x_click, "x_click")
    _assert_finite_scalar(y_click, "y_click")

    Nt = len(times)

    if Nt >= 2:
        dt_saved = float(times[1] - times[0])
        _assert(
            np.isclose(dt_saved, tau_step, rtol=1e-10, atol=1e-12),
            f"tau_step={tau_step} must match saved-frame spacing dt_saved={dt_saved}",
        )

    phi_cur = theory.initialize_click_state(x_click, y_click, sigma_click)
    _assert_state_shape_matches_grid(phi_cur, grid, "phi_cur(initialized click state)")

    phi0_vis = _crop_visible_state(phi_cur, grid)
    phi_tau_frames = np.zeros((Nt,) + phi0_vis.shape, dtype=np.complex128)

    dt_small = tau_step / save_every
    _assert_positive_scalar(dt_small, "dt_small")

    for i in range(Nt):
        phi_vis = _crop_visible_state(phi_cur, grid)
        _assert(
            phi_vis.shape == phi0_vis.shape,
            f"phi_vis shape changed at frame {i}: {phi_vis.shape} != {phi0_vis.shape}",
        )

        phi_tau_frames[i] = phi_vis

        _assert(np.all(np.isfinite(phi_vis.real)), f"phi_vis.real non-finite at frame {i}")
        _assert(np.all(np.isfinite(phi_vis.imag)), f"phi_vis.imag non-finite at frame {i}")

        if (i % print_every_frames) == 0 or i == Nt - 1:
            rho_full = theory.density(phi_cur)
            Ny, Nx = _grid_shape(grid)
            _assert(
                rho_full.shape == (Ny, Nx),
                f"theory.density(phi_cur) shape {rho_full.shape} != {(Ny, Nx)}",
            )
            _assert(np.all(np.isfinite(rho_full)), f"rho_full non-finite at backward frame {i}")
            _assert(np.all(rho_full >= -1e-14), f"rho_full negative at backward frame {i}")

            norm_phi = float(np.sum(rho_full) * grid.dx * grid.dy)
            tau = i * tau_step
            print(f"[BWD] frame {i:4d}/{Nt-1}, tau={tau:7.3f}, norm≈{norm_phi:.6f}")

        if i < Nt - 1:
            for sub in range(save_every):
                res = theory.step_backward_adjoint(phi_cur, dt_small)
                _assert(
                    hasattr(res, "state"),
                    "theory.step_backward_adjoint(...) must return object with .state",
                )
                phi_cur = res.state
                _assert_state_shape_matches_grid(
                    phi_cur,
                    grid,
                    f"phi_cur(after backward substep {sub} at frame {i})",
                )

    _assert_phi_tau_shape(phi_tau_frames, Nt)
    return phi_tau_frames


# ============================================================
# Emix builders
# ============================================================

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

    IMPORTANT:
    This intentionally follows the old working logic closely:
      - no re-normalization of valid weights
      - nearest-frame lookup with round(), not interpolation

    Returns:
        - Schrödinger: complex array (Nt, Ny, Nx)
        - Dirac-like:  complex array (Nt, C, Ny, Nx)
        - Entangled:   complex array (Nt, Ny, Nx, 2, 2)
    """
    _assert_times(times)
    _assert_finite_scalar(t_det, "t_det")
    _assert_finite_scalar(sigmaT, "sigmaT")
    _assert_positive_scalar(tau_step, "tau_step")
    _assert(
        isinstance(K_JITTER, int) and K_JITTER >= 1,
        f"K_JITTER must be int >= 1, got {K_JITTER}",
    )

    Nt_ = len(times)
    _assert_phi_tau_shape(phi_tau_frames, Nt_)

    halfK = K_JITTER // 2
    idx_det2 = int(np.argmin(np.abs(times - t_det)))
    _assert(0 <= idx_det2 < Nt_, "idx_det2 out of bounds")

    k_inds = np.arange(idx_det2 - halfK, idx_det2 + halfK + 1)
    k_inds = np.clip(k_inds, 0, Nt_ - 1)
    k_inds = np.unique(k_inds)

    _assert(k_inds.size >= 1, "k_inds must be non-empty")
    Tk = times[k_inds]
    _assert(
        np.all(Tk >= times[0]) and np.all(Tk <= times[-1]),
        "Tk values out of times range",
    )

    w_full = gaussian_weights(Tk, t_det, sigmaT)
    _assert(w_full.shape == Tk.shape, "w_full shape must match Tk")
    _assert(np.all(w_full >= 0.0), "w_full contains negative weights")

    Emix = np.zeros_like(phi_tau_frames, dtype=np.complex128)

    for i, ti in enumerate(times):
        tau = Tk - ti
        valid = tau >= 0.0

        if not np.any(valid):
            continue

        tau_valid = tau[valid]
        _assert(np.all(tau_valid >= 0.0), "tau_valid must be non-negative")

        wv = w_full[valid].astype(float)
        _assert(np.all(np.isfinite(wv)), "wv contains non-finite values")
        _assert(np.all(wv >= 0.0), "wv contains negative values")

        if not np.any(wv > 0.0):
            continue

        j = np.rint(tau_valid / tau_step).astype(int)
        j = np.clip(j, 0, Nt_ - 1)

        _assert(np.all((0 <= j) & (j < Nt_)), "j indices out of bounds")

        if phi_tau_frames.ndim == 3:
            Emix[i] = np.sum(wv[:, None, None] * phi_tau_frames[j], axis=0)

        elif phi_tau_frames.ndim == 4:
            Emix[i] = np.sum(wv[:, None, None, None] * phi_tau_frames[j], axis=0)

        elif phi_tau_frames.ndim == 5:
            Emix[i] = np.sum(wv[:, None, None, None, None] * phi_tau_frames[j], axis=0)

        else:
            raise ValueError(f"Unsupported phi_tau_frames ndim={phi_tau_frames.ndim}")

        _assert(np.all(np.isfinite(Emix[i].real)), f"Emix.real non-finite at frame {i}")
        _assert(np.all(np.isfinite(Emix[i].imag)), f"Emix.imag non-finite at frame {i}")

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

    IMPORTANT:
    This intentionally follows the old working logic closely:
      - no re-normalization of valid weights
      - nearest-frame lookup with round(), not interpolation

        Emix_density(t) = sum_k w_k * rho_phi(T_k - t)

    where rho_phi is density of each backward frame BEFORE mixing.

    Returns:
        real array shape (Nt, Ny, Nx)
    """
    _assert_times(times)
    _assert_finite_scalar(t_det, "t_det")
    _assert_finite_scalar(sigmaT, "sigmaT")
    _assert_positive_scalar(tau_step, "tau_step")
    _assert(
        isinstance(K_JITTER, int) and K_JITTER >= 1,
        f"K_JITTER must be int >= 1, got {K_JITTER}",
    )

    Nt_ = len(times)
    _assert_phi_tau_shape(phi_tau_frames, Nt_)

    halfK = K_JITTER // 2
    idx_det2 = int(np.argmin(np.abs(times - t_det)))
    _assert(0 <= idx_det2 < Nt_, "idx_det2 out of bounds")

    k_inds = np.arange(idx_det2 - halfK, idx_det2 + halfK + 1)
    k_inds = np.clip(k_inds, 0, Nt_ - 1)
    k_inds = np.unique(k_inds)

    _assert(k_inds.size >= 1, "k_inds must be non-empty")

    Tk = times[k_inds]
    w_full = gaussian_weights(Tk, t_det, sigmaT)

    phi_tau_density = _frames_density_from_complex_frames(phi_tau_frames)
    _assert_nonnegative_density_cube(phi_tau_density, "phi_tau_density")

    Emix_density = np.zeros_like(phi_tau_density, dtype=float)

    for i, ti in enumerate(times):
        tau = Tk - ti
        valid = tau >= 0.0

        if not np.any(valid):
            continue

        tau_valid = tau[valid]
        wv = w_full[valid].astype(float)

        _assert(np.all(tau_valid >= 0.0), "density tau_valid must be non-negative")
        _assert(np.all(np.isfinite(wv)), "density weights contain non-finite values")
        _assert(np.all(wv >= 0.0), "density weights contain negative values")

        if not np.any(wv > 0.0):
            continue

        j = np.rint(tau_valid / tau_step).astype(int)
        j = np.clip(j, 0, Nt_ - 1)

        _assert(np.all((0 <= j) & (j < Nt_)), "density j indices out of bounds")

        Emix_density[i] = np.sum(wv[:, None, None] * phi_tau_density[j], axis=0)

        _assert(np.all(np.isfinite(Emix_density[i])), f"Emix_density non-finite at frame {i}")
        _assert(np.all(Emix_density[i] >= -1e-14), f"Emix_density negative at frame {i}")

    return Emix_density


# ============================================================
# Overlap / densities
# ============================================================

def local_amplitude_overlap(frames_psi, Emix: np.ndarray) -> np.ndarray:
    """
    Local amplitude overlap field between forward state and mixed backward state.

    For Schrödinger:
        A(x,y,t) = conj(Emix) * psi

    For Dirac-like:
        A(x,y,t) = sum_a conj(Emix_a) * psi_a

    For entangled spinor:
        A(x,y,t) = sum_ab conj(Emix_ab) * psi_ab

    Returns:
        complex array shape (Nt, Ny, Nx)
    """
    _assert_frames_psi_emix_compatible(frames_psi, Emix)

    if frames_psi.ndim == 3 and Emix.ndim == 3:
        out = np.conjugate(Emix) * frames_psi

    elif frames_psi.ndim == 4 and Emix.ndim == 4:
        out = np.sum(np.conjugate(Emix) * frames_psi, axis=1)

    elif frames_psi.ndim == 5 and Emix.ndim == 5:
        out = np.sum(np.conjugate(Emix) * frames_psi, axis=(-2, -1))

    else:
        raise ValueError(
            f"Incompatible shapes: frames_psi.ndim={frames_psi.ndim}, Emix.ndim={Emix.ndim}"
        )

    _assert(out.ndim == 3, f"local_amplitude_overlap output must be ndim=3, got {out.ndim}")
    _assert(np.all(np.isfinite(out.real)), "local_amplitude_overlap.real non-finite")
    _assert(np.all(np.isfinite(out.imag)), "local_amplitude_overlap.imag non-finite")
    return out


def make_emix_density(Emix: np.ndarray) -> np.ndarray:
    """
    Density of amplitude-level Emix.

    Returns:
        - Schrödinger: |Emix|^2
        - Dirac-like:  sum_a |Emix_a|^2
        - Entangled:   sum_ab |Emix_ab|^2
    """
    _assert(isinstance(Emix, np.ndarray), "Emix must be np.ndarray")
    _assert(Emix.ndim in (3, 4, 5), f"Unsupported Emix ndim={Emix.ndim}")

    rho = _frames_density_from_complex_frames(Emix)

    _assert_nonnegative_density_cube(rho, "make_emix_density output")
    return rho


def _forward_density_from_frames(frames_psi) -> np.ndarray:
    """
    Convert forward complex frames to density.

    Supports:
        - Schrödinger: (Nt, Ny, Nx)
        - Dirac-like:  (Nt, C, Ny, Nx)
        - Entangled:   (Nt, Ny, Nx, 2, 2)
    """
    rho = _frames_density_from_complex_frames(frames_psi)
    _assert_nonnegative_density_cube(rho, "_forward_density_from_frames output")
    return rho


def _normalize_framewise(rho: np.ndarray, dx: float, dy: float) -> np.ndarray:
    _assert_nonnegative_density_cube(rho, "rho before normalization")
    _assert_positive_scalar(dx, "dx")
    _assert_positive_scalar(dy, "dy")

    out = np.zeros_like(rho, dtype=float)

    for i in range(rho.shape[0]):
        ri = rho[i].astype(float)
        s = float(np.sum(ri) * dx * dy)

        _assert(np.isfinite(s), f"frame normalization sum non-finite at i={i}")
        if s > 0:
            ri = ri / s
            s2 = float(np.sum(ri) * dx * dy)
            _assert(np.isfinite(s2), f"normalized frame sum non-finite at i={i}")
            _assert(
                np.isclose(s2, 1.0, atol=1e-10),
                f"normalized frame integral must be 1 at i={i}, got {s2}",
            )

        out[i] = ri

    _assert_nonnegative_density_cube(out, "normalized rho")
    return out


# ============================================================
# rho builders
# ============================================================

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

    _assert(
        rho_fwd.shape == rho_emix.shape,
        f"rho_fwd shape {rho_fwd.shape} != rho_emix shape {rho_emix.shape}",
    )

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
    _assert_nonnegative_density_cube(Emix_density, "Emix_density")

    _assert(
        rho_fwd.shape == Emix_density.shape,
        f"rho_fwd shape {rho_fwd.shape} != Emix_density shape {Emix_density.shape}",
    )

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
    _assert_finite_scalar(blend_alpha, "blend_alpha")
    a = float(np.clip(blend_alpha, 0.0, 1.0))

    rho_dp = make_rho_density_product(frames_psi, Emix, dx, dy)
    rho_ao = make_rho_amplitude_overlap(frames_psi, Emix, dx, dy)

    _assert(
        rho_dp.shape == rho_ao.shape,
        f"rho_dp shape {rho_dp.shape} != rho_ao shape {rho_ao.shape}",
    )

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

    "blended"
        rho = (1-a)*density_product + a*amplitude_overlap
    """
    _assert(
        mode in {
            "amplitude_overlap",
            "density_product",
            "density_product_oldstyle",
            "blended",
        },
        f"Unknown make_rho mode: {mode}",
    )

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