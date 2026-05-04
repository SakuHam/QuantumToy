from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from analysis.trf_branch import TrfBranchDecision


# ============================================================
# Validation helpers
# ============================================================

def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def _assert_finite_scalar(x, name: str):
    _assert(np.isscalar(x), f"{name} must be scalar, got type={type(x)}")
    xf = float(x)
    _assert(np.isfinite(xf), f"{name} must be finite, got {x}")
    return xf


def _assert_positive_scalar(x, name: str):
    xf = _assert_finite_scalar(x, name)
    _assert(xf > 0.0, f"{name} must be > 0, got {x}")
    return xf


def _assert_nonnegative_scalar(x, name: str):
    xf = _assert_finite_scalar(x, name)
    _assert(xf >= 0.0, f"{name} must be >= 0, got {x}")
    return xf


def _assert_real_2d(arr: np.ndarray, name: str):
    _assert(isinstance(arr, np.ndarray), f"{name} must be np.ndarray")
    _assert(arr.ndim == 2, f"{name} must be 2D, got ndim={arr.ndim}")
    _assert(np.all(np.isfinite(arr)), f"{name} contains non-finite values")


def _assert_nonnegative_density_cube(arr: np.ndarray, name: str):
    _assert(isinstance(arr, np.ndarray), f"{name} must be np.ndarray")
    _assert(arr.ndim == 3, f"{name} must be 3D, got ndim={arr.ndim}")
    _assert(arr.shape[0] >= 1, f"{name} must have at least one frame")
    _assert(np.all(np.isfinite(arr)), f"{name} contains non-finite values")
    _assert(np.all(arr >= -1e-14), f"{name} contains significantly negative values")


def _assert_visible_xy(X_vis: np.ndarray, Y_vis: np.ndarray):
    _assert_real_2d(X_vis, "X_vis")
    _assert_real_2d(Y_vis, "Y_vis")
    _assert(X_vis.shape == Y_vis.shape,
            f"X_vis shape {X_vis.shape} must match Y_vis shape {Y_vis.shape}")


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class WorldlineResult:
    rho_selected: np.ndarray
    path_y: np.ndarray
    path_x: np.ndarray
    tube: np.ndarray
    time_gate: np.ndarray
    seed_x: float
    seed_y: float
    seed_side: str


# ============================================================
# Helpers
# ============================================================

def extract_local_peak(arr: np.ndarray, iy0: int, ix0: int, radius_px: int = 16):
    _assert_real_2d(arr, "arr")
    _assert(isinstance(radius_px, int) and radius_px >= 1,
            f"radius_px must be int >= 1, got {radius_px}")
    _assert(0 <= iy0 < arr.shape[0], f"iy0 out of bounds: {iy0}")
    _assert(0 <= ix0 < arr.shape[1], f"ix0 out of bounds: {ix0}")

    ny, nx = arr.shape
    y0 = max(0, iy0 - radius_px)
    y1 = min(ny, iy0 + radius_px + 1)
    x0 = max(0, ix0 - radius_px)
    x1 = min(nx, ix0 + radius_px + 1)

    sub = arr[y0:y1, x0:x1]
    _assert(sub.size > 0, "local peak subarray is empty")

    flat_idx = int(np.argmax(sub))
    val = float(sub.ravel()[flat_idx])
    sy, sx = np.unravel_index(flat_idx, sub.shape)

    iy = y0 + sy
    ix = x0 + sx

    _assert(0 <= iy < ny, f"peak iy out of bounds: {iy}")
    _assert(0 <= ix < nx, f"peak ix out of bounds: {ix}")

    return val, iy, ix


def track_worldline_from_seed(
    rho_frames: np.ndarray,
    start_i: int,
    start_iy: int,
    start_ix: int,
    radius_px: int = 16,
    min_local_rel: float = 0.02,
):
    _assert_nonnegative_density_cube(rho_frames, "rho_frames")
    _assert(isinstance(start_i, int) and 0 <= start_i < rho_frames.shape[0],
            f"start_i out of bounds: {start_i}")
    _assert(isinstance(start_iy, int) and 0 <= start_iy < rho_frames.shape[1],
            f"start_iy out of bounds: {start_iy}")
    _assert(isinstance(start_ix, int) and 0 <= start_ix < rho_frames.shape[2],
            f"start_ix out of bounds: {start_ix}")
    _assert(isinstance(radius_px, int) and radius_px >= 1,
            f"radius_px must be int >= 1, got {radius_px}")
    _assert_nonnegative_scalar(min_local_rel, "min_local_rel")

    Nt, _, _ = rho_frames.shape
    path_y = np.full(Nt, start_iy, dtype=int)
    path_x = np.full(Nt, start_ix, dtype=int)

    ref_amp = float(max(rho_frames[start_i, start_iy, start_ix], 1e-30))

    iy, ix = start_iy, start_ix
    for i in range(start_i + 1, Nt):
        val, iy_new, ix_new = extract_local_peak(rho_frames[i], iy, ix, radius_px=radius_px)
        if val >= float(min_local_rel) * ref_amp:
            iy, ix = iy_new, ix_new
        path_y[i] = iy
        path_x[i] = ix

    iy, ix = start_iy, start_ix
    for i in range(start_i - 1, -1, -1):
        val, iy_new, ix_new = extract_local_peak(rho_frames[i], iy, ix, radius_px=radius_px)
        if val >= float(min_local_rel) * ref_amp:
            iy, ix = iy_new, ix_new
        path_y[i] = iy
        path_x[i] = ix

    return path_y, path_x


def build_worldline_tube(shape, path_y: np.ndarray, path_x: np.ndarray, sigma_px: float):
    _assert(isinstance(shape, tuple) and len(shape) == 3,
            f"shape must be a 3-tuple (Nt, Ny, Nx), got {shape}")
    Nt, ny, nx = shape
    _assert(Nt >= 1 and ny >= 1 and nx >= 1,
            f"invalid tube shape: {shape}")

    _assert(isinstance(path_y, np.ndarray) and path_y.ndim == 1,
            "path_y must be 1D np.ndarray")
    _assert(isinstance(path_x, np.ndarray) and path_x.ndim == 1,
            "path_x must be 1D np.ndarray")
    _assert(len(path_y) == Nt, f"path_y length {len(path_y)} must equal Nt={Nt}")
    _assert(len(path_x) == Nt, f"path_x length {len(path_x)} must equal Nt={Nt}")
    _assert_positive_scalar(sigma_px, "sigma_px")

    yy = np.arange(ny)[:, None]
    xx = np.arange(nx)[None, :]
    tube = np.zeros(shape, dtype=np.float32)
    inv2s2 = 1.0 / max(2.0 * float(sigma_px) * float(sigma_px), 1e-12)

    for i in range(Nt):
        _assert(0 <= int(path_y[i]) < ny, f"path_y[{i}] out of bounds: {path_y[i]}")
        _assert(0 <= int(path_x[i]) < nx, f"path_x[{i}] out of bounds: {path_x[i]}")

        dy2 = (yy - int(path_y[i])) ** 2
        dx2 = (xx - int(path_x[i])) ** 2
        tube[i] = np.exp(-(dy2 + dx2) * inv2s2).astype(np.float32)

    _assert_nonnegative_density_cube(tube.astype(float), "tube")
    return tube


def smooth_time_ramp(Nt: int, center_idx: int, ramp_frac: float = 0.1):
    _assert(isinstance(Nt, int) and Nt >= 1, f"Nt must be int >= 1, got {Nt}")
    _assert(isinstance(center_idx, int) and 0 <= center_idx < Nt,
            f"center_idx out of bounds: {center_idx}")
    _assert_nonnegative_scalar(ramp_frac, "ramp_frac")

    ramp_len = max(3, int(float(ramp_frac) * Nt))
    gate = np.zeros(Nt, dtype=np.float32)

    for i in range(Nt):
        if i <= center_idx - ramp_len:
            gate[i] = 0.0
        elif i >= center_idx + ramp_len:
            gate[i] = 1.0
        else:
            u = (i - (center_idx - ramp_len)) / (2.0 * ramp_len)
            gate[i] = 0.5 - 0.5 * np.cos(np.pi * u)

    _assert(np.all(np.isfinite(gate)), "time gate contains non-finite values")
    _assert(np.all(gate >= -1e-14), "time gate contains negative values")
    _assert(np.all(gate <= 1.0 + 1e-14), "time gate exceeds 1")
    return gate


def apply_worldline_selection(
    rho_frames: np.ndarray,
    tube: np.ndarray,
    dx: float,
    dy: float,
    gain_strength: float = 2.0,
    outside_damp: float = 0.2,
    time_gate: np.ndarray | None = None,
):
    _assert_nonnegative_density_cube(rho_frames, "rho_frames")
    _assert_nonnegative_density_cube(tube.astype(float), "tube")
    _assert(rho_frames.shape == tube.shape,
            f"rho_frames shape {rho_frames.shape} must equal tube shape {tube.shape}")
    _assert_positive_scalar(dx, "dx")
    _assert_positive_scalar(dy, "dy")
    _assert_nonnegative_scalar(gain_strength, "gain_strength")
    _assert_nonnegative_scalar(outside_damp, "outside_damp")

    if time_gate is None:
        time_gate = np.ones(rho_frames.shape[0], dtype=np.float32)
    else:
        _assert(isinstance(time_gate, np.ndarray), "time_gate must be np.ndarray")
        _assert(time_gate.ndim == 1, f"time_gate must be 1D, got ndim={time_gate.ndim}")
        _assert(len(time_gate) == rho_frames.shape[0],
                f"time_gate length {len(time_gate)} must equal Nt={rho_frames.shape[0]}")
        _assert(np.all(np.isfinite(time_gate)), "time_gate contains non-finite values")

    out = np.zeros_like(rho_frames, dtype=np.float32)

    for i in range(rho_frames.shape[0]):
        g = float(time_gate[i])
        field = (
            float(gain_strength) * g * tube[i]
            - float(outside_damp) * g * (1.0 - tube[i])
        )
        rho = rho_frames[i] * np.exp(field)
        s = float(np.sum(rho) * float(dx) * float(dy))
        _assert(np.isfinite(s), f"worldline frame normalization sum non-finite at i={i}")
        if s > 0:
            rho /= s
        out[i] = rho.astype(np.float32)

    _assert_nonnegative_density_cube(out.astype(float), "rho_selected")
    return out


# ============================================================
# Orchestrator
# ============================================================

def compute_posthoc_worldline_selected_rho(
    base_rho: np.ndarray,
    X_vis: np.ndarray,
    Y_vis: np.ndarray,
    trf_info: TrfBranchDecision,
    dx: float,
    dy: float,
    track_radius_px: int = 20,
    min_local_rel: float = 0.03,
    tube_sigma_px: float = 10.0,
    gain_strength: float = 2.0,
    outside_damp: float = 0.20,
    time_ramp_frac: float = 0.12,
) -> WorldlineResult:
    _assert_nonnegative_density_cube(base_rho, "base_rho")
    _assert_visible_xy(X_vis, Y_vis)
    _assert(base_rho.shape[1:] == X_vis.shape,
            f"base_rho visible shape {base_rho.shape[1:]} must equal X_vis shape {X_vis.shape}")
    _assert_positive_scalar(dx, "dx")
    _assert_positive_scalar(dy, "dy")
    _assert(trf_info.valid, "trf_info must be valid for worldline selection")
    _assert(trf_info.chosen_side in {"upper", "lower"},
            f"trf_info.chosen_side must be 'upper' or 'lower', got {trf_info.chosen_side!r}")

    if trf_info.chosen_side == "upper":
        seed_x = float(trf_info.upper_seed_x)
        seed_y = float(trf_info.upper_seed_y)
    else:
        seed_x = float(trf_info.lower_seed_x)
        seed_y = float(trf_info.lower_seed_y)

    ix = int(np.argmin(np.abs(X_vis[0, :] - seed_x)))
    iy = int(np.argmin(np.abs(Y_vis[:, 0] - seed_y)))

    _assert(0 <= ix < X_vis.shape[1], f"seed ix out of bounds: {ix}")
    _assert(0 <= iy < Y_vis.shape[0], f"seed iy out of bounds: {iy}")

    path_y, path_x = track_worldline_from_seed(
        rho_frames=base_rho,
        start_i=int(trf_info.ref_idx),
        start_iy=iy,
        start_ix=ix,
        radius_px=track_radius_px,
        min_local_rel=min_local_rel,
    )

    tube = build_worldline_tube(
        shape=base_rho.shape,
        path_y=path_y,
        path_x=path_x,
        sigma_px=tube_sigma_px,
    )

    time_gate = smooth_time_ramp(
        Nt=base_rho.shape[0],
        center_idx=int(trf_info.ref_idx),
        ramp_frac=time_ramp_frac,
    )

    rho_selected = apply_worldline_selection(
        rho_frames=base_rho,
        tube=tube,
        dx=dx,
        dy=dy,
        gain_strength=gain_strength,
        outside_damp=outside_damp,
        time_gate=time_gate,
    )

    return WorldlineResult(
        rho_selected=rho_selected,
        path_y=path_y,
        path_x=path_x,
        tube=tube,
        time_gate=time_gate,
        seed_x=float(seed_x),
        seed_y=float(seed_y),
        seed_side=str(trf_info.chosen_side),
    )