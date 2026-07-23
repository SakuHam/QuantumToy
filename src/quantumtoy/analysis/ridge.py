from __future__ import annotations

import numpy as np
from scipy.ndimage import label


def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def _assert_frame_pair_compatible(frame_psi, frame_emix):
    """
    Validate one forward/backward frame pair.

    Supported single-frame shapes:
      - scalar Schrödinger:
          (Ny, Nx)

      - Dirac-like / component-first spinor:
          (C, Ny, Nx)

      - entangled spinor:
          (Ny, Nx, 2, 2)
    """
    _assert(isinstance(frame_psi, np.ndarray), "frame_psi must be np.ndarray")
    _assert(isinstance(frame_emix, np.ndarray), "frame_emix must be np.ndarray")

    _assert(
        frame_psi.shape == frame_emix.shape,
        f"frame_psi.shape {frame_psi.shape} != frame_emix.shape {frame_emix.shape}",
    )

    _assert(
        frame_psi.ndim in (2, 3, 4),
        f"frame_psi.ndim must be 2, 3, or 4, got {frame_psi.ndim}",
    )

    _assert(
        frame_emix.ndim in (2, 3, 4),
        f"frame_emix.ndim must be 2, 3, or 4, got {frame_emix.ndim}",
    )

    if frame_psi.ndim == 2:
        _assert(
            frame_emix.ndim == 2,
            f"scalar frame_psi requires scalar frame_emix, got ndim={frame_emix.ndim}",
        )

    elif frame_psi.ndim == 3:
        _assert(
            frame_emix.ndim == 3,
            f"spinor frame_psi requires spinor frame_emix, got ndim={frame_emix.ndim}",
        )
        _assert(
            frame_psi.shape[0] >= 1,
            f"spinor frame_psi must have at least one component, got shape={frame_psi.shape}",
        )

    elif frame_psi.ndim == 4:
        _assert(
            frame_emix.ndim == 4,
            f"entangled frame_psi requires entangled frame_emix, got ndim={frame_emix.ndim}",
        )
        _assert(
            frame_psi.shape[-2:] == (2, 2),
            f"entangled frame_psi must end with spin axes (2,2), got shape={frame_psi.shape}",
        )

    _assert(np.all(np.isfinite(frame_psi.real)), "frame_psi.real contains non-finite values")
    _assert(np.all(np.isfinite(frame_psi.imag)), "frame_psi.imag contains non-finite values")
    _assert(np.all(np.isfinite(frame_emix.real)), "frame_emix.real contains non-finite values")
    _assert(np.all(np.isfinite(frame_emix.imag)), "frame_emix.imag contains non-finite values")


def _assert_frames_pair_compatible(frames_psi, Emix):
    """
    Validate full forward/backward frame arrays.

    Supported full-frame shapes:
      - scalar Schrödinger:
          (Nt, Ny, Nx)

      - Dirac-like / component-first spinor:
          (Nt, C, Ny, Nx)

      - entangled spinor:
          (Nt, Ny, Nx, 2, 2)
    """
    _assert(isinstance(frames_psi, np.ndarray), "frames_psi must be np.ndarray")
    _assert(isinstance(Emix, np.ndarray), "Emix must be np.ndarray")

    _assert(
        frames_psi.shape == Emix.shape,
        f"frames_psi.shape {frames_psi.shape} != Emix.shape {Emix.shape}",
    )

    _assert(
        frames_psi.ndim in (3, 4, 5),
        f"frames_psi.ndim must be 3, 4, or 5, got {frames_psi.ndim}",
    )

    _assert(
        Emix.ndim in (3, 4, 5),
        f"Emix.ndim must be 3, 4, or 5, got {Emix.ndim}",
    )

    _assert(frames_psi.shape[0] >= 1, "frames_psi must contain at least one frame")

    if frames_psi.ndim == 3:
        # (Nt, Ny, Nx)
        _assert(Emix.ndim == 3, f"scalar frames require scalar Emix, got ndim={Emix.ndim}")
        _assert(frames_psi.shape[1] > 0 and frames_psi.shape[2] > 0, "scalar frames have empty spatial axes")

    elif frames_psi.ndim == 4:
        # (Nt, C, Ny, Nx)
        _assert(Emix.ndim == 4, f"spinor frames require spinor Emix, got ndim={Emix.ndim}")
        _assert(frames_psi.shape[1] >= 1, "spinor frames must have at least one component")
        _assert(frames_psi.shape[2] > 0 and frames_psi.shape[3] > 0, "spinor frames have empty spatial axes")

    elif frames_psi.ndim == 5:
        # (Nt, Ny, Nx, 2, 2)
        _assert(Emix.ndim == 5, f"entangled frames require entangled Emix, got ndim={Emix.ndim}")
        _assert(
            frames_psi.shape[-2:] == (2, 2),
            f"entangled frames must end with spin axes (2,2), got shape={frames_psi.shape}",
        )
        _assert(frames_psi.shape[1] > 0 and frames_psi.shape[2] > 0, "entangled frames have empty spatial axes")

    _assert(np.all(np.isfinite(frames_psi.real)), "frames_psi.real contains non-finite values")
    _assert(np.all(np.isfinite(frames_psi.imag)), "frames_psi.imag contains non-finite values")
    _assert(np.all(np.isfinite(Emix.real)), "Emix.real contains non-finite values")
    _assert(np.all(np.isfinite(Emix.imag)), "Emix.imag contains non-finite values")


def _assert_vis_axes(x_vis_1d, y_vis_1d, Gamma_shape):
    _assert(isinstance(x_vis_1d, np.ndarray), "x_vis_1d must be np.ndarray")
    _assert(isinstance(y_vis_1d, np.ndarray), "y_vis_1d must be np.ndarray")
    _assert(x_vis_1d.ndim == 1, f"x_vis_1d must be 1D, got {x_vis_1d.ndim}")
    _assert(y_vis_1d.ndim == 1, f"y_vis_1d must be 1D, got {y_vis_1d.ndim}")
    _assert(
        Gamma_shape == (len(y_vis_1d), len(x_vis_1d)),
        f"Gamma shape {Gamma_shape} != {(len(y_vis_1d), len(x_vis_1d))}",
    )
    _assert(np.all(np.isfinite(x_vis_1d)), "x_vis_1d contains non-finite values")
    _assert(np.all(np.isfinite(y_vis_1d)), "y_vis_1d contains non-finite values")


def local_gamma_field(frame_psi, frame_emix):
    """
    Build positive scalar ridge field Gamma from local amplitude overlap.

    Scalar Schrödinger:
        A[y,x] = conj(emix[y,x]) * psi[y,x]
        Gamma = |A|^2

    Dirac-like:
        A[y,x] = sum_c conj(emix[c,y,x]) * psi[c,y,x]
        Gamma = |A|^2

    Entangled spinor:
        A[y,x] = sum_ab conj(emix[y,x,a,b]) * psi[y,x,a,b]
        Gamma = |A|^2
    """
    _assert_frame_pair_compatible(frame_psi, frame_emix)

    if frame_psi.ndim == 2:
        A = np.conjugate(frame_emix) * frame_psi

    elif frame_psi.ndim == 3:
        A = np.sum(np.conjugate(frame_emix) * frame_psi, axis=0)

    elif frame_psi.ndim == 4 and frame_psi.shape[-2:] == (2, 2):
        A = np.sum(np.conjugate(frame_emix) * frame_psi, axis=(-2, -1))

    else:
        raise ValueError(
            f"Incompatible frame shapes: frame_psi.ndim={frame_psi.ndim}, "
            f"frame_emix.ndim={frame_emix.ndim}"
        )

    _assert(A.ndim == 2, f"local amplitude overlap A must be 2D, got ndim={A.ndim}")

    Gamma = (np.abs(A) ** 2).astype(float)

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


def ridge_top_components(Gamma, x_vis_1d, y_vis_1d, n_tops=2, top_q=0.02, eps=1e-30):
    """
    Return weighted centroids for distinct high-Gamma ridge components.

    Components are extracted from the top-q mask and sorted from upper to lower
    y, which keeps split-slit ridge indicators visually stable across frames.
    """
    _assert_vis_axes(x_vis_1d, y_vis_1d, Gamma.shape)
    _assert(isinstance(n_tops, int) and n_tops >= 1, f"n_tops must be int >= 1, got {n_tops}")
    _assert(0.0 < top_q < 1.0, f"top_q must be in (0,1), got {top_q}")

    g = Gamma.astype(float)
    gmax = float(np.max(g))

    out = []
    if gmax <= 0:
        xg, yg, sg = ridge_argmax(g, x_vis_1d, y_vis_1d)
        return [(xg, yg, sg)]

    thr = np.quantile(g.ravel(), 1.0 - top_q)
    mask = (g >= thr) & (g > 0.0)
    labels, n_labels = label(mask)

    for lab in range(1, n_labels + 1):
        comp = labels == lab
        w = g[comp] + eps
        if w.size == 0 or float(np.sum(w)) <= 0:
            continue

        iy_idx, ix_idx = np.where(comp)
        xs_ = x_vis_1d[ix_idx]
        ys_ = y_vis_1d[iy_idx]

        wsum = float(np.sum(w))
        xc = float(np.sum(xs_ * w) / wsum)
        yc = float(np.sum(ys_ * w) / wsum)
        score = float(np.max(g[comp]))
        mass = float(np.sum(g[comp]))

        out.append((xc, yc, score, mass))

    if not out:
        xg, yg, sg = ridge_argmax(g, x_vis_1d, y_vis_1d)
        return [(xg, yg, sg)]

    out.sort(key=lambda item: item[3], reverse=True)
    selected = out[:n_tops]
    selected.sort(key=lambda item: item[1], reverse=True)
    return [(xc, yc, score) for xc, yc, score, _ in selected]


def compute_ridge_top_xy(
    frames_psi,
    Emix,
    x_vis_1d,
    y_vis_1d,
    n_tops=2,
    top_q=0.02,
):
    """
    Compute up to n_tops distinct high-Gamma ridge indicator trajectories.

    Missing indicators are filled with NaN, so callers can hide unavailable
    markers without special casing frame counts.
    """
    _assert_frames_pair_compatible(frames_psi, Emix)
    _assert(isinstance(n_tops, int) and n_tops >= 1, f"n_tops must be int >= 1, got {n_tops}")

    Nt_ = frames_psi.shape[0]
    ridge_x = np.full((n_tops, Nt_), np.nan, dtype=float)
    ridge_y = np.full((n_tops, Nt_), np.nan, dtype=float)
    ridge_s = np.full((n_tops, Nt_), np.nan, dtype=float)

    for i in range(Nt_):
        Gamma = local_gamma_field(frames_psi[i], Emix[i])
        _assert_vis_axes(x_vis_1d, y_vis_1d, Gamma.shape)

        tops = ridge_top_components(
            Gamma,
            x_vis_1d,
            y_vis_1d,
            n_tops=n_tops,
            top_q=top_q,
        )

        for k, (xi, yi, si) in enumerate(tops):
            ridge_x[k, i] = xi
            ridge_y[k, i] = yi
            ridge_s[k, i] = si

    return ridge_x, ridge_y, ridge_s


def _segment_cells(iy0: int, ix0: int, iy1: int, ix1: int) -> list[tuple[int, int]]:
    """Integer cells crossed by a line segment, including both endpoints."""
    n = max(abs(int(iy1) - int(iy0)), abs(int(ix1) - int(ix0)), 1)
    ys = np.rint(np.linspace(iy0, iy1, n + 1)).astype(int)
    xs = np.rint(np.linspace(ix0, ix1, n + 1)).astype(int)
    return list(dict.fromkeys(zip(ys.tolist(), xs.tolist())))


def contiguous_finite_track_slice(xs: np.ndarray, ys: np.ndarray, end_index: int) -> slice | None:
    """Latest contiguous finite segment at or before ``end_index``."""
    end_index = min(int(end_index), len(xs) - 1)
    finite = np.isfinite(xs[: end_index + 1]) & np.isfinite(ys[: end_index + 1])
    indices = np.flatnonzero(finite)
    if not indices.size:
        return None
    end = int(indices[-1])
    start = end
    while start > 0 and finite[start - 1]:
        start -= 1
    return slice(start, end + 1)


def compute_detector_anchored_ridge(
    density_frames: np.ndarray,
    x_vis_1d: np.ndarray,
    y_vis_1d: np.ndarray,
    *,
    click_frame_idx: int,
    click_x: float,
    click_y: float,
    forbidden_mask: np.ndarray | None = None,
    current_x_frames: np.ndarray | None = None,
    current_y_frames: np.ndarray | None = None,
    radius_px: int = 8,
    distance_weight: float = 0.35,
    current_weight: float = 0.25,
    hysteresis_weight: float = 0.20,
    min_global_rel: float = 1e-8,
    branch_switch_cos: float = -0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Track one conditional ridge backward from its selected detector cell.

    Unlike ``compute_ridge_top_xy``, identities are never assigned by peak
    rank. A track has one anchor and one temporally adjacent continuation.
    When no valid continuation exists, earlier samples remain NaN.
    """
    _assert(isinstance(density_frames, np.ndarray) and density_frames.ndim == 3,
            "density_frames must have shape (Nt, Ny, Nx)")
    _assert(np.all(np.isfinite(density_frames)), "density_frames contains non-finite values")
    _assert(np.all(density_frames >= -1e-14), "density_frames contains negative values")
    nt, ny, nx = density_frames.shape
    _assert(0 <= int(click_frame_idx) < nt, "click_frame_idx out of range")
    _assert(isinstance(radius_px, int) and radius_px >= 1, "radius_px must be >= 1")
    _assert_vis_axes(x_vis_1d, y_vis_1d, (ny, nx))

    blocked = np.zeros((ny, nx), dtype=bool) if forbidden_mask is None else np.asarray(forbidden_mask, dtype=bool)
    _assert(blocked.shape == (ny, nx), "forbidden_mask shape mismatch")
    if current_x_frames is not None or current_y_frames is not None:
        _assert(current_x_frames is not None and current_y_frames is not None,
                "both current component arrays are required")
        _assert(current_x_frames.shape == density_frames.shape, "current_x_frames shape mismatch")
        _assert(current_y_frames.shape == density_frames.shape, "current_y_frames shape mismatch")

    xs = np.full(nt, np.nan, dtype=float)
    ys = np.full(nt, np.nan, dtype=float)
    scores = np.full(nt, np.nan, dtype=float)
    ix_next = int(np.argmin(np.abs(x_vis_1d - float(click_x))))
    iy_next = int(np.argmin(np.abs(y_vis_1d - float(click_y))))
    _assert(not blocked[iy_next, ix_next], "selected detector anchor lies inside an obstacle")
    xs[click_frame_idx] = float(x_vis_1d[ix_next])
    ys[click_frame_idx] = float(y_vis_1d[iy_next])
    scores[click_frame_idx] = float(density_frames[click_frame_idx, iy_next, ix_next])

    max_jump = 0.0
    branch_switches = 0
    obstacle_intersections = 0
    previous_forward_step = None
    terminated_at = None

    for i in range(int(click_frame_idx) - 1, -1, -1):
        frame = np.maximum(density_frames[i], 0.0)
        global_max = float(np.max(frame))
        if global_max <= 0.0:
            terminated_at = i
            break

        y0, y1 = max(0, iy_next - radius_px), min(ny, iy_next + radius_px + 1)
        x0, x1 = max(0, ix_next - radius_px), min(nx, ix_next + radius_px + 1)
        candidates = []
        local_max = 0.0
        for iy in range(y0, y1):
            for ix in range(x0, x1):
                dpx = float(np.hypot(ix - ix_next, iy - iy_next))
                if dpx > radius_px or blocked[iy, ix]:
                    continue
                cells = _segment_cells(iy, ix, iy_next, ix_next)
                if any(blocked[cy, cx] for cy, cx in cells):
                    continue
                local_max = max(local_max, float(frame[iy, ix]))
                candidates.append((iy, ix, dpx))

        if not candidates or local_max < min_global_rel * global_max:
            terminated_at = i
            break

        best = None
        best_score = -np.inf
        for iy, ix, dpx in candidates:
            dens_score = float(frame[iy, ix]) / max(local_max, 1e-300)
            forward_step = np.asarray([ix_next - ix, iy_next - iy], dtype=float)
            step_norm = float(np.linalg.norm(forward_step))
            flow_score = 0.0
            if current_x_frames is not None and step_norm > 0.0:
                flow = np.asarray([current_x_frames[i, iy, ix], current_y_frames[i, iy, ix]], dtype=float)
                flow_norm = float(np.linalg.norm(flow))
                if np.isfinite(flow_norm) and flow_norm > 1e-14:
                    flow_score = float(np.dot(forward_step, flow) / (step_norm * flow_norm))
            hysteresis_score = 0.0
            if previous_forward_step is not None and step_norm > 0.0:
                prev_norm = float(np.linalg.norm(previous_forward_step))
                if prev_norm > 0.0:
                    hysteresis_score = float(
                        np.dot(forward_step, previous_forward_step) / (step_norm * prev_norm)
                    )
            total_score = (
                dens_score
                - distance_weight * (dpx / radius_px)
                + current_weight * flow_score
                + hysteresis_weight * hysteresis_score
            )
            if total_score > best_score:
                best_score = total_score
                best = (iy, ix, dpx, forward_step, hysteresis_score)

        if best is None:
            terminated_at = i
            break
        iy, ix, dpx, forward_step, direction_cos = best
        if previous_forward_step is not None and direction_cos < branch_switch_cos:
            branch_switches += 1
            terminated_at = i
            break
        if any(blocked[cy, cx] for cy, cx in _segment_cells(iy, ix, iy_next, ix_next)):
            obstacle_intersections += 1
            terminated_at = i
            break
        xs[i], ys[i], scores[i] = x_vis_1d[ix], y_vis_1d[iy], frame[iy, ix]
        max_jump = max(max_jump, dpx)
        previous_forward_step = forward_step
        iy_next, ix_next = iy, ix

    finite_idx = np.flatnonzero(np.isfinite(xs) & np.isfinite(ys))
    diagnostics = {
        "click_cell": (int(np.argmin(np.abs(y_vis_1d - click_y))), int(np.argmin(np.abs(x_vis_1d - click_x)))),
        "ridge_endpoint_cell": (
            int(np.argmin(np.abs(y_vis_1d - ys[click_frame_idx]))),
            int(np.argmin(np.abs(x_vis_1d - xs[click_frame_idx]))),
        ),
        "endpoint_distance_px": float(np.hypot(
            (xs[click_frame_idx] - click_x) / max(np.median(np.diff(x_vis_1d)), 1e-300),
            (ys[click_frame_idx] - click_y) / max(np.median(np.diff(y_vis_1d)), 1e-300),
        )),
        "max_jump_px": float(max_jump),
        "branch_switches": int(branch_switches),
        "obstacle_intersections": int(obstacle_intersections),
        "terminated_at_frame": terminated_at,
        "frame_indices": finite_idx,
        "times_monotonic": bool(np.all(np.diff(finite_idx) > 0)),
    }
    return xs, ys, scores, diagnostics


def snap_to_localmax_near_point(Gamma, x_vis_1d, y_vis_1d, xc, yc, radius=12):
    """
    Snap a continuous point (xc, yc) to the strongest local maximum in a window.
    """
    _assert_vis_axes(x_vis_1d, y_vis_1d, Gamma.shape)
    _assert(isinstance(radius, int) and radius >= 0, f"radius must be int >= 0, got {radius}")

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
    Gamma,
    x_vis_1d,
    y_vis_1d,
    top_q=0.02,
    snap_radius=12,
):
    xc, yc, _ = ridge_centroid_top(Gamma, x_vis_1d, y_vis_1d, top_q=top_q)

    xs, ys, ss, _, _ = snap_to_localmax_near_point(
        Gamma,
        x_vis_1d,
        y_vis_1d,
        xc,
        yc,
        radius=snap_radius,
    )

    return xs, ys, ss


def ridge_localmax_track(Gamma, x_vis_1d, y_vis_1d, prev_ix, prev_iy, radius=20):
    _assert_vis_axes(x_vis_1d, y_vis_1d, Gamma.shape)
    _assert(isinstance(radius, int) and radius >= 0, f"radius must be int >= 0, got {radius}")

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

    Supported frame arrays:
      - scalar:
          frames_psi.shape == Emix.shape == (Nt, Ny, Nx)

      - Dirac-like:
          frames_psi.shape == Emix.shape == (Nt, C, Ny, Nx)

      - entangled spinor:
          frames_psi.shape == Emix.shape == (Nt, Ny, Nx, 2, 2)

    Modes
    -----
    "argmax"
    "centroid_top"
    "centroid_top_snap_localmax"
    "localmax_track"
    """
    _assert_frames_pair_compatible(frames_psi, Emix)

    _assert(
        mode in {"argmax", "centroid_top", "centroid_top_snap_localmax", "localmax_track"},
        f"Unknown RIDGE_MODE: {mode}",
    )

    _assert(np.isscalar(top_q), f"top_q must be scalar, got type={type(top_q)}")
    top_q = float(top_q)
    _assert(0.0 < top_q < 1.0, f"top_q must be in (0,1), got {top_q}")

    _assert(np.isscalar(alpha_smooth), f"alpha_smooth must be scalar, got type={type(alpha_smooth)}")
    alpha_smooth = float(alpha_smooth)
    _assert(0.0 <= alpha_smooth < 1.0, f"alpha_smooth must be in [0,1), got {alpha_smooth}")

    _assert(isinstance(radius, int) and radius >= 0, f"radius must be int >= 0, got {radius}")
    _assert(isinstance(snap_radius, int) and snap_radius >= 0, f"snap_radius must be int >= 0, got {snap_radius}")

    Nt_ = frames_psi.shape[0]

    ridge_x = np.zeros(Nt_, dtype=float)
    ridge_y = np.zeros(Nt_, dtype=float)
    ridge_s = np.zeros(Nt_, dtype=float)

    Gamma0 = local_gamma_field(frames_psi[0], Emix[0])
    _assert_vis_axes(x_vis_1d, y_vis_1d, Gamma0.shape)

    x0r, y0r, s0 = ridge_argmax(Gamma0, x_vis_1d, y_vis_1d)

    ridge_x[0], ridge_y[0], ridge_s[0] = x0r, y0r, s0

    prev_ix = int(np.argmin(np.abs(x_vis_1d - x0r)))
    prev_iy = int(np.argmin(np.abs(y_vis_1d - y0r)))

    for i in range(1, Nt_):
        Gamma = local_gamma_field(frames_psi[i], Emix[i])
        _assert_vis_axes(x_vis_1d, y_vis_1d, Gamma.shape)

        if mode == "argmax":
            xi, yi, si = ridge_argmax(Gamma, x_vis_1d, y_vis_1d)

        elif mode == "centroid_top":
            xi, yi, si = ridge_centroid_top(
                Gamma,
                x_vis_1d,
                y_vis_1d,
                top_q=top_q,
            )

        elif mode == "centroid_top_snap_localmax":
            xi, yi, si = ridge_centroid_top_snap_localmax(
                Gamma,
                x_vis_1d,
                y_vis_1d,
                top_q=top_q,
                snap_radius=snap_radius,
            )

        elif mode == "localmax_track":
            xi, yi, si, prev_ix, prev_iy = ridge_localmax_track(
                Gamma,
                x_vis_1d,
                y_vis_1d,
                prev_ix,
                prev_iy,
                radius=radius,
            )

        else:
            raise ValueError(f"Unknown RIDGE_MODE: {mode}")

        if alpha_smooth > 0.0:
            xi = (1.0 - alpha_smooth) * xi + alpha_smooth * ridge_x[i - 1]
            yi = (1.0 - alpha_smooth) * yi + alpha_smooth * ridge_y[i - 1]

        ridge_x[i], ridge_y[i], ridge_s[i] = xi, yi, si

        if mode != "localmax_track":
            prev_ix = int(np.argmin(np.abs(x_vis_1d - xi)))
            prev_iy = int(np.argmin(np.abs(y_vis_1d - yi)))

    _assert(np.all(np.isfinite(ridge_x)), "ridge_x contains non-finite values")
    _assert(np.all(np.isfinite(ridge_y)), "ridge_y contains non-finite values")
    _assert(np.all(np.isfinite(ridge_s)), "ridge_s contains non-finite values")

    return ridge_x, ridge_y, ridge_s
