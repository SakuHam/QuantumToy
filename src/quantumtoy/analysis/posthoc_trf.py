from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ============================================================
# Small local validation helpers
# ============================================================

def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def _assert_real_array_2d(arr: np.ndarray, name: str):
    _assert(isinstance(arr, np.ndarray), f"{name} must be np.ndarray")
    _assert(arr.ndim == 2, f"{name} must be 2D, got ndim={arr.ndim}")
    _assert(np.all(np.isfinite(arr)), f"{name} contains non-finite values")


def _assert_real_array_3d(arr: np.ndarray, name: str):
    _assert(isinstance(arr, np.ndarray), f"{name} must be np.ndarray")
    _assert(arr.ndim == 3, f"{name} must be 3D, got ndim={arr.ndim}")
    _assert(np.all(np.isfinite(arr)), f"{name} contains non-finite values")


def _assert_finite_scalar(x, name: str):
    _assert(np.isscalar(x), f"{name} must be scalar, got {type(x)}")
    xf = float(x)
    _assert(np.isfinite(xf), f"{name} must be finite, got {x}")
    return xf


# ============================================================
# Config / result dataclasses
# ============================================================

@dataclass
class PosthocTRFConfig:
    """
    Post hoc TRF selection from saved frames.

    Typical usage:
      1) build base_rho_frames (or some other support field)
      2) choose reference frame by corridor evidence
      3) optionally build worldline-selected rho from that branch choice
    """

    enabled: bool = True

    # Reference frame search
    use_adaptive_ref: bool = True
    ref_t_min_frac: float = 0.30
    ref_t_max_frac: float = 0.95
    valid_total_evidence_eps: float = 1e-12

    # Corridor geometry
    corridor_x_frac_start: float = 0.70
    corridor_y_sigma: float = 1.30
    corridor_x_weight_power: float = 2.50

    # Optional worldline refinement after corridor branch choice
    use_posthoc_worldline: bool = True
    wl_track_radius_px: int = 20
    wl_min_local_rel: float = 0.03
    wl_tube_sigma_px: float = 10.0
    wl_gain_strength: float = 2.0
    wl_outside_damp: float = 0.20
    wl_time_ramp_frac: float = 0.12


@dataclass
class PosthocTRFResult:
    valid: bool
    ref_idx: int | None
    ref_time: float | None
    chosen_side: str | None

    upper_evidence: float
    lower_evidence: float
    total_evidence: float
    abs_margin: float
    rel_margin: float
    ratio: float
    dominance: float
    adaptive_score: float

    upper_seed_x: float | None
    upper_seed_y: float | None
    lower_seed_x: float | None
    lower_seed_y: float | None

    worldline_used: bool
    worldline_seed_side: str | None
    worldline_seed_x: float | None
    worldline_seed_y: float | None

    aux: dict


# ============================================================
# Basic helpers
# ============================================================

def gaussian_weights(Tk: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    _assert(isinstance(Tk, np.ndarray), "Tk must be np.ndarray")
    _assert(Tk.ndim == 1, "Tk must be 1D")
    _assert(np.all(np.isfinite(Tk)), "Tk contains non-finite values")
    _assert_finite_scalar(mu, "mu")
    sigma = float(sigma)

    if sigma <= 0.0:
        w = np.zeros_like(Tk, dtype=float)
        w[int(np.argmin(np.abs(Tk - mu)))] = 1.0
        return w

    z = (Tk - mu) / sigma
    w = np.exp(-0.5 * z * z)
    s = float(np.sum(w))
    return (w / s) if s > 0.0 else w


def build_Emix_from_phi_tau(
    phi_tau_frames: np.ndarray,
    times: np.ndarray,
    t_det: float,
    sigmaT: float,
    tau_step: float,
    K_JITTER: int = 13,
) -> np.ndarray:
    """
    Same idea as the old script:
      - take a small time neighborhood around t_det
      - Gaussian weight those backward-library frames
      - for each forward time i, mix compatible tau slices
    """
    _assert_real_array_3d(phi_tau_frames, "phi_tau_frames")
    _assert(isinstance(times, np.ndarray), "times must be np.ndarray")
    _assert(times.ndim == 1, "times must be 1D")
    _assert(np.all(np.isfinite(times)), "times contains non-finite values")
    _assert_finite_scalar(t_det, "t_det")
    _assert_finite_scalar(sigmaT, "sigmaT")
    _assert_finite_scalar(tau_step, "tau_step")
    _assert(len(times) == phi_tau_frames.shape[0], "times length must match phi_tau_frames")

    Nt = len(times)
    halfK = int(K_JITTER) // 2
    idx_det2 = int(np.argmin(np.abs(times - t_det)))

    k_inds = np.arange(idx_det2 - halfK, idx_det2 + halfK + 1)
    k_inds = np.clip(k_inds, 0, Nt - 1)
    k_inds = np.unique(k_inds)

    Tk = times[k_inds]
    w = gaussian_weights(Tk, t_det, sigmaT)

    Emix = np.zeros_like(phi_tau_frames, dtype=np.float32)

    for i, ti in enumerate(times):
        tau = Tk - ti
        valid = tau >= 0.0
        if not np.any(valid):
            continue

        j = np.rint(tau[valid] / tau_step).astype(int)
        j = np.clip(j, 0, Nt - 1)

        Emix[i] = np.sum(
            (w[valid])[:, None, None] * phi_tau_frames[j],
            axis=0,
        ).astype(np.float32)

    return Emix


def make_rho_from_density_product(
    frames_density: np.ndarray,
    Emix_density: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    Old-style density product:
        rho_i ~ frames_density[i] * Emix_density[i]
    then renormalized per frame.
    """
    _assert_real_array_3d(frames_density, "frames_density")
    _assert_real_array_3d(Emix_density, "Emix_density")
    _assert(frames_density.shape == Emix_density.shape, "shape mismatch in make_rho_from_density_product")
    _assert_finite_scalar(dx, "dx")
    _assert_finite_scalar(dy, "dy")

    out = np.zeros_like(frames_density, dtype=np.float32)

    for i in range(frames_density.shape[0]):
        rho = (frames_density[i] * Emix_density[i]).astype(np.float64)
        s = float(np.sum(rho) * dx * dy)
        if s > 0.0:
            rho /= s
        out[i] = rho.astype(np.float32)

    return out


# ============================================================
# Corridor TRF logic
# ============================================================

def build_trf_corridor_masks_vis(
    X_vis: np.ndarray,
    Y_vis: np.ndarray,
    barrier_center_x: float,
    screen_center_x: float,
    slit_center_offset: float,
    cfg: PosthocTRFConfig,
):
    _assert_real_array_2d(X_vis, "X_vis")
    _assert_real_array_2d(Y_vis, "Y_vis")
    _assert(X_vis.shape == Y_vis.shape, "X_vis and Y_vis must match")

    x_left = float(barrier_center_x + 0.5)
    x_right = float(screen_center_x - 0.5)
    x_start = x_left + float(cfg.corridor_x_frac_start) * (x_right - x_left)

    x_u = np.clip((X_vis - x_start) / max(x_right - x_start, 1e-12), 0.0, 1.0)
    x_weight = np.power(x_u, float(cfg.corridor_x_weight_power)).astype(np.float32)
    end_mask = (X_vis >= x_start).astype(np.float32)

    sigma_y = max(float(cfg.corridor_y_sigma), 1e-12)

    upper_band = np.exp(-0.5 * ((Y_vis - float(slit_center_offset)) / sigma_y) ** 2)
    lower_band = np.exp(-0.5 * ((Y_vis + float(slit_center_offset)) / sigma_y) ** 2)

    upper = (upper_band * x_weight * end_mask).astype(np.float32)
    lower = (lower_band * x_weight * end_mask).astype(np.float32)

    return upper, lower, float(x_start)


def compute_trf_corridor_evidence_for_frame(
    rho_frame: np.ndarray,
    upper_corridor_vis: np.ndarray,
    lower_corridor_vis: np.ndarray,
):
    _assert_real_array_2d(rho_frame, "rho_frame")
    _assert_real_array_2d(upper_corridor_vis, "upper_corridor_vis")
    _assert_real_array_2d(lower_corridor_vis, "lower_corridor_vis")
    _assert(
        rho_frame.shape == upper_corridor_vis.shape == lower_corridor_vis.shape,
        "corridor evidence shapes must match",
    )

    upper_ev = float(np.sum(rho_frame * upper_corridor_vis))
    lower_ev = float(np.sum(rho_frame * lower_corridor_vis))
    total_ev = float(upper_ev + lower_ev)

    if total_ev <= 0.0:
        dominance = 0.0
        rel_margin = 0.0
        ratio = 0.0
        chosen_side = None
    else:
        dominance = float(max(upper_ev, lower_ev) / max(total_ev, 1e-12))
        rel_margin = float(abs(upper_ev - lower_ev) / max(max(upper_ev, lower_ev), 1e-12))
        ratio = float(max(upper_ev, lower_ev) / max(min(upper_ev, lower_ev), 1e-12))
        chosen_side = "upper" if upper_ev >= lower_ev else "lower"

    adaptive_score = float(total_ev * dominance)

    return {
        "upper_evidence": float(upper_ev),
        "lower_evidence": float(lower_ev),
        "total_evidence": float(total_ev),
        "dominance": float(dominance),
        "rel_margin": float(rel_margin),
        "ratio": float(ratio),
        "chosen_side": chosen_side,
        "adaptive_score": float(adaptive_score),
    }


def choose_trf_branch_by_corridor(
    base_rho: np.ndarray,
    times: np.ndarray,
    X_vis: np.ndarray,
    Y_vis: np.ndarray,
    barrier_center_x: float,
    screen_center_x: float,
    slit_center_offset: float,
    cfg: PosthocTRFConfig,
):
    _assert_real_array_3d(base_rho, "base_rho")
    _assert(isinstance(times, np.ndarray), "times must be np.ndarray")
    _assert(times.ndim == 1, "times must be 1D")
    _assert(np.all(np.isfinite(times)), "times contains non-finite values")
    _assert(len(times) == base_rho.shape[0], "times length must match base_rho")
    _assert_real_array_2d(X_vis, "X_vis")
    _assert_real_array_2d(Y_vis, "Y_vis")
    _assert(base_rho.shape[1:] == X_vis.shape == Y_vis.shape, "base_rho frame shape must match X_vis/Y_vis")

    Nt = len(times)

    upper_corridor_vis, lower_corridor_vis, corridor_x_start = build_trf_corridor_masks_vis(
        X_vis=X_vis,
        Y_vis=Y_vis,
        barrier_center_x=barrier_center_x,
        screen_center_x=screen_center_x,
        slit_center_offset=slit_center_offset,
        cfg=cfg,
    )

    upper_seed_x = float(0.5 * (corridor_x_start + (screen_center_x - 0.5)))
    upper_seed_y = float(slit_center_offset)
    lower_seed_x = float(0.5 * (corridor_x_start + (screen_center_x - 0.5)))
    lower_seed_y = float(-slit_center_offset)

    if not cfg.use_adaptive_ref:
        idx_ref = int(np.argmin(np.abs(times - 0.55 * times[-1])))
        ev = compute_trf_corridor_evidence_for_frame(
            base_rho[idx_ref],
            upper_corridor_vis,
            lower_corridor_vis,
        )
        valid = bool(
            ev["total_evidence"] >= float(cfg.valid_total_evidence_eps)
            and ev["chosen_side"] is not None
        )

        return {
            "valid": valid,
            "ref_idx": int(idx_ref),
            "ref_time": float(times[idx_ref]),
            "chosen_side": ev["chosen_side"],
            "upper_evidence": ev["upper_evidence"],
            "lower_evidence": ev["lower_evidence"],
            "total_evidence": ev["total_evidence"],
            "abs_margin": float(abs(ev["upper_evidence"] - ev["lower_evidence"])),
            "rel_margin": ev["rel_margin"],
            "ratio": ev["ratio"],
            "dominance": ev["dominance"],
            "adaptive_score": ev["adaptive_score"],
            "upper_seed_x": upper_seed_x,
            "upper_seed_y": upper_seed_y,
            "lower_seed_x": lower_seed_x,
            "lower_seed_y": lower_seed_y,
            "corridor_x_start": float(corridor_x_start),
        }

    t_min = float(cfg.ref_t_min_frac * times[-1])
    t_max = float(cfg.ref_t_max_frac * times[-1])

    cand_inds = np.where((times >= t_min) & (times <= t_max))[0]
    if len(cand_inds) == 0:
        cand_inds = np.arange(Nt)

    best = None
    for idx in cand_inds:
        ev = compute_trf_corridor_evidence_for_frame(
            base_rho[idx],
            upper_corridor_vis,
            lower_corridor_vis,
        )

        rec = {
            "idx": int(idx),
            "time": float(times[idx]),
            **ev,
        }

        if best is None:
            best = rec
            continue

        if rec["adaptive_score"] > best["adaptive_score"]:
            best = rec
        elif rec["adaptive_score"] == best["adaptive_score"] and rec["dominance"] > best["dominance"]:
            best = rec

    valid = bool(
        best["total_evidence"] >= float(cfg.valid_total_evidence_eps)
        and best["chosen_side"] is not None
    )

    return {
        "valid": valid,
        "ref_idx": int(best["idx"]),
        "ref_time": float(best["time"]),
        "chosen_side": best["chosen_side"],
        "upper_evidence": float(best["upper_evidence"]),
        "lower_evidence": float(best["lower_evidence"]),
        "total_evidence": float(best["total_evidence"]),
        "abs_margin": float(abs(best["upper_evidence"] - best["lower_evidence"])),
        "rel_margin": float(best["rel_margin"]),
        "ratio": float(best["ratio"]),
        "dominance": float(best["dominance"]),
        "adaptive_score": float(best["adaptive_score"]),
        "upper_seed_x": upper_seed_x,
        "upper_seed_y": upper_seed_y,
        "lower_seed_x": lower_seed_x,
        "lower_seed_y": lower_seed_y,
        "corridor_x_start": float(corridor_x_start),
    }


# ============================================================
# Optional post hoc worldline refinement
# ============================================================

def extract_local_peak(arr: np.ndarray, iy0: int, ix0: int, radius_px: int = 16):
    _assert_real_array_2d(arr, "arr(extract_local_peak)")

    ny, nx = arr.shape
    y0 = max(0, int(iy0) - int(radius_px))
    y1 = min(ny, int(iy0) + int(radius_px) + 1)
    x0 = max(0, int(ix0) - int(radius_px))
    x1 = min(nx, int(ix0) + int(radius_px) + 1)

    sub = arr[y0:y1, x0:x1]
    flat_idx = int(np.argmax(sub))
    val = float(sub.ravel()[flat_idx])
    sy, sx = np.unravel_index(flat_idx, sub.shape)

    return val, y0 + sy, x0 + sx


def track_worldline_from_seed(
    rho_frames: np.ndarray,
    start_i: int,
    start_iy: int,
    start_ix: int,
    radius_px: int = 16,
    min_local_rel: float = 0.02,
):
    _assert_real_array_3d(rho_frames, "rho_frames(track_worldline)")

    Nt, _, _ = rho_frames.shape
    path_y = np.full(Nt, int(start_iy), dtype=int)
    path_x = np.full(Nt, int(start_ix), dtype=int)

    ref_amp = float(max(rho_frames[start_i, start_iy, start_ix], 1e-30))

    iy, ix = int(start_iy), int(start_ix)
    for i in range(start_i + 1, Nt):
        val, iy_new, ix_new = extract_local_peak(rho_frames[i], iy, ix, radius_px=radius_px)
        if val >= float(min_local_rel) * ref_amp:
            iy, ix = int(iy_new), int(ix_new)
        path_y[i] = iy
        path_x[i] = ix

    iy, ix = int(start_iy), int(start_ix)
    for i in range(start_i - 1, -1, -1):
        val, iy_new, ix_new = extract_local_peak(rho_frames[i], iy, ix, radius_px=radius_px)
        if val >= float(min_local_rel) * ref_amp:
            iy, ix = int(iy_new), int(ix_new)
        path_y[i] = iy
        path_x[i] = ix

    return path_y, path_x


def build_worldline_tube(shape, path_y: np.ndarray, path_x: np.ndarray, sigma_px: float):
    Nt, ny, nx = shape
    yy = np.arange(ny)[:, None]
    xx = np.arange(nx)[None, :]

    tube = np.zeros(shape, dtype=np.float32)
    inv2s2 = 1.0 / max(2.0 * float(sigma_px) * float(sigma_px), 1e-12)

    for i in range(Nt):
        dy2 = (yy - int(path_y[i])) ** 2
        dx2 = (xx - int(path_x[i])) ** 2
        tube[i] = np.exp(-(dy2 + dx2) * inv2s2).astype(np.float32)

    return tube


def smooth_time_ramp(Nt: int, center_idx: int, ramp_frac: float = 0.1):
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
    _assert_real_array_3d(rho_frames, "rho_frames(apply_worldline_selection)")
    _assert_real_array_3d(tube, "tube(apply_worldline_selection)")
    _assert(rho_frames.shape == tube.shape, "rho_frames and tube shape mismatch")
    _assert_finite_scalar(dx, "dx")
    _assert_finite_scalar(dy, "dy")

    out = np.zeros_like(rho_frames, dtype=np.float32)

    if time_gate is None:
        time_gate = np.ones(rho_frames.shape[0], dtype=np.float32)

    for i in range(rho_frames.shape[0]):
        g = float(time_gate[i])
        field = float(gain_strength) * g * tube[i] - float(outside_damp) * g * (1.0 - tube[i])
        rho = rho_frames[i] * np.exp(field)
        s = float(np.sum(rho) * dx * dy)
        if s > 0.0:
            rho /= s
        out[i] = rho.astype(np.float32)

    return out


def compute_posthoc_worldline_selected_rho(
    base_rho: np.ndarray,
    trf_info: dict,
    X_vis: np.ndarray,
    Y_vis: np.ndarray,
    dx: float,
    dy: float,
    cfg: PosthocTRFConfig,
):
    _assert_real_array_3d(base_rho, "base_rho(worldline)")
    _assert_real_array_2d(X_vis, "X_vis(worldline)")
    _assert_real_array_2d(Y_vis, "Y_vis(worldline)")

    idx_ref = int(trf_info["ref_idx"])

    if trf_info["chosen_side"] == "upper":
        seed_x = float(trf_info["upper_seed_x"])
        seed_y = float(trf_info["upper_seed_y"])
    else:
        seed_x = float(trf_info["lower_seed_x"])
        seed_y = float(trf_info["lower_seed_y"])

    ix = int(np.argmin(np.abs(X_vis[0, :] - seed_x)))
    iy = int(np.argmin(np.abs(Y_vis[:, 0] - seed_y)))

    path_y, path_x = track_worldline_from_seed(
        base_rho,
        idx_ref,
        iy,
        ix,
        radius_px=int(cfg.wl_track_radius_px),
        min_local_rel=float(cfg.wl_min_local_rel),
    )

    tube = build_worldline_tube(
        base_rho.shape,
        path_y,
        path_x,
        sigma_px=float(cfg.wl_tube_sigma_px),
    )

    time_gate = smooth_time_ramp(
        base_rho.shape[0],
        idx_ref,
        ramp_frac=float(cfg.wl_time_ramp_frac),
    )

    rho_wl = apply_worldline_selection(
        base_rho,
        tube,
        dx=dx,
        dy=dy,
        gain_strength=float(cfg.wl_gain_strength),
        outside_damp=float(cfg.wl_outside_damp),
        time_gate=time_gate,
    )

    return rho_wl, {
        "seed_x": float(seed_x),
        "seed_y": float(seed_y),
        "seed_side": trf_info["chosen_side"],
        "seed_ix": int(ix),
        "seed_iy": int(iy),
        "path_x": path_x,
        "path_y": path_y,
        "tube": tube,
        "time_gate": time_gate,
    }


# ============================================================
# High-level runner
# ============================================================

def run_posthoc_trf(
    base_rho: np.ndarray,
    times: np.ndarray,
    X_vis: np.ndarray,
    Y_vis: np.ndarray,
    barrier_center_x: float,
    screen_center_x: float,
    slit_center_offset: float,
    dx: float,
    dy: float,
    cfg: PosthocTRFConfig,
):
    if not cfg.enabled:
        return PosthocTRFResult(
            valid=False,
            ref_idx=None,
            ref_time=None,
            chosen_side=None,
            upper_evidence=0.0,
            lower_evidence=0.0,
            total_evidence=0.0,
            abs_margin=0.0,
            rel_margin=0.0,
            ratio=0.0,
            dominance=0.0,
            adaptive_score=0.0,
            upper_seed_x=None,
            upper_seed_y=None,
            lower_seed_x=None,
            lower_seed_y=None,
            worldline_used=False,
            worldline_seed_side=None,
            worldline_seed_x=None,
            worldline_seed_y=None,
            aux={"enabled": False},
        )

    trf_info = choose_trf_branch_by_corridor(
        base_rho=base_rho,
        times=times,
        X_vis=X_vis,
        Y_vis=Y_vis,
        barrier_center_x=barrier_center_x,
        screen_center_x=screen_center_x,
        slit_center_offset=slit_center_offset,
        cfg=cfg,
    )

    wl_used = False
    wl_seed_side = None
    wl_seed_x = None
    wl_seed_y = None
    wl_aux = {}
    rho_worldline = None

    if cfg.use_posthoc_worldline and trf_info["valid"]:
        rho_worldline, wl_aux = compute_posthoc_worldline_selected_rho(
            base_rho=base_rho,
            trf_info=trf_info,
            X_vis=X_vis,
            Y_vis=Y_vis,
            dx=dx,
            dy=dy,
            cfg=cfg,
        )
        wl_used = True
        wl_seed_side = str(wl_aux["seed_side"])
        wl_seed_x = float(wl_aux["seed_x"])
        wl_seed_y = float(wl_aux["seed_y"])

    aux = {
        "enabled": True,
        "corridor_x_start": float(trf_info["corridor_x_start"]),
        "rho_worldline": rho_worldline,
        "worldline_aux": wl_aux,
    }

    return PosthocTRFResult(
        valid=bool(trf_info["valid"]),
        ref_idx=int(trf_info["ref_idx"]),
        ref_time=float(trf_info["ref_time"]),
        chosen_side=trf_info["chosen_side"],

        upper_evidence=float(trf_info["upper_evidence"]),
        lower_evidence=float(trf_info["lower_evidence"]),
        total_evidence=float(trf_info["total_evidence"]),
        abs_margin=float(trf_info["abs_margin"]),
        rel_margin=float(trf_info["rel_margin"]),
        ratio=float(trf_info["ratio"]),
        dominance=float(trf_info["dominance"]),
        adaptive_score=float(trf_info["adaptive_score"]),

        upper_seed_x=float(trf_info["upper_seed_x"]),
        upper_seed_y=float(trf_info["upper_seed_y"]),
        lower_seed_x=float(trf_info["lower_seed_x"]),
        lower_seed_y=float(trf_info["lower_seed_y"]),

        worldline_used=bool(wl_used),
        worldline_seed_side=wl_seed_side,
        worldline_seed_x=wl_seed_x,
        worldline_seed_y=wl_seed_y,

        aux=aux,
    )