from __future__ import annotations

from dataclasses import dataclass
import numpy as np


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


def _assert_visible_xy(X_vis: np.ndarray, Y_vis: np.ndarray):
    _assert(isinstance(X_vis, np.ndarray), "X_vis must be np.ndarray")
    _assert(isinstance(Y_vis, np.ndarray), "Y_vis must be np.ndarray")
    _assert(X_vis.ndim == 2, f"X_vis must be 2D, got ndim={X_vis.ndim}")
    _assert(Y_vis.ndim == 2, f"Y_vis must be 2D, got ndim={Y_vis.ndim}")
    _assert(X_vis.shape == Y_vis.shape,
            f"X_vis shape {X_vis.shape} must match Y_vis shape {Y_vis.shape}")
    _assert(np.all(np.isfinite(X_vis)), "X_vis contains non-finite values")
    _assert(np.all(np.isfinite(Y_vis)), "Y_vis contains non-finite values")


def _assert_nonnegative_density_cube(arr: np.ndarray, name: str):
    _assert(isinstance(arr, np.ndarray), f"{name} must be np.ndarray")
    _assert(arr.ndim == 3, f"{name} must be 3D, got ndim={arr.ndim}")
    _assert(arr.shape[0] >= 1, f"{name} must have at least one frame")
    _assert(arr.shape[1] >= 1 and arr.shape[2] >= 1, f"{name} spatial dims must be non-empty")
    _assert(np.all(np.isfinite(arr)), f"{name} contains non-finite values")
    _assert(np.all(arr >= -1e-14), f"{name} contains significantly negative values")


def _assert_frame_shape(frame: np.ndarray, ref_shape: tuple[int, int], name: str):
    _assert(isinstance(frame, np.ndarray), f"{name} must be np.ndarray")
    _assert(frame.ndim == 2, f"{name} must be 2D, got ndim={frame.ndim}")
    _assert(frame.shape == ref_shape,
            f"{name} shape {frame.shape} must match reference shape {ref_shape}")
    _assert(np.all(np.isfinite(frame)), f"{name} contains non-finite values")


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class CorridorMasks:
    upper: np.ndarray
    lower: np.ndarray
    x_start: float


@dataclass
class TrfBranchDecision:
    valid: bool
    ref_idx: int
    ref_time: float
    chosen_side: str | None

    upper_evidence: float
    lower_evidence: float
    total_evidence: float

    abs_margin: float
    rel_margin: float
    ratio: float
    dominance: float
    adaptive_score: float

    upper_seed_x: float
    upper_seed_y: float
    lower_seed_x: float
    lower_seed_y: float


# ============================================================
# Geometry helpers
# ============================================================

def find_reference_frame_index(
    times: np.ndarray,
    x0: float,
    barrier_center_x: float,
    screen_center_x: float,
    v_est: float,
    frac_gap: float,
):
    _assert_times(times)
    _assert_finite_scalar(x0, "x0")
    _assert_finite_scalar(barrier_center_x, "barrier_center_x")
    _assert_finite_scalar(screen_center_x, "screen_center_x")
    _assert_positive_scalar(v_est, "v_est")
    _assert_nonnegative_scalar(frac_gap, "frac_gap")

    t_barrier = (float(barrier_center_x) - float(x0)) / max(float(v_est), 1e-12)
    t_gap = (float(screen_center_x) - float(barrier_center_x)) / max(float(v_est), 1e-12)
    t_ref = t_barrier + float(frac_gap) * t_gap

    idx = int(np.argmin(np.abs(times - t_ref)))
    _assert(0 <= idx < len(times), f"reference idx out of bounds: {idx}")
    return idx, float(t_ref)


def build_trf_corridor_masks_vis(
    X_vis: np.ndarray,
    Y_vis: np.ndarray,
    barrier_center_x: float,
    screen_center_x: float,
    slit_center_offset: float,
    corridor_x_frac_start: float = 0.70,
    corridor_y_sigma: float = 1.3,
    corridor_x_weight_power: float = 2.5,
) -> CorridorMasks:
    """
    Build upper/lower corridor evidence masks on visible grid.
    """
    _assert_visible_xy(X_vis, Y_vis)
    _assert_finite_scalar(barrier_center_x, "barrier_center_x")
    _assert_finite_scalar(screen_center_x, "screen_center_x")
    _assert_finite_scalar(slit_center_offset, "slit_center_offset")
    _assert_nonnegative_scalar(corridor_x_frac_start, "corridor_x_frac_start")
    _assert_positive_scalar(corridor_y_sigma, "corridor_y_sigma")
    _assert_nonnegative_scalar(corridor_x_weight_power, "corridor_x_weight_power")

    x_left = float(barrier_center_x + 0.5)
    x_right = float(screen_center_x - 0.5)
    _assert(x_right > x_left,
            f"invalid corridor x-range: x_right={x_right} must be > x_left={x_left}")

    x_start = x_left + float(corridor_x_frac_start) * (x_right - x_left)

    x_u = np.clip((X_vis - x_start) / max(x_right - x_start, 1e-12), 0.0, 1.0)
    x_weight = np.power(x_u, float(corridor_x_weight_power)).astype(np.float32)
    end_mask = (X_vis >= x_start).astype(np.float32)

    upper_band = np.exp(
        -0.5 * ((Y_vis - float(slit_center_offset)) / max(float(corridor_y_sigma), 1e-12)) ** 2
    )
    lower_band = np.exp(
        -0.5 * ((Y_vis + float(slit_center_offset)) / max(float(corridor_y_sigma), 1e-12)) ** 2
    )

    upper = (upper_band * x_weight * end_mask).astype(np.float32)
    lower = (lower_band * x_weight * end_mask).astype(np.float32)

    _assert_frame_shape(upper, X_vis.shape, "upper corridor mask")
    _assert_frame_shape(lower, X_vis.shape, "lower corridor mask")
    _assert(np.all(upper >= -1e-14), "upper corridor mask contains negative values")
    _assert(np.all(lower >= -1e-14), "lower corridor mask contains negative values")

    return CorridorMasks(
        upper=upper,
        lower=lower,
        x_start=float(x_start),
    )


# ============================================================
# Evidence
# ============================================================

def compute_trf_corridor_evidence_for_frame(
    rho_frame: np.ndarray,
    masks: CorridorMasks,
):
    _assert_frame_shape(rho_frame, masks.upper.shape, "rho_frame")
    _assert(np.all(rho_frame >= -1e-14), "rho_frame contains significantly negative values")

    upper_ev = float(np.sum(rho_frame * masks.upper))
    lower_ev = float(np.sum(rho_frame * masks.lower))
    total_ev = float(upper_ev + lower_ev)

    _assert(np.isfinite(upper_ev), "upper_evidence is non-finite")
    _assert(np.isfinite(lower_ev), "lower_evidence is non-finite")
    _assert(np.isfinite(total_ev), "total_evidence is non-finite")

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


# ============================================================
# Branch choice
# ============================================================

def choose_trf_branch_by_corridor(
    base_rho: np.ndarray,
    times: np.ndarray,
    masks: CorridorMasks,
    x0: float,
    barrier_center_x: float,
    screen_center_x: float,
    slit_center_offset: float,
    v_est: float,
    use_adaptive_ref: bool = True,
    ref_t_min_frac: float = 0.30,
    ref_t_max_frac: float = 0.95,
    valid_total_evidence_eps: float = 1e-12,
) -> TrfBranchDecision:
    _assert_nonnegative_density_cube(base_rho, "base_rho")
    _assert_times(times)
    _assert(base_rho.shape[0] == len(times),
            f"base_rho Nt={base_rho.shape[0]} must equal len(times)={len(times)}")

    _assert_finite_scalar(x0, "x0")
    _assert_finite_scalar(barrier_center_x, "barrier_center_x")
    _assert_finite_scalar(screen_center_x, "screen_center_x")
    _assert_finite_scalar(slit_center_offset, "slit_center_offset")
    _assert_positive_scalar(v_est, "v_est")
    _assert_nonnegative_scalar(ref_t_min_frac, "ref_t_min_frac")
    _assert_nonnegative_scalar(ref_t_max_frac, "ref_t_max_frac")
    _assert_nonnegative_scalar(valid_total_evidence_eps, "valid_total_evidence_eps")

    Nt = len(times)

    upper_seed_x = float(0.5 * (masks.x_start + (float(screen_center_x) - 0.5)))
    upper_seed_y = float(slit_center_offset)
    lower_seed_x = float(0.5 * (masks.x_start + (float(screen_center_x) - 0.5)))
    lower_seed_y = float(-float(slit_center_offset))

    if not use_adaptive_ref:
        idx_ref, t_ref = find_reference_frame_index(
            times=times,
            x0=x0,
            barrier_center_x=barrier_center_x,
            screen_center_x=screen_center_x,
            v_est=v_est,
            frac_gap=0.55,
        )

        ev = compute_trf_corridor_evidence_for_frame(base_rho[idx_ref], masks)
        valid = bool(
            ev["total_evidence"] >= float(valid_total_evidence_eps)
            and ev["chosen_side"] is not None
        )

        return TrfBranchDecision(
            valid=valid,
            ref_idx=int(idx_ref),
            ref_time=float(t_ref),
            chosen_side=ev["chosen_side"],
            upper_evidence=float(ev["upper_evidence"]),
            lower_evidence=float(ev["lower_evidence"]),
            total_evidence=float(ev["total_evidence"]),
            abs_margin=float(abs(ev["upper_evidence"] - ev["lower_evidence"])),
            rel_margin=float(ev["rel_margin"]),
            ratio=float(ev["ratio"]),
            dominance=float(ev["dominance"]),
            adaptive_score=float(ev["adaptive_score"]),
            upper_seed_x=upper_seed_x,
            upper_seed_y=upper_seed_y,
            lower_seed_x=lower_seed_x,
            lower_seed_y=lower_seed_y,
        )

    t_min = float(ref_t_min_frac) * float(times[-1])
    t_max = float(ref_t_max_frac) * float(times[-1])

    cand_inds = np.where((times >= t_min) & (times <= t_max))[0]
    if len(cand_inds) == 0:
        cand_inds = np.arange(Nt)

    best = None
    for idx in cand_inds:
        ev = compute_trf_corridor_evidence_for_frame(base_rho[idx], masks)
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

    _assert(best is not None, "adaptive branch search failed to produce a best record")

    valid = bool(
        best["total_evidence"] >= float(valid_total_evidence_eps)
        and best["chosen_side"] is not None
    )

    return TrfBranchDecision(
        valid=valid,
        ref_idx=int(best["idx"]),
        ref_time=float(best["time"]),
        chosen_side=best["chosen_side"],
        upper_evidence=float(best["upper_evidence"]),
        lower_evidence=float(best["lower_evidence"]),
        total_evidence=float(best["total_evidence"]),
        abs_margin=float(abs(best["upper_evidence"] - best["lower_evidence"])),
        rel_margin=float(best["rel_margin"]),
        ratio=float(best["ratio"]),
        dominance=float(best["dominance"]),
        adaptive_score=float(best["adaptive_score"]),
        upper_seed_x=upper_seed_x,
        upper_seed_y=upper_seed_y,
        lower_seed_x=lower_seed_x,
        lower_seed_y=lower_seed_y,
    )