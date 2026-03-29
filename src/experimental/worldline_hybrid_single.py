from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


# ====================================================
# CLI
# ====================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--case-name", type=str, default=None)
    p.add_argument(
        "--save",
        action="store_true",
        help="Save debug npz payloads.",
    )
    return p.parse_args()


# ====================================================
# 0) Geometry / grid
# ====================================================
VISIBLE_LX = 40.0
VISIBLE_LY = 20.0
N_VISIBLE_X = 512
N_VISIBLE_Y = 256

PAD_FACTOR = 3

Lx = VISIBLE_LX * PAD_FACTOR
Ly = VISIBLE_LY * PAD_FACTOR
Nx = N_VISIBLE_X * PAD_FACTOR
Ny = N_VISIBLE_Y * PAD_FACTOR

dx = Lx / Nx
dy = Ly / Ny

x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
X, Y = np.meshgrid(x, y)

m_mass = 1.0
hbar = 1.0

cx = Nx // 2
cy = Ny // 2
hx = N_VISIBLE_X // 2
hy = N_VISIBLE_Y // 2

xs = slice(cx - hx, cx + hx)
ys = slice(cy - hy, cy + hy)

X_vis = X[ys, xs]
Y_vis = Y[ys, xs]

kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2


def kinetic_phase(dt: float) -> np.ndarray:
    return np.exp(-1j * K2 * dt / (2 * m_mass))


# ====================================================
# 1) Double slit
# ====================================================
barrier_center_x = 0.0
barrier_thickness = 0.4
V_barrier = 80.0

slit_center_offset = 2.0
slit_half_height = 0.5

BARRIER_SMOOTH = 0.15

barrier_core = np.abs(X - barrier_center_x) < (barrier_thickness / 2.0)
slit1_mask = np.abs(Y - slit_center_offset) < slit_half_height
slit2_mask = np.abs(Y + slit_center_offset) < slit_half_height

V_real = np.zeros_like(X, dtype=float)

if BARRIER_SMOOTH <= 0.0:
    barrier_mask = barrier_core.copy()
    barrier_mask[slit1_mask] = False
    barrier_mask[slit2_mask] = False
    V_real[barrier_mask] = V_barrier
else:
    dist = np.abs(X - barrier_center_x) - (barrier_thickness / 2.0)
    wall = 1.0 / (1.0 + np.exp(dist / BARRIER_SMOOTH))
    wall[slit1_mask] = 0.0
    wall[slit2_mask] = 0.0
    V_real = V_barrier * wall


# ====================================================
# 2) CAP
# ====================================================
def smooth_cap_edge(X, Y, Lx, Ly, cap_width=8.0, strength=2.0, power=4):
    dist_to_x = (Lx / 2) - np.abs(X)
    dist_to_y = (Ly / 2) - np.abs(Y)
    dist_to_edge = np.minimum(dist_to_x, dist_to_y)

    W = np.zeros_like(X, dtype=float)
    mask = dist_to_edge < cap_width
    s = (cap_width - dist_to_edge[mask]) / cap_width
    W[mask] = strength * (s**power)
    return W


CAP_WIDTH = 10.0
CAP_STRENGTH = 2.0
CAP_POWER = 4

W_edge = smooth_cap_edge(
    X, Y, Lx, Ly, cap_width=CAP_WIDTH, strength=CAP_STRENGTH, power=CAP_POWER
)

screen_center_x = 10.0
screen_eval_width = 1.5
screen_mask_full = np.abs(X - screen_center_x) < screen_eval_width
screen_mask_vis = screen_mask_full[ys, xs]

USE_SCREEN_CAP = False
SCREEN_CAP_STRENGTH = 1.5

W_screen = np.zeros_like(X, dtype=float)
if USE_SCREEN_CAP:
    W_screen[screen_mask_full] = SCREEN_CAP_STRENGTH

W = W_edge + W_screen

V_fwd = V_real - 1j * W
V_adj = np.conjugate(V_fwd)


def potential_phase(V: np.ndarray, dt: float) -> np.ndarray:
    return np.exp(-1j * V * dt / hbar)


# ====================================================
# 3) Helpers
# ====================================================
def norm_L2(field: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.abs(field) ** 2) * dx * dy))


def normalize_unit(field: np.ndarray):
    n = norm_L2(field)
    if n <= 0:
        return field, 0.0
    return field / n, n


def expval_xy_unitnorm(psi_unit: np.ndarray):
    p = np.abs(psi_unit) ** 2
    norm = float(np.sum(p) * dx * dy)
    if norm <= 0:
        return 0.0, 0.0
    mx = float(np.sum(p * X) * dx * dy / norm)
    my = float(np.sum(p * Y) * dx * dy / norm)
    return mx, my


def step_field(field: np.ndarray, K_phase: np.ndarray, P_half: np.ndarray) -> np.ndarray:
    if not np.iscomplexobj(field):
        field = field.astype(np.complex128)
    field = field * P_half
    f_k = np.fft.fft2(field)
    f_k = f_k * K_phase
    field = np.fft.ifft2(f_k)
    field = field * P_half
    return field


def extract_local_peak(arr: np.ndarray, iy0: int, ix0: int, radius_px: int = 16):
    ny, nx = arr.shape
    y0 = max(0, iy0 - radius_px)
    y1 = min(ny, iy0 + radius_px + 1)
    x0 = max(0, ix0 - radius_px)
    x1 = min(nx, ix0 + radius_px + 1)
    sub = arr[y0:y1, x0:x1]

    flat_idx = int(np.argmax(sub))
    val = float(sub.ravel()[flat_idx])
    sy, sx = np.unravel_index(flat_idx, sub.shape)
    return val, y0 + sy, x0 + sx


# ====================================================
# 4) Initial state
# ====================================================
def make_packet(x0, y0, sigma0, k0x, k0y):
    XR = X - x0
    YR = Y - y0
    amp = np.exp(-(XR**2 + YR**2) / (2 * sigma0**2))
    phase = np.exp(1j * (k0x * X + k0y * Y))
    return amp * phase


sigma0 = 1.0
k0x = 5.0
k0y = 0.0
x0 = -15.0
y0 = 0.0


# ====================================================
# 5) Time stepping
# ====================================================
dt = 0.003
n_steps = 2200
save_every = 5

K_phase_fwd = kinetic_phase(dt)
K_phase_bwd = kinetic_phase(-dt)

P_half_fwd = potential_phase(V_fwd, dt / 2.0)
P_half_bwd_adj = potential_phase(V_adj, -dt / 2.0)


# ====================================================
# 6) Continuous measurement
# ====================================================
USE_CONTINUOUS_MEAS = True
KAPPA_MEAS = 0.02


def continuous_measurement_update_preserve_norm(psi, dt, kappa, rng):
    if kappa <= 0:
        return psi

    psi_u, n0 = normalize_unit(psi)
    if n0 <= 0:
        return psi

    mx, my = expval_xy_unitnorm(psi_u)
    Xc = (X - mx)
    Yc = (Y - my)

    dWx = rng.normal(0.0, np.sqrt(dt))
    dWy = rng.normal(0.0, np.sqrt(dt))

    drift = -0.5 * kappa * (Xc**2 + Yc**2) * dt
    stoch = np.sqrt(kappa) * (Xc * dWx + Yc * dWy)

    psi_u2 = psi_u * np.exp(drift + stoch)
    psi_u2, _ = normalize_unit(psi_u2)

    return psi_u2 * n0


# ====================================================
# 7) Posthoc / TRF selection + forward-only guess
# ====================================================
USE_POSTHOC_TRF_SELECTION = True

sigma_click = 0.4
TRF_SIGMAT_FRAC = 0.60
TRF_K_JITTER = 13

TRF_REF_T_MIN_FRAC = 0.30
TRF_REF_T_MAX_FRAC = 0.95

TRF_CORRIDOR_X_FRAC_START = 0.70
TRF_CORRIDOR_Y_SIGMA = 1.3
TRF_CORRIDOR_X_WEIGHT_POWER = 2.5

TRF_USE_ADAPTIVE_REF = True
TRF_VALID_TOTAL_EVIDENCE_EPS = 1e-12

USE_POSTHOC_WORLDLINE = True
WL_TRACK_RADIUS_PX = 20
WL_MIN_LOCAL_REL = 0.03
WL_TUBE_SIGMA_PX = 10.0
WL_GAIN_STRENGTH = 2.0
WL_OUTSIDE_DAMP = 0.20
WL_TIME_RAMP_FRAC = 0.12


# ====================================================
# 7b) Pre-slit + slit-pass debug analysis
# ====================================================
USE_PRE_SLIT_DEBUG = True
USE_SLIT_PASS_DEBUG = True

# Pre-slit window: left of barrier
PRE_SLIT_DEBUG_X_SIGMA = 0.60
PRE_SLIT_DEBUG_Y_SIGMA = 0.90
PRE_SLIT_DEBUG_X_CENTER = barrier_center_x - 1.20
PRE_SLIT_DEBUG_T_MIN_FRAC = 0.08
PRE_SLIT_DEBUG_T_MAX_FRAC = 0.55
PRE_SLIT_DEBUG_VALID_TOTAL_EVIDENCE_EPS = 1e-14

# Slit window: near barrier / just after
SLIT_DEBUG_X_SIGMA = 0.45
SLIT_DEBUG_Y_SIGMA = 0.60
SLIT_DEBUG_X_CENTER = barrier_center_x + 0.15
SLIT_DEBUG_T_MIN_FRAC = 0.15
SLIT_DEBUG_T_MAX_FRAC = 0.75
SLIT_DEBUG_VALID_TOTAL_EVIDENCE_EPS = 1e-14


def make_phi_at_click(x_click: float, y_click: float):
    Xc = X - x_click
    Yc = Y - y_click
    phi = np.exp(-(Xc**2 + Yc**2) / (2 * sigma_click**2)).astype(np.complex128)
    phi, _ = normalize_unit(phi)
    return phi


def gaussian_weights(Tk, mu, sigma):
    if sigma <= 0:
        w = np.zeros_like(Tk)
        w[np.argmin(np.abs(Tk - mu))] = 1.0
        return w
    z = (Tk - mu) / sigma
    w = np.exp(-0.5 * z * z)
    s = w.sum()
    return w / s if s > 0 else w


def build_Emix_from_phi_tau(phi_tau_frames, times, t_det, sigmaT, tau_step, K_JITTER=13):
    Nt = len(times)
    halfK = K_JITTER // 2
    idx_det2 = int(np.argmin(np.abs(times - t_det)))

    k_inds = np.arange(idx_det2 - halfK, idx_det2 + halfK + 1)
    k_inds = np.clip(k_inds, 0, Nt - 1)
    k_inds = np.unique(k_inds)

    Tk = times[k_inds]
    w = gaussian_weights(Tk, t_det, sigmaT)

    Emix = np.zeros((Nt, phi_tau_frames.shape[1], phi_tau_frames.shape[2]), dtype=np.float32)

    for i, ti in enumerate(times):
        tau = Tk - ti
        valid = tau >= 0.0
        if not np.any(valid):
            continue

        j = np.rint(tau[valid] / tau_step).astype(int)
        j = np.clip(j, 0, Nt - 1)
        Emix[i] = np.sum((w[valid])[:, None, None] * phi_tau_frames[j], axis=0)

    return Emix


def make_rho(frames_psi, Emix):
    out = np.zeros_like(frames_psi, dtype=np.float32)
    for i in range(frames_psi.shape[0]):
        rho = frames_psi[i] * Emix[i]
        s = float(np.sum(rho) * dx * dy)
        if s > 0:
            rho /= s
        out[i] = rho.astype(np.float32)
    return out


def build_trf_corridor_masks_vis():
    x_left = float(barrier_center_x + 0.5)
    x_right = float(screen_center_x - 0.5)
    x_start = x_left + float(TRF_CORRIDOR_X_FRAC_START) * (x_right - x_left)

    x_u = np.clip((X_vis - x_start) / max(x_right - x_start, 1e-12), 0.0, 1.0)
    x_weight = np.power(x_u, float(TRF_CORRIDOR_X_WEIGHT_POWER)).astype(np.float32)
    end_mask = (X_vis >= x_start).astype(np.float32)

    upper_band = np.exp(-0.5 * ((Y_vis - slit_center_offset) / max(TRF_CORRIDOR_Y_SIGMA, 1e-12)) ** 2)
    lower_band = np.exp(-0.5 * ((Y_vis + slit_center_offset) / max(TRF_CORRIDOR_Y_SIGMA, 1e-12)) ** 2)

    upper = (upper_band * x_weight * end_mask).astype(np.float32)
    lower = (lower_band * x_weight * end_mask).astype(np.float32)
    return upper, lower, float(x_start)


TRF_UPPER_CORRIDOR_VIS, TRF_LOWER_CORRIDOR_VIS, TRF_CORRIDOR_X_START = build_trf_corridor_masks_vis()


def build_side_debug_masks_vis(x_center: float, x_sigma: float, y_sigma: float):
    xg = np.exp(
        -0.5 * ((X_vis - float(x_center)) / max(float(x_sigma), 1e-12)) ** 2
    ).astype(np.float32)

    upper_y = np.exp(
        -0.5 * ((Y_vis - float(slit_center_offset)) / max(float(y_sigma), 1e-12)) ** 2
    ).astype(np.float32)

    lower_y = np.exp(
        -0.5 * ((Y_vis + float(slit_center_offset)) / max(float(y_sigma), 1e-12)) ** 2
    ).astype(np.float32)

    upper = (xg * upper_y).astype(np.float32)
    lower = (xg * lower_y).astype(np.float32)
    return upper, lower


PRE_SLIT_UPPER_MASK_VIS, PRE_SLIT_LOWER_MASK_VIS = build_side_debug_masks_vis(
    x_center=PRE_SLIT_DEBUG_X_CENTER,
    x_sigma=PRE_SLIT_DEBUG_X_SIGMA,
    y_sigma=PRE_SLIT_DEBUG_Y_SIGMA,
)

SLIT_UPPER_MASK_VIS, SLIT_LOWER_MASK_VIS = build_side_debug_masks_vis(
    x_center=SLIT_DEBUG_X_CENTER,
    x_sigma=SLIT_DEBUG_X_SIGMA,
    y_sigma=SLIT_DEBUG_Y_SIGMA,
)


def compute_corridor_evidence_for_frame(frame_2d: np.ndarray):
    upper_ev = float(np.sum(frame_2d * TRF_UPPER_CORRIDOR_VIS))
    lower_ev = float(np.sum(frame_2d * TRF_LOWER_CORRIDOR_VIS))
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
    abs_margin = float(abs(upper_ev - lower_ev))

    return {
        "upper_evidence": float(upper_ev),
        "lower_evidence": float(lower_ev),
        "total_evidence": float(total_ev),
        "dominance": float(dominance),
        "rel_margin": float(rel_margin),
        "ratio": float(ratio),
        "abs_margin": float(abs_margin),
        "chosen_side": chosen_side,
        "adaptive_score": float(adaptive_score),
    }


def choose_branch_adaptive_from_frames(frames_2d: np.ndarray, times: np.ndarray):
    Nt = len(times)

    t_min = float(TRF_REF_T_MIN_FRAC * times[-1])
    t_max = float(TRF_REF_T_MAX_FRAC * times[-1])

    cand_inds = np.where((times >= t_min) & (times <= t_max))[0]
    if len(cand_inds) == 0:
        cand_inds = np.arange(Nt)

    best = None
    for idx in cand_inds:
        ev = compute_corridor_evidence_for_frame(frames_2d[idx])
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

    upper_seed_x = float(0.5 * (TRF_CORRIDOR_X_START + (screen_center_x - 0.5)))
    upper_seed_y = float(slit_center_offset)
    lower_seed_x = float(0.5 * (TRF_CORRIDOR_X_START + (screen_center_x - 0.5)))
    lower_seed_y = float(-slit_center_offset)

    valid = bool(best["total_evidence"] >= TRF_VALID_TOTAL_EVIDENCE_EPS and best["chosen_side"] is not None)

    return {
        "valid": valid,
        "ref_idx": int(best["idx"]),
        "ref_time": float(best["time"]),
        "chosen_side": best["chosen_side"],
        "upper_evidence": float(best["upper_evidence"]),
        "lower_evidence": float(best["lower_evidence"]),
        "total_evidence": float(best["total_evidence"]),
        "abs_margin": float(best["abs_margin"]),
        "rel_margin": float(best["rel_margin"]),
        "ratio": float(best["ratio"]),
        "dominance": float(best["dominance"]),
        "adaptive_score": float(best["adaptive_score"]),
        "upper_seed_x": upper_seed_x,
        "upper_seed_y": upper_seed_y,
        "lower_seed_x": lower_seed_x,
        "lower_seed_y": lower_seed_y,
    }


def compute_side_evidence_series(frames_2d: np.ndarray, upper_mask: np.ndarray, lower_mask: np.ndarray):
    Nt = frames_2d.shape[0]
    upper = np.zeros(Nt, dtype=np.float64)
    lower = np.zeros(Nt, dtype=np.float64)
    total = np.zeros(Nt, dtype=np.float64)

    for i in range(Nt):
        u = float(np.sum(frames_2d[i] * upper_mask))
        l = float(np.sum(frames_2d[i] * lower_mask))
        upper[i] = u
        lower[i] = l
        total[i] = u + l

    return upper, lower, total


def choose_side_pass_from_frames(
    frames_2d: np.ndarray,
    times: np.ndarray,
    upper_mask: np.ndarray,
    lower_mask: np.ndarray,
    t_min_frac: float,
    t_max_frac: float,
    valid_total_eps: float,
):
    upper_series, lower_series, total_series = compute_side_evidence_series(frames_2d, upper_mask, lower_mask)

    t_min = float(t_min_frac * times[-1])
    t_max = float(t_max_frac * times[-1])

    cand_inds = np.where((times >= t_min) & (times <= t_max))[0]
    if len(cand_inds) == 0:
        cand_inds = np.arange(len(times))

    best = None
    for idx in cand_inds:
        u = float(upper_series[idx])
        l = float(lower_series[idx])
        total = float(total_series[idx])

        if total <= 0.0:
            chosen_side = None
            dominance = 0.0
            rel_margin = 0.0
            ratio = 0.0
            abs_margin = 0.0
            score = 0.0
        else:
            chosen_side = "upper" if u >= l else "lower"
            dominance = float(max(u, l) / max(total, 1e-12))
            rel_margin = float(abs(u - l) / max(max(u, l), 1e-12))
            ratio = float(max(u, l) / max(min(u, l), 1e-12))
            abs_margin = float(abs(u - l))
            score = float(total * dominance)

        rec = {
            "ref_idx": int(idx),
            "ref_time": float(times[idx]),
            "upper_evidence": u,
            "lower_evidence": l,
            "total_evidence": total,
            "abs_margin": abs_margin,
            "rel_margin": rel_margin,
            "ratio": ratio,
            "dominance": dominance,
            "score": score,
            "chosen_side": chosen_side,
        }

        if best is None:
            best = rec
            continue

        if rec["score"] > best["score"]:
            best = rec
        elif rec["score"] == best["score"] and rec["dominance"] > best["dominance"]:
            best = rec

    valid = bool(
        best["total_evidence"] >= valid_total_eps
        and best["chosen_side"] is not None
    )

    return {
        "valid": valid,
        "ref_idx": int(best["ref_idx"]),
        "ref_time": float(best["ref_time"]),
        "chosen_side": best["chosen_side"],
        "upper_evidence": float(best["upper_evidence"]),
        "lower_evidence": float(best["lower_evidence"]),
        "total_evidence": float(best["total_evidence"]),
        "abs_margin": float(best["abs_margin"]),
        "rel_margin": float(best["rel_margin"]),
        "ratio": float(best["ratio"]),
        "dominance": float(best["dominance"]),
        "score": float(best["score"]),
        "upper_series": upper_series,
        "lower_series": lower_series,
        "total_series": total_series,
    }


def choose_pre_slit_pass_from_frames(frames_2d: np.ndarray, times: np.ndarray):
    return choose_side_pass_from_frames(
        frames_2d=frames_2d,
        times=times,
        upper_mask=PRE_SLIT_UPPER_MASK_VIS,
        lower_mask=PRE_SLIT_LOWER_MASK_VIS,
        t_min_frac=PRE_SLIT_DEBUG_T_MIN_FRAC,
        t_max_frac=PRE_SLIT_DEBUG_T_MAX_FRAC,
        valid_total_eps=PRE_SLIT_DEBUG_VALID_TOTAL_EVIDENCE_EPS,
    )


def choose_slit_pass_from_frames(frames_2d: np.ndarray, times: np.ndarray):
    return choose_side_pass_from_frames(
        frames_2d=frames_2d,
        times=times,
        upper_mask=SLIT_UPPER_MASK_VIS,
        lower_mask=SLIT_LOWER_MASK_VIS,
        t_min_frac=SLIT_DEBUG_T_MIN_FRAC,
        t_max_frac=SLIT_DEBUG_T_MAX_FRAC,
        valid_total_eps=SLIT_DEBUG_VALID_TOTAL_EVIDENCE_EPS,
    )


def track_worldline_from_seed(rho_frames, start_i, start_iy, start_ix, radius_px=16, min_local_rel=0.02):
    Nt, _, _ = rho_frames.shape
    path_y = np.full(Nt, start_iy, dtype=int)
    path_x = np.full(Nt, start_ix, dtype=int)

    ref_amp = float(max(rho_frames[start_i, start_iy, start_ix], 1e-30))

    iy, ix = start_iy, start_ix
    for i in range(start_i + 1, Nt):
        val, iy_new, ix_new = extract_local_peak(rho_frames[i], iy, ix, radius_px=radius_px)
        if val >= min_local_rel * ref_amp:
            iy, ix = iy_new, ix_new
        path_y[i] = iy
        path_x[i] = ix

    iy, ix = start_iy, start_ix
    for i in range(start_i - 1, -1, -1):
        val, iy_new, ix_new = extract_local_peak(rho_frames[i], iy, ix, radius_px=radius_px)
        if val >= min_local_rel * ref_amp:
            iy, ix = iy_new, ix_new
        path_y[i] = iy
        path_x[i] = ix

    return path_y, path_x


def build_worldline_tube(shape, path_y, path_x, sigma_px):
    Nt, ny, nx = shape
    yy = np.arange(ny)[:, None]
    xx = np.arange(nx)[None, :]

    tube = np.zeros(shape, dtype=np.float32)
    inv2s2 = 1.0 / max(2.0 * sigma_px * sigma_px, 1e-12)

    for i in range(Nt):
        dy2 = (yy - path_y[i]) ** 2
        dx2 = (xx - path_x[i]) ** 2
        tube[i] = np.exp(-(dy2 + dx2) * inv2s2).astype(np.float32)

    return tube


def smooth_time_ramp(Nt, center_idx, ramp_frac=0.1):
    ramp_len = max(3, int(ramp_frac * Nt))
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


def apply_worldline_selection(rho_frames, tube, gain_strength=2.0, outside_damp=0.2, time_gate=None):
    out = np.zeros_like(rho_frames, dtype=np.float32)

    if time_gate is None:
        time_gate = np.ones(rho_frames.shape[0], dtype=np.float32)

    for i in range(rho_frames.shape[0]):
        g = float(time_gate[i])
        field = gain_strength * g * tube[i] - outside_damp * g * (1.0 - tube[i])
        rho = rho_frames[i] * np.exp(field)
        s = float(np.sum(rho) * dx * dy)
        if s > 0:
            rho /= s
        out[i] = rho.astype(np.float32)

    return out


def compute_posthoc_worldline_selected_rho(base_rho, trf_info):
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
        radius_px=WL_TRACK_RADIUS_PX,
        min_local_rel=WL_MIN_LOCAL_REL,
    )

    tube = build_worldline_tube(base_rho.shape, path_y, path_x, sigma_px=WL_TUBE_SIGMA_PX)
    time_gate = smooth_time_ramp(base_rho.shape[0], idx_ref, ramp_frac=WL_TIME_RAMP_FRAC)

    rho_wl = apply_worldline_selection(
        base_rho,
        tube,
        gain_strength=WL_GAIN_STRENGTH,
        outside_damp=WL_OUTSIDE_DAMP,
        time_gate=time_gate,
    )

    return rho_wl, {
        "seed_x": float(seed_x),
        "seed_y": float(seed_y),
        "seed_side": trf_info["chosen_side"],
    }


# ====================================================
# 8) Main run
# ====================================================
def main() -> int:
    args = parse_args()
    seed = int(args.seed)

    case_name = args.case_name or f"seed_{seed}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / f"{case_name}.log"
    summary_path = out_dir / f"{case_name}_summary.json"

    t0 = time.time()

    def log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"CASE {case_name}\n")
        f.write(f"SEED {seed}\n\n")

    rng_meas = np.random.default_rng(seed)
    rng_click = np.random.default_rng(seed + 10_000)

    psi = make_packet(x0, y0, sigma0, k0x, k0y).astype(np.complex128)
    psi, _ = normalize_unit(psi)

    frames_psi = []
    times_arr = []
    norms_psi = []

    psi_cur = psi.copy()

    log(f"[START] case={case_name} seed={seed}")

    for n in range(n_steps + 1):
        prob_psi = np.abs(psi_cur) ** 2
        norm_now = float(np.sum(prob_psi) * dx * dy)

        if n % save_every == 0:
            frames_psi.append(prob_psi[ys, xs].astype(np.float32, copy=True))
            times_arr.append(n * dt)
            norms_psi.append(norm_now)

        if n < n_steps:
            psi_cur = step_field(psi_cur, K_phase_fwd, P_half_fwd)

            if USE_CONTINUOUS_MEAS:
                psi_cur = continuous_measurement_update_preserve_norm(
                    psi_cur, dt, KAPPA_MEAS, rng_meas
                )

    frames_psi = np.array(frames_psi, dtype=np.float32)
    times_arr = np.array(times_arr, dtype=float)
    norms_psi = np.array(norms_psi, dtype=float)
    Nt = len(times_arr)

    # Forward-only guess before any click-conditioned backward construction
    forward_guess_info = choose_branch_adaptive_from_frames(frames_psi, times_arr)
    log(
        f"[FWD_GUESS] valid={forward_guess_info['valid']} "
        f"chosen={forward_guess_info['chosen_side']} "
        f"upper_ev={forward_guess_info['upper_evidence']:.6e} "
        f"lower_ev={forward_guess_info['lower_evidence']:.6e} "
        f"total_ev={forward_guess_info['total_evidence']:.6e} "
        f"abs_margin={forward_guess_info['abs_margin']:.6e} "
        f"rel_margin={forward_guess_info['rel_margin']:.6f} "
        f"dominance={forward_guess_info['dominance']:.6f} "
        f"ratio={forward_guess_info['ratio']:.6f} "
        f"adaptive_score={forward_guess_info['adaptive_score']:.6e} "
        f"ref_idx={forward_guess_info['ref_idx']} "
        f"ref_time={forward_guess_info['ref_time']:.6f}"
    )

    # Pre-slit debug guess from forward density before the barrier
    pre_slit_info = None
    pre_slit_upper_series = None
    pre_slit_lower_series = None
    pre_slit_total_series = None

    if USE_PRE_SLIT_DEBUG:
        pre_slit_info = choose_pre_slit_pass_from_frames(frames_psi, times_arr)
        pre_slit_upper_series = pre_slit_info.pop("upper_series")
        pre_slit_lower_series = pre_slit_info.pop("lower_series")
        pre_slit_total_series = pre_slit_info.pop("total_series")

        log(
            f"[PRE_SLIT] valid={pre_slit_info['valid']} "
            f"chosen={pre_slit_info['chosen_side']} "
            f"upper_ev={pre_slit_info['upper_evidence']:.6e} "
            f"lower_ev={pre_slit_info['lower_evidence']:.6e} "
            f"total_ev={pre_slit_info['total_evidence']:.6e} "
            f"abs_margin={pre_slit_info['abs_margin']:.6e} "
            f"rel_margin={pre_slit_info['rel_margin']:.6f} "
            f"dominance={pre_slit_info['dominance']:.6f} "
            f"ratio={pre_slit_info['ratio']:.6f} "
            f"score={pre_slit_info['score']:.6e} "
            f"ref_idx={pre_slit_info['ref_idx']} "
            f"ref_time={pre_slit_info['ref_time']:.6f}"
        )

    # Slit-pass debug guess from forward density near the barrier/slits
    slit_pass_info = None
    slit_upper_series = None
    slit_lower_series = None
    slit_total_series = None

    if USE_SLIT_PASS_DEBUG:
        slit_pass_info = choose_slit_pass_from_frames(frames_psi, times_arr)
        slit_upper_series = slit_pass_info.pop("upper_series")
        slit_lower_series = slit_pass_info.pop("lower_series")
        slit_total_series = slit_pass_info.pop("total_series")

        log(
            f"[SLIT_PASS] valid={slit_pass_info['valid']} "
            f"chosen={slit_pass_info['chosen_side']} "
            f"upper_ev={slit_pass_info['upper_evidence']:.6e} "
            f"lower_ev={slit_pass_info['lower_evidence']:.6e} "
            f"total_ev={slit_pass_info['total_evidence']:.6e} "
            f"abs_margin={slit_pass_info['abs_margin']:.6e} "
            f"rel_margin={slit_pass_info['rel_margin']:.6f} "
            f"dominance={slit_pass_info['dominance']:.6f} "
            f"ratio={slit_pass_info['ratio']:.6f} "
            f"score={slit_pass_info['score']:.6e} "
            f"ref_idx={slit_pass_info['ref_idx']} "
            f"ref_time={slit_pass_info['ref_time']:.6f}"
        )

    screen_int = np.array(
        [np.sum(frames_psi[i][screen_mask_vis]) * dx * dy for i in range(Nt)],
        dtype=float,
    )
    idx_det = int(np.argmax(screen_int))
    t_det = float(times_arr[idx_det])

    w = frames_psi[idx_det].copy()
    w = np.where(screen_mask_vis, w, 0.0)
    wsum = float(np.sum(w))
    if wsum <= 0:
        raise RuntimeError("No detector intensity at t_det")

    p = (w / wsum).ravel()
    flat_idx = int(rng_click.choice(p.size, p=p))
    iy_vis, ix_vis = np.unravel_index(flat_idx, w.shape)

    iy_click = (cy - hy) + iy_vis
    ix_click = (cx - hx) + ix_vis
    x_click = float(X[iy_click, ix_click])
    y_click = float(Y[iy_click, ix_click])
    click_side = "upper" if y_click > 0 else "lower"

    log(f"[CLICK] t={t_det:.6f}, x={x_click:.6f}, y={y_click:.6f}, side={click_side}")

    phi_cur = make_phi_at_click(x_click, y_click)
    phi_tau_frames = np.zeros((Nt, N_VISIBLE_Y, N_VISIBLE_X), dtype=np.float32)
    tau_step = save_every * dt

    for i in range(Nt):
        phi_tau_frames[i] = (np.abs(phi_cur) ** 2)[ys, xs].astype(np.float32)
        if i < Nt - 1:
            for _ in range(save_every):
                phi_cur = step_field(phi_cur, K_phase_bwd, P_half_bwd_adj)

    v_est = k0x / m_mass
    L_gap = screen_center_x - barrier_center_x
    t_gap = L_gap / v_est
    sigmaT = TRF_SIGMAT_FRAC * t_gap

    Emix = build_Emix_from_phi_tau(
        phi_tau_frames,
        times_arr,
        t_det,
        sigmaT=sigmaT,
        tau_step=tau_step,
        K_JITTER=TRF_K_JITTER,
    )
    base_rho = make_rho(frames_psi, Emix)

    trf_info = None
    if USE_POSTHOC_TRF_SELECTION:
        trf_info = choose_branch_adaptive_from_frames(base_rho, times_arr)
        log(
            f"[TRF] valid={trf_info['valid']} "
            f"chosen={trf_info['chosen_side']} "
            f"upper_ev={trf_info['upper_evidence']:.6e} "
            f"lower_ev={trf_info['lower_evidence']:.6e} "
            f"total_ev={trf_info['total_evidence']:.6e} "
            f"abs_margin={trf_info['abs_margin']:.6e} "
            f"rel_margin={trf_info['rel_margin']:.6f} "
            f"dominance={trf_info['dominance']:.6f} "
            f"ratio={trf_info['ratio']:.6f} "
            f"adaptive_score={trf_info['adaptive_score']:.6e} "
            f"ref_idx={trf_info['ref_idx']} "
            f"ref_time={trf_info['ref_time']:.6f}"
        )

    forward_vs_trf_different = False
    forward_vs_click_different = False
    trf_vs_click_match = False
    interesting_forward_trf_click_case = False

    if forward_guess_info["chosen_side"] is not None and trf_info is not None and trf_info["chosen_side"] is not None:
        forward_vs_trf_different = bool(forward_guess_info["chosen_side"] != trf_info["chosen_side"])
        if forward_vs_trf_different:
            log(
                f"[DIFF] seed={seed} "
                f"forward={forward_guess_info['chosen_side']} "
                f"trf={trf_info['chosen_side']} "
                f"click={click_side} "
                f"fwd_ref_t={forward_guess_info['ref_time']:.6f} "
                f"trf_ref_t={trf_info['ref_time']:.6f}"
            )

    if forward_guess_info["chosen_side"] is not None:
        forward_vs_click_different = bool(forward_guess_info["chosen_side"] != click_side)

    if trf_info is not None and trf_info["chosen_side"] is not None:
        trf_vs_click_match = bool(trf_info["chosen_side"] == click_side)

    if forward_vs_trf_different and trf_vs_click_match:
        interesting_forward_trf_click_case = True
        log(
            f"[INTERESTING] seed={seed} "
            f"forward={forward_guess_info['chosen_side']} "
            f"trf={trf_info['chosen_side']} "
            f"click={click_side} "
            f"fwd_dom={forward_guess_info['dominance']:.6f} "
            f"trf_dom={trf_info['dominance']:.6f} "
            f"fwd_ref_t={forward_guess_info['ref_time']:.6f} "
            f"trf_ref_t={trf_info['ref_time']:.6f}"
        )

    # Pre-slit vs click
    pre_slit_vs_click_different = False
    pre_slit_to_click_transition = None
    interesting_pre_slit_click_case = False

    if pre_slit_info is not None and pre_slit_info["chosen_side"] is not None:
        pre_slit_vs_click_different = bool(pre_slit_info["chosen_side"] != click_side)
        pre_slit_to_click_transition = f"{pre_slit_info['chosen_side']}_pre->{click_side}_click"

        if pre_slit_vs_click_different:
            interesting_pre_slit_click_case = True
            log(
                f"[PRE_CLICK_DIFF] seed={seed} "
                f"pre_slit={pre_slit_info['chosen_side']} "
                f"click={click_side} "
                f"pre_ref_t={pre_slit_info['ref_time']:.6f} "
                f"pre_dom={pre_slit_info['dominance']:.6f}"
            )
        else:
            log(
                f"[PRE_CLICK_MATCH] seed={seed} "
                f"pre_slit={pre_slit_info['chosen_side']} "
                f"click={click_side} "
                f"pre_ref_t={pre_slit_info['ref_time']:.6f} "
                f"pre_dom={pre_slit_info['dominance']:.6f}"
            )

    # Slit-pass vs click
    slit_vs_click_different = False
    slit_to_click_transition = None
    interesting_slit_click_case = False

    if slit_pass_info is not None and slit_pass_info["chosen_side"] is not None:
        slit_vs_click_different = bool(slit_pass_info["chosen_side"] != click_side)
        slit_to_click_transition = f"{slit_pass_info['chosen_side']}_slit->{click_side}_click"
        if slit_vs_click_different:
            interesting_slit_click_case = True
            log(
                f"[SLIT_CLICK_DIFF] seed={seed} "
                f"slit_pass={slit_pass_info['chosen_side']} "
                f"click={click_side} "
                f"slit_ref_t={slit_pass_info['ref_time']:.6f} "
                f"slit_dom={slit_pass_info['dominance']:.6f}"
            )
        else:
            log(
                f"[SLIT_CLICK_MATCH] seed={seed} "
                f"slit_pass={slit_pass_info['chosen_side']} "
                f"click={click_side} "
                f"slit_ref_t={slit_pass_info['ref_time']:.6f} "
                f"slit_dom={slit_pass_info['dominance']:.6f}"
            )

    # Combined pre+slit vs click
    interesting_clean_path_mismatch_case = False
    clean_path_transition = None
    if (
        pre_slit_info is not None
        and slit_pass_info is not None
        and pre_slit_info["chosen_side"] in {"upper", "lower"}
        and slit_pass_info["chosen_side"] in {"upper", "lower"}
    ):
        if (
            pre_slit_info["chosen_side"] == slit_pass_info["chosen_side"]
            and slit_pass_info["chosen_side"] != click_side
        ):
            interesting_clean_path_mismatch_case = True
            clean_path_transition = (
                f"{pre_slit_info['chosen_side']}_pre"
                f"->{slit_pass_info['chosen_side']}_slit"
                f"->{click_side}_click"
            )
            log(
                f"[CLEAN_PATH_MISMATCH] seed={seed} "
                f"pre_slit={pre_slit_info['chosen_side']} "
                f"slit_pass={slit_pass_info['chosen_side']} "
                f"click={click_side} "
                f"pre_dom={pre_slit_info['dominance']:.6f} "
                f"slit_dom={slit_pass_info['dominance']:.6f}"
            )

    wl_info = None
    if USE_POSTHOC_WORLDLINE and trf_info is not None and trf_info["valid"]:
        _, wl_info = compute_posthoc_worldline_selected_rho(base_rho, trf_info)

    result = {
        "case_name": case_name,
        "seed": int(seed),
        "returncode": 0,
        "elapsed_sec": float(time.time() - t0),

        "clicked": True,
        "click_time": float(t_det),
        "click_x": float(x_click),
        "click_y": float(y_click),
        "click_side": click_side,

        "trf_use_adaptive_ref": bool(TRF_USE_ADAPTIVE_REF),
        "trf_ref_t_min_frac": float(TRF_REF_T_MIN_FRAC),
        "trf_ref_t_max_frac": float(TRF_REF_T_MAX_FRAC),
        "trf_sigmat_frac": float(TRF_SIGMAT_FRAC),
        "trf_k_jitter": int(TRF_K_JITTER),

        "trf_corridor_x_frac_start": float(TRF_CORRIDOR_X_FRAC_START),
        "trf_corridor_y_sigma": float(TRF_CORRIDOR_Y_SIGMA),
        "trf_corridor_x_weight_power": float(TRF_CORRIDOR_X_WEIGHT_POWER),

        # forward-only guess
        "forward_guess_valid": bool(forward_guess_info["valid"]),
        "forward_guess_ref_idx": int(forward_guess_info["ref_idx"]),
        "forward_guess_ref_time": float(forward_guess_info["ref_time"]),
        "forward_guess_chosen_side": forward_guess_info["chosen_side"],
        "forward_guess_upper_evidence": float(forward_guess_info["upper_evidence"]),
        "forward_guess_lower_evidence": float(forward_guess_info["lower_evidence"]),
        "forward_guess_total_evidence": float(forward_guess_info["total_evidence"]),
        "forward_guess_abs_margin": float(forward_guess_info["abs_margin"]),
        "forward_guess_rel_margin": float(forward_guess_info["rel_margin"]),
        "forward_guess_ratio": float(forward_guess_info["ratio"]),
        "forward_guess_dominance": float(forward_guess_info["dominance"]),
        "forward_guess_adaptive_score": float(forward_guess_info["adaptive_score"]),

        # pre-slit debug
        "pre_slit_debug_used": bool(USE_PRE_SLIT_DEBUG),
        "pre_slit_valid": None if pre_slit_info is None else bool(pre_slit_info["valid"]),
        "pre_slit_ref_idx": None if pre_slit_info is None else int(pre_slit_info["ref_idx"]),
        "pre_slit_ref_time": None if pre_slit_info is None else float(pre_slit_info["ref_time"]),
        "pre_slit_chosen_side": None if pre_slit_info is None else pre_slit_info["chosen_side"],
        "pre_slit_upper_evidence": None if pre_slit_info is None else float(pre_slit_info["upper_evidence"]),
        "pre_slit_lower_evidence": None if pre_slit_info is None else float(pre_slit_info["lower_evidence"]),
        "pre_slit_total_evidence": None if pre_slit_info is None else float(pre_slit_info["total_evidence"]),
        "pre_slit_abs_margin": None if pre_slit_info is None else float(pre_slit_info["abs_margin"]),
        "pre_slit_rel_margin": None if pre_slit_info is None else float(pre_slit_info["rel_margin"]),
        "pre_slit_ratio": None if pre_slit_info is None else float(pre_slit_info["ratio"]),
        "pre_slit_dominance": None if pre_slit_info is None else float(pre_slit_info["dominance"]),
        "pre_slit_score": None if pre_slit_info is None else float(pre_slit_info["score"]),

        # slit-pass debug
        "slit_pass_debug_used": bool(USE_SLIT_PASS_DEBUG),
        "slit_pass_valid": None if slit_pass_info is None else bool(slit_pass_info["valid"]),
        "slit_pass_ref_idx": None if slit_pass_info is None else int(slit_pass_info["ref_idx"]),
        "slit_pass_ref_time": None if slit_pass_info is None else float(slit_pass_info["ref_time"]),
        "slit_pass_chosen_side": None if slit_pass_info is None else slit_pass_info["chosen_side"],
        "slit_pass_upper_evidence": None if slit_pass_info is None else float(slit_pass_info["upper_evidence"]),
        "slit_pass_lower_evidence": None if slit_pass_info is None else float(slit_pass_info["lower_evidence"]),
        "slit_pass_total_evidence": None if slit_pass_info is None else float(slit_pass_info["total_evidence"]),
        "slit_pass_abs_margin": None if slit_pass_info is None else float(slit_pass_info["abs_margin"]),
        "slit_pass_rel_margin": None if slit_pass_info is None else float(slit_pass_info["rel_margin"]),
        "slit_pass_ratio": None if slit_pass_info is None else float(slit_pass_info["ratio"]),
        "slit_pass_dominance": None if slit_pass_info is None else float(slit_pass_info["dominance"]),
        "slit_pass_score": None if slit_pass_info is None else float(slit_pass_info["score"]),

        # posthoc TRF
        "posthoc_trf_valid": None if trf_info is None else bool(trf_info["valid"]),
        "posthoc_trf_ref_idx": None if trf_info is None else int(trf_info["ref_idx"]),
        "posthoc_trf_ref_time": None if trf_info is None else float(trf_info["ref_time"]),
        "posthoc_trf_chosen_side": None if trf_info is None else trf_info["chosen_side"],
        "posthoc_trf_upper_evidence": None if trf_info is None else float(trf_info["upper_evidence"]),
        "posthoc_trf_lower_evidence": None if trf_info is None else float(trf_info["lower_evidence"]),
        "posthoc_trf_total_evidence": None if trf_info is None else float(trf_info["total_evidence"]),
        "posthoc_trf_abs_margin": None if trf_info is None else float(trf_info["abs_margin"]),
        "posthoc_trf_rel_margin": None if trf_info is None else float(trf_info["rel_margin"]),
        "posthoc_trf_ratio": None if trf_info is None else float(trf_info["ratio"]),
        "posthoc_trf_dominance": None if trf_info is None else float(trf_info["dominance"]),
        "posthoc_trf_adaptive_score": None if trf_info is None else float(trf_info["adaptive_score"]),
        "posthoc_trf_upper_seed_x": None if trf_info is None else float(trf_info["upper_seed_x"]),
        "posthoc_trf_upper_seed_y": None if trf_info is None else float(trf_info["upper_seed_y"]),
        "posthoc_trf_lower_seed_x": None if trf_info is None else float(trf_info["lower_seed_x"]),
        "posthoc_trf_lower_seed_y": None if trf_info is None else float(trf_info["lower_seed_y"]),

        # compatibility alias
        "posthoc_chosen_side": None if trf_info is None else trf_info["chosen_side"],

        # comparisons
        "forward_vs_trf_different": bool(forward_vs_trf_different),
        "forward_vs_click_different": bool(forward_vs_click_different),
        "trf_vs_click_match": bool(trf_vs_click_match),
        "interesting_forward_trf_click_case": bool(interesting_forward_trf_click_case),

        "pre_slit_vs_click_different": bool(pre_slit_vs_click_different),
        "pre_slit_to_click_transition": pre_slit_to_click_transition,
        "interesting_pre_slit_click_case": bool(interesting_pre_slit_click_case),

        "slit_vs_click_different": bool(slit_vs_click_different),
        "slit_to_click_transition": slit_to_click_transition,
        "interesting_slit_click_case": bool(interesting_slit_click_case),

        "interesting_clean_path_mismatch_case": bool(interesting_clean_path_mismatch_case),
        "clean_path_transition": clean_path_transition,

        "posthoc_worldline_used": bool(USE_POSTHOC_WORLDLINE and trf_info is not None and trf_info["valid"]),
        "posthoc_worldline_seed_side": None if wl_info is None else wl_info["seed_side"],
        "posthoc_worldline_seed_x": None if wl_info is None else float(wl_info["seed_x"]),
        "posthoc_worldline_seed_y": None if wl_info is None else float(wl_info["seed_y"]),

        "norm_final": float(norms_psi[-1]),
        "norm_min": float(np.min(norms_psi)),
        "norm_max": float(np.max(norms_psi)),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log(f"[DONE] case={case_name} elapsed={result['elapsed_sec']:.2f}s")

    if args.save:
        debug_npz_path = out_dir / f"{case_name}_debug.npz"
        np.savez_compressed(
            debug_npz_path,
            frames_psi=frames_psi,
            base_rho=base_rho,
            times_arr=times_arr,
            screen_int=screen_int,
            screen_mask_vis=screen_mask_vis.astype(np.uint8),

            pre_slit_upper_mask_vis=PRE_SLIT_UPPER_MASK_VIS.astype(np.float32),
            pre_slit_lower_mask_vis=PRE_SLIT_LOWER_MASK_VIS.astype(np.float32),
            pre_slit_upper_series=np.array([] if pre_slit_upper_series is None else pre_slit_upper_series, dtype=np.float64),
            pre_slit_lower_series=np.array([] if pre_slit_lower_series is None else pre_slit_lower_series, dtype=np.float64),
            pre_slit_total_series=np.array([] if pre_slit_total_series is None else pre_slit_total_series, dtype=np.float64),

            slit_upper_mask_vis=SLIT_UPPER_MASK_VIS.astype(np.float32),
            slit_lower_mask_vis=SLIT_LOWER_MASK_VIS.astype(np.float32),
            slit_upper_series=np.array([] if slit_upper_series is None else slit_upper_series, dtype=np.float64),
            slit_lower_series=np.array([] if slit_lower_series is None else slit_lower_series, dtype=np.float64),
            slit_total_series=np.array([] if slit_total_series is None else slit_total_series, dtype=np.float64),
        )
        log(f"[DEBUG_SAVE] saved={debug_npz_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())