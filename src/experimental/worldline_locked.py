import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# ====================================================
# 0) Alustus: näkyvä alue + padding
# ====================================================
VISIBLE_LX   = 40.0
VISIBLE_LY   = 20.0
N_VISIBLE_X  = 512
N_VISIBLE_Y  = 256

PAD_FACTOR   = 3

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
hbar   = 1.0

cx = Nx // 2
cy = Ny // 2
hx = N_VISIBLE_X // 2
hy = N_VISIBLE_Y // 2

xs = slice(cx - hx, cx + hx)
ys = slice(cy - hy, cy + hy)

X_vis = X[ys, xs]
Y_vis = Y[ys, xs]

x_vis_1d = X_vis[0, :]
y_vis_1d = Y_vis[:, 0]

extent = (-VISIBLE_LX / 2, VISIBLE_LX / 2, -VISIBLE_LY / 2, VISIBLE_LY / 2)

# Fourier-hilat
kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2

def kinetic_phase(dt):
    return np.exp(-1j * K2 * dt / (2 * m_mass))

# ====================================================
# 1) Kaksoisrako
# ====================================================
barrier_center_x  = 0.0
barrier_thickness = 0.4
V_barrier         = 80.0

slit_center_offset = 2.0
slit_half_height   = 0.5

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
# 2) CAP reunoille + ruutualue
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

CAP_WIDTH    = 10.0
CAP_STRENGTH = 2.0
CAP_POWER    = 4

W_edge = smooth_cap_edge(
    X, Y, Lx, Ly,
    cap_width=CAP_WIDTH,
    strength=CAP_STRENGTH,
    power=CAP_POWER
)

screen_center_x   = 10.0
screen_eval_width = 1.5
screen_mask_full  = np.abs(X - screen_center_x) < screen_eval_width
screen_mask_vis   = screen_mask_full[ys, xs]

USE_SCREEN_CAP = False
SCREEN_CAP_STRENGTH = 1.5

W_screen = np.zeros_like(X, dtype=float)
if USE_SCREEN_CAP:
    W_screen[screen_mask_full] = SCREEN_CAP_STRENGTH

W = W_edge + W_screen

V_fwd = V_real - 1j * W
V_adj = np.conjugate(V_fwd)

def potential_phase(V, dt):
    return np.exp(-1j * V * dt / hbar)

# ====================================================
# 3) Normalisointi
# ====================================================
def norm_L2(field):
    return float(np.sqrt(np.sum(np.abs(field)**2) * dx * dy))

def normalize_unit(field):
    n = norm_L2(field)
    if n <= 0:
        return field, 0.0
    return field / n, n

def expval_xy_unitnorm(psi_unit):
    p = np.abs(psi_unit)**2
    norm = float(np.sum(p) * dx * dy)
    if norm <= 0:
        return 0.0, 0.0
    mx = float(np.sum(p * X) * dx * dy / norm)
    my = float(np.sum(p * Y) * dx * dy / norm)
    return mx, my

# ====================================================
# 4) Jatkuva mittaus
# ====================================================
USE_CONTINUOUS_MEAS = True
KAPPA_MEAS = 0.02
rng_meas = np.random.default_rng(1234)

def continuous_measurement_update_preserve_norm(psi, dt, kappa, rng):
    if kappa <= 0:
        return psi, (0.0, 0.0, 0.0, 0.0)

    psi_u, n0 = normalize_unit(psi)
    if n0 <= 0:
        return psi, (0.0, 0.0, 0.0, 0.0)

    mx, my = expval_xy_unitnorm(psi_u)
    Xc = (X - mx)
    Yc = (Y - my)

    dWx = rng.normal(0.0, np.sqrt(dt))
    dWy = rng.normal(0.0, np.sqrt(dt))

    drift = -0.5 * kappa * (Xc**2 + Yc**2) * dt
    stoch = np.sqrt(kappa) * (Xc * dWx + Yc * dWy)

    psi_u2 = psi_u * np.exp(drift + stoch)
    psi_u2, _ = normalize_unit(psi_u2)

    return psi_u2 * n0, (dWx, dWy, mx, my)

# ====================================================
# 5) Alkuarvo ψ
# ====================================================
def make_packet(x0, y0, sigma0, k0x, k0y):
    XR = X - x0
    YR = Y - y0
    amp   = np.exp(-(XR**2 + YR**2) / (2 * sigma0**2))
    phase = np.exp(1j * (k0x * X + k0y * Y))
    return amp * phase

sigma0 = 1.0
k0x    = 5.0
k0y    = 0.0
x0     = -15.0
y0     = 0.0

psi = make_packet(x0, y0, sigma0, k0x, k0y).astype(np.complex128)
psi, _ = normalize_unit(psi)

# ====================================================
# 6) Split-operator askel
# ====================================================
dt         = 0.003
n_steps    = 2200
save_every = 5

K_phase_fwd = kinetic_phase(dt)
K_phase_bwd = kinetic_phase(-dt)

P_half_fwd     = potential_phase(V_fwd,  dt / 2.0)
P_half_bwd_adj = potential_phase(V_adj, -dt / 2.0)

def step_field(field, K_phase, P_half):
    if not np.iscomplexobj(field):
        field = field.astype(np.complex128)
    field = field * P_half
    f_k = np.fft.fft2(field)
    f_k = f_k * K_phase
    field = np.fft.ifft2(f_k)
    field = field * P_half
    return field

# ====================================================
# 6.5) BF / counterfactual parametrit
# ====================================================
USE_BRIDGE_FRONT_BIAS = True
BF_BRANCH_POLICY = "opposite_of_natural_post_branch"

PRE_X_MIN  = -5.0
PRE_X_MAX  = -1.8
POST_X_MIN =  0.8
POST_X_MAX =  5.5

PRE_Y_SIGN  = "upper"
PRE_Y_MIN_ABS = 1.0
POST_Y_MIN_ABS = 1.0

BRIDGE_PRE_MIN_PEAK  = 5e-4
BRIDGE_POST_MIN_PEAK = 1e-4

BRANCH_DECISION_MARGIN = 1.30

v_est = k0x / m_mass
t_barrier_est = (barrier_center_x - x0) / max(v_est, 1e-12)
BRIDGE_EARLIEST_TIME = t_barrier_est - 0.60
BRIDGE_REBUILD_EVERY_STEPS = 60

TUBE_SIGMA_LONG  = 2.2
TUBE_SIGMA_TRANS = 0.55
TUBE_AHEAD_SHIFT = 0.10

TUBE_GAIN = 1.60
TUBE_DAMP = 0.12
TUBE_RAMP_TIME = 0.25

DP_Y_WINDOW_HALF = 18
DP_STEP_Y_PENALTY = 0.90
DP_CURVE_PENALTY  = 0.30
DP_BRIGHTNESS_WEIGHT = 1.0
DP_LOG_EPS = 1e-14

ANCHOR_LOG_EPS = 1e-14
ANCHOR_BRIGHTNESS_WEIGHT = 1.0
ANCHOR_PRE_Y_BONUS  = 1.10
ANCHOR_POST_Y_BONUS = 0.90
ANCHOR_X_CENTER_BONUS = 0.15
ANCHOR_SOFT_POWER = 1.0

# Pre-bias
USE_PRE_BRANCH_BIAS = True
PRE_BIAS_START_TIME = t_barrier_est - 1.20
PRE_BIAS_RAMP_TIME  = 0.50

PRE_BIAS_GAIN = 1.15
PRE_BIAS_DAMP_CENTER = 0.14

PRE_BIAS_X_MIN = -6.0
PRE_BIAS_X_MAX = -0.5

PRE_BIAS_TARGET = "upper"
PRE_BIAS_Y_CENTER = 2.0
PRE_BIAS_SIGMA_Y = 0.7
PRE_BIAS_SIGMA_X = 2.2

# Branch mass evaluation
POST_BRANCH_Y_CUT = 1.0

# UUSI: pre-anchor lukitusikkuna activationin jälkeen
LOCK_PRE_ANCHOR_WINDOW = True
PRE_LOCK_RADIUS_X = 1.0
PRE_LOCK_RADIUS_Y = 0.8

# Debug
PRINT_BRIDGE_DEBUG = True
PRINT_READINESS_DEBUG = True
READINESS_DEBUG_EVERY_STEPS = 25

# ====================================================
# 6.6) Apufunktiot
# ====================================================
def ramp01(t, t0, tau):
    if tau <= 0:
        return 1.0 if t >= t0 else 0.0
    z = (t - t0) / tau
    if z <= 0.0:
        return 0.0
    if z >= 1.0:
        return 1.0
    return 0.5 - 0.5 * np.cos(np.pi * z)

def build_anchor_masks(post_branch_choice="lower"):
    pre_x_mask = (X_vis >= PRE_X_MIN) & (X_vis <= PRE_X_MAX)
    post_x_mask = (X_vis >= POST_X_MIN) & (X_vis <= POST_X_MAX)

    if PRE_Y_SIGN == "upper":
        pre_y_mask = Y_vis >= PRE_Y_MIN_ABS
    else:
        pre_y_mask = Y_vis <= -PRE_Y_MIN_ABS

    if post_branch_choice == "upper":
        post_y_mask = Y_vis >= POST_Y_MIN_ABS
    else:
        post_y_mask = Y_vis <= -POST_Y_MIN_ABS

    pre_mask = pre_x_mask & pre_y_mask
    post_mask = post_x_mask & post_y_mask
    return pre_mask, post_mask

def build_locked_pre_window_mask(center_x, center_y):
    if center_x is None or center_y is None:
        return np.zeros_like(X_vis, dtype=bool)

    return (
        (X_vis >= center_x - PRE_LOCK_RADIUS_X) &
        (X_vis <= center_x + PRE_LOCK_RADIUS_X) &
        (Y_vis >= center_y - PRE_LOCK_RADIUS_Y) &
        (Y_vis <= center_y + PRE_LOCK_RADIUS_Y)
    )

def anchor_score_field(
    prob_vis,
    region="pre",
    post_branch_choice="lower",
    locked_pre_center=None,
):
    logp = np.log(prob_vis + ANCHOR_LOG_EPS)

    if region == "pre":
        x_mid = 0.5 * (PRE_X_MIN + PRE_X_MAX)
        x_span = max(PRE_X_MAX - PRE_X_MIN, 1e-12)
        x_pref = 1.0 - np.abs((X_vis - x_mid) / (0.5 * x_span))
        x_pref = np.clip(x_pref, 0.0, 1.0)

        if PRE_Y_SIGN == "upper":
            y_pref = np.clip(
                (Y_vis - PRE_Y_MIN_ABS) / max(VISIBLE_LY / 2 - PRE_Y_MIN_ABS, 1e-12),
                0.0, 1.0
            )
        else:
            y_pref = np.clip(
                (-Y_vis - PRE_Y_MIN_ABS) / max(VISIBLE_LY / 2 - PRE_Y_MIN_ABS, 1e-12),
                0.0, 1.0
            )

        score = (
            ANCHOR_BRIGHTNESS_WEIGHT * logp
            + ANCHOR_PRE_Y_BONUS * (y_pref ** ANCHOR_SOFT_POWER)
            + ANCHOR_X_CENTER_BONUS * x_pref
        )

        pre_mask, _ = build_anchor_masks(post_branch_choice=post_branch_choice)

        # UUSI: jos pre-anchor on lukittu, rajoita haku ikkunaan
        if LOCK_PRE_ANCHOR_WINDOW and (locked_pre_center is not None):
            cx, cy = locked_pre_center
            win_mask = build_locked_pre_window_mask(cx, cy)
            pre_mask = pre_mask & win_mask

        score = np.where(pre_mask, score, -np.inf)
        return score

    x_mid = 0.5 * (POST_X_MIN + POST_X_MAX)
    x_span = max(POST_X_MAX - POST_X_MIN, 1e-12)
    x_pref = 1.0 - np.abs((X_vis - x_mid) / (0.5 * x_span))
    x_pref = np.clip(x_pref, 0.0, 1.0)

    if post_branch_choice == "upper":
        y_pref = np.clip(
            (Y_vis - POST_Y_MIN_ABS) / max(VISIBLE_LY / 2 - POST_Y_MIN_ABS, 1e-12),
            0.0, 1.0
        )
    else:
        y_pref = np.clip(
            (-Y_vis - POST_Y_MIN_ABS) / max(VISIBLE_LY / 2 - POST_Y_MIN_ABS, 1e-12),
            0.0, 1.0
        )

    score = (
        ANCHOR_BRIGHTNESS_WEIGHT * logp
        + ANCHOR_POST_Y_BONUS * (y_pref ** ANCHOR_SOFT_POWER)
        + ANCHOR_X_CENTER_BONUS * x_pref
    )

    _, post_mask = build_anchor_masks(post_branch_choice=post_branch_choice)
    score = np.where(post_mask, score, -np.inf)
    return score

def best_anchor_by_score(
    prob_vis,
    region="pre",
    post_branch_choice="lower",
    locked_pre_center=None,
):
    score = anchor_score_field(
        prob_vis,
        region=region,
        post_branch_choice=post_branch_choice,
        locked_pre_center=locked_pre_center,
    )
    flat = int(np.argmax(score))
    best_score = float(score.ravel()[flat])
    if not np.isfinite(best_score):
        return None

    iy, ix = np.unravel_index(flat, score.shape)
    prob_val = float(prob_vis[iy, ix])
    return {
        "prob": prob_val,
        "score": best_score,
        "iy": iy,
        "ix": ix,
    }

def compute_post_branch_masses(prob_vis):
    post_x_mask = (X_vis >= POST_X_MIN) & (X_vis <= POST_X_MAX)
    upper_mask = post_x_mask & (Y_vis >= POST_BRANCH_Y_CUT)
    lower_mask = post_x_mask & (Y_vis <= -POST_BRANCH_Y_CUT)

    upper_mass = float(np.sum(prob_vis[upper_mask]) * dx * dy)
    lower_mass = float(np.sum(prob_vis[lower_mask]) * dx * dy)
    return upper_mass, lower_mass

def decide_locked_counterfactual_branch(prob_vis):
    upper_mass, lower_mass = compute_post_branch_masses(prob_vis)

    max_mass = max(upper_mass, lower_mass)
    min_mass = min(upper_mass, lower_mass)

    if min_mass <= 0.0:
        ratio = np.inf if max_mass > 0 else 1.0
    else:
        ratio = max_mass / min_mass

    decision_ready = ratio >= BRANCH_DECISION_MARGIN

    natural = "upper" if upper_mass >= lower_mass else "lower"

    if BF_BRANCH_POLICY == "opposite_of_natural_post_branch":
        chosen = "lower" if natural == "upper" else "upper"
    elif BF_BRANCH_POLICY == "weaker_post_branch":
        chosen = "upper" if upper_mass <= lower_mass else "lower"
    else:
        chosen = natural

    return {
        "upper_mass": upper_mass,
        "lower_mass": lower_mass,
        "ratio": ratio,
        "decision_ready": decision_ready,
        "natural_post_branch": natural,
        "chosen_post_branch": chosen,
    }

def solve_bridge_path_dp(prob_vis, start_iy, start_ix, end_iy, end_ix):
    if end_ix <= start_ix:
        return None

    ixs = np.arange(start_ix, end_ix + 1)
    nxp = len(ixs)
    ny = prob_vis.shape[0]

    score = np.full((nxp, ny), -np.inf, dtype=float)
    parent = np.full((nxp, ny), -1, dtype=int)
    prev_dy_store = np.zeros((nxp, ny), dtype=int)

    score[0, start_iy] = DP_BRIGHTNESS_WEIGHT * np.log(prob_vis[start_iy, start_ix] + DP_LOG_EPS)

    for k in range(1, nxp):
        ix = ixs[k]

        for iy in range(ny):
            local_bright = DP_BRIGHTNESS_WEIGHT * np.log(prob_vis[iy, ix] + DP_LOG_EPS)

            y0 = max(0, iy - DP_Y_WINDOW_HALF)
            y1 = min(ny, iy + DP_Y_WINDOW_HALF + 1)

            best_val = -np.inf
            best_py = -1
            best_prev_dy = 0

            for py in range(y0, y1):
                if not np.isfinite(score[k - 1, py]):
                    continue

                dy_step = iy - py
                step_cost = DP_STEP_Y_PENALTY * (dy_step * dy_step)

                prev_prev_dy = prev_dy_store[k - 1, py]
                curve_cost = DP_CURVE_PENALTY * ((dy_step - prev_prev_dy) ** 2)

                cand = score[k - 1, py] + local_bright - step_cost - curve_cost
                if cand > best_val:
                    best_val = cand
                    best_py = py
                    best_prev_dy = dy_step

            score[k, iy] = best_val
            parent[k, iy] = best_py
            prev_dy_store[k, iy] = best_prev_dy

    if not np.isfinite(score[-1, end_iy]):
        return None

    path_y = np.zeros(nxp, dtype=int)
    path_y[-1] = end_iy

    for k in range(nxp - 1, 0, -1):
        py = parent[k, path_y[k]]
        if py < 0:
            return None
        path_y[k - 1] = py

    path_x = ixs.copy()
    return path_y, path_x

def extend_path_piecewise(prob_vis, anchor_left, anchor_right):
    iy0, ix0 = anchor_left["iy"], anchor_left["ix"]
    iy1, ix1 = anchor_right["iy"], anchor_right["ix"]

    if ix1 <= ix0:
        return None

    solved = solve_bridge_path_dp(prob_vis, iy0, ix0, iy1, ix1)
    if solved is None:
        return None

    path_y, path_x = solved
    return path_y, path_x

def path_to_direction_fields(path_x, path_y):
    n = len(path_x)
    vx = np.zeros(n, dtype=float)
    vy = np.zeros(n, dtype=float)

    for i in range(n):
        if i == 0:
            dxv = x_vis_1d[path_x[min(i + 1, n - 1)]] - x_vis_1d[path_x[i]]
            dyv = y_vis_1d[path_y[min(i + 1, n - 1)]] - y_vis_1d[path_y[i]]
        elif i == n - 1:
            dxv = x_vis_1d[path_x[i]] - x_vis_1d[path_x[i - 1]]
            dyv = y_vis_1d[path_y[i]] - y_vis_1d[path_y[i - 1]]
        else:
            dxv = x_vis_1d[path_x[i + 1]] - x_vis_1d[path_x[i - 1]]
            dyv = y_vis_1d[path_y[i + 1]] - y_vis_1d[path_y[i - 1]]

        vx[i] = dxv
        vy[i] = dyv

    return vx, vy

def build_tube_from_bridge(path_x, path_y, sigma_long, sigma_trans, ahead_shift):
    tube = np.zeros_like(X_vis, dtype=float)
    vx_path, vy_path = path_to_direction_fields(path_x, path_y)

    for px, py, vx, vy in zip(path_x, path_y, vx_path, vy_path):
        xc = x_vis_1d[px]
        yc = y_vis_1d[py]

        vnorm = np.hypot(vx, vy)
        if vnorm < 1e-12:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = vx / vnorm, vy / vnorm

        nxp, nyp = -uy, ux

        xc2 = xc + ahead_shift * ux
        yc2 = yc + ahead_shift * uy

        dxg = X_vis - xc2
        dyg = Y_vis - yc2

        s_long  = dxg * ux  + dyg * uy
        s_trans = dxg * nxp + dyg * nyp

        blob = np.exp(
            -0.5 * (s_long / sigma_long) ** 2
            -0.5 * (s_trans / sigma_trans) ** 2
        )
        tube = np.maximum(tube, blob)

    return tube

def expand_visible_tube_to_full_grid(tube_vis):
    full = np.zeros_like(X, dtype=float)
    full[ys, xs] = tube_vis
    return full

def apply_tube_bias_preserve_norm(psi, tube_full, dt, gain, damp, ramp_scale):
    if ramp_scale <= 0:
        return psi

    n0 = norm_L2(psi)
    if n0 <= 0:
        return psi

    field = gain * tube_full - damp * (1.0 - tube_full)
    psi2 = psi * np.exp(dt * ramp_scale * field)

    psi2_u, _ = normalize_unit(psi2)
    return psi2_u * n0

# ====================================================
# 6.7) Pre-bias kenttä
# ====================================================
def smooth_box_x(Xv, x_min, x_max, smooth=0.5):
    left = 1.0 / (1.0 + np.exp(-(Xv - x_min) / smooth))
    right = 1.0 / (1.0 + np.exp((Xv - x_max) / smooth))
    return left * right

def build_pre_branch_bias_field():
    x_gate = smooth_box_x(X, PRE_BIAS_X_MIN, PRE_BIAS_X_MAX, smooth=0.5)

    if PRE_BIAS_TARGET == "upper":
        y0 = PRE_BIAS_Y_CENTER
    else:
        y0 = -PRE_BIAS_Y_CENTER

    target = np.exp(-0.5 * ((Y - y0) / PRE_BIAS_SIGMA_Y) ** 2)
    target *= np.exp(-0.5 * ((X - 0.5 * (PRE_BIAS_X_MIN + PRE_BIAS_X_MAX)) / PRE_BIAS_SIGMA_X) ** 2)

    center_damp = np.exp(-0.5 * (Y / 0.9) ** 2)

    field = PRE_BIAS_GAIN * x_gate * target - PRE_BIAS_DAMP_CENTER * x_gate * center_damp
    return field

def apply_scalar_bias_preserve_norm(psi, bias_field, dt, ramp_scale):
    if ramp_scale <= 0:
        return psi

    n0 = norm_L2(psi)
    if n0 <= 0:
        return psi

    psi2 = psi * np.exp(dt * ramp_scale * bias_field)
    psi2_u, _ = normalize_unit(psi2)
    return psi2_u * n0

pre_branch_bias_field = build_pre_branch_bias_field() if USE_PRE_BRANCH_BIAS else None

# ====================================================
# 6.8) BF readiness / build
# ====================================================
def evaluate_bridge_readiness(
    prob_vis,
    locked_chosen_post_branch=None,
    locked_natural_post_branch=None,
    locked_pre_center=None,
):
    decision = decide_locked_counterfactual_branch(prob_vis)

    if locked_chosen_post_branch is not None:
        chosen_post_branch = locked_chosen_post_branch
        natural_post_branch = locked_natural_post_branch
        decision_ready = True
    else:
        chosen_post_branch = decision["chosen_post_branch"]
        natural_post_branch = decision["natural_post_branch"]
        decision_ready = decision["decision_ready"]

    pre_anchor = best_anchor_by_score(
        prob_vis,
        region="pre",
        post_branch_choice=chosen_post_branch,
        locked_pre_center=locked_pre_center,
    )
    post_anchor = best_anchor_by_score(
        prob_vis,
        region="post",
        post_branch_choice=chosen_post_branch,
        locked_pre_center=None,
    )

    pre_val = pre_anchor["prob"] if pre_anchor is not None else 0.0
    post_val = post_anchor["prob"] if post_anchor is not None else 0.0

    ready = (
        decision_ready and
        (pre_anchor is not None) and
        (post_anchor is not None) and
        (pre_val >= BRIDGE_PRE_MIN_PEAK) and
        (post_val >= BRIDGE_POST_MIN_PEAK)
    )

    return {
        "ready": ready,
        "decision_ready": decision_ready,
        "pre_anchor": pre_anchor,
        "post_anchor": post_anchor,
        "pre_val": pre_val,
        "post_val": post_val,
        "natural_post_branch": natural_post_branch,
        "chosen_post_branch": chosen_post_branch,
        "upper_mass": decision["upper_mass"],
        "lower_mass": decision["lower_mass"],
        "ratio": decision["ratio"],
    }

def build_bridge_front(prob_vis, readiness=None):
    if readiness is None:
        readiness = evaluate_bridge_readiness(prob_vis)

    pre_anchor = readiness["pre_anchor"]
    post_anchor = readiness["post_anchor"]

    if pre_anchor is None or post_anchor is None:
        return None

    bridge = extend_path_piecewise(prob_vis, pre_anchor, post_anchor)
    if bridge is None:
        return None

    path_y, path_x = bridge
    tube_vis = build_tube_from_bridge(
        path_x, path_y,
        sigma_long=TUBE_SIGMA_LONG,
        sigma_trans=TUBE_SIGMA_TRANS,
        ahead_shift=TUBE_AHEAD_SHIFT
    )
    tube_full = expand_visible_tube_to_full_grid(tube_vis)

    return {
        "pre_anchor": pre_anchor,
        "post_anchor": post_anchor,
        "path_y": path_y,
        "path_x": path_x,
        "tube_vis": tube_vis,
        "tube_full": tube_full,
        "natural_post_branch": readiness["natural_post_branch"],
        "chosen_post_branch": readiness["chosen_post_branch"],
        "upper_mass": readiness["upper_mass"],
        "lower_mass": readiness["lower_mass"],
        "ratio": readiness["ratio"],
    }

# ====================================================
# 6.9) Bridge-tila
# ====================================================
bridge_state = {
    "enabled": USE_BRIDGE_FRONT_BIAS,
    "active": False,
    "last_build_step": None,
    "activation_time": None,
    "front": None,
    "locked_natural_post_branch": None,
    "locked_chosen_post_branch": None,
    "decision_time": None,
    "locked_pre_anchor_x": None,
    "locked_pre_anchor_y": None,
}

# ====================================================
# 7) Forward
# ====================================================
frames_psi = []
times      = []
norms_psi  = []

bridge_overlay_path_x = []
bridge_overlay_path_y = []
bridge_overlay_pre_x  = []
bridge_overlay_pre_y  = []
bridge_overlay_post_x = []
bridge_overlay_post_y = []
bridge_overlay_active = []
bridge_overlay_nat_branch = []
bridge_overlay_chosen_branch = []

psi_cur = psi.copy()

print("Forward: simulaatio alkaa...")
if USE_CONTINUOUS_MEAS:
    print("  - continuous measurement ON")
if USE_PRE_BRANCH_BIAS:
    print("  - pre-branch bias ON")
    print(f"  - pre-bias start = {PRE_BIAS_START_TIME:.3f}")
if USE_BRIDGE_FRONT_BIAS:
    print("  - bridge-front tube bias ON")
    print(f"  - branch policy = {BF_BRANCH_POLICY}")
    print(f"  - earliest check time = {BRIDGE_EARLIEST_TIME:.3f}")
    print(f"  - branch decision margin = {BRANCH_DECISION_MARGIN:.2f}")
    print(f"  - pre/post thresholds = {BRIDGE_PRE_MIN_PEAK:.2e} / {BRIDGE_POST_MIN_PEAK:.2e}")

for n in range(n_steps + 1):
    t_now = n * dt
    prob_psi = np.abs(psi_cur)**2
    norm_now = float(np.sum(prob_psi) * dx * dy)

    if n % save_every == 0:
        frames_psi.append(prob_psi[ys, xs].copy())
        times.append(t_now)
        norms_psi.append(norm_now)

        if bridge_state["active"] and bridge_state["front"] is not None:
            front = bridge_state["front"]
            bridge_overlay_path_x.append(front["path_x"].copy())
            bridge_overlay_path_y.append(front["path_y"].copy())

            pre = front["pre_anchor"]
            post = front["post_anchor"]
            bridge_overlay_pre_x.append(x_vis_1d[pre["ix"]])
            bridge_overlay_pre_y.append(y_vis_1d[pre["iy"]])
            bridge_overlay_post_x.append(x_vis_1d[post["ix"]])
            bridge_overlay_post_y.append(y_vis_1d[post["iy"]])
            bridge_overlay_active.append(True)
            bridge_overlay_nat_branch.append(front["natural_post_branch"])
            bridge_overlay_chosen_branch.append(front["chosen_post_branch"])
        else:
            bridge_overlay_path_x.append(None)
            bridge_overlay_path_y.append(None)
            bridge_overlay_pre_x.append(np.nan)
            bridge_overlay_pre_y.append(np.nan)
            bridge_overlay_post_x.append(np.nan)
            bridge_overlay_post_y.append(np.nan)
            bridge_overlay_active.append(False)
            bridge_overlay_nat_branch.append("searching")
            bridge_overlay_chosen_branch.append("searching")

        if (len(frames_psi) % 20) == 0:
            status = "bridge_on" if bridge_state["active"] else "searching"
            print(f"[FWD] step {n:5d}/{n_steps}, t={t_now:7.3f}, norm≈{norm_now:.6f}, BF={status}")

    if n < n_steps:
        psi_cur = step_field(psi_cur, K_phase_fwd, P_half_fwd)

        if USE_CONTINUOUS_MEAS:
            psi_cur, _rec = continuous_measurement_update_preserve_norm(
                psi_cur, dt, KAPPA_MEAS, rng_meas
            )

        # pre-bias ennen BF-aktivointia
        if USE_PRE_BRANCH_BIAS and (not bridge_state["active"]):
            pre_ramp = ramp01(t_now, PRE_BIAS_START_TIME, PRE_BIAS_RAMP_TIME)
            psi_cur = apply_scalar_bias_preserve_norm(
                psi_cur,
                pre_branch_bias_field,
                dt=dt,
                ramp_scale=pre_ramp
            )

        # BF readiness
        if bridge_state["enabled"] and t_now >= BRIDGE_EARLIEST_TIME:
            prob_vis_now = (np.abs(psi_cur) ** 2)[ys, xs]

            locked_pre_center = None
            if (bridge_state["locked_pre_anchor_x"] is not None) and (bridge_state["locked_pre_anchor_y"] is not None):
                locked_pre_center = (
                    bridge_state["locked_pre_anchor_x"],
                    bridge_state["locked_pre_anchor_y"],
                )

            readiness = evaluate_bridge_readiness(
                prob_vis_now,
                locked_chosen_post_branch=bridge_state["locked_chosen_post_branch"],
                locked_natural_post_branch=bridge_state["locked_natural_post_branch"],
                locked_pre_center=locked_pre_center,
            )

            # Lukitse counterfactual-haara vain kerran, kun ero on riittävä
            if bridge_state["locked_chosen_post_branch"] is None and readiness["decision_ready"]:
                bridge_state["locked_natural_post_branch"] = readiness["natural_post_branch"]
                bridge_state["locked_chosen_post_branch"] = readiness["chosen_post_branch"]
                bridge_state["decision_time"] = t_now
                print(
                    f"[BF-LOCK] t={t_now:.3f}, "
                    f"upper_mass={readiness['upper_mass']:.6e}, "
                    f"lower_mass={readiness['lower_mass']:.6e}, "
                    f"ratio={readiness['ratio']:.3f}, "
                    f"locked natural={bridge_state['locked_natural_post_branch']}, "
                    f"locked chosen={bridge_state['locked_chosen_post_branch']}"
                )

                # evaluoi uudestaan lukitulla branchilla
                readiness = evaluate_bridge_readiness(
                    prob_vis_now,
                    locked_chosen_post_branch=bridge_state["locked_chosen_post_branch"],
                    locked_natural_post_branch=bridge_state["locked_natural_post_branch"],
                    locked_pre_center=None,
                )

            # Kun first valid front löytyy, lukitse myös pre-anchor center
            if (
                bridge_state["locked_chosen_post_branch"] is not None and
                bridge_state["locked_pre_anchor_x"] is None and
                readiness["pre_anchor"] is not None
            ):
                pre0 = readiness["pre_anchor"]
                bridge_state["locked_pre_anchor_x"] = float(x_vis_1d[pre0["ix"]])
                bridge_state["locked_pre_anchor_y"] = float(y_vis_1d[pre0["iy"]])
                print(
                    f"[BF-PRE-LOCK] t={t_now:.3f}, "
                    f"locked pre center x={bridge_state['locked_pre_anchor_x']:.3f}, "
                    f"y={bridge_state['locked_pre_anchor_y']:.3f}, "
                    f"window=({PRE_LOCK_RADIUS_X:.2f}, {PRE_LOCK_RADIUS_Y:.2f})"
                )

                # evaluoi vielä kerran lukitulla pre-ikkunalla
                readiness = evaluate_bridge_readiness(
                    prob_vis_now,
                    locked_chosen_post_branch=bridge_state["locked_chosen_post_branch"],
                    locked_natural_post_branch=bridge_state["locked_natural_post_branch"],
                    locked_pre_center=(
                        bridge_state["locked_pre_anchor_x"],
                        bridge_state["locked_pre_anchor_y"],
                    ),
                )

            if PRINT_READINESS_DEBUG and (n % READINESS_DEBUG_EVERY_STEPS == 0) and (not bridge_state["active"]):
                print(
                    f"[BF-READY] t={t_now:.3f}, "
                    f"upper_mass={readiness['upper_mass']:.6e}, "
                    f"lower_mass={readiness['lower_mass']:.6e}, "
                    f"ratio={readiness['ratio']:.3f}, "
                    f"decision_ready={readiness['decision_ready']}, "
                    f"natural={readiness['natural_post_branch']}, "
                    f"chosen={readiness['chosen_post_branch']}, "
                    f"pre={readiness['pre_val']:.6e}, "
                    f"post={readiness['post_val']:.6e}, "
                    f"ready={readiness['ready']}"
                )

            must_build = (
                readiness["ready"] and (
                    (not bridge_state["active"]) or
                    (bridge_state["last_build_step"] is None) or
                    ((n - bridge_state["last_build_step"]) >= BRIDGE_REBUILD_EVERY_STEPS)
                )
            )

            if must_build:
                front = build_bridge_front(prob_vis_now, readiness=readiness)

                if front is not None:
                    was_inactive = not bridge_state["active"]

                    bridge_state["active"] = True
                    bridge_state["front"] = front
                    bridge_state["last_build_step"] = n
                    if was_inactive:
                        bridge_state["activation_time"] = t_now

                    if PRINT_BRIDGE_DEBUG:
                        pre = front["pre_anchor"]
                        post = front["post_anchor"]
                        print("[BRIDGE] built/updated front")
                        print(
                            f"  locked natural post branch={front['natural_post_branch']}, "
                            f"locked chosen opposite={front['chosen_post_branch']}"
                        )
                        print(
                            f"  upper_mass={front['upper_mass']:.6e}, "
                            f"lower_mass={front['lower_mass']:.6e}, "
                            f"ratio={front['ratio']:.3f}"
                        )
                        print(
                            f"  locked pre center=({bridge_state['locked_pre_anchor_x']:.3f}, "
                            f"{bridge_state['locked_pre_anchor_y']:.3f})"
                        )
                        print(
                            f"  pre anchor : prob={pre['prob']:.6e}, score={pre['score']:.6e}, "
                            f"x={x_vis_1d[pre['ix']]:.3f}, y={y_vis_1d[pre['iy']]:.3f}"
                        )
                        print(
                            f"  post anchor: prob={post['prob']:.6e}, score={post['score']:.6e}, "
                            f"x={x_vis_1d[post['ix']]:.3f}, y={y_vis_1d[post['iy']]:.3f}"
                        )
                        print(
                            f"  path len={len(front['path_x'])}, "
                            f"x-range=({x_vis_1d[front['path_x'][0]]:.3f} -> {x_vis_1d[front['path_x'][-1]]:.3f})"
                        )

        # bridge tube bias
        if bridge_state["active"] and bridge_state["front"] is not None:
            activation_time = bridge_state["activation_time"]
            ramp_scale = ramp01(t_now, activation_time, TUBE_RAMP_TIME)
            psi_cur = apply_tube_bias_preserve_norm(
                psi_cur,
                bridge_state["front"]["tube_full"],
                dt=dt,
                gain=TUBE_GAIN,
                damp=TUBE_DAMP,
                ramp_scale=ramp_scale
            )

frames_psi = np.array(frames_psi)
times      = np.array(times)
norms_psi  = np.array(norms_psi)
Nt = len(times)

print("Forward valmis.")

# ====================================================
# 8) Detektioajan arvio + Born-otanta klikille
# ====================================================
if not np.any(screen_mask_vis):
    raise RuntimeError("screen_mask_vis tyhjä: tarkista ruudun parametrit.")

screen_int = np.array([
    np.sum(frames_psi[i][screen_mask_vis]) * dx * dy
    for i in range(Nt)
])

idx_det = int(np.argmax(screen_int))
t_det = float(times[idx_det])
print(f"t_det≈{t_det:.3f} (screen_int max≈{screen_int[idx_det]:.3e})")

w = frames_psi[idx_det].copy()
w = np.where(screen_mask_vis, w, 0.0)
wsum = float(np.sum(w))
if wsum <= 0:
    raise RuntimeError("Ruudulla ei ole intensiteettiä t_det:llä.")

p = (w / wsum).ravel()
rng = np.random.default_rng()

flat_idx = int(rng.choice(p.size, p=p))
iy_vis_click, ix_vis_click = np.unravel_index(flat_idx, w.shape)

iy_click = (cy - hy) + iy_vis_click
ix_click = (cx - hx) + ix_vis_click
x_click = float(X[iy_click, ix_click])
y_click = float(Y[iy_click, ix_click])

print(f"Click: x≈{x_click:.3f}, y≈{y_click:.3f}")

# ====================================================
# 9) Backward library
# ====================================================
sigma_click = 0.4

def make_phi_at_click():
    Xc = X - x_click
    Yc = Y - y_click
    phi = np.exp(-(Xc**2 + Yc**2) / (2 * sigma_click**2)).astype(np.complex128)
    phi, _ = normalize_unit(phi)
    return phi

print("Backward library: lasketaan phi_tau...")

phi_cur = make_phi_at_click()
phi_tau_frames = np.zeros((Nt, N_VISIBLE_Y, N_VISIBLE_X), dtype=float)
tau_step = save_every * dt
print_every_frames = 20

for i in range(Nt):
    phi_tau_frames[i] = (np.abs(phi_cur) ** 2)[ys, xs]

    if (i % print_every_frames) == 0 or i == Nt - 1:
        tau = i * tau_step
        norm_phi = float(np.sum(np.abs(phi_cur) ** 2) * dx * dy)
        print(f"[BWD] frame {i:4d}/{Nt-1}, tau={tau:7.3f}, norm≈{norm_phi:.6f}")

    if i < Nt - 1:
        for _ in range(save_every):
            phi_cur = step_field(phi_cur, K_phase_bwd, P_half_bwd_adj)

print("phi_tau-kirjasto valmis.")

# ====================================================
# 10) Emix ja rho
# ====================================================
def gaussian_weights(Tk, mu, sigma):
    if sigma <= 0:
        w = np.zeros_like(Tk)
        w[np.argmin(np.abs(Tk - mu))] = 1.0
        return w
    z = (Tk - mu) / sigma
    w = np.exp(-0.5 * z * z)
    s = w.sum()
    return w / s if s > 0 else w

def build_Emix_from_phi_tau(phi_tau_frames, times, t_det, sigmaT, K_JITTER=13):
    Nt_local = len(times)
    halfK = K_JITTER // 2
    idx_det2 = int(np.argmin(np.abs(times - t_det)))

    k_inds = np.arange(idx_det2 - halfK, idx_det2 + halfK + 1)
    k_inds = np.clip(k_inds, 0, Nt_local - 1)
    k_inds = np.unique(k_inds)

    Tk = times[k_inds]
    w = gaussian_weights(Tk, t_det, sigmaT)

    Emix = np.zeros((Nt_local, phi_tau_frames.shape[1], phi_tau_frames.shape[2]), dtype=float)

    for i, ti in enumerate(times):
        tau = Tk - ti
        valid = tau >= 0.0
        if not np.any(valid):
            continue

        j = np.rint(tau[valid] / tau_step).astype(int)
        j = np.clip(j, 0, Nt_local - 1)

        Emix[i] = np.sum((w[valid])[:, None, None] * phi_tau_frames[j], axis=0)

    return Emix

def make_rho(frames_psi, Emix):
    out = np.zeros_like(frames_psi, dtype=float)
    for i in range(frames_psi.shape[0]):
        rho = frames_psi[i] * Emix[i]
        s = float(np.sum(rho) * dx * dy)
        if s > 0:
            rho /= s
        out[i] = rho
    return out

# ====================================================
# 11) Slider / sigmaT
# ====================================================
L_gap = screen_center_x - barrier_center_x
t_gap = L_gap / max(v_est, 1e-12)

SIGMA_MIN  = 0.05 * t_gap
SIGMA_MAX  = 2.00 * t_gap
SIGMA_INIT = 0.60 * t_gap

def recompute_rho_for_sigma(new_sigma):
    Emix = build_Emix_from_phi_tau(phi_tau_frames, times, t_det, sigmaT=new_sigma, K_JITTER=13)
    return make_rho(frames_psi, Emix)

rho_init = recompute_rho_for_sigma(SIGMA_INIT)

# ====================================================
# 12) Visualisointi
# ====================================================
def gamma_display(arr, gamma=0.5):
    m = np.max(arr)
    if m <= 0:
        return arr
    disp = arr / m
    return disp ** gamma

fig = plt.figure(figsize=(10, 7))
ax = fig.add_axes([0.07, 0.18, 0.86, 0.78])

im = ax.imshow(
    gamma_display(rho_init[0]),
    extent=extent,
    origin='lower',
    vmin=0.0,
    vmax=1.0,
    cmap='magma'
)

ax.axvline(barrier_center_x, color='white', linestyle='--', alpha=0.6)
ax.axvline(screen_center_x,  color='cyan',  linestyle='--', alpha=0.4)
ax.set_xlabel("x")
ax.set_ylabel("y")

initial_status = "bridge_on" if bridge_overlay_active[0] else "searching"
title = ax.set_title(
    rf"ρ(t): σT={SIGMA_INIT:.3f}, t={times[0]:.3f}, norm≈{norms_psi[0]:.4f}, BF={initial_status}"
)

bridge_line, = ax.plot([], [], color='lime', linewidth=1.2, alpha=0.9)
pre_dot, = ax.plot([], [], marker='o', color='cyan', linestyle='None', markersize=5, alpha=0.9)
post_dot, = ax.plot([], [], marker='o', color='white', linestyle='None', markersize=5, alpha=0.9)

ax_sigma = fig.add_axes([0.10, 0.08, 0.80, 0.04])
sigma_slider = Slider(
    ax=ax_sigma,
    label="sigmaT (time thickness)",
    valmin=SIGMA_MIN,
    valmax=SIGMA_MAX,
    valinit=SIGMA_INIT,
)

rho_current = [rho_init]
sigma_current = [SIGMA_INIT]

def update_bridge_overlay(i):
    if bridge_overlay_active[i] and bridge_overlay_path_x[i] is not None:
        path_x = bridge_overlay_path_x[i]
        path_y = bridge_overlay_path_y[i]
        xs_line = x_vis_1d[path_x]
        ys_line = y_vis_1d[path_y]
        bridge_line.set_data(xs_line, ys_line)

        pre_dot.set_data([bridge_overlay_pre_x[i]], [bridge_overlay_pre_y[i]])
        post_dot.set_data([bridge_overlay_post_x[i]], [bridge_overlay_post_y[i]])
    else:
        bridge_line.set_data([], [])
        pre_dot.set_data([], [])
        post_dot.set_data([], [])

def on_sigma_change(_val):
    new_sigma = float(sigma_slider.val)
    sigma_current[0] = new_sigma
    rho_current[0] = recompute_rho_for_sigma(new_sigma)

    i = getattr(on_sigma_change, "last_i", 0)
    im.set_data(gamma_display(rho_current[0][i]))
    update_bridge_overlay(i)

    status = "bridge_on" if bridge_overlay_active[i] else "searching"
    nat = bridge_overlay_nat_branch[i]
    chosen = bridge_overlay_chosen_branch[i]
    title.set_text(
        rf"ρ(t): σT={sigma_current[0]:.3f}, t={times[i]:.3f}, "
        rf"norm≈{norms_psi[i]:.4f}, BF={status}, nat={nat}, chosen={chosen}"
    )
    fig.canvas.draw_idle()

sigma_slider.on_changed(on_sigma_change)

def update(i):
    on_sigma_change.last_i = i
    im.set_data(gamma_display(rho_current[0][i]))
    update_bridge_overlay(i)

    status = "bridge_on" if bridge_overlay_active[i] else "searching"
    nat = bridge_overlay_nat_branch[i]
    chosen = bridge_overlay_chosen_branch[i]
    title.set_text(
        rf"ρ(t): σT={sigma_current[0]:.3f}, t={times[i]:.3f}, "
        rf"norm≈{norms_psi[i]:.4f}, BF={status}, nat={nat}, chosen={chosen}"
    )
    return (im, bridge_line, pre_dot, post_dot)

ani = FuncAnimation(fig, update, frames=Nt, interval=40, blit=False)

ani.save(
    "worldline_locked.mp4",
    writer="ffmpeg",
    fps=25,
    dpi=150
)

plt.show()