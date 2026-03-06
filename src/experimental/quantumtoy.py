import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# ====================================================
# RIDGE MODE
# ====================================================
RIDGE_MODE = "centroid_top"
CENTROID_TOP_Q = 0.02
LOCALMAX_RADIUS = 20
LOCALMAX_SMOOTH_ALPHA = 0.0

# ====================================================
# FLOW / ALIGNMENT SETTINGS
# ====================================================
SAVE_COMPLEX_PSI_FRAMES = True
DRAW_FLOW_ARROW = True
ARROW_SCALE = 3.0

ALIGN_EPS_RHO = 1e-10
ALIGN_EPS_SPEED = 1e-12

ARROW_SPATIAL_AVG = True
ARROW_AVG_RADIUS = 3
ARROW_AVG_GAUSS_SIGMA = 1.5

ARROW_TEMPORAL_SMOOTH = True
ARROW_SMOOTH_ALPHA = 0.20

ARROW_HOLD_LAST_WHEN_INVALID = True
ARROW_HIDE_WHEN_INVALID = False

SHOW_TRAIL = True
TRAIL_LEN = 40

PRINT_ALIGNMENT_STATS = True

# ====================================================
# DISPLAY SETTINGS
# ====================================================
USE_FIXED_DISPLAY_SCALE = True
DISPLAY_Q = 0.995
GAMMA = 0.5
IM_INTERPOLATION = "nearest"

# ====================================================
# OPTIONAL EXTRA DIAGNOSTIC
# ====================================================
ENABLE_DIVERGENCE_DIAGNOSTIC = True
PRINT_DIVERGENCE_STATS = True

# ====================================================
# BOHMIAN TRAJECTORY OVERLAY (heavy -> flagin takana)
# ====================================================
ENABLE_BOHMIAN_OVERLAY = True

# init modes:
#   "born_initial"  : sample from initial |psi|^2 in visible region
#   "packet_center" : center + optional jitter
#   "ridge_start"   : start from initial ridge
#   "custom"        : use BOHMIAN_CUSTOM_POINTS
BOHMIAN_INIT_MODE = "born_initial"
BOHMIAN_N_TRAJ = 5
BOHMIAN_CUSTOM_POINTS = [(-15.0, 0.0)]

BOHMIAN_INIT_JITTER = 0.0
BOHMIAN_WITH_REPLACEMENT = False
BOHMIAN_RNG_SEED = 20260306

BOHMIAN_STOP_ON_LOW_RHO = True
BOHMIAN_MIN_RHO = 1e-8
BOHMIAN_STOP_OUTSIDE_VISIBLE = True

# RK4 settings
BOHMIAN_USE_RK4 = True

# drawing
BOHMIAN_SHOW_HEAD = True
BOHMIAN_SHOW_TRAIL = True
BOHMIAN_SHOW_FULL_PATH_EACH_FRAME = True
BOHMIAN_TRAIL_LEN = 120
BOHMIAN_COLOR = "cyan"
BOHMIAN_HEAD_COLOR = "deepskyblue"
BOHMIAN_LINEWIDTH = 1.6
BOHMIAN_HEAD_SIZE = 4

PRINT_BOHMIAN_STATS = True

# ====================================================
# 0) Grid / visible region
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

x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
X, Y = np.meshgrid(x, y)

m_mass = 1.0
hbar   = 1.0

cx = Nx // 2
cy = Ny // 2
hx = N_VISIBLE_X // 2
hy = N_VISIBLE_Y // 2

xs = slice(cx - hx, cx + hx)
ys = slice(cy - hy, cy + hy)

mask_visible = np.zeros_like(X, dtype=bool)
mask_visible[ys, xs] = True

X_vis = X[ys, xs]
Y_vis = Y[ys, xs]

x_vis_1d = x[xs]
y_vis_1d = y[ys]

x_vis_min = float(x_vis_1d[0])
x_vis_max = float(x_vis_1d[-1] + dx)
y_vis_min = float(y_vis_1d[0])
y_vis_max = float(y_vis_1d[-1] + dy)

# Fourier grids
kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2

def kinetic_phase(dt):
    return np.exp(-1j * K2 * dt / (2 * m_mass))

# ====================================================
# 1) Double slit barrier
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
# 2) CAP + screen
# ====================================================
def smooth_cap_edge(X, Y, Lx, Ly, cap_width=8.0, strength=2.0, power=4):
    dist_to_x = (Lx/2) - np.abs(X)
    dist_to_y = (Ly/2) - np.abs(Y)
    dist_to_edge = np.minimum(dist_to_x, dist_to_y)

    W = np.zeros_like(X, dtype=float)
    mask = dist_to_edge < cap_width
    s = (cap_width - dist_to_edge[mask]) / cap_width
    W[mask] = strength * (s**power)
    return W

CAP_WIDTH    = 10.0
CAP_STRENGTH = 2.0
CAP_POWER    = 4

W_edge = smooth_cap_edge(X, Y, Lx, Ly, cap_width=CAP_WIDTH, strength=CAP_STRENGTH, power=CAP_POWER)

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
# 3) Helpers
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
# 4) Continuous measurement
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
    psi_new = psi_u2 * n0
    return psi_new, (dWx, dWy, mx, my)

# ====================================================
# 5) Initial packet
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
# 6) Split-operator
# ====================================================
dt         = 0.003
n_steps    = 2200
save_every = 5

K_phase_fwd = kinetic_phase(dt)
K_phase_bwd = kinetic_phase(-dt)

P_half_fwd     = potential_phase(V_fwd,  dt/2.0)
P_half_bwd_adj = potential_phase(V_adj, -dt/2.0)

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
# 7) Forward frames
# ====================================================
frames_psi = []
times      = []
norms_psi  = []
psi_vis_frames = []

psi_cur = psi.copy()
print("Forward: simulaatio alkaa... (continuous measurement ON)" if USE_CONTINUOUS_MEAS else "Forward: simulaatio alkaa...")

for n in range(n_steps + 1):
    prob_psi = np.abs(psi_cur)**2
    norm_now = float(np.sum(prob_psi) * dx * dy)

    if n % save_every == 0:
        frames_psi.append(prob_psi[ys, xs].copy())
        times.append(n * dt)
        norms_psi.append(norm_now)
        if SAVE_COMPLEX_PSI_FRAMES:
            psi_vis_frames.append(psi_cur[ys, xs].copy())

        if (len(frames_psi) % 20) == 0:
            print(f"[FWD] step {n:5d}/{n_steps}, t={times[-1]:7.3f}, norm≈{norm_now:.6f}, frames={len(frames_psi)}")

    if n < n_steps:
        psi_cur = step_field(psi_cur, K_phase_fwd, P_half_fwd)
        if USE_CONTINUOUS_MEAS:
            psi_cur, _rec = continuous_measurement_update_preserve_norm(
                psi_cur, dt, KAPPA_MEAS, rng_meas
            )

frames_psi = np.array(frames_psi)
times      = np.array(times)
norms_psi  = np.array(norms_psi)
psi_vis_frames = np.array(psi_vis_frames) if SAVE_COMPLEX_PSI_FRAMES else None
Nt         = len(times)
tau_step   = save_every * dt

print("Forward valmis.")

# ====================================================
# 8) Detection time + click
# ====================================================
if not np.any(screen_mask_vis):
    raise RuntimeError("screen_mask_vis tyhjä: tarkista ruudun parametrit.")

screen_int = np.array([np.sum(frames_psi[i][screen_mask_vis]) * dx * dy for i in range(Nt)])
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
iy_vis, ix_vis = np.unravel_index(flat_idx, w.shape)

iy_click = (cy - hy) + iy_vis
ix_click = (cx - hx) + ix_vis
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

print("Backward library: lasketaan phi_tau (vain 1×)...")

phi_cur = make_phi_at_click()
phi_tau_frames = np.zeros((Nt, N_VISIBLE_Y, N_VISIBLE_X), dtype=float)

print_every_frames = 20

for i in range(Nt):
    phi_tau_frames[i] = (np.abs(phi_cur)**2)[ys, xs]

    if (i % print_every_frames) == 0 or i == Nt - 1:
        tau = i * tau_step
        norm_phi = float(np.sum(np.abs(phi_cur)**2) * dx * dy)
        print(f"[BWD] frame {i:4d}/{Nt-1}, tau={tau:7.3f}, norm≈{norm_phi:.6f}")

    if i < Nt - 1:
        for _ in range(save_every):
            phi_cur = step_field(phi_cur, K_phase_bwd, P_half_bwd_adj)

print("phi_tau-kirjasto valmis.")

# ====================================================
# 10) Emix with temporal interpolation
# ====================================================
def gaussian_weights(Tk, mu, sigma):
    if sigma <= 0:
        w = np.zeros_like(Tk)
        w[np.argmin(np.abs(Tk - mu))] = 1.0
        return w
    z = (Tk - mu) / sigma
    w = np.exp(-0.5 * z*z)
    s = w.sum()
    return w / s if s > 0 else w

def build_Emix_from_phi_tau(phi_tau_frames, times, t_det, sigmaT, K_JITTER=13):
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
            (1.0 - alpha)[:, None, None] * phi_tau_frames[j0] +
            alpha[:, None, None] * phi_tau_frames[j1]
        )
        Emix[i] = np.sum((w[valid])[:, None, None] * phi_interp, axis=0)

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
# 11) Schrödinger current / velocity
# ====================================================
def schrodinger_current_visible(psi_vis):
    dpsi_dx = (np.roll(psi_vis, -1, axis=1) - np.roll(psi_vis, 1, axis=1)) / (2.0 * dx)
    dpsi_dy = (np.roll(psi_vis, -1, axis=0) - np.roll(psi_vis, 1, axis=0)) / (2.0 * dy)

    rho = (np.abs(psi_vis)**2).astype(float)
    jx = ((hbar / m_mass) * np.imag(np.conjugate(psi_vis) * dpsi_dx)).astype(float)
    jy = ((hbar / m_mass) * np.imag(np.conjugate(psi_vis) * dpsi_dy)).astype(float)
    return jx, jy, rho

def velocity_from_current(jx, jy, rho, eps=ALIGN_EPS_RHO):
    denom = np.maximum(rho, eps)
    vx = jx / denom
    vy = jy / denom
    sp = np.hypot(vx, vy)
    return vx, vy, sp

# ====================================================
# 12) Ridge helpers
# ====================================================
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
    x0i = int(np.clip(prev_ix, 0, W_-1))
    y0i = int(np.clip(prev_iy, 0, H-1))
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

def compute_ridge_xy(frames_psi, Emix, x_vis_1d, y_vis_1d,
                     mode="argmax",
                     top_q=0.02,
                     radius=20,
                     alpha_smooth=0.0):
    Nt_ = frames_psi.shape[0]
    ridge_x = np.zeros(Nt_, dtype=float)
    ridge_y = np.zeros(Nt_, dtype=float)
    ridge_s = np.zeros(Nt_, dtype=float)

    Gamma0 = frames_psi[0] * Emix[0]
    x0r, y0r, s0 = ridge_argmax(Gamma0, x_vis_1d, y_vis_1d)
    ridge_x[0], ridge_y[0], ridge_s[0] = x0r, y0r, s0
    prev_ix = int(np.argmin(np.abs(x_vis_1d - x0r)))
    prev_iy = int(np.argmin(np.abs(y_vis_1d - y0r)))

    for i in range(1, Nt_):
        Gamma = frames_psi[i] * Emix[i]

        if mode == "argmax":
            xi, yi, si = ridge_argmax(Gamma, x_vis_1d, y_vis_1d)
        elif mode == "centroid_top":
            xi, yi, si = ridge_centroid_top(Gamma, x_vis_1d, y_vis_1d, top_q=top_q)
        elif mode == "localmax_track":
            xi, yi, si, prev_ix, prev_iy = ridge_localmax_track(
                Gamma, x_vis_1d, y_vis_1d, prev_ix, prev_iy, radius=radius
            )
        else:
            raise ValueError(f"Unknown RIDGE_MODE: {mode}")

        if alpha_smooth and alpha_smooth > 0.0:
            xi = (1.0 - alpha_smooth) * xi + alpha_smooth * ridge_x[i-1]
            yi = (1.0 - alpha_smooth) * yi + alpha_smooth * ridge_y[i-1]

        ridge_x[i], ridge_y[i], ridge_s[i] = xi, yi, si

        if mode != "localmax_track":
            prev_ix = int(np.argmin(np.abs(x_vis_1d - xi)))
            prev_iy = int(np.argmin(np.abs(y_vis_1d - yi)))

    return ridge_x, ridge_y, ridge_s

# ====================================================
# 13) Ridge tangent + local weighted direction
# ====================================================
def ridge_tangent_unit(ridge_x, ridge_y):
    Nt_ = len(ridge_x)
    tx = np.zeros(Nt_, dtype=float)
    ty = np.zeros(Nt_, dtype=float)
    for i in range(1, Nt_ - 1):
        dxp = ridge_x[i+1] - ridge_x[i-1]
        dyp = ridge_y[i+1] - ridge_y[i-1]
        n = np.hypot(dxp, dyp)
        if n > 0:
            tx[i] = dxp / n
            ty[i] = dyp / n
    if Nt_ >= 2:
        tx[0], ty[0] = tx[1], ty[1]
        tx[-1], ty[-1] = tx[-2], ty[-2]
    return tx, ty

def gaussian_kernel_2d(radius, sigma):
    r = int(radius)
    if r <= 0:
        return np.array([[1.0]], dtype=float)
    ax = np.arange(-r, r+1, dtype=float)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx*xx + yy*yy) / (2.0 * sigma*sigma))
    s = float(np.sum(k))
    return k / s if s > 0 else k

_GK = gaussian_kernel_2d(ARROW_AVG_RADIUS, ARROW_AVG_GAUSS_SIGMA)

def local_weighted_mean(vx, vy, speed, ix, iy, kernel=_GK):
    H, W_ = vx.shape
    r = kernel.shape[0] // 2
    x1 = max(0, ix - r); x2 = min(W_, ix + r + 1)
    y1 = max(0, iy - r); y2 = min(H, iy + r + 1)

    sub_vx = vx[y1:y2, x1:x2]
    sub_vy = vy[y1:y2, x1:x2]
    sub_sp = speed[y1:y2, x1:x2]

    ky1 = y1 - (iy - r); ky2 = ky1 + (y2 - y1)
    kx1 = x1 - (ix - r); kx2 = kx1 + (x2 - x1)
    k = kernel[ky1:ky2, kx1:kx2]

    w = k * sub_sp
    wsum = float(np.sum(w))
    if wsum <= 0:
        return np.nan, np.nan, np.nan

    vxm = float(np.sum(w * sub_vx) / wsum)
    vym = float(np.sum(w * sub_vy) / wsum)
    spd = float(np.hypot(vxm, vym))
    if spd <= 0:
        return np.nan, np.nan, np.nan

    return vxm / spd, vym / spd, spd

# ====================================================
# 14) Alignment + divergence
# ====================================================
def divergence_of_velocity(vx, vy):
    dvx_dx = (np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1)) / (2.0 * dx)
    dvy_dy = (np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0)) / (2.0 * dy)
    return dvx_dx + dvy_dy

def alignment_and_diagnostics_from_psi(psi_vis_frames, ridge_x, ridge_y, x_vis_1d, y_vis_1d):
    Nt_ = len(ridge_x)
    cos_th = np.full(Nt_, np.nan, dtype=float)
    speed  = np.full(Nt_, np.nan, dtype=float)
    ux     = np.full(Nt_, np.nan, dtype=float)
    uy     = np.full(Nt_, np.nan, dtype=float)
    div_v_at_ridge = np.full(Nt_, np.nan, dtype=float)

    tx, ty = ridge_tangent_unit(ridge_x, ridge_y)

    for i in range(Nt_):
        ix = int(np.argmin(np.abs(x_vis_1d - ridge_x[i])))
        iy = int(np.argmin(np.abs(y_vis_1d - ridge_y[i])))

        jx, jy, rho_s = schrodinger_current_visible(psi_vis_frames[i])
        vx, vy, sp = velocity_from_current(jx, jy, rho_s, eps=ALIGN_EPS_RHO)

        if ARROW_SPATIAL_AVG:
            uxi, uyi, spd = local_weighted_mean(vx, vy, sp, ix, iy)
        else:
            spd = float(sp[iy, ix])
            if spd > 0:
                uxi = float(vx[iy, ix] / spd)
                uyi = float(vy[iy, ix] / spd)
            else:
                uxi, uyi = np.nan, np.nan

        if np.isfinite(uxi) and np.isfinite(uyi) and np.isfinite(spd) and (spd > ALIGN_EPS_SPEED):
            ux[i] = uxi
            uy[i] = uyi
            speed[i] = spd
            cos_th[i] = float(np.clip(uxi * tx[i] + uyi * ty[i], -1.0, 1.0))

        if ENABLE_DIVERGENCE_DIAGNOSTIC:
            div_v = divergence_of_velocity(vx, vy)
            div_v_at_ridge[i] = float(div_v[iy, ix])

    if ARROW_TEMPORAL_SMOOTH:
        a = float(np.clip(ARROW_SMOOTH_ALPHA, 0.0, 1.0))
        ux_f = ux.copy()
        uy_f = uy.copy()
        sp_f = speed.copy()

        last_u = None
        last_v = None
        last_s = None
        for i in range(Nt_):
            if np.isfinite(ux_f[i]) and np.isfinite(uy_f[i]):
                if last_u is not None:
                    uu = (1-a) * ux_f[i] + a * last_u
                    vv = (1-a) * uy_f[i] + a * last_v
                    nn = float(np.hypot(uu, vv))
                    if nn > 0:
                        uu /= nn
                        vv /= nn
                    ux_f[i], uy_f[i] = uu, vv
                    if np.isfinite(sp_f[i]) and last_s is not None:
                        sp_f[i] = (1-a) * sp_f[i] + a * last_s
                last_u, last_v = ux_f[i], uy_f[i]
                last_s = sp_f[i] if np.isfinite(sp_f[i]) else last_s

        ux, uy, speed = ux_f, uy_f, sp_f

        tx, ty = ridge_tangent_unit(ridge_x, ridge_y)
        for i in range(Nt_):
            if np.isfinite(ux[i]) and np.isfinite(uy[i]):
                cos_th[i] = float(np.clip(ux[i] * tx[i] + uy[i] * ty[i], -1.0, 1.0))

    return cos_th, speed, ux, uy, div_v_at_ridge

# ====================================================
# 15) Bohmian helpers
# ====================================================
def bilinear_interpolate_scalar(field, xq, yq, x0_arr, y0_arr, dx_, dy_):
    nx_ = len(x0_arr)
    ny_ = len(y0_arr)

    fx = (xq - x0_arr[0]) / dx_
    fy = (yq - y0_arr[0]) / dy_

    if not (0.0 <= fx < nx_ - 1 and 0.0 <= fy < ny_ - 1):
        return np.nan

    ix0 = int(np.floor(fx))
    iy0 = int(np.floor(fy))
    txi = fx - ix0
    tyi = fy - iy0

    f00 = field[iy0, ix0]
    f10 = field[iy0, ix0 + 1]
    f01 = field[iy0 + 1, ix0]
    f11 = field[iy0 + 1, ix0 + 1]

    return (
        (1.0 - txi) * (1.0 - tyi) * f00 +
        txi * (1.0 - tyi) * f10 +
        (1.0 - txi) * tyi * f01 +
        txi * tyi * f11
    )

def bilinear_interpolate_vector(vx, vy, xq, yq, x0_arr, y0_arr, dx_, dy_):
    vxq = bilinear_interpolate_scalar(vx, xq, yq, x0_arr, y0_arr, dx_, dy_)
    vyq = bilinear_interpolate_scalar(vy, xq, yq, x0_arr, y0_arr, dx_, dy_)
    return vxq, vyq

def is_inside_visible(xp, yp):
    return (x_vis_min <= xp < x_vis_max) and (y_vis_min <= yp < y_vis_max)

def build_velocity_frames_from_psi(psi_vis_frames):
    Nt_ = psi_vis_frames.shape[0]
    vx_frames = np.zeros((Nt_, N_VISIBLE_Y, N_VISIBLE_X), dtype=float)
    vy_frames = np.zeros((Nt_, N_VISIBLE_Y, N_VISIBLE_X), dtype=float)
    rho_frames = np.zeros((Nt_, N_VISIBLE_Y, N_VISIBLE_X), dtype=float)

    for i in range(Nt_):
        jx, jy, rho_s = schrodinger_current_visible(psi_vis_frames[i])
        vx, vy, _sp = velocity_from_current(jx, jy, rho_s, eps=ALIGN_EPS_RHO)
        vx_frames[i] = vx
        vy_frames[i] = vy
        rho_frames[i] = rho_s

    return vx_frames, vy_frames, rho_frames

def velocity_rho_at_time(vx_frames, vy_frames, rho_frames, t_query):
    """
    Linear interpolation in time between saved frames.
    Returns (vx_t, vy_t, rho_t) full 2D fields on visible grid.
    """
    if t_query <= times[0]:
        return vx_frames[0], vy_frames[0], rho_frames[0]
    if t_query >= times[-1]:
        return vx_frames[-1], vy_frames[-1], rho_frames[-1]

    s = (t_query - times[0]) / tau_step
    i0 = int(np.floor(s))
    i1 = min(i0 + 1, len(times) - 1)
    a = float(s - i0)

    vx_t = (1.0 - a) * vx_frames[i0] + a * vx_frames[i1]
    vy_t = (1.0 - a) * vy_frames[i0] + a * vy_frames[i1]
    rho_t = (1.0 - a) * rho_frames[i0] + a * rho_frames[i1]
    return vx_t, vy_t, rho_t

def velocity_sample_time_space(vx_frames, vy_frames, rho_frames, t_query, xq, yq):
    vx_t, vy_t, rho_t = velocity_rho_at_time(vx_frames, vy_frames, rho_frames, t_query)
    rho_q = bilinear_interpolate_scalar(rho_t, xq, yq, x_vis_1d, y_vis_1d, dx, dy)
    vx_q, vy_q = bilinear_interpolate_vector(vx_t, vy_t, xq, yq, x_vis_1d, y_vis_1d, dx, dy)
    return vx_q, vy_q, rho_q

def sample_born_initial_points_from_visible_psi(psi0_vis, ntraj, rng, with_replacement=False):
    rho0 = np.abs(psi0_vis)**2
    w = rho0.ravel().astype(float)
    s = float(np.sum(w))
    if s <= 0:
        return [(x0, y0)]
    p = w / s

    nsel = min(ntraj, p.size) if not with_replacement else ntraj
    idxs = rng.choice(p.size, size=nsel, replace=with_replacement, p=p)

    pts = []
    for idx in np.atleast_1d(idxs):
        iy0, ix0 = np.unravel_index(int(idx), rho0.shape)
        pts.append((float(x_vis_1d[ix0]), float(y_vis_1d[iy0])))
    return pts

def make_bohmian_initial_points(mode, ntraj, custom_points, ridge_x0, ridge_y0,
                                x0_packet, y0_packet, psi0_vis, jitter=0.0):
    rng_b = np.random.default_rng(BOHMIAN_RNG_SEED)

    if mode == "born_initial":
        pts = sample_born_initial_points_from_visible_psi(
            psi0_vis, ntraj, rng_b, with_replacement=BOHMIAN_WITH_REPLACEMENT
        )
        return pts

    if mode == "packet_center":
        cx0, cy0 = x0_packet, y0_packet
        if ntraj <= 1:
            return [(cx0, cy0)]
        offsets = np.linspace(-(ntraj-1)/2.0, (ntraj-1)/2.0, ntraj) * max(jitter, 0.15)
        return [(cx0, cy0 + off) for off in offsets]

    if mode == "ridge_start":
        cx0, cy0 = ridge_x0, ridge_y0
        if ntraj <= 1:
            return [(cx0, cy0)]
        offsets = np.linspace(-(ntraj-1)/2.0, (ntraj-1)/2.0, ntraj) * max(jitter, 0.15)
        return [(cx0, cy0 + off) for off in offsets]

    if mode == "custom":
        return list(custom_points[:ntraj])

    raise ValueError(f"Unknown BOHMIAN_INIT_MODE: {mode}")

def bohmian_rhs(vx_frames, vy_frames, rho_frames, t_query, xq, yq):
    vx_q, vy_q, rho_q = velocity_sample_time_space(vx_frames, vy_frames, rho_frames, t_query, xq, yq)

    if BOHMIAN_STOP_OUTSIDE_VISIBLE and not is_inside_visible(xq, yq):
        return np.nan, np.nan, np.nan

    if BOHMIAN_STOP_ON_LOW_RHO and (not np.isfinite(rho_q) or rho_q < BOHMIAN_MIN_RHO):
        return np.nan, np.nan, rho_q

    if not (np.isfinite(vx_q) and np.isfinite(vy_q)):
        return np.nan, np.nan, rho_q

    return float(vx_q), float(vy_q), float(rho_q)

def rk4_step_bohmian(vx_frames, vy_frames, rho_frames, t0, xcur, ycur, h):
    k1x, k1y, _ = bohmian_rhs(vx_frames, vy_frames, rho_frames, t0, xcur, ycur)
    if not (np.isfinite(k1x) and np.isfinite(k1y)):
        return np.nan, np.nan

    k2x, k2y, _ = bohmian_rhs(vx_frames, vy_frames, rho_frames, t0 + 0.5*h, xcur + 0.5*h*k1x, ycur + 0.5*h*k1y)
    if not (np.isfinite(k2x) and np.isfinite(k2y)):
        return np.nan, np.nan

    k3x, k3y, _ = bohmian_rhs(vx_frames, vy_frames, rho_frames, t0 + 0.5*h, xcur + 0.5*h*k2x, ycur + 0.5*h*k2y)
    if not (np.isfinite(k3x) and np.isfinite(k3y)):
        return np.nan, np.nan

    k4x, k4y, _ = bohmian_rhs(vx_frames, vy_frames, rho_frames, t0 + h, xcur + h*k3x, ycur + h*k3y)
    if not (np.isfinite(k4x) and np.isfinite(k4y)):
        return np.nan, np.nan

    xnext = xcur + (h/6.0) * (k1x + 2.0*k2x + 2.0*k3x + k4x)
    ynext = ycur + (h/6.0) * (k1y + 2.0*k2y + 2.0*k3y + k4y)
    return xnext, ynext

def euler_step_bohmian(vx_frames, vy_frames, rho_frames, t0, xcur, ycur, h):
    vx_q, vy_q, _ = bohmian_rhs(vx_frames, vy_frames, rho_frames, t0, xcur, ycur)
    if not (np.isfinite(vx_q) and np.isfinite(vy_q)):
        return np.nan, np.nan
    return xcur + h * vx_q, ycur + h * vy_q

def integrate_bohmian_trajectories(vx_frames, vy_frames, rho_frames, init_points):
    Nt_ = len(times)
    ntraj = len(init_points)

    traj_x = np.full((ntraj, Nt_), np.nan, dtype=float)
    traj_y = np.full((ntraj, Nt_), np.nan, dtype=float)
    traj_alive = np.zeros((ntraj, Nt_), dtype=bool)

    stepper = rk4_step_bohmian if BOHMIAN_USE_RK4 else euler_step_bohmian

    for k, (x_init, y_init) in enumerate(init_points):
        xcur = float(x_init)
        ycur = float(y_init)

        if is_inside_visible(xcur, ycur):
            traj_x[k, 0] = xcur
            traj_y[k, 0] = ycur
            traj_alive[k, 0] = True

        for i in range(0, Nt_ - 1):
            if not traj_alive[k, i]:
                break

            if BOHMIAN_STOP_OUTSIDE_VISIBLE and not is_inside_visible(xcur, ycur):
                break

            xnext, ynext = stepper(vx_frames, vy_frames, rho_frames, times[i], xcur, ycur, tau_step)

            if not (np.isfinite(xnext) and np.isfinite(ynext)):
                break

            if BOHMIAN_STOP_OUTSIDE_VISIBLE and not is_inside_visible(xnext, ynext):
                break

            traj_x[k, i+1] = xnext
            traj_y[k, i+1] = ynext
            traj_alive[k, i+1] = True

            xcur, ycur = xnext, ynext

    return traj_x, traj_y, traj_alive

# ====================================================
# 16) Build rho + diagnostics for sigmaT
# ====================================================
def build_all_for_sigma(sigmaT):
    Emix = build_Emix_from_phi_tau(phi_tau_frames, times, t_det, sigmaT=sigmaT, K_JITTER=13)
    rho  = make_rho(frames_psi, Emix)

    rx, ry, rs = compute_ridge_xy(
        frames_psi, Emix, x_vis_1d, y_vis_1d,
        mode=RIDGE_MODE,
        top_q=CENTROID_TOP_Q,
        radius=LOCALMAX_RADIUS,
        alpha_smooth=LOCALMAX_SMOOTH_ALPHA
    )

    cos_th = speed = ux = uy = div_v = None
    if SAVE_COMPLEX_PSI_FRAMES and (psi_vis_frames is not None):
        cos_th, speed, ux, uy, div_v = alignment_and_diagnostics_from_psi(
            psi_vis_frames, rx, ry, x_vis_1d, y_vis_1d
        )

    return rho, Emix, rx, ry, rs, cos_th, speed, ux, uy, div_v

# ====================================================
# 17) Sigma slider setup
# ====================================================
v_est = k0x / m_mass
L_gap = screen_center_x - barrier_center_x
t_gap = L_gap / (v_est + 1e-12)

SIGMA_MIN  = 0.05 * t_gap
SIGMA_MAX  = 2.00 * t_gap
SIGMA_INIT = 0.60 * t_gap

rho_init, Emix_init, ridge_x_init, ridge_y_init, ridge_s_init, cos_th_init, speed_init, ux_init, uy_init, div_v_init = build_all_for_sigma(SIGMA_INIT)

if USE_FIXED_DISPLAY_SCALE:
    vref = float(np.quantile(rho_init, DISPLAY_Q))
    if vref <= 0:
        vref = 1.0
else:
    vref = 1.0

def gamma_display(arr, gamma=GAMMA):
    if USE_FIXED_DISPLAY_SCALE:
        disp = np.clip(arr / (vref + 1e-30), 0.0, 1.0)
        return disp**gamma
    m = np.max(arr)
    if m <= 0:
        return arr
    return (arr / m)**gamma

if speed_init is not None:
    vv = speed_init[np.isfinite(speed_init)]
    speed_ref = float(np.quantile(vv, 0.80)) if vv.size > 0 else 1.0
else:
    speed_ref = 1.0

# ====================================================
# 18) Precompute Bohmian trajectories
# ====================================================
bohm_traj_x = bohm_traj_y = bohm_traj_alive = None
bohm_init_points = []

if ENABLE_BOHMIAN_OVERLAY:
    if not SAVE_COMPLEX_PSI_FRAMES or (psi_vis_frames is None):
        raise RuntimeError("ENABLE_BOHMIAN_OVERLAY vaatii SAVE_COMPLEX_PSI_FRAMES=True.")

    print("Bohmian overlay: lasketaan velocity frames...")
    vx_frames, vy_frames, rho_frames = build_velocity_frames_from_psi(psi_vis_frames)

    bohm_init_points = make_bohmian_initial_points(
        mode=BOHMIAN_INIT_MODE,
        ntraj=BOHMIAN_N_TRAJ,
        custom_points=BOHMIAN_CUSTOM_POINTS,
        ridge_x0=ridge_x_init[0],
        ridge_y0=ridge_y_init[0],
        x0_packet=x0,
        y0_packet=y0,
        psi0_vis=psi_vis_frames[0],
        jitter=BOHMIAN_INIT_JITTER
    )

    print(f"Bohmian overlay: integroidaan {len(bohm_init_points)} trajektoria ({'RK4' if BOHMIAN_USE_RK4 else 'Euler'})...")
    bohm_traj_x, bohm_traj_y, bohm_traj_alive = integrate_bohmian_trajectories(
        vx_frames, vy_frames, rho_frames, bohm_init_points
    )

# ====================================================
# 19) Visualization
# ====================================================
extent = (-VISIBLE_LX/2, VISIBLE_LX/2, -VISIBLE_LY/2, VISIBLE_LY/2)

fig = plt.figure(figsize=(10.8, 7.2))
ax = fig.add_axes([0.07, 0.18, 0.86, 0.78])

im = ax.imshow(
    gamma_display(rho_init[0]),
    extent=extent, origin='lower',
    vmin=0.0, vmax=1.0,
    cmap='magma',
    interpolation=IM_INTERPOLATION
)

ax.axvline(barrier_center_x, color='white', linestyle='--', alpha=0.6)
ax.axvline(screen_center_x,  color='cyan',  linestyle='--', alpha=0.4)
ax.set_xlabel("x")
ax.set_ylabel("y")

title = ax.set_title(
    rf"ρ(t): σT={SIGMA_INIT:.3f}, t={times[0]:.3f}, ridge={RIDGE_MODE}"
)

ridge_marker, = ax.plot(
    [ridge_x_init[0]], [ridge_y_init[0]],
    marker='o', markersize=7, linestyle='None',
    color='lime', alpha=0.9, label=f"ridge ({RIDGE_MODE})"
)

ridge_trail, = ax.plot([], [], linestyle='-', linewidth=1.5, color='lime', alpha=0.5)

flow_quiver = None
if DRAW_FLOW_ARROW and SAVE_COMPLEX_PSI_FRAMES and (ux_init is not None):
    flow_quiver = ax.quiver(
        [ridge_x_init[0]], [ridge_y_init[0]],
        [0.0], [0.0],
        angles='xy', scale_units='xy', scale=1.0,
        color='cyan', alpha=0.9, width=0.006
    )

bohm_lines = []
bohm_heads = []

if ENABLE_BOHMIAN_OVERLAY and (bohm_traj_x is not None):
    for k in range(bohm_traj_x.shape[0]):
        line_k, = ax.plot([], [], linestyle='-', linewidth=BOHMIAN_LINEWIDTH,
                          color=BOHMIAN_COLOR, alpha=0.85,
                          label="Bohmian traj" if k == 0 else None)
        bohm_lines.append(line_k)

        if BOHMIAN_SHOW_HEAD:
            head_k, = ax.plot([], [], marker='o', markersize=BOHMIAN_HEAD_SIZE,
                              linestyle='None', color=BOHMIAN_HEAD_COLOR, alpha=0.95)
        else:
            head_k = None
        bohm_heads.append(head_k)

ax.legend(loc="upper right", framealpha=0.35)

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
ridge_x = [ridge_x_init]
ridge_y = [ridge_y_init]
ridge_s = [ridge_s_init]
cos_th = [cos_th_init]
speed = [speed_init]
ux = [ux_init]
uy = [uy_init]
div_v_ridge = [div_v_init]

arrow_state = {"ux": np.nan, "uy": np.nan, "spd": np.nan}

def recompute_for_sigma(new_sigma):
    return build_all_for_sigma(new_sigma)

def update_flow_arrow(i):
    if flow_quiver is None or ux[0] is None or speed[0] is None:
        return

    uxi = ux[0][i]
    uyi = uy[0][i]
    spd = speed[0][i]

    valid = np.isfinite(uxi) and np.isfinite(uyi) and np.isfinite(spd) and (spd > ALIGN_EPS_SPEED)

    if not valid:
        if ARROW_HIDE_WHEN_INVALID:
            flow_quiver.set_offsets([[ridge_x[0][i], ridge_y[0][i]]])
            flow_quiver.set_UVC([0.0], [0.0])
            return

        if ARROW_HOLD_LAST_WHEN_INVALID and np.isfinite(arrow_state["ux"]) and np.isfinite(arrow_state["uy"]):
            uxi, uyi = arrow_state["ux"], arrow_state["uy"]
            spd = arrow_state["spd"] if np.isfinite(arrow_state["spd"]) else 0.0
        else:
            flow_quiver.set_offsets([[ridge_x[0][i], ridge_y[0][i]]])
            flow_quiver.set_UVC([0.0], [0.0])
            return

    if ARROW_TEMPORAL_SMOOTH and np.isfinite(arrow_state["ux"]) and np.isfinite(arrow_state["uy"]):
        a = float(np.clip(ARROW_SMOOTH_ALPHA, 0.0, 1.0))
        uu = (1-a) * uxi + a * arrow_state["ux"]
        vv = (1-a) * uyi + a * arrow_state["uy"]
        nn = float(np.hypot(uu, vv))
        if nn > 0:
            uu /= nn
            vv /= nn
        uxi, uyi = uu, vv

    arrow_state["ux"] = uxi
    arrow_state["uy"] = uyi
    arrow_state["spd"] = spd

    flow_quiver.set_offsets([[ridge_x[0][i], ridge_y[0][i]]])
    L = ARROW_SCALE * float(np.clip(spd / (speed_ref + 1e-30), 0.0, 2.5))
    flow_quiver.set_UVC([L * uxi], [L * uyi])

def update_bohmian_overlay(i):
    if not ENABLE_BOHMIAN_OVERLAY or (bohm_traj_x is None):
        return

    for k in range(bohm_traj_x.shape[0]):
        alive = bohm_traj_alive[k]

        if not np.any(alive[:i+1]):
            bohm_lines[k].set_data([], [])
            if bohm_heads[k] is not None:
                bohm_heads[k].set_data([], [])
            continue

        if BOHMIAN_SHOW_FULL_PATH_EACH_FRAME:
            j0 = 0
        else:
            j0 = max(0, i - BOHMIAN_TRAIL_LEN + 1)

        mask = alive[j0:i+1]
        xs_seg = bohm_traj_x[k, j0:i+1][mask]
        ys_seg = bohm_traj_y[k, j0:i+1][mask]
        bohm_lines[k].set_data(xs_seg, ys_seg)

        if BOHMIAN_SHOW_HEAD and bohm_heads[k] is not None:
            alive_idx = np.where(alive[:i+1])[0]
            if alive_idx.size > 0:
                ilast = int(alive_idx[-1])
                bohm_heads[k].set_data([bohm_traj_x[k, ilast]], [bohm_traj_y[k, ilast]])
            else:
                bohm_heads[k].set_data([], [])

def make_title(i):
    parts = [
        rf"ρ(t): σT={sigma_current[0]:.3f}",
        rf"t={times[i]:.3f}",
        rf"ridge={RIDGE_MODE}",
        rf"norm≈{norms_psi[i]:.4f}",
        rf"Γ≈{ridge_s[0][i]:.3e}",
    ]

    if cos_th[0] is not None and np.isfinite(cos_th[0][i]):
        parts.append(rf"cosθ≈{cos_th[0][i]:.3f}")

    if ENABLE_DIVERGENCE_DIAGNOSTIC and div_v_ridge[0] is not None and np.isfinite(div_v_ridge[0][i]):
        parts.append(rf"div v≈{div_v_ridge[0][i]:.3e}")

    if ENABLE_BOHMIAN_OVERLAY:
        parts.append("Bohm=RK4" if BOHMIAN_USE_RK4 else "Bohm=Euler")

    return " | ".join(parts)

def on_sigma_change(_val):
    new_sigma = float(sigma_slider.val)
    sigma_current[0] = new_sigma

    rho_new, _Emix, rx, ry, rs, cth, spd, uxx, uyy, divv = recompute_for_sigma(new_sigma)

    rho_current[0] = rho_new
    ridge_x[0] = rx
    ridge_y[0] = ry
    ridge_s[0] = rs
    cos_th[0] = cth
    speed[0] = spd
    ux[0] = uxx
    uy[0] = uyy
    div_v_ridge[0] = divv

    i = getattr(on_sigma_change, "last_i", 0)

    im.set_data(gamma_display(rho_current[0][i]))
    ridge_marker.set_data([ridge_x[0][i]], [ridge_y[0][i]])

    if SHOW_TRAIL:
        j0 = max(0, i - TRAIL_LEN + 1)
        ridge_trail.set_data(ridge_x[0][j0:i+1], ridge_y[0][j0:i+1])

    update_flow_arrow(i)
    update_bohmian_overlay(i)

    title.set_text(make_title(i))
    fig.canvas.draw_idle()

sigma_slider.on_changed(on_sigma_change)

def update(i):
    on_sigma_change.last_i = i

    im.set_data(gamma_display(rho_current[0][i]))
    ridge_marker.set_data([ridge_x[0][i]], [ridge_y[0][i]])

    if SHOW_TRAIL:
        j0 = max(0, i - TRAIL_LEN + 1)
        ridge_trail.set_data(ridge_x[0][j0:i+1], ridge_y[0][j0:i+1])

    update_flow_arrow(i)
    update_bohmian_overlay(i)
    title.set_text(make_title(i))

    artists = [im, ridge_marker, ridge_trail]
    if flow_quiver is not None:
        artists.append(flow_quiver)
    artists.extend([obj for obj in bohm_lines if obj is not None])
    artists.extend([obj for obj in bohm_heads if obj is not None])
    return tuple(artists)

ani = FuncAnimation(fig, update, frames=Nt, interval=40, blit=False)

# ====================================================
# 20) Summary stats
# ====================================================
if PRINT_ALIGNMENT_STATS and SAVE_COMPLEX_PSI_FRAMES and (cos_th_init is not None):
    valid = np.isfinite(cos_th_init)
    if np.any(valid):
        mean_c = float(np.mean(cos_th_init[valid]))
        med_c  = float(np.median(cos_th_init[valid]))
        frac_pos = float(np.mean(cos_th_init[valid] > 0.0))
        frac_hi  = float(np.mean(cos_th_init[valid] > 0.7))
        print(
            f"[ALIGN] ridge vs velocity: mean cosθ≈{mean_c:.3f}, "
            f"median≈{med_c:.3f}, frac(cosθ>0)≈{frac_pos:.3f}, frac(cosθ>0.7)≈{frac_hi:.3f}"
        )
    else:
        print("[ALIGN] no valid cosθ.")

if ENABLE_DIVERGENCE_DIAGNOSTIC and PRINT_DIVERGENCE_STATS and (div_v_init is not None):
    valid = np.isfinite(div_v_init)
    if np.any(valid):
        mean_div = float(np.mean(div_v_init[valid]))
        med_div  = float(np.median(div_v_init[valid]))
        frac_neg = float(np.mean(div_v_init[valid] < 0.0))
        frac_pos = float(np.mean(div_v_init[valid] > 0.0))
        print(
            f"[DIV] at ridge: mean div(v)≈{mean_div:.3e}, "
            f"median≈{med_div:.3e}, frac(<0)≈{frac_neg:.3f}, frac(>0)≈{frac_pos:.3f}"
        )
    else:
        print("[DIV] no valid div(v) values.")

if ENABLE_BOHMIAN_OVERLAY and PRINT_BOHMIAN_STATS and (bohm_traj_alive is not None):
    alive_counts = np.sum(bohm_traj_alive, axis=1)
    for k in range(bohm_traj_alive.shape[0]):
        if alive_counts[k] > 0:
            i_last = int(alive_counts[k] - 1)
            print(
                f"[BOHM] traj {k}: steps={alive_counts[k]}, "
                f"start=({bohm_traj_x[k,0]:.3f},{bohm_traj_y[k,0]:.3f}), "
                f"end=({bohm_traj_x[k,i_last]:.3f},{bohm_traj_y[k,i_last]:.3f})"
            )
        else:
            print(f"[BOHM] traj {k}: no valid steps")

ani.save("schrodinger_ridge_bohmian_rk4_born5.mp4", writer="ffmpeg", fps=25, dpi=150)
plt.show()