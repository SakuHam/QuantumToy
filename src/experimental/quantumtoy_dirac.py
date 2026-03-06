import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# ====================================================
# RIDGE MODE (choose one)
#   "argmax"        : ridge is the global maximum of Gamma each frame (can jump)
#   "centroid_top"  : ridge is centroid of top-q% Gamma mass (smooth)
#   "localmax_track": follow nearest strong local maximum (reduces jumps)
# ====================================================
RIDGE_MODE = "centroid_top"   # <-- change to "argmax" / "localmax_track"
CENTROID_TOP_Q = 0.02         # top 2% pixels for centroid ridge
LOCALMAX_RADIUS = 20          # pixels, search window around previous ridge
LOCALMAX_SMOOTH_ALPHA = 0.0   # 0=no smoothing, e.g. 0.2 gives mild smoothing

# ====================================================
# FLOW / ALIGNMENT TEST SETTINGS
#   - save forward spinor frames so we can compute Dirac current j
#   - draw a current/velocity arrow at ridge point
#   - compute alignment cos(theta) between ridge tangent and velocity direction
# ====================================================
SAVE_SPINOR_FRAMES = True
DRAW_FLOW_ARROW = True
ARROW_SCALE = 3.0
ARROW_STRIDE = 1

PRINT_ALIGNMENT_STATS = True

# Robustness knobs
ALIGN_EPS_RHO = 1e-10   # avoid v=j/rho blowups when rho ~ 0
ALIGN_EPS_SPEED = 1e-12 # consider direction invalid when speed ~ 0

# ====================================================
# ARROW STABILIZATION (spatial avg + temporal smoothing + hysteresis)
# ====================================================
ARROW_SPATIAL_AVG = True
ARROW_AVG_RADIUS = 3          # pixels (3 => 7x7 neighborhood)
ARROW_AVG_GAUSS_SIGMA = 1.5   # gaussian weighting sigma in pixels

ARROW_TEMPORAL_SMOOTH = True
ARROW_SMOOTH_ALPHA = 0.20     # 0=no smoothing, 0.2..0.4 good

ARROW_HOLD_LAST_WHEN_INVALID = True
ARROW_HIDE_WHEN_INVALID = False

# ====================================================
# DISPLAY STABILITY (optional): fixed reference scale reduces banding/glitchy look
# ====================================================
USE_FIXED_DISPLAY_SCALE = True
DISPLAY_Q = 0.995
GAMMA = 0.5
IM_INTERPOLATION = "nearest"

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

x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
X, Y = np.meshgrid(x, y)

# ====================================================
# 1) Dirac (2+1D) parametrit
# ====================================================
hbar = 1.0
c    = 25.0
m    = 1.0

# Näkyvän ikkunan indeksit (keskelle isoa hilaa)
cx = Nx // 2
cy = Ny // 2
hx = N_VISIBLE_X // 2
hy = N_VISIBLE_Y // 2
xs = slice(cx - hx, cx + hx)
ys = slice(cy - hy, cy + hy)

mask_visible = np.zeros((Ny, Nx), dtype=bool)
mask_visible[ys, xs] = True

X_vis = X[ys, xs]
Y_vis = Y[ys, xs]

# näkyvän ikkunan koordinaatit 1D (marker-plotille)
x_vis_1d = x[xs]
y_vis_1d = y[ys]

# Fourier-hilat
kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky)

# momentit p = ħ k
PX = hbar * KX
PY = hbar * KY

# ====================================================
# 2) Massagap-este
# ====================================================
barrier_center_x  = 0.0
barrier_thickness = 0.4

slit_center_offset = 2.0
slit_half_height   = 0.5

S0 = 80.0
BARRIER_SMOOTH = 0.15

barrier_core = np.abs(X - barrier_center_x) < (barrier_thickness / 2.0)
slit1_mask = np.abs(Y - slit_center_offset) < slit_half_height
slit2_mask = np.abs(Y + slit_center_offset) < slit_half_height

S = np.zeros((Ny, Nx), dtype=float)

if BARRIER_SMOOTH <= 0.0:
    barrier_mask = barrier_core.copy()
    barrier_mask[slit1_mask] = False
    barrier_mask[slit2_mask] = False
    S[barrier_mask] = S0
else:
    dist = np.abs(X - barrier_center_x) - (barrier_thickness / 2.0)
    wall = 1.0 / (1.0 + np.exp(dist / BARRIER_SMOOTH))
    wall[slit1_mask] = 0.0
    wall[slit2_mask] = 0.0
    S = S0 * wall

# ====================================================
# 3) CAP + ruutualue
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

V_fwd = -1j * W
V_adj = np.conjugate(V_fwd)

def cap_phase(V_cap, dt):
    return np.exp(-1j * V_cap * dt / hbar)

def scalar_mass_phase(S, dt):
    ph_u = np.exp(-1j * S * dt / (2.0 * hbar))
    ph_l = np.exp(+1j * S * dt / (2.0 * hbar))
    return ph_u, ph_l

# ====================================================
# 4) Spinori-apufunktiot
# ====================================================
def density(psi):
    return (np.abs(psi[0])**2 + np.abs(psi[1])**2)

def norm_L2(psi):
    return float(np.sqrt(np.sum(density(psi)) * dx * dy))

def normalize_unit(psi):
    n = norm_L2(psi)
    if n <= 0:
        return psi, 0.0
    return psi / n, n

def expval_xy_unitnorm(psi_unit):
    p = density(psi_unit)
    n = float(np.sum(p) * dx * dy)
    if n <= 0:
        return 0.0, 0.0
    mx = float(np.sum(p * X) * dx * dy / n)
    my = float(np.sum(p * Y) * dx * dy / n)
    return mx, my

# ====================================================
# 5) Jatkuva mittaus (SSE)
# ====================================================
USE_CONTINUOUS_MEAS = True
KAPPA_MEAS = 0.02
MEAS_SEED = 1234

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
    factor = np.exp(drift + stoch)

    psi_u2 = psi_u.copy()
    psi_u2[0] *= factor
    psi_u2[1] *= factor

    psi_u2, _ = normalize_unit(psi_u2)
    psi_new = psi_u2 * n0
    return psi_new, (dWx, dWy, mx, my)

# ====================================================
# 6) Dirac free-step (k-avaruudessa)
# ====================================================
mc2 = m * c * c

def dirac_free_step(psi, dt):
    psi0_k = np.fft.fft2(psi[0])
    psi1_k = np.fft.fft2(psi[1])

    H11 = mc2
    H22 = -mc2
    H12 = c * (PX - 1j * PY)
    H21 = c * (PX + 1j * PY)

    cp2 = (c*PX)**2 + (c*PY)**2
    omega = np.sqrt(cp2 + mc2*mc2)

    a = omega * (dt / hbar)
    cos_a = np.cos(a)
    sin_a = np.sin(a)

    inv_omega = np.where(omega > 0, 1.0/omega, 0.0)

    U11 = cos_a - 1j * sin_a * (H11 * inv_omega)
    U22 = cos_a - 1j * sin_a * (H22 * inv_omega)
    U12 = -1j * sin_a * (H12 * inv_omega)
    U21 = -1j * sin_a * (H21 * inv_omega)

    out0_k = U11 * psi0_k + U12 * psi1_k
    out1_k = U21 * psi0_k + U22 * psi1_k

    out = np.empty_like(psi)
    out[0] = np.fft.ifft2(out0_k)
    out[1] = np.fft.ifft2(out1_k)
    return out

# ====================================================
# 7) Split-step: (massgap + CAP)/2 -> free -> (massgap + CAP)/2
# ====================================================
def step_dirac_massgap_cap(psi, dt, S_field, Vcap):
    ph_u_mass, ph_l_mass = scalar_mass_phase(S_field, dt)
    ph_cap = cap_phase(Vcap, dt/2.0)

    out = psi.copy()
    out[0] *= (ph_u_mass * ph_cap)
    out[1] *= (ph_l_mass * ph_cap)

    out = dirac_free_step(out, dt)

    out[0] *= (ph_u_mass * ph_cap)
    out[1] *= (ph_l_mass * ph_cap)
    return out

# ====================================================
# 8) Alkuarvo: Dirac-spinori
# ====================================================
def make_spinor_packet(x0, y0, sigma0, k0x, k0y):
    XR = X - x0
    YR = Y - y0
    amp = np.exp(-(XR**2 + YR**2) / (2 * sigma0**2))
    phase = np.exp(1j * (k0x * X + k0y * Y))
    scalar = amp * phase

    psi = np.zeros((2, Ny, Nx), dtype=np.complex128)
    psi[0] = scalar
    psi[1] = 0.0
    psi, _ = normalize_unit(psi)
    return psi

sigma0 = 1.0
k0x    = 5.0
k0y    = 0.0
x0     = -15.0
y0     = 0.0

psi0 = make_spinor_packet(x0, y0, sigma0, k0x, k0y)

# ====================================================
# 9) Aika-askeleet
# ====================================================
dt         = 0.003
n_steps    = 1400
save_every = 5
tau_step   = save_every * dt

# ====================================================
# 10) Forward-ajon funktio + printit
#     + option to store spinor frames (visible window) for current computation
# ====================================================
PRINT_EVERY_STEPS_FWD = 200

def run_forward(store_prob_frames=True, store_spinor_frames=False, target_step=None):
    rng_local = np.random.default_rng(MEAS_SEED)
    psi = psi0.copy()

    frames = []
    times = []
    norms = []
    spinor_frames = []  # (Nt, 2, Hy, Hx)

    psi_target = None

    for n in range(n_steps + 1):
        if (n % save_every == 0):
            if store_prob_frames:
                prob = density(psi)
                frames.append(prob[ys, xs].copy())
                times.append(n * dt)
                norms.append(float(np.sum(prob) * dx * dy))

            if store_spinor_frames:
                spinor_frames.append(psi[:, ys, xs].copy())

        if (n % PRINT_EVERY_STEPS_FWD) == 0 or n == n_steps:
            prob = density(psi)
            norm_now = float(np.sum(prob) * dx * dy)
            print(f"[FWD] step {n:5d}/{n_steps}, t={n*dt:7.3f}, norm≈{norm_now:.6f}")

        if target_step is not None and n == target_step:
            psi_target = psi.copy()
            break

        if n < n_steps:
            psi = step_dirac_massgap_cap(psi, dt, S, V_fwd)
            if USE_CONTINUOUS_MEAS:
                psi, _ = continuous_measurement_update_preserve_norm(psi, dt, KAPPA_MEAS, rng_local)

    frames_arr = np.array(frames) if store_prob_frames else None
    times_arr  = np.array(times) if store_prob_frames else None
    norms_arr  = np.array(norms) if store_prob_frames else None
    spinor_arr = np.array(spinor_frames) if store_spinor_frames else None

    return frames_arr, times_arr, norms_arr, spinor_arr, psi_target

# ====================================================
# 11) Forward: tallenna |psi|^2 (+ spinor, jos testi)
# ====================================================
print("Forward (Dirac + massgap barrier): alkaa... (continuous measurement ON)" if USE_CONTINUOUS_MEAS else "Forward alkaa...")
frames_psi, times, norms_psi, spinor_vis, _ = run_forward(
    store_prob_frames=True,
    store_spinor_frames=SAVE_SPINOR_FRAMES,
    target_step=None
)
Nt = len(times)
print("Forward valmis.")

# ====================================================
# 12) Detektioajan arvio + Born-otanta klikille
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
rng_click = np.random.default_rng()
flat_idx = int(rng_click.choice(p.size, p=p))
iy_vis, ix_vis = np.unravel_index(flat_idx, w.shape)

iy_click = (cy - hy) + iy_vis
ix_click = (cx - hx) + ix_vis
x_click = float(X[iy_click, ix_click])
y_click = float(Y[iy_click, ix_click])
print(f"Click: x≈{x_click:.3f}, y≈{y_click:.3f}")

# ====================================================
# 13) Backward library: phi_tau_frames[j] = |phi(τ_j)|^2 + printit
# ====================================================
sigma_click = 0.4

def make_phi_at_click():
    Xc = X - x_click
    Yc = Y - y_click
    g = np.exp(-(Xc**2 + Yc**2) / (2 * sigma_click**2)).astype(np.complex128)
    phi = np.zeros((2, Ny, Nx), dtype=np.complex128)
    phi[0] = g
    phi[1] = 0.0
    phi, _ = normalize_unit(phi)
    return phi

PRINT_EVERY_FRAMES_BWD = 20

print("Backward library (Dirac + massgap): lasketaan phi_tau (vain 1×)...")
phi = make_phi_at_click()

phi_tau_frames = np.zeros((Nt, N_VISIBLE_Y, N_VISIBLE_X), dtype=float)

for i in range(Nt):
    phi_tau_frames[i] = density(phi)[ys, xs].copy()

    if (i % PRINT_EVERY_FRAMES_BWD) == 0 or i == Nt - 1:
        tau = i * tau_step
        norm_phi = float(np.sum(density(phi)) * dx * dy)
        print(f"[BWD] frame {i:4d}/{Nt-1}, tau={tau:7.3f}, norm≈{norm_phi:.6f}")

    if i < Nt - 1:
        for _ in range(save_every):
            phi = step_dirac_massgap_cap(phi, -dt, S, V_adj)

print("phi_tau-kirjasto valmis.")

# ====================================================
# 14) Emix(t; sigmaT) aikasiirrolla (with time interpolation -> reduces banding)
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

def build_Emix_from_phi_tau(phi_tau_frames, times, t_det, sigmaT, tau_step, K_JITTER=13):
    Nt = len(times)
    halfK = K_JITTER // 2
    idx_det2 = int(np.argmin(np.abs(times - t_det)))

    k_inds = np.arange(idx_det2 - halfK, idx_det2 + halfK + 1)
    k_inds = np.clip(k_inds, 0, Nt - 1)
    k_inds = np.unique(k_inds)

    Tk = times[k_inds]
    w = gaussian_weights(Tk, t_det, sigmaT)

    Emix = np.zeros((Nt, phi_tau_frames.shape[1], phi_tau_frames.shape[2]), dtype=float)

    for i, ti in enumerate(times):
        tau = Tk - ti
        valid = tau >= 0.0
        if not np.any(valid):
            continue

        j_float = tau[valid] / tau_step

        j0 = np.floor(j_float).astype(int)
        j1 = j0 + 1

        j0 = np.clip(j0, 0, Nt-1)
        j1 = np.clip(j1, 0, Nt-1)

        alpha = (j_float - j0).astype(float)

        phi_interp = (1 - alpha)[:, None, None] * phi_tau_frames[j0] + \
                     alpha[:, None, None] * phi_tau_frames[j1]

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
# 15) Dirac probability current (visible window) + velocity field v=j/rho
#   jx = c * psi^† σx psi = 2c Re(psi0^* psi1)
#   jy = c * psi^† σy psi = 2c Im(psi0^* psi1)
# ====================================================
def dirac_current_visible(psi_vis):
    psi0 = psi_vis[0]
    psi1 = psi_vis[1]
    rho = (np.abs(psi0)**2 + np.abs(psi1)**2).astype(float)

    z = np.conjugate(psi0) * psi1
    jx = (2.0 * c * np.real(z)).astype(float)
    jy = (2.0 * c * np.imag(z)).astype(float)
    return jx, jy, rho

def velocity_from_current(jx, jy, rho, eps=ALIGN_EPS_RHO):
    """
    v = j / rho (robust: clamp rho from below by eps).
    Returns vx, vy, speed.
    """
    denom = np.maximum(rho, eps)
    vx = jx / denom
    vy = jy / denom
    sp = np.hypot(vx, vy)
    return vx, vy, sp

# ====================================================
# 16) RIDGE extraction helpers
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
    x0 = int(np.clip(prev_ix, 0, W_-1))
    y0 = int(np.clip(prev_iy, 0, H-1))
    x1 = max(0, x0 - radius)
    x2 = min(W_, x0 + radius + 1)
    y1 = max(0, y0 - radius)
    y2 = min(H, y0 + radius + 1)

    sub = Gamma[y1:y2, x1:x2]
    if sub.size == 0:
        return ridge_argmax(Gamma, x_vis_1d, y_vis_1d) + (x0, y0)

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
    Nt = frames_psi.shape[0]
    ridge_x = np.zeros(Nt, dtype=float)
    ridge_y = np.zeros(Nt, dtype=float)
    ridge_s = np.zeros(Nt, dtype=float)

    Gamma0 = frames_psi[0] * Emix[0]
    x0, y0, s0 = ridge_argmax(Gamma0, x_vis_1d, y_vis_1d)
    ridge_x[0], ridge_y[0], ridge_s[0] = x0, y0, s0
    prev_ix = int(np.argmin(np.abs(x_vis_1d - x0)))
    prev_iy = int(np.argmin(np.abs(y_vis_1d - y0)))

    for i in range(1, Nt):
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
# 17) Ridge tangent (unit) + robust cos(theta) from velocity direction
# ====================================================
def ridge_tangent_unit(ridge_x, ridge_y):
    Nt = len(ridge_x)
    tx = np.zeros(Nt, dtype=float)
    ty = np.zeros(Nt, dtype=float)
    for i in range(1, Nt-1):
        dxp = ridge_x[i+1] - ridge_x[i-1]
        dyp = ridge_y[i+1] - ridge_y[i-1]
        n = np.hypot(dxp, dyp)
        if n > 0:
            tx[i] = dxp / n
            ty[i] = dyp / n
    if Nt >= 2:
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
    """
    Weighted mean of velocity components around (iy, ix).
    Weight = kernel * speed (so low-speed pixels contribute less)
    Returns unit direction (ux, uy) and avg_speed.
    """
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

def alignment_series_from_spinor(spinor_vis, ridge_x, ridge_y, x_vis_1d, y_vis_1d,
                                 eps_rho=ALIGN_EPS_RHO, eps_speed=ALIGN_EPS_SPEED):
    """
    Robust cos(theta) and velocity direction at ridge:
      - compute v=j/rho with rho-clamp
      - take local gaussian-speed-weighted average around ridge pixel
      - compute cosθ = t_hat · v_hat
    Returns: cos_th, speed, ux, uy  (arrays length Nt; NaN when invalid)
    """
    Nt = len(ridge_x)
    cos_th = np.full(Nt, np.nan, dtype=float)
    speed  = np.full(Nt, np.nan, dtype=float)
    ux     = np.full(Nt, np.nan, dtype=float)
    uy     = np.full(Nt, np.nan, dtype=float)

    tx, ty = ridge_tangent_unit(ridge_x, ridge_y)

    for i in range(Nt):
        ix = int(np.argmin(np.abs(x_vis_1d - ridge_x[i])))
        iy = int(np.argmin(np.abs(y_vis_1d - ridge_y[i])))

        jx, jy, rho_dirac = dirac_current_visible(spinor_vis[i])
        vx, vy, sp = velocity_from_current(jx, jy, rho_dirac, eps=eps_rho)

        if ARROW_SPATIAL_AVG:
            uxi, uyi, spd = local_weighted_mean(vx, vy, sp, ix, iy)
        else:
            spd = float(sp[iy, ix])
            if spd > 0:
                uxi = float(vx[iy, ix] / spd)
                uyi = float(vy[iy, ix] / spd)
            else:
                uxi, uyi = np.nan, np.nan

        if not (np.isfinite(uxi) and np.isfinite(uyi) and np.isfinite(spd) and spd > eps_speed):
            continue

        ux[i] = uxi
        uy[i] = uyi
        speed[i] = spd
        cos_th[i] = float(uxi * tx[i] + uyi * ty[i])

    # Optional temporal smoothing for direction + cosθ (helps end-of-run jitter)
    if ARROW_TEMPORAL_SMOOTH:
        a = float(np.clip(ARROW_SMOOTH_ALPHA, 0.0, 1.0))
        ux_f = ux.copy()
        uy_f = uy.copy()
        sp_f = speed.copy()

        # forward pass EMA on valid segments
        last_u = None
        last_v = None
        last_s = None
        for i in range(Nt):
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
        for i in range(Nt):
            if np.isfinite(ux[i]) and np.isfinite(uy[i]):
                cos_th[i] = float(ux[i] * tx[i] + uy[i] * ty[i])

    # hard clip numerical overshoots
    cos_th = np.clip(cos_th, -1.0, 1.0)
    return cos_th, speed, ux, uy

# ====================================================
# 18) Build rho + ridge + (optional) velocity/align test for given sigmaT
# ====================================================
def build_all_for_sigma(sigmaT):
    Emix = build_Emix_from_phi_tau(phi_tau_frames, times, t_det, sigmaT=sigmaT,
                                   tau_step=tau_step, K_JITTER=13)
    rho  = make_rho(frames_psi, Emix)
    rx, ry, rs = compute_ridge_xy(
        frames_psi, Emix, x_vis_1d, y_vis_1d,
        mode=RIDGE_MODE,
        top_q=CENTROID_TOP_Q,
        radius=LOCALMAX_RADIUS,
        alpha_smooth=LOCALMAX_SMOOTH_ALPHA
    )

    cos_th = speed = ux = uy = None
    if SAVE_SPINOR_FRAMES and (spinor_vis is not None):
        cos_th, speed, ux, uy = alignment_series_from_spinor(
            spinor_vis, rx, ry, x_vis_1d, y_vis_1d
        )

    return rho, Emix, rx, ry, rs, cos_th, speed, ux, uy

# ====================================================
# 19) Slider range for sigmaT
# ====================================================
p0 = hbar * k0x
E0 = np.sqrt((m*c*c)**2 + (p0*p0)*(c*c))
v_g = (c*c * p0) / (E0 + 1e-12)

L_gap = screen_center_x - barrier_center_x
t_gap = L_gap / (v_g + 1e-12)

SIGMA_MIN  = 0.05 * t_gap
SIGMA_MAX  = 2.00 * t_gap
SIGMA_INIT = 0.60 * t_gap

rho_init, Emix_init, ridge_x_init, ridge_y_init, ridge_s_init, cos_th_init, speed_init, ux_init, uy_init = build_all_for_sigma(SIGMA_INIT)

# fixed display scale reference
if USE_FIXED_DISPLAY_SCALE:
    vref = float(np.quantile(rho_init, DISPLAY_Q) + 1e-30)
else:
    vref = 1.0

def gamma_display(arr, gamma=GAMMA):
    if USE_FIXED_DISPLAY_SCALE:
        disp = np.clip(arr / vref, 0.0, 1.0)
        return disp**gamma
    else:
        mmax = np.max(arr)
        if mmax <= 0:
            return arr
        disp = arr / mmax
        return disp**gamma

# speed reference for arrow scaling (keep stable)
if speed_init is not None:
    vv = speed_init[np.isfinite(speed_init)]
    speed_ref = float(np.quantile(vv, 0.80)) if vv.size > 0 else 1.0
else:
    speed_ref = 1.0

# ====================================================
# 20) Visualisointi: rho + ridge marker (+ trail) + flow arrow + slider + animaatio
# ====================================================
extent = (-VISIBLE_LX/2, VISIBLE_LX/2, -VISIBLE_LY/2, VISIBLE_LY/2)

fig = plt.figure(figsize=(10.8, 7.2))
ax = fig.add_axes([0.07, 0.18, 0.86, 0.78])

im = ax.imshow(
    gamma_display(rho_init[0]),
    extent=extent,
    origin='lower',
    vmin=0.0, vmax=1.0,
    cmap='magma',
    interpolation=IM_INTERPOLATION
)

ax.axvline(barrier_center_x, color='white', linestyle='--', alpha=0.6)
ax.axvline(screen_center_x,  color='cyan',  linestyle='--', alpha=0.4)
ax.set_xlabel("x")
ax.set_ylabel("y")

title = ax.set_title(
    rf"Dirac+massgap: ρ(t), σT={SIGMA_INIT:.3f}, t={times[0]:.3f} | ridge={RIDGE_MODE}"
)

ridge_marker, = ax.plot([ridge_x_init[0]], [ridge_y_init[0]],
                        marker='o', markersize=7, linestyle='None',
                        color='lime', alpha=0.9, label=f"ridge ({RIDGE_MODE})")

SHOW_TRAIL = True
TRAIL_LEN = 40
ridge_trail, = ax.plot([], [], linestyle='-', linewidth=1.5, color='lime', alpha=0.5)

# flow arrow (use quiver with 1 vector)
flow_quiver = None
if DRAW_FLOW_ARROW and SAVE_SPINOR_FRAMES and (ux_init is not None):
    flow_quiver = ax.quiver(
        [ridge_x_init[0]], [ridge_y_init[0]],
        [0.0], [0.0],
        angles='xy', scale_units='xy', scale=1.0,
        color='cyan', alpha=0.9, width=0.006
    )

ax.legend(loc="upper right", framealpha=0.35)

# Slider
ax_sigma = fig.add_axes([0.10, 0.08, 0.80, 0.04])
sigma_slider = Slider(
    ax=ax_sigma,
    label="sigmaT (time thickness)",
    valmin=SIGMA_MIN,
    valmax=SIGMA_MAX,
    valinit=SIGMA_INIT,
)

# state holders
rho_current = [rho_init]
sigma_current = [SIGMA_INIT]
ridge_x = [ridge_x_init]
ridge_y = [ridge_y_init]
ridge_s = [ridge_s_init]
cos_th = [cos_th_init]
speed = [speed_init]
ux = [ux_init]
uy = [uy_init]

# arrow hysteresis state
arrow_state = {"ux": np.nan, "uy": np.nan, "spd": np.nan}

def recompute_for_sigma(new_sigma):
    rho, Emix, rx, ry, rs, cth, spd, uxx, uyy = build_all_for_sigma(new_sigma)
    return rho, rx, ry, rs, cth, spd, uxx, uyy

def update_flow_arrow(i):
    if flow_quiver is None or ux[0] is None or speed[0] is None:
        return
    if (i % ARROW_STRIDE) != 0:
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

    # temporal smoothing on arrow direction (extra layer on top of precomputed smoothing)
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

    # Arrow length from speed (clamped)
    L = ARROW_SCALE * float(np.clip(spd / (speed_ref + 1e-30), 0.0, 2.5))
    flow_quiver.set_UVC([L * uxi], [L * uyi])

def on_sigma_change(_val):
    new_sigma = float(sigma_slider.val)
    sigma_current[0] = new_sigma

    rho_new, rx, ry, rs, cth, spd, uxx, uyy = recompute_for_sigma(new_sigma)
    rho_current[0] = rho_new
    ridge_x[0] = rx
    ridge_y[0] = ry
    ridge_s[0] = rs
    cos_th[0] = cth
    speed[0] = spd
    ux[0] = uxx
    uy[0] = uyy

    i = getattr(on_sigma_change, "last_i", 0)

    im.set_data(gamma_display(rho_current[0][i]))
    ridge_marker.set_data([ridge_x[0][i]], [ridge_y[0][i]])

    if SHOW_TRAIL:
        j0 = max(0, i - TRAIL_LEN + 1)
        ridge_trail.set_data(ridge_x[0][j0:i+1], ridge_y[0][j0:i+1])

    update_flow_arrow(i)

    cth_txt = ""
    if cos_th[0] is not None and np.isfinite(cos_th[0][i]):
        cth_txt = rf" | cosθ≈{cos_th[0][i]:.3f}"

    title.set_text(
        rf"Dirac+massgap: ρ(t), σT={sigma_current[0]:.3f}, t={times[i]:.3f} | "
        rf"ridge={RIDGE_MODE} | norm≈{norms_psi[i]:.4f} | Γ≈{ridge_s[0][i]:.3e}"
        + cth_txt
    )
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

    cth_txt = ""
    if cos_th[0] is not None and np.isfinite(cos_th[0][i]):
        cth_txt = rf" | cosθ≈{cos_th[0][i]:.3f}"

    title.set_text(
        rf"Dirac+massgap: ρ(t), σT={sigma_current[0]:.3f}, t={times[i]:.3f} | "
        rf"ridge={RIDGE_MODE} | norm≈{norms_psi[i]:.4f} | Γ≈{ridge_s[0][i]:.3e}"
        + cth_txt
    )
    return (im, ridge_marker, ridge_trail)

ani = FuncAnimation(fig, update, frames=Nt, interval=40, blit=False)

# ====================================================
# 21) Alignment stats print
# ====================================================
if PRINT_ALIGNMENT_STATS and SAVE_SPINOR_FRAMES and (cos_th_init is not None):
    valid = np.isfinite(cos_th_init)
    if np.any(valid):
        mean_c = float(np.mean(cos_th_init[valid]))
        med_c  = float(np.median(cos_th_init[valid]))
        frac_pos = float(np.mean(cos_th_init[valid] > 0.0))
        frac_hi  = float(np.mean(cos_th_init[valid] > 0.7))
        print(f"[ALIGN] ridge vs velocity: mean cosθ≈{mean_c:.3f}, median≈{med_c:.3f}, "
              f"frac(cosθ>0)≈{frac_pos:.3f}, frac(cosθ>0.7)≈{frac_hi:.3f}")
    else:
        print("[ALIGN] no valid cosθ (velocity too small / rho too small at ridge points).")

# ====================================================
# 22) Save video (optional)
#   (no extra savefig args to avoid breaking, per your note)
# ====================================================
ani.save(
    f"dirac_massgap_ridge_{RIDGE_MODE}_flow.mp4",
    writer="ffmpeg",
    fps=25,
    dpi=150
)

plt.show()