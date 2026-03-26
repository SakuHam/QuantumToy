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

# Näkyvän ikkunan indeksit
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
# 3) Normalisointi ja odotusarvot
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

    psi_new = psi_u2 * n0
    return psi_new, (dWx, dWy, mx, my)

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
# 6.5) Worldline tube bias forwardiin
# ====================================================
USE_WORLDLINE_FORWARD_BIAS = True

# Valintatapa:
#   "weaker_branch"  -> valitse heikompi haara
#   "stronger_branch"-> valitse vahvempi haara
#   "upper_branch"   -> pakota y > 0
#   "lower_branch"   -> pakota y < 0
WL_BRANCH_MODE = "weaker_branch"

WL_TOPK_PER_BRANCH   = 3
WL_MIN_PEAK_DIST_PX  = 18
WL_TRACK_RADIUS_PX   = 20
WL_MIN_LOCAL_REL     = 0.05

# Tube geometry
WL_SIGMA_LONG  = 1.8
WL_SIGMA_TRANS = 0.55
WL_AHEAD_SHIFT = 0.60

# Bias strength
WL_GAIN = 1.15
WL_DAMP = 0.05

# Smooth turn-on
v_est = k0x / m_mass
t_barrier_est = (barrier_center_x - x0) / max(v_est, 1e-12)
t_gap_est = (screen_center_x - barrier_center_x) / max(v_est, 1e-12)
WL_ENABLE_TIME = t_barrier_est + 0.18 * t_gap_est
WL_RAMP_TIME   = 0.30

# Flow / velocity blend
WL_VEL_SMOOTH = 0.75
WL_FLOW_BLEND = 0.85   # 1.0 = vain local current, 0.0 = vain tracker velocity
WL_FLOW_SAMPLE_RADIUS_PX = 3
WL_RHO_FLOOR = 1e-12

# Debug
WL_PRINT_TOP_PEAKS = 3

# ====================================================
# 6.6) Worldline apufunktiot
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

def find_top_peaks_2d(arr, top_k=3, min_dist_px=12, mask=None):
    work = np.array(arr, dtype=float, copy=True)
    if mask is not None:
        work = np.where(mask, work, -np.inf)

    peaks = []
    ny, nx = work.shape

    for _ in range(top_k):
        flat_idx = int(np.argmax(work))
        val = float(work.ravel()[flat_idx])
        if not np.isfinite(val):
            break

        iy, ix = np.unravel_index(flat_idx, work.shape)
        peaks.append((val, iy, ix))

        y0 = max(0, iy - min_dist_px)
        y1 = min(ny, iy + min_dist_px + 1)
        x0 = max(0, ix - min_dist_px)
        x1 = min(nx, ix + min_dist_px + 1)

        yy, xx = np.ogrid[y0:y1, x0:x1]
        rr2 = (yy - iy)**2 + (xx - ix)**2

        block = work[y0:y1, x0:x1]
        block[rr2 <= min_dist_px**2] = -np.inf
        work[y0:y1, x0:x1] = block

    return peaks

def extract_local_peak(prob_vis, iy0, ix0, radius_px=16):
    ny, nx = prob_vis.shape
    y0 = max(0, iy0 - radius_px)
    y1 = min(ny, iy0 + radius_px + 1)
    x0 = max(0, ix0 - radius_px)
    x1 = min(nx, ix0 + radius_px + 1)

    sub = prob_vis[y0:y1, x0:x1]
    flat_idx = int(np.argmax(sub))
    val = float(sub.ravel()[flat_idx])
    sy, sx = np.unravel_index(flat_idx, sub.shape)
    return val, y0 + sy, x0 + sx

def choose_branch_seed(prob_vis):
    branch_zone = (
        (X_vis > (barrier_center_x + 0.4)) &
        (X_vis < (screen_center_x - 0.8))
    )

    mask_upper = branch_zone & (Y_vis > 0.0)
    mask_lower = branch_zone & (Y_vis < 0.0)

    peaks_upper = find_top_peaks_2d(
        prob_vis,
        top_k=WL_TOPK_PER_BRANCH,
        min_dist_px=WL_MIN_PEAK_DIST_PX,
        mask=mask_upper
    )
    peaks_lower = find_top_peaks_2d(
        prob_vis,
        top_k=WL_TOPK_PER_BRANCH,
        min_dist_px=WL_MIN_PEAK_DIST_PX,
        mask=mask_lower
    )

    print("[WL-TUBE] upper peaks:")
    for j, (val, iy, ix) in enumerate(peaks_upper[:WL_PRINT_TOP_PEAKS]):
        print(
            f"  upper#{j+1}: val={val:.6e}, "
            f"x={X_vis[iy, ix]:.3f}, y={Y_vis[iy, ix]:.3f}, iy={iy}, ix={ix}"
        )

    print("[WL-TUBE] lower peaks:")
    for j, (val, iy, ix) in enumerate(peaks_lower[:WL_PRINT_TOP_PEAKS]):
        print(
            f"  lower#{j+1}: val={val:.6e}, "
            f"x={X_vis[iy, ix]:.3f}, y={Y_vis[iy, ix]:.3f}, iy={iy}, ix={ix}"
        )

    best_upper = peaks_upper[0] if len(peaks_upper) > 0 else None
    best_lower = peaks_lower[0] if len(peaks_lower) > 0 else None

    if best_upper is None and best_lower is None:
        return None, None

    if best_upper is None:
        return best_lower, "lower"
    if best_lower is None:
        return best_upper, "upper"

    val_u = best_upper[0]
    val_l = best_lower[0]

    if WL_BRANCH_MODE == "upper_branch":
        return best_upper, "upper"
    if WL_BRANCH_MODE == "lower_branch":
        return best_lower, "lower"
    if WL_BRANCH_MODE == "stronger_branch":
        if val_u >= val_l:
            return best_upper, "upper"
        return best_lower, "lower"

    # weaker_branch
    if val_u <= val_l:
        return best_upper, "upper"
    return best_lower, "lower"

def rotated_tube_field(xc, yc, vx, vy, sigma_long, sigma_trans, ahead_shift):
    vnorm = np.hypot(vx, vy)
    if vnorm < 1e-12:
        ux, uy = 1.0, 0.0
    else:
        ux, uy = vx / vnorm, vy / vnorm

    # kohtisuora
    nxp, nyp = -uy, ux

    # hieman eteenpäin
    xc2 = xc + ahead_shift * ux
    yc2 = yc + ahead_shift * uy

    dxg = X - xc2
    dyg = Y - yc2

    s_long  = dxg * ux  + dyg * uy
    s_trans = dxg * nxp + dyg * nyp

    tube = np.exp(
        -0.5 * (s_long / sigma_long)**2
        -0.5 * (s_trans / sigma_trans)**2
    )
    return tube

def apply_forward_worldline_tube_bias_preserve_norm(
    psi,
    xc,
    yc,
    vx,
    vy,
    dt,
    gain,
    damp,
    sigma_long,
    sigma_trans,
    ahead_shift,
    ramp_scale
):
    if ramp_scale <= 0:
        return psi

    n0 = norm_L2(psi)
    if n0 <= 0:
        return psi

    tube = rotated_tube_field(
        xc, yc, vx, vy,
        sigma_long=sigma_long,
        sigma_trans=sigma_trans,
        ahead_shift=ahead_shift
    )

    field = gain * tube - damp * (1.0 - tube)
    psi2 = psi * np.exp(dt * ramp_scale * field)

    psi2_u, _ = normalize_unit(psi2)
    psi2 = psi2_u * n0
    return psi2

def schrodinger_current(psi):
    dpsi_dx = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2.0 * dx)
    dpsi_dy = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2.0 * dy)

    rho = np.abs(psi)**2
    jx = (hbar / m_mass) * np.imag(np.conjugate(psi) * dpsi_dx)
    jy = (hbar / m_mass) * np.imag(np.conjugate(psi) * dpsi_dy)
    return rho, jx, jy

def local_mean_flow_velocity(psi, iy_full, ix_full, radius_px=2):
    rho, jx, jy = schrodinger_current(psi)

    y0 = max(0, iy_full - radius_px)
    y1 = min(rho.shape[0], iy_full + radius_px + 1)
    x0 = max(0, ix_full - radius_px)
    x1 = min(rho.shape[1], ix_full + radius_px + 1)

    rho_sub = rho[y0:y1, x0:x1]
    jx_sub  = jx[y0:y1, x0:x1]
    jy_sub  = jy[y0:y1, x0:x1]

    w = np.maximum(rho_sub, 0.0)
    wsum = float(np.sum(w))
    if wsum <= WL_RHO_FLOOR:
        return 0.0, 0.0

    jx_mean = float(np.sum(jx_sub * w) / wsum)
    jy_mean = float(np.sum(jy_sub * w) / wsum)
    rho_mean = float(np.sum(rho_sub * w) / wsum)

    if rho_mean <= WL_RHO_FLOOR:
        return 0.0, 0.0

    vx = jx_mean / rho_mean
    vy = jy_mean / rho_mean
    return vx, vy

def blend_directions(vx_flow, vy_flow, vx_track, vy_track, blend):
    vx = blend * vx_flow + (1.0 - blend) * vx_track
    vy = blend * vy_flow + (1.0 - blend) * vy_track
    return vx, vy

# ====================================================
# 6.7) Worldline tila
# ====================================================
wl_state = {
    "enabled": USE_WORLDLINE_FORWARD_BIAS,
    "locked": False,
    "branch_name": None,
    "ref_amp": None,
    "iy_vis": None,
    "ix_vis": None,
    "iy_full": None,
    "ix_full": None,
    "x": None,
    "y": None,
    "vx_track": v_est,
    "vy_track": 0.0,
    "vx_flow": v_est,
    "vy_flow": 0.0,
    "vx": v_est,
    "vy": 0.0,
    "history_x": [],
    "history_y": [],
    "history_t": [],
}

def maybe_initialize_or_update_worldline(psi_cur, t_now, dt, wl_state):
    if not wl_state["enabled"]:
        return wl_state

    if t_now < WL_ENABLE_TIME:
        return wl_state

    prob_vis = (np.abs(psi_cur)**2)[ys, xs]

    if not wl_state["locked"]:
        chosen, branch_name = choose_branch_seed(prob_vis)
        if chosen is None:
            return wl_state

        val, iy_vis, ix_vis = chosen
        iy_full = (cy - hy) + iy_vis
        ix_full = (cx - hx) + ix_vis

        x_sel = float(X_vis[iy_vis, ix_vis])
        y_sel = float(Y_vis[iy_vis, ix_vis])

        vx_flow, vy_flow = local_mean_flow_velocity(
            psi_cur, iy_full, ix_full,
            radius_px=WL_FLOW_SAMPLE_RADIUS_PX
        )

        wl_state["locked"] = True
        wl_state["branch_name"] = branch_name
        wl_state["ref_amp"] = max(float(val), 1e-30)
        wl_state["iy_vis"] = int(iy_vis)
        wl_state["ix_vis"] = int(ix_vis)
        wl_state["iy_full"] = int(iy_full)
        wl_state["ix_full"] = int(ix_full)
        wl_state["x"] = x_sel
        wl_state["y"] = y_sel

        wl_state["vx_track"] = v_est
        wl_state["vy_track"] = 0.0
        wl_state["vx_flow"] = vx_flow
        wl_state["vy_flow"] = vy_flow
        wl_state["vx"], wl_state["vy"] = blend_directions(
            vx_flow, vy_flow,
            wl_state["vx_track"], wl_state["vy_track"],
            WL_FLOW_BLEND
        )

        wl_state["history_x"].append(wl_state["x"])
        wl_state["history_y"].append(wl_state["y"])
        wl_state["history_t"].append(t_now)

        print(
            f"[WL-TUBE] chosen {branch_name} branch: "
            f"x={wl_state['x']:.3f}, y={wl_state['y']:.3f}, val={val:.6e}"
        )
        print(
            f"[WL-TUBE] initial dirs: "
            f"flow=({wl_state['vx_flow']:.3f}, {wl_state['vy_flow']:.3f}), "
            f"track=({wl_state['vx_track']:.3f}, {wl_state['vy_track']:.3f}), "
            f"blend=({wl_state['vx']:.3f}, {wl_state['vy']:.3f})"
        )
        return wl_state

    # trackkaa lokaali peak
    iy0 = wl_state["iy_vis"]
    ix0 = wl_state["ix_vis"]

    val, iy_new, ix_new = extract_local_peak(
        prob_vis, iy0, ix0,
        radius_px=WL_TRACK_RADIUS_PX
    )

    old_x = wl_state["x"]
    old_y = wl_state["y"]

    if val >= WL_MIN_LOCAL_REL * wl_state["ref_amp"]:
        new_x = float(X_vis[iy_new, ix_new])
        new_y = float(Y_vis[iy_new, ix_new])

        iy_full = (cy - hy) + iy_new
        ix_full = (cx - hx) + ix_new

        vx_inst = (new_x - old_x) / max(dt, 1e-12)
        vy_inst = (new_y - old_y) / max(dt, 1e-12)

        wl_state["vx_track"] = WL_VEL_SMOOTH * wl_state["vx_track"] + (1.0 - WL_VEL_SMOOTH) * vx_inst
        wl_state["vy_track"] = WL_VEL_SMOOTH * wl_state["vy_track"] + (1.0 - WL_VEL_SMOOTH) * vy_inst

        vx_flow, vy_flow = local_mean_flow_velocity(
            psi_cur, iy_full, ix_full,
            radius_px=WL_FLOW_SAMPLE_RADIUS_PX
        )
        wl_state["vx_flow"] = WL_VEL_SMOOTH * wl_state["vx_flow"] + (1.0 - WL_VEL_SMOOTH) * vx_flow
        wl_state["vy_flow"] = WL_VEL_SMOOTH * wl_state["vy_flow"] + (1.0 - WL_VEL_SMOOTH) * vy_flow

        wl_state["vx"], wl_state["vy"] = blend_directions(
            wl_state["vx_flow"], wl_state["vy_flow"],
            wl_state["vx_track"], wl_state["vy_track"],
            WL_FLOW_BLEND
        )

        wl_state["iy_vis"] = int(iy_new)
        wl_state["ix_vis"] = int(ix_new)
        wl_state["iy_full"] = int(iy_full)
        wl_state["ix_full"] = int(ix_full)
        wl_state["x"] = new_x
        wl_state["y"] = new_y

    wl_state["history_x"].append(wl_state["x"])
    wl_state["history_y"].append(wl_state["y"])
    wl_state["history_t"].append(t_now)
    return wl_state

# ====================================================
# 7) Forward: ψ(t) + continuous measurement + tube bias
# ====================================================
frames_psi = []
times      = []
norms_psi  = []

wl_path_saved_x = []
wl_path_saved_y = []
wl_dir_saved_x  = []
wl_dir_saved_y  = []
wl_branch_saved = []

psi_cur = psi.copy()

print("Forward: simulaatio alkaa...")
if USE_CONTINUOUS_MEAS:
    print("  - continuous measurement ON")
if USE_WORLDLINE_FORWARD_BIAS:
    print("  - worldline tube forward bias ON")
    print(f"  - WL branch mode = {WL_BRANCH_MODE}")
    print(f"  - WL flow blend  = {WL_FLOW_BLEND:.2f}")

for n in range(n_steps + 1):
    t_now = n * dt
    prob_psi = np.abs(psi_cur)**2
    norm_now = float(np.sum(prob_psi) * dx * dy)

    if n % save_every == 0:
        frames_psi.append(prob_psi[ys, xs].copy())
        times.append(t_now)
        norms_psi.append(norm_now)

        if wl_state["locked"]:
            wl_path_saved_x.append(wl_state["x"])
            wl_path_saved_y.append(wl_state["y"])
            wl_dir_saved_x.append(wl_state["vx"])
            wl_dir_saved_y.append(wl_state["vy"])
            wl_branch_saved.append(wl_state["branch_name"])
        else:
            wl_path_saved_x.append(np.nan)
            wl_path_saved_y.append(np.nan)
            wl_dir_saved_x.append(np.nan)
            wl_dir_saved_y.append(np.nan)
            wl_branch_saved.append("searching")

        if (len(frames_psi) % 20) == 0:
            lock_txt = wl_state["branch_name"] if wl_state["locked"] else "searching"
            print(f"[FWD] step {n:5d}/{n_steps}, t={t_now:7.3f}, norm≈{norm_now:.6f}, WL={lock_txt}")

    if n < n_steps:
        psi_cur = step_field(psi_cur, K_phase_fwd, P_half_fwd)

        if USE_CONTINUOUS_MEAS:
            psi_cur, _rec = continuous_measurement_update_preserve_norm(
                psi_cur, dt, KAPPA_MEAS, rng_meas
            )

        wl_state = maybe_initialize_or_update_worldline(psi_cur, t_now, dt, wl_state)

        if wl_state["locked"]:
            ramp_scale = ramp01(t_now, WL_ENABLE_TIME, WL_RAMP_TIME)
            psi_cur = apply_forward_worldline_tube_bias_preserve_norm(
                psi_cur,
                wl_state["x"],
                wl_state["y"],
                wl_state["vx"],
                wl_state["vy"],
                dt=dt,
                gain=WL_GAIN,
                damp=WL_DAMP,
                sigma_long=WL_SIGMA_LONG,
                sigma_trans=WL_SIGMA_TRANS,
                ahead_shift=WL_AHEAD_SHIFT,
                ramp_scale=ramp_scale
            )

frames_psi = np.array(frames_psi)
times      = np.array(times)
norms_psi  = np.array(norms_psi)

wl_path_saved_x = np.array(wl_path_saved_x, dtype=float)
wl_path_saved_y = np.array(wl_path_saved_y, dtype=float)
wl_dir_saved_x  = np.array(wl_dir_saved_x, dtype=float)
wl_dir_saved_y  = np.array(wl_dir_saved_y, dtype=float)

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
# 10) Emix(t; sigmaT)
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

        j = np.rint(tau[valid] / tau_step).astype(int)
        j = np.clip(j, 0, Nt - 1)

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
    return disp**gamma

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

initial_branch = wl_branch_saved[0] if len(wl_branch_saved) > 0 else "searching"
title = ax.set_title(
    rf"ρ(t): σT={SIGMA_INIT:.3f}, t={times[0]:.3f}, WL={initial_branch}"
)

# worldline overlay
wl_line, = ax.plot([], [], color='lime', linewidth=1.0, alpha=0.9)
wl_dot,  = ax.plot([], [], marker='o', color='cyan', linestyle='None', markersize=5, alpha=0.9)
wl_dir,  = ax.plot([], [], color='white', linewidth=1.0, alpha=0.8)

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

def update_worldline_overlay(i):
    good = np.isfinite(wl_path_saved_x[:i+1]) & np.isfinite(wl_path_saved_y[:i+1])

    if np.any(good):
        wl_line.set_data(wl_path_saved_x[:i+1][good], wl_path_saved_y[:i+1][good])
    else:
        wl_line.set_data([], [])

    if np.isfinite(wl_path_saved_x[i]) and np.isfinite(wl_path_saved_y[i]):
        wl_dot.set_data([wl_path_saved_x[i]], [wl_path_saved_y[i]])

        vx = wl_dir_saved_x[i]
        vy = wl_dir_saved_y[i]
        vnorm = np.hypot(vx, vy)

        if np.isfinite(vnorm) and vnorm > 1e-12:
            scale = 1.0
            ux = vx / vnorm
            uy = vy / vnorm
            x0 = wl_path_saved_x[i]
            y0 = wl_path_saved_y[i]
            x1 = x0 + scale * ux
            y1 = y0 + scale * uy
            wl_dir.set_data([x0, x1], [y0, y1])
        else:
            wl_dir.set_data([], [])
    else:
        wl_dot.set_data([], [])
        wl_dir.set_data([], [])

def on_sigma_change(_val):
    new_sigma = float(sigma_slider.val)
    sigma_current[0] = new_sigma
    rho_current[0] = recompute_rho_for_sigma(new_sigma)

    i = getattr(on_sigma_change, "last_i", 0)
    im.set_data(gamma_display(rho_current[0][i]))
    update_worldline_overlay(i)

    branch_now = wl_branch_saved[i] if i < len(wl_branch_saved) else "searching"
    title.set_text(
        rf"ρ(t): σT={sigma_current[0]:.3f}, t={times[i]:.3f}, "
        rf"norm≈{norms_psi[i]:.4f}, WL={branch_now}"
    )
    fig.canvas.draw_idle()

sigma_slider.on_changed(on_sigma_change)

def update(i):
    on_sigma_change.last_i = i
    im.set_data(gamma_display(rho_current[0][i]))
    update_worldline_overlay(i)

    branch_now = wl_branch_saved[i] if i < len(wl_branch_saved) else "searching"
    title.set_text(
        rf"ρ(t): σT={sigma_current[0]:.3f}, t={times[i]:.3f}, "
        rf"norm≈{norms_psi[i]:.4f}, WL={branch_now}"
    )
    return (im, wl_line, wl_dot, wl_dir)

ani = FuncAnimation(fig, update, frames=Nt, interval=40, blit=False)

ani.save("worldline_tube.mp4",
         writer="ffmpeg", fps=25, dpi=150)

plt.show()