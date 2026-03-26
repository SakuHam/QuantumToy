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

x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
X, Y = np.meshgrid(x, y)

m_mass = 1.0
hbar   = 1.0

# Näkyvän ikkunan indeksit (keskelle isoa hilaa)
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

# Fourier-hilat
kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2

def kinetic_phase(dt):
    return np.exp(-1j * K2 * dt / (2 * m_mass))

# ====================================================
# 1) Kaksoisrako: este + kaksi rakoa (pehmeä reuna optiona)
# ====================================================
barrier_center_x  = 0.0
barrier_thickness = 0.4
V_barrier         = 80.0

slit_center_offset = 2.0
slit_half_height   = 0.5

BARRIER_SMOOTH = 0.15  # 0.0 = kova, >0 pehmeä

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

# Forward: V = V_real - iW ; Adjoint backward: conj(V_fwd) = V_real + iW
V_fwd = V_real - 1j * W
V_adj = np.conjugate(V_fwd)

def potential_phase(V, dt):
    return np.exp(-1j * V * dt / hbar)

# ====================================================
# 3) Normalisointi ja odotusarvot (apufunktiot)
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
# 4) Jatkuva mittaus (heikko paikkamittaus) - SSE
# ====================================================
USE_CONTINUOUS_MEAS = True
KAPPA_MEAS = 0.02
rng_meas = np.random.default_rng(1234)

def continuous_measurement_update_preserve_norm(psi, dt, kappa, rng):
    """
    Jatkuva 2D paikkamittaus (SSE) mutta säilyttää psi:n alkuperäisen normin
    (jotta CAP absorptio ei "katoa" renormalisointiin).
    """
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
# 7) Forward: ψ(t) + (valinnainen) jatkuva mittaus
# ====================================================
frames_psi = []
times      = []
norms_psi  = []

psi_cur = psi.copy()
print("Forward: simulaatio alkaa... (continuous measurement ON)" if USE_CONTINUOUS_MEAS else "Forward: simulaatio alkaa...")

for n in range(n_steps + 1):
    prob_psi = np.abs(psi_cur)**2
    norm_now = float(np.sum(prob_psi) * dx * dy)

    if n % save_every == 0:
        frames_psi.append(prob_psi[ys, xs].copy())
        times.append(n * dt)
        norms_psi.append(norm_now)
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
Nt         = len(times)

print("Forward valmis.")

# ====================================================
# 8) Detektioajan arvio + Born-otanta klikille
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
# 9) Backward library (vain 1×): phi_tau_frames[j] = |phi(τ_j)|^2
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
# 10) Emix(t; sigmaT) aikasiirrolla
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
# 11) WORLDLINE SELECTION - koemalli
# ====================================================
USE_WORLDLINE_SELECTION = True

# Valitaan tarkoituksella "toiseksi paras" haara
WORLDLINE_CHOOSE_SECOND_PEAK = True

# Huippujen etsintä
WL_TOPK_PEAKS = 3
WL_MIN_PEAK_DIST_PX = 18

# Mistä ajasta haaran valinta tehdään
# 0 = barrierin jälkeen heti, 1 = lähellä screeniä
WL_REF_FRAC_GAP = 0.55

# Tracker
WL_TRACK_RADIUS_PX = 20
WL_MIN_LOCAL_REL = 0.03   # jos paikallinen huippu on liian heikko, pidä vanha paikka

# Putken muoto
WL_TUBE_SIGMA_PX = 10.0

# Vahvistus / vaimennus
WL_GAIN_STRENGTH = 2.0
WL_OUTSIDE_DAMP  = 0.20

# Aika-gate: kytke bias vähitellen päälle
WL_TIME_RAMP_FRAC = 0.12  # kuinka suuri osa [0..Nt) käytetään smooth rampiin

# Tulostus
WL_PRINT_TOP_PEAKS = 3

def gaussian_blur_fft(arr, sigma_px):
    """
    Kevyt FFT-Gaussian blur ilman SciPyä.
    sigma_px pikseleissä.
    """
    if sigma_px <= 0:
        return arr.copy()

    ny, nx = arr.shape
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=1.0)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=1.0)
    KXg, KYg = np.meshgrid(kx, ky)
    G = np.exp(-0.5 * sigma_px * sigma_px * (KXg**2 + KYg**2))
    out = np.fft.ifft2(np.fft.fft2(arr) * G).real
    return out

def pick_top_peaks_2d(arr, top_k=3, min_dist_px=12, mask=None):
    """
    Greedy peak picker ilman SciPyä.
    Palauttaa listan: [(value, iy, ix), ...] suurimmasta alkaen.
    """
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
        work[y0:y1, x0:x1][rr2 <= min_dist_px**2] = -np.inf

    return peaks

def find_reference_frame_index(times, x0, barrier_center_x, screen_center_x, v_est, frac_gap):
    t_barrier = (barrier_center_x - x0) / max(v_est, 1e-12)
    t_gap = (screen_center_x - barrier_center_x) / max(v_est, 1e-12)
    t_ref = t_barrier + frac_gap * t_gap
    idx = int(np.argmin(np.abs(times - t_ref)))
    return idx, t_ref

def extract_local_peak(arr, iy0, ix0, radius_px=16):
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

def track_worldline_from_seed(rho_frames, start_i, start_iy, start_ix,
                              radius_px=16, min_local_rel=0.02):
    """
    Trackaa lokaalia huippua eteen- ja taaksepäin.
    Jos paikallinen signaali romahtaa, jäädään vanhaan pisteeseen.
    """
    Nt, ny, nx = rho_frames.shape
    path_y = np.full(Nt, start_iy, dtype=int)
    path_x = np.full(Nt, start_ix, dtype=int)

    ref_amp = float(max(rho_frames[start_i, start_iy, start_ix], 1e-30))

    # eteenpäin
    iy, ix = start_iy, start_ix
    for i in range(start_i + 1, Nt):
        val, iy_new, ix_new = extract_local_peak(rho_frames[i], iy, ix, radius_px=radius_px)
        if val >= min_local_rel * ref_amp:
            iy, ix = iy_new, ix_new
        path_y[i] = iy
        path_x[i] = ix

    # taaksepäin
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

    tube = np.zeros(shape, dtype=float)
    inv2s2 = 1.0 / max(2.0 * sigma_px * sigma_px, 1e-12)

    for i in range(Nt):
        dy2 = (yy - path_y[i])**2
        dx2 = (xx - path_x[i])**2
        tube[i] = np.exp(-(dy2 + dx2) * inv2s2)

    return tube

def smooth_time_ramp(Nt, center_idx, ramp_frac=0.1):
    """
    Tekee 0..1 rampin ajan yli.
    Bias ei tule kerralla täydellä voimalla.
    """
    ramp_len = max(3, int(ramp_frac * Nt))
    gate = np.zeros(Nt, dtype=float)

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
    """
    Yksinkertainen worldline selection:
      rho' = rho * exp(gain * tube - damp * (1 - tube))
    ja renormalisointi per frame.
    """
    out = np.zeros_like(rho_frames, dtype=float)

    if time_gate is None:
        time_gate = np.ones(rho_frames.shape[0], dtype=float)

    for i in range(rho_frames.shape[0]):
        g = float(time_gate[i])
        field = gain_strength * g * tube[i] - outside_damp * g * (1.0 - tube[i])
        rho = rho_frames[i] * np.exp(field)
        s = float(np.sum(rho) * dx * dy)
        if s > 0:
            rho /= s
        out[i] = rho

    return out

def compute_worldline_selected_rho(base_rho, times, x0, barrier_center_x, screen_center_x, v_est):
    """
    Valitse referenssiframe, etsi top-peaks, ota toisen vahvin haara,
    trackkaa se ajassa, rakenna tube ja vahvista sitä.
    """
    idx_ref, t_ref = find_reference_frame_index(
        times, x0, barrier_center_x, screen_center_x, v_est, WL_REF_FRAC_GAP
    )

    rho_ref = base_rho[idx_ref]

    # kevyt blur peakkien löytöön
    rho_ref_blur = gaussian_blur_fft(rho_ref, sigma_px=2.0)

    # kiinnostaa vain barrierin oikea puoli ja ennen screeniä
    mask_branch_zone = (
        (X_vis > (barrier_center_x + 0.5)) &
        (X_vis < (screen_center_x - 0.5))
    )

    peaks = pick_top_peaks_2d(
        rho_ref_blur,
        top_k=WL_TOPK_PEAKS,
        min_dist_px=WL_MIN_PEAK_DIST_PX,
        mask=mask_branch_zone
    )

    print(f"[WL] reference frame idx={idx_ref}, t_ref≈{t_ref:.3f}, actual t={times[idx_ref]:.3f}")
    if len(peaks) == 0:
        print("[WL] peaks: ei löytynyt. Palautetaan base_rho ilman worldline-biasia.")
        return base_rho, None

    print("[WL] strongest peaks in reference frame:")
    for j, (val, iy, ix) in enumerate(peaks[:WL_PRINT_TOP_PEAKS]):
        print(
            f"  peak#{j+1}: value={val:.6e}, "
            f"x={X_vis[iy, ix]: .3f}, y={Y_vis[iy, ix]: .3f}, iy={iy}, ix={ix}"
        )

    if WORLDLINE_CHOOSE_SECOND_PEAK and len(peaks) >= 2:
        chosen = peaks[1]
        chosen_rank = 2
    else:
        chosen = peaks[0]
        chosen_rank = 1

    chosen_val, chosen_iy, chosen_ix = chosen
    print(
        f"[WL] chosen peak#{chosen_rank}: "
        f"value={chosen_val:.6e}, x={X_vis[chosen_iy, chosen_ix]:.3f}, y={Y_vis[chosen_iy, chosen_ix]:.3f}"
    )

    path_y, path_x = track_worldline_from_seed(
        base_rho,
        idx_ref,
        chosen_iy,
        chosen_ix,
        radius_px=WL_TRACK_RADIUS_PX,
        min_local_rel=WL_MIN_LOCAL_REL
    )

    tube = build_worldline_tube(base_rho.shape, path_y, path_x, sigma_px=WL_TUBE_SIGMA_PX)
    time_gate = smooth_time_ramp(base_rho.shape[0], idx_ref, ramp_frac=WL_TIME_RAMP_FRAC)

    rho_wl = apply_worldline_selection(
        base_rho,
        tube,
        gain_strength=WL_GAIN_STRENGTH,
        outside_damp=WL_OUTSIDE_DAMP,
        time_gate=time_gate
    )

    wl_info = {
        "idx_ref": idx_ref,
        "t_ref": float(times[idx_ref]),
        "path_y": path_y,
        "path_x": path_x,
        "tube": tube,
        "chosen_iy": chosen_iy,
        "chosen_ix": chosen_ix,
    }
    return rho_wl, wl_info

# ====================================================
# 12) Slider: sigmaT valittavissa reaaliajassa
# ====================================================
v_est = k0x / m_mass
L_gap = screen_center_x - barrier_center_x
t_gap = L_gap / v_est

SIGMA_MIN  = 0.05 * t_gap
SIGMA_MAX  = 2.00 * t_gap
SIGMA_INIT = 0.60 * t_gap

def recompute_rho_for_sigma(new_sigma):
    Emix = build_Emix_from_phi_tau(phi_tau_frames, times, t_det, sigmaT=new_sigma, K_JITTER=13)
    base_rho = make_rho(frames_psi, Emix)

    if USE_WORLDLINE_SELECTION:
        rho_wl, wl_info = compute_worldline_selected_rho(
            base_rho, times, x0, barrier_center_x, screen_center_x, v_est
        )
        return rho_wl, wl_info
    else:
        return base_rho, None

rho_init, wl_info_init = recompute_rho_for_sigma(SIGMA_INIT)

# ====================================================
# 13) Visualisointi: yksi paneeli + slider + animaatio
# ====================================================
def gamma_display(arr, gamma=0.5):
    m = np.max(arr)
    if m <= 0:
        return arr
    disp = arr / m
    return disp**gamma

extent = (-VISIBLE_LX/2, VISIBLE_LX/2, -VISIBLE_LY/2, VISIBLE_LY/2)

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

mode_txt = "WL ON" if USE_WORLDLINE_SELECTION else "WL OFF"
title = ax.set_title(rf"ρ(t): σT={SIGMA_INIT:.3f}, t={times[0]:.3f}, {mode_txt}")

# overlay: valittu worldline
wl_line, = ax.plot([], [], color='lime', linewidth=1.2, alpha=0.8)
wl_seed_plot, = ax.plot([], [], marker='o', color='cyan', markersize=5, linestyle='None', alpha=0.9)

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
wl_info_current = [wl_info_init]

def update_worldline_overlay(i):
    wl_info = wl_info_current[0]
    if wl_info is None:
        wl_line.set_data([], [])
        wl_seed_plot.set_data([], [])
        return

    path_y = wl_info["path_y"]
    path_x = wl_info["path_x"]

    # näytetään polku frameen i asti
    xs_line = X_vis[path_y[:i+1], path_x[:i+1]]
    ys_line = Y_vis[path_y[:i+1], path_x[:i+1]]
    wl_line.set_data(xs_line, ys_line)

    seed_x = X_vis[wl_info["chosen_iy"], wl_info["chosen_ix"]]
    seed_y = Y_vis[wl_info["chosen_iy"], wl_info["chosen_ix"]]
    wl_seed_plot.set_data([seed_x], [seed_y])

def on_sigma_change(_val):
    new_sigma = float(sigma_slider.val)
    sigma_current[0] = new_sigma
    rho_new, wl_info_new = recompute_rho_for_sigma(new_sigma)
    rho_current[0] = rho_new
    wl_info_current[0] = wl_info_new

    i = getattr(on_sigma_change, "last_i", 0)
    im.set_data(gamma_display(rho_current[0][i]))
    update_worldline_overlay(i)
    title.set_text(
        rf"ρ(t): σT={sigma_current[0]:.3f}, t={times[i]:.3f}, norm≈{norms_psi[i]:.4f}, {mode_txt}"
    )
    fig.canvas.draw_idle()

sigma_slider.on_changed(on_sigma_change)

def update(i):
    on_sigma_change.last_i = i
    im.set_data(gamma_display(rho_current[0][i]))
    update_worldline_overlay(i)
    title.set_text(
        rf"ρ(t): σT={sigma_current[0]:.3f}, t={times[i]:.3f}, norm≈{norms_psi[i]:.4f}, {mode_txt}"
    )
    return (im, wl_line, wl_seed_plot)

ani = FuncAnimation(fig, update, frames=Nt, interval=40, blit=False)

# Tallennus käyttää sen hetkistä sigmaT-arvoa
ani.save("worldline.mp4", writer="ffmpeg", fps=25, dpi=150)

plt.show()