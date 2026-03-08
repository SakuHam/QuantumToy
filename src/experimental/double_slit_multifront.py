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
    # oletetaan ||psi||=1 (tai lähes), mutta lasketaan silti robustisti
    p = np.abs(psi_unit)**2
    norm = float(np.sum(p) * dx * dy)
    if norm <= 0:
        return 0.0, 0.0
    mx = float(np.sum(p * X) * dx * dy / norm)
    my = float(np.sum(p * Y) * dx * dy / norm)
    return mx, my

# ====================================================
# 4) Jatkuva mittaus (heikko paikkamittaus) - SSE
#     HUOM: Säilytetään CAP:n normihäviö:
#     tehdään mittauspäivitys unit-norm muodossa, ja palautetaan alkuperäinen normi takaisin.
# ====================================================
USE_CONTINUOUS_MEAS = True
KAPPA_MEAS = 0.02          # kokeile 0.005..0.08
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

    # Drift + stochastic backaction
    drift = -0.5 * kappa * (Xc**2 + Yc**2) * dt
    stoch = np.sqrt(kappa) * (Xc * dWx + Yc * dWy)

    psi_u2 = psi_u * np.exp(drift + stoch)

    # SSE normaalisti renormalisoidaan => unit norm
    psi_u2, _ = normalize_unit(psi_u2)

    # Palautetaan alkuperäinen normi (CAP:n häviö säilyy)
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
        # Hamilton + CAP
        psi_cur = step_field(psi_cur, K_phase_fwd, P_half_fwd)

        # Jatkuva mittaus
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
#     + step-printit
# ====================================================
sigma_click = 0.4

def make_phi_at_click():
    Xc = X - x_click
    Yc = Y - y_click
    phi = np.exp(-(Xc**2 + Yc**2) / (2 * sigma_click**2)).astype(np.complex128)
    phi, _ = normalize_unit(phi)  # effectin siemen normiin 1
    return phi

print("Backward library: lasketaan phi_tau (vain 1×)...")

phi_cur = make_phi_at_click()

phi_tau_frames = np.zeros((Nt, N_VISIBLE_Y, N_VISIBLE_X), dtype=float)
tau_step = save_every * dt

print_every_frames = 20  # säädä: esim 10 tiheämmäksi

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
# 11) Slider: sigmaT valittavissa reaaliajassa
# ====================================================
v_est = k0x / m_mass
L_gap = screen_center_x - barrier_center_x
t_gap = L_gap / v_est

SIGMA_MIN  = 0.05 * t_gap
SIGMA_MAX  = 2.00 * t_gap
SIGMA_INIT = 0.60 * t_gap

Emix_init = build_Emix_from_phi_tau(phi_tau_frames, times, t_det, sigmaT=SIGMA_INIT, K_JITTER=13)
rho_init  = make_rho(frames_psi, Emix_init)

# ====================================================
# 12) Visualisointi: yksi paneeli + slider + animaatio
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

im = ax.imshow(gamma_display(rho_init[0]),
               extent=extent, origin='lower', vmin=0.0, vmax=1.0, cmap='magma')

ax.axvline(barrier_center_x, color='white', linestyle='--', alpha=0.6)
ax.axvline(screen_center_x,  color='cyan',  linestyle='--', alpha=0.4)
ax.set_xlabel("x")
ax.set_ylabel("y")
title = ax.set_title(rf"ρ(t): σT={SIGMA_INIT:.3f}, t={times[0]:.3f}")

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

def recompute_rho_for_sigma(new_sigma):
    Emix = build_Emix_from_phi_tau(phi_tau_frames, times, t_det, sigmaT=new_sigma, K_JITTER=13)
    return make_rho(frames_psi, Emix)

def on_sigma_change(_val):
    new_sigma = float(sigma_slider.val)
    sigma_current[0] = new_sigma
    rho_current[0] = recompute_rho_for_sigma(new_sigma)

    i = getattr(on_sigma_change, "last_i", 0)
    im.set_data(gamma_display(rho_current[0][i]))
    title.set_text(rf"ρ(t): σT={sigma_current[0]:.3f}, t={times[i]:.3f}, norm≈{norms_psi[i]:.4f}")
    fig.canvas.draw_idle()

sigma_slider.on_changed(on_sigma_change)

def update(i):
    on_sigma_change.last_i = i
    im.set_data(gamma_display(rho_current[0][i]))
    title.set_text(rf"ρ(t): σT={sigma_current[0]:.3f}, t={times[i]:.3f}, norm≈{norms_psi[i]:.4f}")
    return (im,)

ani = FuncAnimation(fig, update, frames=Nt, interval=40, blit=False)

# Tallennus käyttää sen hetkistä sigmaT-arvoa (aseta slider ennen savea, jos haluat tietyn).
ani.save("single_pulse_slider_sigmaT_onebackward_contmeas.mp4", writer="ffmpeg", fps=25, dpi=150)

plt.show()
