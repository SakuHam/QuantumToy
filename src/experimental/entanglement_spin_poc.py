from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# Helpers
# ============================================================

def norm_l2_spinor(psi: np.ndarray, dx: float, dy: float) -> float:
    return float(np.sqrt(np.sum(np.abs(psi) ** 2) * dx * dy))


def normalize_unit_spinor(psi: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, float]:
    n = norm_l2_spinor(psi, dx, dy)
    if n <= 0.0:
        return psi, 0.0
    return psi / n, n


def total_density(psi: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(psi) ** 2, axis=(-2, -1))


def gaussian_cap_1d(
    x: np.ndarray,
    half_extent: float,
    cap_width: float,
    strength: float,
    power: float = 4.0,
) -> np.ndarray:
    dist_to_edge = half_extent - np.abs(x)
    W = np.zeros_like(x, dtype=float)
    mask = dist_to_edge < cap_width
    s = (cap_width - dist_to_edge[mask]) / max(cap_width, 1e-12)
    W[mask] = strength * (s ** power)
    return W


def safe_frame_normalize(arr: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    amax = float(np.max(arr))
    if amax <= eps:
        return np.zeros_like(arr, dtype=float)
    return arr / amax


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# Spin matrices
# ============================================================

sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)


def pauli_along_axis(theta: float) -> np.ndarray:
    return np.sin(theta) * sigma_x + np.cos(theta) * sigma_z


def spin_rotation_to_axis(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def projector_along_axis(theta: float, sign: int) -> np.ndarray:
    sig = pauli_along_axis(theta)
    return 0.5 * (I2 + sign * sig)


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    Lx: float = 80.0
    Nx: int = 192

    dt: float = 0.006
    n_steps: int = 1800
    save_every: int = 6

    hbar: float = 1.0
    m_a: float = 1.0
    m_b: float = 1.0

    sigma_cm: float = 4.0
    sigma_rel: float = 1.6
    k0: float = 2.4
    x_cm0: float = 0.0

    cap_width: float = 12.0
    cap_strength: float = 2.0
    cap_power: float = 4.0

    sg_region_halfwidth: float = 30.0
    sg_center_a: float = 12.0
    sg_center_b: float = -12.0
    sg_gradient_a: float = 0.18
    sg_gradient_b: float = 0.18

    theta_a: float = 0.0
    theta_b: float = np.pi / 3.0

    detector_halfwidth: float = 2.5
    print_every_frames: int = 20
    click_weight_threshold: float = 1e-3

    no_anim: bool = False


# ============================================================
# Fast spin solver
# ============================================================

class TwoParticleSpin1DFastSolver:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.x = np.linspace(-cfg.Lx / 2.0, cfg.Lx / 2.0, cfg.Nx, endpoint=False)
        self.dx = float(self.x[1] - self.x[0])

        self.XA, self.XB = np.meshgrid(self.x, self.x, indexing="xy")

        k = 2.0 * np.pi * np.fft.fftfreq(cfg.Nx, d=self.dx)
        self.KA, self.KB = np.meshgrid(k, k, indexing="xy")

        self.K_phase = np.exp(
            -1j
            * cfg.dt
            * ((self.KA ** 2) / (2.0 * cfg.m_a) + (self.KB ** 2) / (2.0 * cfg.m_b))
            / cfg.hbar
        ).astype(np.complex128)

        W_a = gaussian_cap_1d(
            self.x,
            half_extent=cfg.Lx / 2.0,
            cap_width=cfg.cap_width,
            strength=cfg.cap_strength,
            power=cfg.cap_power,
        )
        W_b = gaussian_cap_1d(
            self.x,
            half_extent=cfg.Lx / 2.0,
            cap_width=cfg.cap_width,
            strength=cfg.cap_strength,
            power=cfg.cap_power,
        )
        self.WA, self.WB = np.meshgrid(W_a, W_b, indexing="xy")
        self.W = self.WA + self.WB

        gate_a = np.exp(-((self.XA - cfg.sg_center_a) ** 2) / (2.0 * cfg.sg_region_halfwidth ** 2))
        gate_b = np.exp(-((self.XB - cfg.sg_center_b) ** 2) / (2.0 * cfg.sg_region_halfwidth ** 2))

        self.scalar_a = (cfg.sg_gradient_a * (self.XA - cfg.sg_center_a) * gate_a).astype(float)
        self.scalar_b = (cfg.sg_gradient_b * (self.XB - cfg.sg_center_b) * gate_b).astype(float)

        self.sigma_a = pauli_along_axis(cfg.theta_a)
        self.sigma_b = pauli_along_axis(cfg.theta_b)

        self._precompute_half_step_spin_propagators()

        self.PA_plus = projector_along_axis(cfg.theta_a, +1)
        self.PA_minus = projector_along_axis(cfg.theta_a, -1)
        self.PB_plus = projector_along_axis(cfg.theta_b, +1)
        self.PB_minus = projector_along_axis(cfg.theta_b, -1)

        self.Ua_basis = spin_rotation_to_axis(cfg.theta_a)
        self.Ub_basis = spin_rotation_to_axis(cfg.theta_b)

    def _precompute_half_step_spin_propagators(self):
        cfg = self.cfg
        tau = cfg.dt / (2.0 * cfg.hbar)

        theta_a = tau * self.scalar_a
        theta_b = tau * self.scalar_b

        self.ca = np.cos(theta_a).astype(np.complex128)
        self.sa = np.sin(theta_a).astype(np.complex128)
        self.cb = np.cos(theta_b).astype(np.complex128)
        self.sb = np.sin(theta_b).astype(np.complex128)

        self.damp_half = np.exp(-self.W * (cfg.dt / (2.0 * cfg.hbar))).astype(np.complex128)

    def make_singlet_entangled_state(self) -> np.ndarray:
        cfg = self.cfg

        x_cm = 0.5 * (self.XA + self.XB)
        x_rel = self.XA - self.XB

        amp_cm = np.exp(-((x_cm - cfg.x_cm0) ** 2) / (2.0 * cfg.sigma_cm ** 2))
        amp_rel = np.exp(-(x_rel ** 2) / (2.0 * cfg.sigma_rel ** 2))
        phase = np.exp(1j * 0.5 * cfg.k0 * (self.XA - self.XB))

        spatial = (amp_cm * amp_rel * phase).astype(np.complex128)

        psi = np.zeros((cfg.Nx, cfg.Nx, 2, 2), dtype=np.complex128)
        psi[:, :, 0, 1] = spatial / np.sqrt(2.0)
        psi[:, :, 1, 0] = -spatial / np.sqrt(2.0)

        psi, _ = normalize_unit_spinor(psi, self.dx, self.dx)
        return psi

    def _apply_spin_unitary_A(self, psi: np.ndarray) -> np.ndarray:
        term0 = self.ca[:, :, None, None] * psi
        term1 = -1j * self.sa[:, :, None, None] * np.einsum("ab,xybc->xyac", self.sigma_a, psi)
        return term0 + term1

    def _apply_spin_unitary_B(self, psi: np.ndarray) -> np.ndarray:
        term0 = self.cb[:, :, None, None] * psi
        term1 = -1j * self.sb[:, :, None, None] * np.einsum("xyab,bc->xyac", psi, self.sigma_b)
        return term0 + term1

    def _apply_potential_half_step(self, psi: np.ndarray) -> np.ndarray:
        psi = self._apply_spin_unitary_A(psi)
        psi = self._apply_spin_unitary_B(psi)
        psi = self.damp_half[:, :, None, None] * psi
        return psi

    def step(self, psi: np.ndarray) -> np.ndarray:
        psi = self._apply_potential_half_step(psi)
        psi_k = np.fft.fft2(psi, axes=(0, 1))
        psi_k *= self.K_phase[:, :, None, None]
        psi = np.fft.ifft2(psi_k, axes=(0, 1))
        psi = self._apply_potential_half_step(psi)
        return psi

    def evolve(self, psi0: np.ndarray):
        cfg = self.cfg

        joint_frames = []
        psi_frames = []
        times = []
        norms = []

        psi = psi0.copy()
        t_start = time.perf_counter()

        for n in range(cfg.n_steps + 1):
            rho = total_density(psi)
            norm_now = float(np.sum(rho) * self.dx * self.dx)

            if n % cfg.save_every == 0:
                frame_idx = len(times)

                joint_frames.append(rho.astype(np.float32))
                psi_frames.append(psi.astype(np.complex64))
                times.append(n * cfg.dt)
                norms.append(norm_now)

                if (frame_idx % max(1, cfg.print_every_frames)) == 0:
                    elapsed = time.perf_counter() - t_start
                    eab = bell_correlation_E(psi, self)
                    print(
                        f"[FWD] step {n:5d}/{cfg.n_steps}, "
                        f"frame={frame_idx:4d}, "
                        f"t={times[-1]:7.3f}, "
                        f"norm≈{norm_now:.6f}, "
                        f"Eproj≈{eab:.4f}, "
                        f"elapsed={elapsed:.2f}s"
                    )

            if n < cfg.n_steps:
                psi = self.step(psi)

        total_elapsed = time.perf_counter() - t_start
        print(f"[DONE/FWD] evolution finished in {total_elapsed:.2f}s")

        return {
            "joint_frames": np.asarray(joint_frames, dtype=np.float32),
            "psi_frames": np.asarray(psi_frames, dtype=np.complex64),
            "times": np.asarray(times, dtype=float),
            "norms": np.asarray(norms, dtype=float),
            "elapsed_sec": float(total_elapsed),
        }


# ============================================================
# Spin observables
# ============================================================

def expectation_onebody_A(psi: np.ndarray, OA: np.ndarray, dx: float, dy: float) -> float:
    val = np.einsum("xyab,ac,xycb->", np.conjugate(psi), OA, psi)
    return float(np.real(val) * dx * dy)


def expectation_onebody_B(psi: np.ndarray, OB: np.ndarray, dx: float, dy: float) -> float:
    val = np.einsum("xyab,bd,xyad->", np.conjugate(psi), OB, psi)
    return float(np.real(val) * dx * dy)


def expectation_twobody(psi: np.ndarray, OA: np.ndarray, OB: np.ndarray, dx: float, dy: float) -> float:
    val = np.einsum("xyab,ac,bd,xycd->", np.conjugate(psi), OA, OB, psi)
    return float(np.real(val) * dx * dy)


def joint_spin_probs(psi: np.ndarray, solver: TwoParticleSpin1DFastSolver) -> dict[str, float]:
    return {
        "++": expectation_twobody(psi, solver.PA_plus, solver.PB_plus, solver.dx, solver.dx),
        "+-": expectation_twobody(psi, solver.PA_plus, solver.PB_minus, solver.dx, solver.dx),
        "-+": expectation_twobody(psi, solver.PA_minus, solver.PB_plus, solver.dx, solver.dx),
        "--": expectation_twobody(psi, solver.PA_minus, solver.PB_minus, solver.dx, solver.dx),
    }


def bell_correlation_E(psi: np.ndarray, solver: TwoParticleSpin1DFastSolver) -> float:
    jp = joint_spin_probs(psi, solver)
    return float(jp["++"] + jp["--"] - jp["+-"] - jp["-+"])


# ============================================================
# Measurement basis and 2D Voronoi detector
# ============================================================

def rotate_state_to_measurement_basis(
    psi: np.ndarray,
    Ua: np.ndarray,
    Ub: np.ndarray,
) -> np.ndarray:
    tmp = np.einsum("ia,xyab->xyib", Ua.conj().T, psi)
    out = np.einsum("xyib,jb->xyij", tmp, Ub.conj().T)
    return out


def full_basis_component_probs(
    psi: np.ndarray,
    solver: TwoParticleSpin1DFastSolver,
) -> dict[str, float]:
    psi_m = rotate_state_to_measurement_basis(psi, solver.Ua_basis, solver.Ub_basis)
    dx2 = solver.dx * solver.dx
    return {
        "++": float(np.sum(np.abs(psi_m[:, :, 0, 0]) ** 2) * dx2),
        "+-": float(np.sum(np.abs(psi_m[:, :, 0, 1]) ** 2) * dx2),
        "-+": float(np.sum(np.abs(psi_m[:, :, 1, 0]) ** 2) * dx2),
        "--": float(np.sum(np.abs(psi_m[:, :, 1, 1]) ** 2) * dx2),
    }


def channel_component_densities(
    psi: np.ndarray,
    solver: TwoParticleSpin1DFastSolver,
) -> dict[str, np.ndarray]:
    psi_m = rotate_state_to_measurement_basis(psi, solver.Ua_basis, solver.Ub_basis)
    return {
        "++": np.abs(psi_m[:, :, 0, 0]) ** 2,
        "+-": np.abs(psi_m[:, :, 0, 1]) ** 2,
        "-+": np.abs(psi_m[:, :, 1, 0]) ** 2,
        "--": np.abs(psi_m[:, :, 1, 1]) ** 2,
    }


def component_peak_xy(arr: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    iy, ix = np.unravel_index(np.argmax(arr), arr.shape)
    return float(x[ix]), float(x[iy]), float(arr[iy, ix])


def build_voronoi_detector_for_frame(
    psi: np.ndarray,
    solver: TwoParticleSpin1DFastSolver,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[float, float]], dict[str, float]]:
    comps = channel_component_densities(psi, solver)

    peaks = {}
    for ch in ["++", "+-", "-+", "--"]:
        xa, xb, amp = component_peak_xy(comps[ch], solver.x)
        peaks[ch] = (xa, xb, amp)

    XA = solver.XA
    XB = solver.XB

    dists = []
    ordered_channels = ["++", "+-", "-+", "--"]
    for ch in ordered_channels:
        xa, xb, _ = peaks[ch]
        d2 = (XA - xa) ** 2 + (XB - xb) ** 2
        dists.append(d2)

    dist_stack = np.stack(dists, axis=0)
    labels = np.argmin(dist_stack, axis=0)

    masks = {
        "++": labels == 0,
        "+-": labels == 1,
        "-+": labels == 2,
        "--": labels == 3,
    }

    diag = {
        "pp_peak_xA": float(peaks["++"][0]),
        "pp_peak_xB": float(peaks["++"][1]),
        "pm_peak_xA": float(peaks["+-"][0]),
        "pm_peak_xB": float(peaks["+-"][1]),
        "mp_peak_xA": float(peaks["-+"][0]),
        "mp_peak_xB": float(peaks["-+"][1]),
        "mm_peak_xA": float(peaks["--"][0]),
        "mm_peak_xB": float(peaks["--"][1]),
    }

    peak_pos = {
        "++": (float(peaks["++"][0]), float(peaks["++"][1])),
        "+-": (float(peaks["+-"][0]), float(peaks["+-"][1])),
        "-+": (float(peaks["-+"][0]), float(peaks["-+"][1])),
        "--": (float(peaks["--"][0]), float(peaks["--"][1])),
    }

    return masks, peak_pos, diag


def debug_voronoi_partition(masks: dict[str, np.ndarray]) -> None:
    total = (
        masks["++"].astype(np.int32)
        + masks["+-"].astype(np.int32)
        + masks["-+"].astype(np.int32)
        + masks["--"].astype(np.int32)
    )
    cover_all = bool(np.all(total == 1))
    overlap = bool(np.any(total > 1))
    print(f"[VORONOI CHECK] cover_all={cover_all}, overlap={overlap}")


def detector_click_probabilities_from_voronoi_masks(
    psi: np.ndarray,
    solver: TwoParticleSpin1DFastSolver,
    masks: dict[str, np.ndarray],
) -> dict[str, float]:
    comps = channel_component_densities(psi, solver)
    dx2 = solver.dx * solver.dx

    p_pp_raw = float(np.sum(comps["++"] * masks["++"]) * dx2)
    p_pm_raw = float(np.sum(comps["+-"] * masks["+-"]) * dx2)
    p_mp_raw = float(np.sum(comps["-+"] * masks["-+"]) * dx2)
    p_mm_raw = float(np.sum(comps["--"] * masks["--"]) * dx2)

    total = p_pp_raw + p_pm_raw + p_mp_raw + p_mm_raw

    if total > 0.0:
        p_pp_cond = p_pp_raw / total
        p_pm_cond = p_pm_raw / total
        p_mp_cond = p_mp_raw / total
        p_mm_cond = p_mm_raw / total
    else:
        p_pp_cond = 0.0
        p_pm_cond = 0.0
        p_mp_cond = 0.0
        p_mm_cond = 0.0

    return {
        "++": float(p_pp_cond),
        "+-": float(p_pm_cond),
        "-+": float(p_mp_cond),
        "--": float(p_mm_cond),
        "pp_raw": float(p_pp_raw),
        "pm_raw": float(p_pm_raw),
        "mp_raw": float(p_mp_raw),
        "mm_raw": float(p_mm_raw),
        "total_weight": float(total),
    }


def frame_voronoi_detector(
    psi: np.ndarray,
    solver: TwoParticleSpin1DFastSolver,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[float, float]], dict[str, float]]:
    masks, peak_pos, peak_diag = build_voronoi_detector_for_frame(psi, solver)
    det = detector_click_probabilities_from_voronoi_masks(psi, solver, masks)
    proj = joint_spin_probs(psi, solver)

    diag = {
        **peak_diag,
        "err": float(
            (det["++"] - proj["++"]) ** 2
            + (det["+-"] - proj["+-"]) ** 2
            + (det["-+"] - proj["-+"]) ** 2
            + (det["--"] - proj["--"]) ** 2
        ),
        "W": float(det["total_weight"]),
        "E_det_cond": float(det["++"] + det["--"] - det["+-"] - det["-+"]),
        "E_proj": float(proj["++"] + proj["--"] - proj["+-"] - proj["-+"]),
    }
    return masks, peak_pos, diag


def compute_dynamic_voronoi_detector_series(
    run: dict,
    solver: TwoParticleSpin1DFastSolver,
) -> tuple[list[dict[str, float]], list[dict[str, np.ndarray]], list[dict[str, tuple[float, float]]], list[dict[str, float]]]:
    detector_series: list[dict[str, float]] = []
    masks_series: list[dict[str, np.ndarray]] = []
    peaks_series: list[dict[str, tuple[float, float]]] = []
    diag_series: list[dict[str, float]] = []

    for psi in run["psi_frames"]:
        psi128 = psi.astype(np.complex128)
        masks, peak_pos, diag = frame_voronoi_detector(psi128, solver)
        det = detector_click_probabilities_from_voronoi_masks(psi128, solver, masks)

        detector_series.append(det)
        masks_series.append(masks)
        peaks_series.append(peak_pos)
        diag_series.append(diag)

    return detector_series, masks_series, peaks_series, diag_series


# ============================================================
# Click event model
# ============================================================

def find_click_event(
    times: np.ndarray,
    detector_series: list[dict[str, float]],
    psi_frames: np.ndarray,
    solver: TwoParticleSpin1DFastSolver,
    threshold: float,
) -> dict[str, float] | None:
    weights = np.array([d["total_weight"] for d in detector_series], dtype=float)
    if weights.size == 0:
        return None

    idx = int(np.argmax(weights))
    det = detector_series[idx]

    if det["total_weight"] < threshold:
        return None

    psi_click = psi_frames[idx].astype(np.complex128)
    born = joint_spin_probs(psi_click, solver)
    comp = full_basis_component_probs(psi_click, solver)

    return {
        "frame_idx": int(idx),
        "time": float(times[idx]),
        "threshold": float(threshold),
        "total_weight": float(det["total_weight"]),
        "det_pp": float(det["++"]),
        "det_pm": float(det["+-"]),
        "det_mp": float(det["-+"]),
        "det_mm": float(det["--"]),
        "det_pp_raw": float(det["pp_raw"]),
        "det_pm_raw": float(det["pm_raw"]),
        "det_mp_raw": float(det["mp_raw"]),
        "det_mm_raw": float(det["mm_raw"]),
        "E_det_cond": float(det["++"] + det["--"] - det["+-"] - det["-+"]),
        "born_pp": float(born["++"]),
        "born_pm": float(born["+-"]),
        "born_mp": float(born["-+"]),
        "born_mm": float(born["--"]),
        "E_born": float(born["++"] + born["--"] - born["+-"] - born["-+"]),
        "basis_pp": float(comp["++"]),
        "basis_pm": float(comp["+-"]),
        "basis_mp": float(comp["-+"]),
        "basis_mm": float(comp["--"]),
        "E_basis": float(comp["++"] + comp["--"] - comp["+-"] - comp["-+"]),
    }


# ============================================================
# Plot helpers
# ============================================================

CHANNELS = ["++", "+-", "-+", "--"]
CHANNEL_COLORS = {
    "++": "tab:blue",
    "+-": "tab:orange",
    "-+": "tab:green",
    "--": "tab:red",
}


def choose_click_channel(det: dict[str, float]) -> str:
    vals = {ch: det[ch] for ch in CHANNELS}
    return max(vals, key=vals.get)


def joint_density_with_mask_overlay(
    joint_density: np.ndarray,
    masks: dict[str, np.ndarray],
    alpha_scale: float = 0.22,
) -> np.ndarray:
    rho_n = safe_frame_normalize(joint_density)
    overlay = np.zeros((*joint_density.shape, 4), dtype=float)

    color_rgba = {
        "++": np.array([0.12, 0.47, 0.71, alpha_scale]),
        "+-": np.array([1.00, 0.50, 0.05, alpha_scale]),
        "-+": np.array([0.17, 0.63, 0.17, alpha_scale]),
        "--": np.array([0.84, 0.15, 0.16, alpha_scale]),
    }

    for ch in CHANNELS:
        m = masks[ch]
        overlay[m, :] = color_rgba[ch]

    overlay[..., 3] *= np.clip(0.35 + 0.65 * rho_n, 0.0, 1.0)
    return overlay


def add_channel_peak_labels(ax, peaks: dict[str, tuple[float, float]]) -> None:
    label_offsets = {
        "++": (0.6, 0.6),
        "+-": (0.6, -1.4),
        "-+": (-2.2, 0.6),
        "--": (-2.2, -1.4),
    }
    for ch in CHANNELS:
        xa, xb = peaks[ch]
        ax.scatter([xa], [xb], s=30, c=CHANNEL_COLORS[ch], edgecolors="white", linewidths=0.7, zorder=4)
        dx, dy = label_offsets[ch]
        ax.text(
            xa + dx,
            xb + dy,
            ch,
            color="white",
            fontsize=10,
            weight="bold",
            ha="left",
            va="bottom",
            zorder=5,
        )


def detector_series_to_arrays(detector_series: list[dict[str, float]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in ["++", "+-", "-+", "--", "pp_raw", "pm_raw", "mp_raw", "mm_raw", "total_weight"]:
        out[key] = np.array([d[key] for d in detector_series], dtype=float)
    return out


# ============================================================
# Plotting
# ============================================================

def make_detector_calibration_plot(outdir, solver, run, detector_series, diag_series):
    outpath = Path(outdir) / "detector_calibration.png"
    ensure_parent_dir(outpath)

    times = run["times"]
    det_arr = detector_series_to_arrays(detector_series)
    psi_frames = run["psi_frames"]

    born = {ch: [] for ch in CHANNELS}
    basis = {ch: [] for ch in CHANNELS}
    errs = []
    e_det = []
    e_born = []

    for i, psi in enumerate(psi_frames):
        psi128 = psi.astype(np.complex128)
        jp = joint_spin_probs(psi128, solver)
        bp = full_basis_component_probs(psi128, solver)
        for ch in CHANNELS:
            born[ch].append(jp[ch])
            basis[ch].append(bp[ch])
        errs.append(diag_series[i]["err"])
        e_det.append(diag_series[i]["E_det_cond"])
        e_born.append(diag_series[i]["E_proj"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    ax = axes[0, 0]
    for ch in CHANNELS:
        ax.plot(times, det_arr[ch], label=f"det {ch}", color=CHANNEL_COLORS[ch], linewidth=2)
    ax.set_title("Detector conditional channel probabilities")
    ax.set_xlabel("time")
    ax.set_ylabel("probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)

    ax = axes[0, 1]
    for ch in CHANNELS:
        ax.plot(times, born[ch], label=f"Born {ch}", color=CHANNEL_COLORS[ch], linewidth=2)
        ax.plot(times, basis[ch], linestyle="--", color=CHANNEL_COLORS[ch], alpha=0.75, linewidth=1.5)
    ax.set_title("Born global (solid) vs rotated-basis mass (dashed)")
    ax.set_xlabel("time")
    ax.set_ylabel("probability / mass")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)

    ax = axes[1, 0]
    ax.plot(times, det_arr["total_weight"], label="detector total weight", linewidth=2)
    ax.set_title("Detector total coincidence weight")
    ax.set_xlabel("time")
    ax.set_ylabel("weight")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(times, errs, label="sum sq. error(det vs Born)", linewidth=2)
    ax.plot(times, e_det, label="E_det_cond", linewidth=2)
    ax.plot(times, e_born, label="E_proj", linewidth=2, linestyle="--")
    ax.set_title("Calibration diagnostics")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.suptitle("Voronoi detector calibration summary", fontsize=16)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    print(f"[PLOT] saved {outpath}")


def make_click_summary_plot(outdir, solver, run, detector_series, click_event, masks_series, peaks_series):
    if click_event is None:
        return

    outpath = Path(outdir) / "click_summary.png"
    ensure_parent_dir(outpath)

    idx = int(click_event["frame_idx"])
    t_click = float(click_event["time"])
    psi_click = run["psi_frames"][idx].astype(np.complex128)
    joint = total_density(psi_click)
    det = detector_series[idx]
    born = joint_spin_probs(psi_click, solver)
    weights = np.array([d["total_weight"] for d in detector_series], dtype=float)
    click_channel = choose_click_channel(det)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0], width_ratios=[1.25, 1.0])

    ax_img = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_w = fig.add_subplot(gs[1, :])

    extent = [solver.x[0], solver.x[-1], solver.x[0], solver.x[-1]]

    im = ax_img.imshow(
        safe_frame_normalize(joint),
        origin="lower",
        extent=extent,
        cmap="magma",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    overlay = joint_density_with_mask_overlay(joint, masks_series[idx], alpha_scale=0.18)
    ax_img.imshow(overlay, origin="lower", extent=extent, aspect="auto")
    add_channel_peak_labels(ax_img, peaks_series[idx])
    cbar = fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
    cbar.set_label("frame-normalized density")

    info_text = (
        "CLICK FRAME\n"
        f"E_click={click_event['E_born']:.4f}\n"
        f"P++={click_event['det_pp']:.3f}, P+-={click_event['det_pm']:.3f}\n"
        f"P-+={click_event['det_mp']:.3f}, P--={click_event['det_mm']:.3f}"
    )
    ax_img.text(
        0.015,
        0.985,
        info_text,
        transform=ax_img.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        color="black",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.2"),
    )
    ax_img.set_title("Joint density")
    ax_img.set_xlabel("x_A")
    ax_img.set_ylabel("x_B")

    x = np.arange(len(CHANNELS))
    det_vals = [det[ch] for ch in CHANNELS]
    born_vals = [born[ch] for ch in CHANNELS]
    ax_bar.bar(x, det_vals, color=[CHANNEL_COLORS[ch] for ch in CHANNELS], alpha=0.95)
    ax_bar.scatter(x, born_vals, s=110, facecolors="none", edgecolors="black", linewidths=1.4, label="Born click")
    ax_bar.set_xticks(x, CHANNELS)
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.set_ylabel("probability")
    ax_bar.set_title(
        f"Detector vs Born click channels\nW={click_event['total_weight']:.3e} | {click_channel} CLICK"
    )
    ax_bar.legend(loc="upper right")
    ax_bar.grid(True, axis="y", alpha=0.25)

    times = run["times"]
    ax_w.plot(times, weights, label="detector total weight", linewidth=2)
    ax_w.axhline(click_event["threshold"], linestyle="--", linewidth=1.2, label="click threshold")
    ax_w.axvline(t_click, linestyle="--", linewidth=1.2, label=f"click @ t={t_click:.2f}")
    ax_w.scatter([t_click], [click_event["total_weight"]], s=35, zorder=5)
    ax_w.set_title("Detector coincidence weight")
    ax_w.set_xlabel("time")
    ax_w.set_ylabel("weight")
    ax_w.grid(True, alpha=0.25)
    ax_w.legend()

    fig.suptitle(f"t={t_click:.2f} [{click_channel} click]", fontsize=16)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    print(f"[PLOT] saved {outpath}")


def make_click_channels_plot(outdir, solver, run, detector_series, click_event):
    if click_event is None:
        return

    outpath = Path(outdir) / "click_channels_timeseries.png"
    ensure_parent_dir(outpath)

    times = run["times"]
    det_arr = detector_series_to_arrays(detector_series)
    psi_frames = run["psi_frames"]

    born = {ch: [] for ch in CHANNELS}
    for psi in psi_frames:
        psi128 = psi.astype(np.complex128)
        jp = joint_spin_probs(psi128, solver)
        for ch in CHANNELS:
            born[ch].append(jp[ch])

    idx_click = int(click_event["frame_idx"])
    t_click = float(click_event["time"])

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True, constrained_layout=True)

    ax = axes[0]
    for ch in CHANNELS:
        ax.plot(times, det_arr[ch], color=CHANNEL_COLORS[ch], label=f"det {ch}", linewidth=2)
    ax.axvline(t_click, color="k", linestyle="--", alpha=0.7)
    ax.set_title("Detector conditional channel probabilities")
    ax.set_ylabel("probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=4, fontsize=9)

    ax = axes[1]
    for ch in CHANNELS:
        ax.plot(times, born[ch], color=CHANNEL_COLORS[ch], label=f"Born {ch}", linewidth=2)
    ax.axvline(t_click, color="k", linestyle="--", alpha=0.7)
    ax.set_title("Born global channel probabilities")
    ax.set_ylabel("probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=4, fontsize=9)

    ax = axes[2]
    ax.plot(times, det_arr["total_weight"], label="detector total weight", linewidth=2)
    ax.axhline(click_event["threshold"], color="tab:red", linestyle="--", label="threshold")
    ax.axvline(t_click, color="k", linestyle="--", alpha=0.7, label=f"click @ {t_click:.3f}")
    ax.scatter([t_click], [det_arr["total_weight"][idx_click]], zorder=5, s=35)
    ax.set_title("Click trigger series")
    ax.set_xlabel("time")
    ax.set_ylabel("weight")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    print(f"[PLOT] saved {outpath}")


def make_animation(outdir, solver, run, detector_series, click_event, masks_series, peaks_series):
    if click_event is None:
        print("[ANIM] skipped: no click event")
        return

    outdir = Path(outdir)
    out_mp4 = outdir / "click_animation.mp4"
    out_gif = outdir / "click_animation.gif"

    times = run["times"]
    joint_frames = run["joint_frames"]
    weights = np.array([d["total_weight"] for d in detector_series], dtype=float)
    click_idx = int(click_event["frame_idx"])
    click_time = float(click_event["time"])

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0], width_ratios=[1.25, 1.0])

    ax_img = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_w = fig.add_subplot(gs[1, :])

    extent = [solver.x[0], solver.x[-1], solver.x[0], solver.x[-1]]

    rho0 = joint_frames[0]
    im = ax_img.imshow(
        safe_frame_normalize(rho0),
        origin="lower",
        extent=extent,
        cmap="magma",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        animated=True,
    )
    overlay0 = joint_density_with_mask_overlay(rho0, masks_series[0], alpha_scale=0.18)
    im_overlay = ax_img.imshow(
        overlay0,
        origin="lower",
        extent=extent,
        aspect="auto",
        animated=True,
    )

    peak_scatters = {}
    peak_texts = {}
    for ch in CHANNELS:
        xa, xb = peaks_series[0][ch]
        sc = ax_img.scatter([xa], [xb], s=30, c=CHANNEL_COLORS[ch], edgecolors="white", linewidths=0.7, zorder=4)
        peak_scatters[ch] = sc
        txt = ax_img.text(xa, xb, ch, color="white", fontsize=10, weight="bold", zorder=5)
        peak_texts[ch] = txt

    ax_img.set_title("Joint density")
    ax_img.set_xlabel("x_A")
    ax_img.set_ylabel("x_B")
    cbar = fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
    cbar.set_label("frame-normalized density")

    info_box = ax_img.text(
        0.015,
        0.985,
        "",
        transform=ax_img.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        color="black",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.2"),
    )

    x = np.arange(len(CHANNELS))
    bars = ax_bar.bar(x, [0.0] * 4, color=[CHANNEL_COLORS[ch] for ch in CHANNELS], alpha=0.95)
    born_scatter = ax_bar.scatter(
        x, [0.0] * 4, s=110, facecolors="none", edgecolors="black", linewidths=1.4, label="Born click"
    )
    ax_bar.set_xticks(x, CHANNELS)
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.set_ylabel("probability")
    ax_bar.set_title("Detector vs Born click channels")
    ax_bar.legend(loc="upper right")
    ax_bar.grid(True, axis="y", alpha=0.25)

    w_line, = ax_w.plot(times, weights, label="detector total weight", linewidth=2)
    th_line = ax_w.axhline(click_event["threshold"], linestyle="--", linewidth=1.2, label="click threshold")
    click_vline = ax_w.axvline(click_time, linestyle="--", linewidth=1.2, label=f"click @ t={click_time:.2f}")
    current_dot, = ax_w.plot([times[0]], [weights[0]], marker="o", linestyle="None", markersize=5)
    ax_w.set_title("Detector coincidence weight")
    ax_w.set_xlabel("time")
    ax_w.set_ylabel("weight")
    ax_w.grid(True, alpha=0.25)
    ax_w.legend()

    frame_vline = ax_w.axvline(times[0], color="tab:gray", linestyle=":", linewidth=1.4, alpha=0.9)

    def update_peak_annotations(peaks):
        label_offsets = {
            "++": (0.6, 0.6),
            "+-": (0.6, -1.4),
            "-+": (-2.2, 0.6),
            "--": (-2.2, -1.4),
        }
        for ch in CHANNELS:
            xa, xb = peaks[ch]
            peak_scatters[ch].set_offsets(np.array([[xa, xb]]))
            dx, dy = label_offsets[ch]
            peak_texts[ch].set_position((xa + dx, xb + dy))

    def update(frame_idx: int):
        rho = joint_frames[frame_idx]
        im.set_data(safe_frame_normalize(rho))
        im_overlay.set_data(joint_density_with_mask_overlay(rho, masks_series[frame_idx], alpha_scale=0.18))
        update_peak_annotations(peaks_series[frame_idx])

        det = detector_series[frame_idx]
        psi = run["psi_frames"][frame_idx].astype(np.complex128)
        born = joint_spin_probs(psi, solver)
        click_channel = choose_click_channel(det)

        for rect, ch in zip(bars, CHANNELS):
            rect.set_height(det[ch])

        born_scatter.set_offsets(np.column_stack([x, [born[ch] for ch in CHANNELS]]))

        info_text = (
            f"{'CLICK FRAME' if frame_idx == click_idx else 'FRAME'}\n"
            f"E_click={born['++'] + born['--'] - born['+-'] - born['-+']:.4f}\n"
            f"P++={det['++']:.3f}, P+-={det['+-']:.3f}\n"
            f"P-+={det['-+']:.3f}, P--={det['--']:.3f}"
        )
        info_box.set_text(info_text)

        ttl = f"t={times[frame_idx]:.2f}"
        if frame_idx == click_idx:
            ttl += f" [{click_channel} click]"
        elif frame_idx > click_idx:
            ttl += " [post-click]"
        fig.suptitle(ttl, fontsize=16)

        ax_bar.set_title(
            f"Detector vs Born click channels\nW={det['total_weight']:.3e}"
            + (f" | {click_channel} CLICK" if frame_idx == click_idx else "")
        )

        frame_vline.set_xdata([times[frame_idx], times[frame_idx]])
        current_dot.set_data([times[frame_idx]], [weights[frame_idx]])

        artists = [im, im_overlay, info_box, born_scatter, current_dot, frame_vline]
        artists.extend(list(bars))
        artists.extend(peak_scatters.values())
        artists.extend(peak_texts.values())
        return artists

    anim = FuncAnimation(fig, update, frames=len(times), interval=60, blit=False)

    saved = False
    try:
        anim.save(out_mp4, dpi=140, fps=15)
        saved = True
        print(f"[ANIM] saved {out_mp4}")
    except Exception as e:
        print(f"[ANIM] mp4 save failed: {e}")
        try:
            from matplotlib.animation import PillowWriter
            anim.save(out_gif, writer=PillowWriter(fps=15), dpi=120)
            saved = True
            print(f"[ANIM] saved {out_gif}")
        except Exception as e2:
            print(f"[ANIM] gif save failed: {e2}")

    plt.close(fig)

    if not saved:
        print("[ANIM] animation was not saved")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast two-particle 1D spin click-event Bell PoC (Voronoi detector)")
    p.add_argument("--outdir", type=str, default="two_particle_spin_click_out")
    p.add_argument("--nx", type=int, default=192)
    p.add_argument("--n-steps", type=int, default=1800)
    p.add_argument("--dt", type=float, default=0.006)
    p.add_argument("--save-every", type=int, default=6)
    p.add_argument("--k0", type=float, default=2.4)
    p.add_argument("--sigma-cm", type=float, default=4.0)
    p.add_argument("--sigma-rel", type=float, default=1.6)
    p.add_argument("--theta-a", type=float, default=0.0)
    p.add_argument("--theta-b", type=float, default=float(np.pi / 3.0))
    p.add_argument("--sg-gradient-a", type=float, default=0.12)
    p.add_argument("--sg-gradient-b", type=float, default=0.12)
    p.add_argument("--detector-halfwidth", type=float, default=2.5)
    p.add_argument("--click-weight-threshold", type=float, default=1e-3)
    p.add_argument("--print-every-frames", type=int, default=20)
    p.add_argument("--no-anim", action="store_true")
    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        Nx=int(args.nx),
        n_steps=int(args.n_steps),
        dt=float(args.dt),
        save_every=int(args.save_every),
        k0=float(args.k0),
        sigma_cm=float(args.sigma_cm),
        sigma_rel=float(args.sigma_rel),
        theta_a=float(args.theta_a),
        theta_b=float(args.theta_b),
        sg_gradient_a=float(args.sg_gradient_a),
        sg_gradient_b=float(args.sg_gradient_b),
        detector_halfwidth=float(args.detector_halfwidth),
        click_weight_threshold=float(args.click_weight_threshold),
        print_every_frames=int(args.print_every_frames),
        no_anim=bool(args.no_anim),
    )

    print("[START] building fast spin click-event Bell PoC (Voronoi detector)")
    print(cfg)

    solver = TwoParticleSpin1DFastSolver(cfg)
    psi0 = solver.make_singlet_entangled_state()

    print("[INIT] initial singlet state built and normalized")
    run = solver.evolve(psi0)

    print("[DET] computing frame-wise 2D Voronoi detector masks ...")
    detector_series, masks_series, peaks_series, diag_series = compute_dynamic_voronoi_detector_series(run, solver)

    mid_idx = len(run["psi_frames"]) // 2
    last_idx = len(run["psi_frames"]) - 1

    for idx in sorted(set([0, mid_idx, last_idx])):
        diag = diag_series[idx]
        peaks = peaks_series[idx]
        print(
            "[DET FRAME] "
            f"frame_idx={idx}, "
            f"++@({peaks['++'][0]:.3f},{peaks['++'][1]:.3f}), "
            f"+-@({peaks['+-'][0]:.3f},{peaks['+-'][1]:.3f}), "
            f"-+@({peaks['-+'][0]:.3f},{peaks['-+'][1]:.3f}), "
            f"--@({peaks['--'][0]:.3f},{peaks['--'][1]:.3f}), "
            f"err={diag['err']:.3e}, "
            f"W={diag['W']:.3e}, "
            f"E_det_cond={diag['E_det_cond']:.4f}, "
            f"E_proj={diag['E_proj']:.4f}"
        )
        debug_voronoi_partition(masks_series[idx])

    click_event = find_click_event(
        times=run["times"],
        detector_series=detector_series,
        psi_frames=run["psi_frames"],
        solver=solver,
        threshold=cfg.click_weight_threshold,
    )

    psi_last = run["psi_frames"][-1].astype(np.complex128)
    E_last = bell_correlation_E(psi_last, solver)
    E_ideal = -np.cos(cfg.theta_a - cfg.theta_b)

    print(f"[RUN] frames={len(run['times'])}, final saved time={run['times'][-1]:.3f}")
    print(f"[RUN] total forward elapsed={run['elapsed_sec']:.2f}s")
    print(f"[RUN] norm min={np.min(run['norms']):.6e}, max={np.max(run['norms']):.6e}")
    print(f"[RUN] final E_proj(a,b)={E_last:.6f}")
    print(f"[RUN] ideal singlet E(a,b)={E_ideal:.6f}")

    if click_event is None:
        print("[CLICK] no click event found")
    else:
        idx_click = int(click_event["frame_idx"])
        det_click = detector_series[idx_click]
        peaks_click = peaks_series[idx_click]
        diag_click = diag_series[idx_click]
        masks_click = masks_series[idx_click]

        psi_click = run["psi_frames"][idx_click].astype(np.complex128)
        comps_click = channel_component_densities(psi_click, solver)
        dx2 = solver.dx * solver.dx

        total_full = float(
            np.sum(comps_click["++"] + comps_click["+-"] + comps_click["-+"] + comps_click["--"]) * dx2
        )
        total_det = (
            det_click["pp_raw"]
            + det_click["pm_raw"]
            + det_click["mp_raw"]
            + det_click["mm_raw"]
        )

        region_total = float(
            np.sum((comps_click["++"] + comps_click["+-"] + comps_click["-+"] + comps_click["--"]) *
                   (masks_click["++"] | masks_click["+-"] | masks_click["-+"] | masks_click["--"])) * dx2
        )

        print(
            "[CLICK DETECTOR FRAME] "
            f"++@({peaks_click['++'][0]:.3f},{peaks_click['++'][1]:.3f}), "
            f"+-@({peaks_click['+-'][0]:.3f},{peaks_click['+-'][1]:.3f}), "
            f"-+@({peaks_click['-+'][0]:.3f},{peaks_click['-+'][1]:.3f}), "
            f"--@({peaks_click['--'][0]:.3f},{peaks_click['--'][1]:.3f}), "
            f"err={diag_click['err']:.3e}"
        )

        print(
            "[CLICK] "
            f"frame_idx={click_event['frame_idx']}, "
            f"t={click_event['time']:.6f}, "
            f"W={click_event['total_weight']:.6e}, "
            f"E_det_cond={click_event['E_det_cond']:.6f}, "
            f"E_born={click_event['E_born']:.6f}, "
            f"E_basis={click_event['E_basis']:.6f}"
        )

        print(
            "[CLICK/DET CONDITIONAL] "
            f"++={click_event['det_pp']:.4f}, "
            f"+-={click_event['det_pm']:.4f}, "
            f"-+={click_event['det_mp']:.4f}, "
            f"--={click_event['det_mm']:.4f}"
        )

        print(
            "[CLICK/BORN GLOBAL] "
            f"++={click_event['born_pp']:.4f}, "
            f"+-={click_event['born_pm']:.4f}, "
            f"-+={click_event['born_mp']:.4f}, "
            f"--={click_event['born_mm']:.4f}"
        )

        print(
            "[CLICK/BASIS GLOBAL] "
            f"++={click_event['basis_pp']:.4f}, "
            f"+-={click_event['basis_pm']:.4f}, "
            f"-+={click_event['basis_mp']:.4f}, "
            f"--={click_event['basis_mm']:.4f}"
        )

        print(
            "[COMPARE GLOBAL BORN VS BASIS] "
            f"Born: ++={click_event['born_pp']:.4f}, "
            f"+-={click_event['born_pm']:.4f}, "
            f"-+={click_event['born_mp']:.4f}, "
            f"--={click_event['born_mm']:.4f} | "
            f"Basis: ++={click_event['basis_pp']:.4f}, "
            f"+-={click_event['basis_pm']:.4f}, "
            f"-+={click_event['basis_mp']:.4f}, "
            f"--={click_event['basis_mm']:.4f}"
        )

        print(
            "[COMPARE DET CONDITIONAL VS GLOBAL BORN] "
            f"detector_cond: ++={det_click['++']:.4f}, "
            f"+-={det_click['+-']:.4f}, "
            f"-+={det_click['-+']:.4f}, "
            f"--={det_click['--']:.4f} | "
            f"Born_global: ++={click_event['born_pp']:.4f}, "
            f"+-={click_event['born_pm']:.4f}, "
            f"-+={click_event['born_mp']:.4f}, "
            f"--={click_event['born_mm']:.4f}"
        )

        print(
            "[COMPARE DET RAW VS GLOBAL BASIS] "
            f"det_raw: ++={det_click['pp_raw']:.4f}, "
            f"+-={det_click['pm_raw']:.4f}, "
            f"-+={det_click['mp_raw']:.4f}, "
            f"--={det_click['mm_raw']:.4f} | "
            f"basis_global: ++={click_event['basis_pp']:.4f}, "
            f"+-={click_event['basis_pm']:.4f}, "
            f"-+={click_event['basis_mp']:.4f}, "
            f"--={click_event['basis_mm']:.4f}"
        )

        print(
            "[TOTAL CHECK] "
            f"full={total_full:.6f}, det_sum={total_det:.6f}, region_total={region_total:.6f}"
        )

    # plots
    make_detector_calibration_plot(outdir, solver, run, detector_series, diag_series)
    if click_event is not None:
        make_click_summary_plot(outdir, solver, run, detector_series, click_event, masks_series, peaks_series)
        make_click_channels_plot(outdir, solver, run, detector_series, click_event)

    if not cfg.no_anim and click_event is not None:
        make_animation(outdir, solver, run, detector_series, click_event, masks_series, peaks_series)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())