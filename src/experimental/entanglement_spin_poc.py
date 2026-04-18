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
    # arr shape is (xB, xA) because meshgrid(indexing="xy")
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

    # grid coords in (xB, xA) array layout
    XA = solver.XA
    XB = solver.XB

    dists = []
    ordered_channels = ["++", "+-", "-+", "--"]
    for ch in ordered_channels:
        xa, xb, _ = peaks[ch]
        d2 = (XA - xa) ** 2 + (XB - xb) ** 2
        dists.append(d2)

    dist_stack = np.stack(dists, axis=0)  # (4, Ny, Nx)
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
# Minimal placeholders for plotting so main runs
# ============================================================

def make_detector_calibration_plot(outdir, solver, run, peaks, diag):
    pass


def make_click_summary_plot(outdir, solver, run, detector_series, click_event):
    pass


def make_click_channels_plot(outdir, solver, run, detector_series, click_event):
    pass


def make_animation(outdir, solver, run, detector_series, click_event):
    pass


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

        # Optional diagnostic: total component mass in each Voronoi region, regardless of label
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())