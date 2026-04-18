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
    """
    Rotation U such that measuring z after rotation corresponds to measuring
    along axis n=(sin theta,0,cos theta).
    """
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

    # SG region
    sg_region_halfwidth: float = 7.0
    sg_center_a: float = 12.0
    sg_center_b: float = -12.0
    sg_gradient_a: float = 0.12
    sg_gradient_b: float = 0.12

    # measurement axes
    theta_a: float = 0.0
    theta_b: float = np.pi / 3.0

    # calibrated detector half-width
    detector_halfwidth: float = 2.5

    print_every_frames: int = 20

    # click event threshold
    click_weight_threshold: float = 1e-3

    # calibration
    calibration_frame_mode: str = "last"

    no_anim: bool = False


# ============================================================
# Fast spin solver
# ============================================================

class TwoParticleSpin1DFastSolver:
    """
    Internal state shape:
        psi[iy, ix, sA, sB]
    """

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

        # singlet: (|up down> - |down up>) / sqrt(2)
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


def spin_outcome_probs(psi: np.ndarray, solver: TwoParticleSpin1DFastSolver) -> dict[str, float]:
    return {
        "A_plus": expectation_onebody_A(psi, solver.PA_plus, solver.dx, solver.dx),
        "A_minus": expectation_onebody_A(psi, solver.PA_minus, solver.dx, solver.dx),
        "B_plus": expectation_onebody_B(psi, solver.PB_plus, solver.dx, solver.dx),
        "B_minus": expectation_onebody_B(psi, solver.PB_minus, solver.dx, solver.dx),
    }


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
# Measurement basis and calibration
# ============================================================

def rotate_state_to_measurement_basis(
    psi: np.ndarray,
    Ua: np.ndarray,
    Ub: np.ndarray,
) -> np.ndarray:
    tmp = np.einsum("ac,xycb->xyab", Ua.conj().T, psi)
    out = np.einsum("xyab,bd->xyad", tmp, Ub.conj().T)
    return out


def spin_resolved_spatial_marginals(
    psi: np.ndarray,
    solver: TwoParticleSpin1DFastSolver,
) -> dict[str, np.ndarray]:
    psi_m = rotate_state_to_measurement_basis(psi, solver.Ua_basis, solver.Ub_basis)

    rho_A_plus = np.sum(np.abs(psi_m[:, :, 0, :]) ** 2, axis=(0, 2)) * solver.dx
    rho_A_minus = np.sum(np.abs(psi_m[:, :, 1, :]) ** 2, axis=(0, 2)) * solver.dx

    rho_B_plus = np.sum(np.abs(psi_m[:, :, :, 0]) ** 2, axis=(1, 2)) * solver.dx
    rho_B_minus = np.sum(np.abs(psi_m[:, :, :, 1]) ** 2, axis=(1, 2)) * solver.dx

    return {
        "A_plus": rho_A_plus.astype(float),
        "A_minus": rho_A_minus.astype(float),
        "B_plus": rho_B_plus.astype(float),
        "B_minus": rho_B_minus.astype(float),
    }


def find_peak_pair(x: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> tuple[float, float]:
    return float(x[int(np.argmax(y1))]), float(x[int(np.argmax(y2))])


def build_detector_masks(
    x: np.ndarray,
    centers: dict[str, float],
    halfwidth: float,
) -> dict[str, np.ndarray]:
    return {
        "A_plus": np.abs(x - centers["A_plus"]) <= halfwidth,
        "A_minus": np.abs(x - centers["A_minus"]) <= halfwidth,
        "B_plus": np.abs(x - centers["B_plus"]) <= halfwidth,
        "B_minus": np.abs(x - centers["B_minus"]) <= halfwidth,
    }


def detector_click_probabilities(
    psi: np.ndarray,
    solver: TwoParticleSpin1DFastSolver,
    detector_masks: dict[str, np.ndarray],
) -> dict[str, float]:
    psi_m = rotate_state_to_measurement_basis(psi, solver.Ua_basis, solver.Ub_basis)
    dx2 = solver.dx * solver.dx

    mask_pp = detector_masks["B_plus"][:, None] & detector_masks["A_plus"][None, :]
    mask_pm = detector_masks["B_minus"][:, None] & detector_masks["A_plus"][None, :]
    mask_mp = detector_masks["B_plus"][:, None] & detector_masks["A_minus"][None, :]
    mask_mm = detector_masks["B_minus"][:, None] & detector_masks["A_minus"][None, :]

    p_pp_raw = float(np.sum(np.abs(psi_m[:, :, 0, 0]) ** 2 * mask_pp) * dx2)
    p_pm_raw = float(np.sum(np.abs(psi_m[:, :, 0, 1]) ** 2 * mask_pm) * dx2)
    p_mp_raw = float(np.sum(np.abs(psi_m[:, :, 1, 0]) ** 2 * mask_mp) * dx2)
    p_mm_raw = float(np.sum(np.abs(psi_m[:, :, 1, 1]) ** 2 * mask_mm) * dx2)

    total = p_pp_raw + p_pm_raw + p_mp_raw + p_mm_raw

    if total > 0.0:
        return {
            "++": p_pp_raw / total,
            "+-": p_pm_raw / total,
            "-+": p_mp_raw / total,
            "--": p_mm_raw / total,
            "pp_raw": p_pp_raw,
            "pm_raw": p_pm_raw,
            "mp_raw": p_mp_raw,
            "mm_raw": p_mm_raw,
            "total_weight": total,
        }

    return {
        "++": 0.0,
        "+-": 0.0,
        "-+": 0.0,
        "--": 0.0,
        "pp_raw": 0.0,
        "pm_raw": 0.0,
        "mp_raw": 0.0,
        "mm_raw": 0.0,
        "total_weight": 0.0,
    }


def choose_detector_assignment(
    run: dict,
    solver: TwoParticleSpin1DFastSolver,
    mode: str,
    halfwidth: float,
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, float]]:
    if mode == "last":
        idx = len(run["psi_frames"]) - 1
    else:
        idx = len(run["psi_frames"]) - 1

    psi = run["psi_frames"][idx].astype(np.complex128)
    marg = spin_resolved_spatial_marginals(psi, solver)

    a1, a2 = find_peak_pair(solver.x, marg["A_plus"], marg["A_minus"])
    b1, b2 = find_peak_pair(solver.x, marg["B_plus"], marg["B_minus"])

    proj = joint_spin_probs(psi, solver)

    best = None

    for a_swap in [False, True]:
        for b_swap in [False, True]:
            centers = {
                "A_plus": a2 if a_swap else a1,
                "A_minus": a1 if a_swap else a2,
                "B_plus": b2 if b_swap else b1,
                "B_minus": b1 if b_swap else b2,
                "frame_idx": int(idx),
            }
            masks = build_detector_masks(solver.x, centers, halfwidth)
            det = detector_click_probabilities(psi, solver, masks)

            err = (
                (det["++"] - proj["++"]) ** 2
                + (det["+-"] - proj["+-"]) ** 2
                + (det["-+"] - proj["-+"]) ** 2
                + (det["--"] - proj["--"]) ** 2
            )

            rec = {
                "a_swap": a_swap,
                "b_swap": b_swap,
                "centers": centers,
                "masks": masks,
                "det": det,
                "proj": proj,
                "err": float(err),
                "E_proj": float(bell_correlation_E(psi, solver)),
                "E_det": float(det["++"] + det["--"] - det["+-"] - det["-+"]),
                "frame_idx": int(idx),
            }

            if best is None or rec["err"] < best["err"]:
                best = rec

    diag = {
        "a_swap": bool(best["a_swap"]),
        "b_swap": bool(best["b_swap"]),
        "err": float(best["err"]),
        "E_proj_calib": float(best["E_proj"]),
        "E_det_calib": float(best["E_det"]),
        "frame_idx": int(best["frame_idx"]),
        "detector_weight_calib": float(best["det"]["total_weight"]),
    }

    return best["centers"], best["masks"], diag


def compute_detector_series(
    run: dict,
    solver: TwoParticleSpin1DFastSolver,
    detector_masks: dict[str, np.ndarray],
) -> list[dict[str, float]]:
    out = []
    for psi in run["psi_frames"]:
        out.append(detector_click_probabilities(psi.astype(np.complex128), solver, detector_masks))
    return out


# ============================================================
# Click event model
# ============================================================

def find_click_event(
    times: np.ndarray,
    detector_series: list[dict[str, float]],
    psi_frames: np.ndarray,
    solver,
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
    proj = joint_spin_probs(psi_click, solver)

    return {
        "frame_idx": int(idx),
        "time": float(times[idx]),
        "threshold": float(threshold),
        "total_weight": float(det["total_weight"]),
        "Ppp": float(proj["++"]),
        "Ppm": float(proj["+-"]),
        "Pmp": float(proj["-+"]),
        "Pmm": float(proj["--"]),
        "E_click": float(proj["++"] + proj["--"] - proj["+-"] - proj["-+"]),
    }

# ============================================================
# Plotting
# ============================================================

def make_detector_calibration_plot(
    outdir: Path,
    solver: TwoParticleSpin1DFastSolver,
    run: dict,
    centers: dict[str, float],
    diag: dict[str, float],
):
    idx = int(diag["frame_idx"])
    psi = run["psi_frames"][idx].astype(np.complex128)
    marg = spin_resolved_spatial_marginals(psi, solver)
    x = solver.x

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(x, marg["A_plus"], label="rho_A+")
    axes[0].plot(x, marg["A_minus"], label="rho_A-")
    axes[0].axvline(centers["A_plus"], linestyle="--")
    axes[0].axvline(centers["A_minus"], linestyle="--")
    axes[0].axvspan(centers["A_plus"] - solver.cfg.detector_halfwidth, centers["A_plus"] + solver.cfg.detector_halfwidth, alpha=0.12)
    axes[0].axvspan(centers["A_minus"] - solver.cfg.detector_halfwidth, centers["A_minus"] + solver.cfg.detector_halfwidth, alpha=0.12)
    axes[0].set_title(
        f"A-side calibration from frame {idx}\n"
        f"a_swap={diag['a_swap']} b_swap={diag['b_swap']} err={diag['err']:.3e}"
    )
    axes[0].set_ylabel("density")
    axes[0].legend()

    axes[1].plot(x, marg["B_plus"], label="rho_B+")
    axes[1].plot(x, marg["B_minus"], label="rho_B-")
    axes[1].axvline(centers["B_plus"], linestyle="--")
    axes[1].axvline(centers["B_minus"], linestyle="--")
    axes[1].axvspan(centers["B_plus"] - solver.cfg.detector_halfwidth, centers["B_plus"] + solver.cfg.detector_halfwidth, alpha=0.12)
    axes[1].axvspan(centers["B_minus"] - solver.cfg.detector_halfwidth, centers["B_minus"] + solver.cfg.detector_halfwidth, alpha=0.12)
    axes[1].set_title(
        f"B-side calibration\n"
        f"E_proj={diag['E_proj_calib']:.4f}, E_det={diag['E_det_calib']:.4f}, W={diag['detector_weight_calib']:.3e}"
    )
    axes[1].set_xlabel("position")
    axes[1].set_ylabel("density")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(outdir / "spin_detector_calibration.png", dpi=150)
    plt.close(fig)


def make_click_summary_plot(
    outdir: Path,
    solver: TwoParticleSpin1DFastSolver,
    run: dict,
    detector_masks: dict[str, np.ndarray],
    detector_series: list[dict[str, float]],
    click_event: dict[str, float] | None,
):
    x = solver.x
    dx = solver.dx
    times = run["times"]
    joint = run["joint_frames"]
    psi_frames = run["psi_frames"]

    E_proj = []
    W_det = []

    for psi, det in zip(psi_frames, detector_series):
        psi = psi.astype(np.complex128)
        E_proj.append(bell_correlation_E(psi, solver))
        W_det.append(det["total_weight"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

    axes[0].plot(times, E_proj, label="E_proj(a,b)")
    axes[0].axhline(-np.cos(solver.cfg.theta_a - solver.cfg.theta_b), linestyle="--", label="singlet ideal")
    if click_event is not None:
        axes[0].axvline(click_event["time"], linestyle="--", label=f"click @ t={click_event['time']:.2f}")
    axes[0].set_title("Projector correlation and click time")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("correlation")
    axes[0].legend()

    axes[1].plot(times, W_det, label="detector total weight")
    axes[1].axhline(solver.cfg.click_weight_threshold, linestyle="--", label="click threshold")
    if click_event is not None:
        axes[1].axvline(click_event["time"], linestyle="--", label=f"click @ t={click_event['time']:.2f}")
    axes[1].set_title("Detector coincidence weight")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("weight")
    axes[1].legend()

    if click_event is not None:
        idx = int(click_event["frame_idx"])
        rho = joint[idx]

        im = axes[2].imshow(
            safe_frame_normalize(rho),
            origin="lower",
            extent=[x[0], x[-1], x[0], x[-1]],
            aspect="auto",
            cmap="magma",
        )

        centers = {}
        for name, mask in detector_masks.items():
            if np.any(mask):
                centers[name] = float(np.mean(x[np.where(mask)[0]]))
            else:
                centers[name] = 0.0

        for name in ["A_plus", "A_minus"]:
            c = centers[name]
            axes[2].axvspan(c - solver.cfg.detector_halfwidth, c + solver.cfg.detector_halfwidth, alpha=0.10)
        for name in ["B_plus", "B_minus"]:
            c = centers[name]
            axes[2].axhspan(c - solver.cfg.detector_halfwidth, c + solver.cfg.detector_halfwidth, alpha=0.10)

        axes[2].set_title(
            f"Joint density at click frame {idx}, t={click_event['time']:.2f}\n"
            f"E_click={click_event['E_click']:.4f}, "
            f"P++={click_event['Ppp']:.3f}, "
            f"P+-={click_event['Ppm']:.3f}, "
            f"P-+={click_event['Pmp']:.3f}, "
            f"P--={click_event['Pmm']:.3f}"
        )
        axes[2].set_xlabel("x_A")
        axes[2].set_ylabel("x_B")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    else:
        axes[2].text(0.5, 0.5, "No click event found", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_axis_off()

    plt.tight_layout()
    fig.savefig(outdir / "spin_click_summary.png", dpi=150)
    plt.close(fig)


def make_click_channels_plot(
    outdir: Path,
    solver: TwoParticleSpin1DFastSolver,
    run: dict,
    detector_series: list[dict[str, float]],
    click_event: dict[str, float] | None,
):
    times = run["times"]

    Ppp = np.array([d["++"] for d in detector_series], dtype=float)
    Ppm = np.array([d["+-"] for d in detector_series], dtype=float)
    Pmp = np.array([d["-+"] for d in detector_series], dtype=float)
    Pmm = np.array([d["--"] for d in detector_series], dtype=float)
    W = np.array([d["total_weight"] for d in detector_series], dtype=float)

    mask = W >= solver.cfg.click_weight_threshold

    Ppp_plot = Ppp.copy()
    Ppm_plot = Ppm.copy()
    Pmp_plot = Pmp.copy()
    Pmm_plot = Pmm.copy()

    Ppp_plot[~mask] = np.nan
    Ppm_plot[~mask] = np.nan
    Pmp_plot[~mask] = np.nan
    Pmm_plot[~mask] = np.nan

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(times, Ppp_plot, label="P++ detector")
    axes[0].plot(times, Ppm_plot, label="P+- detector")
    axes[0].plot(times, Pmp_plot, label="P-+ detector")
    axes[0].plot(times, Pmm_plot, label="P-- detector")

    if click_event is not None:
        t_click = click_event["time"]
        axes[0].axvline(t_click, linestyle="--", label=f"click @ t={t_click:.2f}")

        dt_vis = 0.06
        axes[0].scatter(
            [t_click - 1.5 * dt_vis], [click_event["Ppp"]],
            s=140, edgecolors="black", linewidths=1.2, zorder=10,
            label="P++ click (Born)"
        )
        axes[0].scatter(
            [t_click - 0.5 * dt_vis], [click_event["Ppm"]],
            s=140, edgecolors="black", linewidths=1.2, zorder=10,
            label="P+- click (Born)"
        )
        axes[0].scatter(
            [t_click + 0.5 * dt_vis], [click_event["Pmp"]],
            s=140, edgecolors="black", linewidths=1.2, zorder=10,
            label="P-+ click (Born)"
        )
        axes[0].scatter(
            [t_click + 1.5 * dt_vis], [click_event["Pmm"]],
            s=140, edgecolors="black", linewidths=1.2, zorder=10,
            label="P-- click (Born)"
        )

    axes[0].set_ylabel("click probability")
    axes[0].set_title("Detector channels vs Born click-event channels")
    axes[0].legend()

    axes[1].plot(times, W, label="detector total weight")
    axes[1].axhline(solver.cfg.click_weight_threshold, linestyle="--", label="click threshold")
    if click_event is not None:
        axes[1].axvline(click_event["time"], linestyle="--", label=f"click @ t={click_event['time']:.2f}")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("weight")
    axes[1].set_title("Detector coincidence weight")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(outdir / "spin_click_channels.png", dpi=150)
    plt.close(fig)

def make_animation(
    outdir: Path,
    solver: TwoParticleSpin1DFastSolver,
    run: dict,
    detector_series: list[dict[str, float]],
    detector_masks: dict[str, np.ndarray],
    click_event: dict[str, float] | None,
    max_frames: int = 220,
):
    x = solver.x
    dx = solver.dx
    joint = run["joint_frames"]
    times = run["times"]

    if len(times) > max_frames:
        idxs = np.linspace(0, len(times) - 1, max_frames).astype(int)
        idxs = np.unique(idxs)
    else:
        idxs = np.arange(len(times))

    joint_sub = joint[idxs]
    det_sub = [detector_series[i] for i in idxs]
    times_sub = times[idxs]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    rho0 = joint_sub[0]
    rho_a0 = np.sum(rho0, axis=0) * dx
    rho_b0 = np.sum(rho0, axis=1) * dx

    det0 = det_sub[0]
    labels = ["++", "+-", "-+", "--"]
    vals0 = [det0[k] for k in labels]

    im = axes[0].imshow(
        safe_frame_normalize(rho0),
        origin="lower",
        extent=[x[0], x[-1], x[0], x[-1]],
        aspect="auto",
        cmap="magma",
    )

    centers = {}
    for name, mask in detector_masks.items():
        if np.any(mask):
            centers[name] = float(np.mean(x[np.where(mask)[0]]))
        else:
            centers[name] = 0.0

    for name in ["A_plus", "A_minus"]:
        c = centers[name]
        axes[0].axvspan(c - solver.cfg.detector_halfwidth, c + solver.cfg.detector_halfwidth, alpha=0.10)
    for name in ["B_plus", "B_minus"]:
        c = centers[name]
        axes[0].axhspan(c - solver.cfg.detector_halfwidth, c + solver.cfg.detector_halfwidth, alpha=0.10)

    axes[0].set_title("Joint density")
    axes[0].set_xlabel("x_A")
    axes[0].set_ylabel("x_B")

    (line_a,) = axes[1].plot(x, rho_a0, label="rho_A")
    (line_b,) = axes[1].plot(x, rho_b0, label="rho_B")
    axes[1].set_title("Position marginals")
    axes[1].legend()

    bars = axes[2].bar(labels, vals0)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title(
        f"Detector click probs\n"
        f"W={det0['total_weight']:.3e}"
    )

    title = fig.suptitle(f"t={times_sub[0]:.2f}")

    plt.tight_layout()

    def update(frame_idx: int):
        rho = joint_sub[frame_idx]
        det = det_sub[frame_idx]

        rho_a = np.sum(rho, axis=0) * dx
        rho_b = np.sum(rho, axis=1) * dx
        vals = [det[k] for k in labels]

        im.set_data(safe_frame_normalize(rho))
        line_a.set_ydata(rho_a)
        line_b.set_ydata(rho_b)

        for bar, v in zip(bars, vals):
            bar.set_height(v)

        suffix = ""
        if click_event is not None and abs(times_sub[frame_idx] - click_event["time"]) < 0.5 * solver.cfg.save_every * solver.cfg.dt:
            suffix = " [click frame]"
        axes[2].set_title(f"Detector click probs\nW={det['total_weight']:.3e}{suffix}")
        title.set_text(f"t={times_sub[frame_idx]:.2f}")
        return [im, line_a, line_b, title, *bars]

    anim = FuncAnimation(fig, update, frames=len(times_sub), interval=50, blit=False)

    mp4_path = outdir / "spin_click_animation.mp4"
    gif_path = outdir / "spin_click_animation.gif"

    try:
        print("[ANIM] saving animation...")
        anim.save(mp4_path, dpi=140)
        print(f"[ANIM] saved {mp4_path}")
    except Exception as e:
        print(f"[ANIM] mp4 save failed: {e}")
        try:
            anim.save(gif_path, dpi=100)
            print(f"[ANIM] saved {gif_path}")
        except Exception as e2:
            print(f"[ANIM] gif save failed: {e2}")

    plt.close(fig)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast two-particle 1D spin click-event Bell PoC")
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
    p.add_argument("--calibration-frame-mode", type=str, default="last", choices=["last"])
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
        calibration_frame_mode=str(args.calibration_frame_mode),
        no_anim=bool(args.no_anim),
    )

    print("[START] building fast spin click-event Bell PoC")
    print(cfg)

    solver = TwoParticleSpin1DFastSolver(cfg)
    psi0 = solver.make_singlet_entangled_state()

    print("[INIT] initial singlet state built and normalized")
    run = solver.evolve(psi0)

    print("[CAL] auto-calibrating detector centers and sign labels ...")
    centers, detector_masks, diag = choose_detector_assignment(
        run=run,
        solver=solver,
        mode=cfg.calibration_frame_mode,
        halfwidth=cfg.detector_halfwidth,
    )

    print(
        "[CAL] chosen assignment: "
        f"a_swap={diag['a_swap']}, "
        f"b_swap={diag['b_swap']}, "
        f"err={diag['err']:.3e}, "
        f"E_proj_calib={diag['E_proj_calib']:.4f}, "
        f"E_det_calib={diag['E_det_calib']:.4f}, "
        f"W_calib={diag['detector_weight_calib']:.3e}"
    )
    print(
        "[CAL] centers: "
        f"A+={centers['A_plus']:.3f}, "
        f"A-={centers['A_minus']:.3f}, "
        f"B+={centers['B_plus']:.3f}, "
        f"B-={centers['B_minus']:.3f}, "
        f"frame_idx={diag['frame_idx']}"
    )

    detector_series = compute_detector_series(run, solver, detector_masks)

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
        print(
            "[CLICK] "
            f"frame_idx={click_event['frame_idx']}, "
            f"t={click_event['time']:.6f}, "
            f"W={click_event['total_weight']:.6e}, "
            f"E_click={click_event['E_click']:.6f}, "
            f"P++={click_event['Ppp']:.4f}, "
            f"P+-={click_event['Ppm']:.4f}, "
            f"P-+={click_event['Pmp']:.4f}, "
            f"P--={click_event['Pmm']:.4f}"
        )

    print("[SAVE] saving run_data.npz ...")
    np.savez_compressed(
        outdir / "run_data.npz",
        x=solver.x,
        joint_frames=run["joint_frames"],
        psi_frames=run["psi_frames"],
        times=run["times"],
        norms=run["norms"],
    )

    print("[PLOT] detector calibration ...")
    make_detector_calibration_plot(outdir, solver, run, centers, diag)

    print("[PLOT] click summary ...")
    make_click_summary_plot(outdir, solver, run, detector_masks, detector_series, click_event)

    print("[PLOT] click channels ...")
    make_click_channels_plot(outdir, solver, run, detector_series, click_event)

    if not cfg.no_anim:
        make_animation(outdir, solver, run, detector_series, detector_masks, click_event)

    print("[DONE] outputs saved to", outdir)
    print("  - spin_detector_calibration.png")
    print("  - spin_click_summary.png")
    print("  - spin_click_channels.png")
    print("  - run_data.npz")
    print("  - spin_click_animation.mp4 or .gif")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())