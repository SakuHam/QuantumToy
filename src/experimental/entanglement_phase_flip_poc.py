from __future__ import annotations

import argparse
import copy
import time
from dataclasses import dataclass, replace
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


def cyclic_texture_rgba(
    value: np.ndarray,
    alpha: np.ndarray,
    cmap: str = "twilight_shifted",
) -> np.ndarray:
    rgba = plt.get_cmap(cmap)(np.mod(value, 1.0))
    rgba[..., 3] = np.clip(alpha, 0.0, 1.0)
    return rgba


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def prob_dict_E(p: dict[str, float]) -> float:
    return float(p["++"] + p["--"] - p["+-"] - p["-+"])


# ============================================================
# Spin matrices
# ============================================================

sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)


def pauli_along_axis(theta: float) -> np.ndarray:
    # Axis in the x-z plane: n(theta) = (sin theta, 0, cos theta)
    return np.sin(theta) * sigma_x + np.cos(theta) * sigma_z


def analyzer_axis(theta: float) -> tuple[float, float, float]:
    return float(np.sin(theta)), 0.0, float(np.cos(theta))


def spin_rotation_to_axis(theta: float) -> np.ndarray:
    # Real rotation that maps z-basis to the analyzer basis in the x-z plane.
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
    sg_gradient_a: float = 0.12
    sg_gradient_b: float = 0.12

    theta_a: float = 0.0
    theta_b: float = np.pi / 3.0

    spin_model: str = "phase_flip_wave"  # "sg", "spinning_wave", or "phase_flip_wave"
    spin_wave_k: float = 0.45
    spin_wave_omega: float = 1.2
    spin_wave_tilt: float = np.pi / 3.0
    spin_wave_phase_a: float = 0.0
    spin_wave_phase_b: float = np.pi

    # Detector changes compared with the first PoC:
    # - detector_mode="dynamic": peaks/masks recomputed every frame.
    # - detector_mode="fixed_final": detector calibrated from final frame and then kept fixed.
    # - detector_mode="fixed_mid": detector calibrated from middle frame and then kept fixed.
    # - detector_mode="fixed_best": first dynamic pass finds max W frame, then detector is fixed there.
    detector_mode: str = "fixed_best"
    detector_response: str = "gaussian"  # "gaussian" or "hard"
    detector_use_voronoi_gate: bool = True
    detector_halfwidth: float = 2.5
    click_model: str = "hazard"  # "hazard" or "argmax"
    click_hazard_rate: float = 12.0
    click_weight_threshold: float = 1e-3
    rng_seed: int = 12345

    print_every_frames: int = 20
    no_anim: bool = False
    no_plots: bool = False


# ============================================================
# Fast spin solver
# ============================================================

class TwoParticleSpin1DFastSolver:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.x = np.linspace(-cfg.Lx / 2.0, cfg.Lx / 2.0, cfg.Nx, endpoint=False)
        self.dx = float(self.x[1] - self.x[0])

        # Shape convention: array[y=x_B, x=x_A] for imshow friendliness.
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

        self._validate_config()

    def _validate_config(self) -> None:
        if self.cfg.spin_model not in {"sg", "spinning_wave", "phase_flip_wave"}:
            raise ValueError("spin_model must be 'sg', 'spinning_wave', or 'phase_flip_wave'")
        if self.cfg.detector_mode not in {"dynamic", "fixed_final", "fixed_mid", "fixed_best"}:
            raise ValueError("detector_mode must be dynamic/fixed_final/fixed_mid/fixed_best")
        if self.cfg.detector_response not in {"gaussian", "hard"}:
            raise ValueError("detector_response must be gaussian or hard")
        if self.cfg.click_model not in {"argmax", "hazard"}:
            raise ValueError("click_model must be argmax or hazard")

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

    def _spinning_wave_axis(
        self,
        theta: float,
        phase: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.cfg
        ax, ay, az = analyzer_axis(theta)

        # Local orthonormal frame around analyzer axis.
        e1x, e1y, e1z = float(np.cos(theta)), 0.0, float(-np.sin(theta))
        e2x, e2y, e2z = 0.0, 1.0, 0.0

        ctilt = np.cos(cfg.spin_wave_tilt)
        stilt = np.sin(cfg.spin_wave_tilt)
        cphase = np.cos(phase)
        sphase = np.sin(phase)

        nx = ctilt * ax + stilt * (cphase * e1x + sphase * e2x)
        ny = ctilt * ay + stilt * (cphase * e1y + sphase * e2y)
        nz = ctilt * az + stilt * (cphase * e1z + sphase * e2z)
        return nx, ny, nz

    def _apply_axis_to_spin_A(
        self,
        psi: np.ndarray,
        nx: np.ndarray,
        ny: np.ndarray,
        nz: np.ndarray,
    ) -> np.ndarray:
        out = np.empty_like(psi)
        off_minus = nx - 1j * ny
        off_plus = nx + 1j * ny
        out[:, :, 0, :] = nz[:, :, None] * psi[:, :, 0, :] + off_minus[:, :, None] * psi[:, :, 1, :]
        out[:, :, 1, :] = off_plus[:, :, None] * psi[:, :, 0, :] - nz[:, :, None] * psi[:, :, 1, :]
        return out

    def _apply_axis_to_spin_B(
        self,
        psi: np.ndarray,
        nx: np.ndarray,
        ny: np.ndarray,
        nz: np.ndarray,
    ) -> np.ndarray:
        out = np.empty_like(psi)
        off_minus = nx - 1j * ny
        off_plus = nx + 1j * ny
        out[:, :, :, 0] = nz[:, :, None] * psi[:, :, :, 0] + off_minus[:, :, None] * psi[:, :, :, 1]
        out[:, :, :, 1] = off_plus[:, :, None] * psi[:, :, :, 0] - nz[:, :, None] * psi[:, :, :, 1]
        return out

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

    def _apply_spinning_wave_unitary_A(self, psi: np.ndarray, t: float) -> np.ndarray:
        cfg = self.cfg
        phase = cfg.spin_wave_k * (self.XA - cfg.sg_center_a) - cfg.spin_wave_omega * t + cfg.spin_wave_phase_a
        nx, ny, nz = self._spinning_wave_axis(cfg.theta_a, phase)
        term0 = self.ca[:, :, None, None] * psi
        term1 = -1j * self.sa[:, :, None, None] * self._apply_axis_to_spin_A(psi, nx, ny, nz)
        return term0 + term1

    def _apply_spinning_wave_unitary_B(self, psi: np.ndarray, t: float) -> np.ndarray:
        cfg = self.cfg
        phase = -cfg.spin_wave_k * (self.XB - cfg.sg_center_b) - cfg.spin_wave_omega * t + cfg.spin_wave_phase_b
        nx, ny, nz = self._spinning_wave_axis(cfg.theta_b, phase)
        term0 = self.cb[:, :, None, None] * psi
        term1 = -1j * self.sb[:, :, None, None] * self._apply_axis_to_spin_B(psi, nx, ny, nz)
        return term0 + term1

    def _apply_phase_flip_wave_unitary_A(self, psi: np.ndarray, t: float) -> np.ndarray:
        """
        Phase-flip / sign-flip SG wave for particle A.

        This keeps the spin operator fixed at the analyzer axis sigma_a and
        only modulates the local coupling strength by cos(kx - omega t + phi).
        Therefore the measurement basis is not rotated in time. Unlike
        spinning_wave, this does not introduce non-commuting sigma_n(t) axes.
        """
        cfg = self.cfg
        phase = cfg.spin_wave_k * (self.XA - cfg.sg_center_a) - cfg.spin_wave_omega * t + cfg.spin_wave_phase_a
        mod = np.cos(phase)
        theta_eff = (cfg.dt / (2.0 * cfg.hbar)) * self.scalar_a * mod
        c = np.cos(theta_eff).astype(np.complex128)
        s = np.sin(theta_eff).astype(np.complex128)
        sigma_psi = np.einsum("ab,xybc->xyac", self.sigma_a, psi)
        return c[:, :, None, None] * psi - 1j * s[:, :, None, None] * sigma_psi

    def _apply_phase_flip_wave_unitary_B(self, psi: np.ndarray, t: float) -> np.ndarray:
        """Phase-flip / sign-flip SG wave for particle B."""
        cfg = self.cfg
        phase = -cfg.spin_wave_k * (self.XB - cfg.sg_center_b) - cfg.spin_wave_omega * t + cfg.spin_wave_phase_b
        mod = np.cos(phase)
        theta_eff = (cfg.dt / (2.0 * cfg.hbar)) * self.scalar_b * mod
        c = np.cos(theta_eff).astype(np.complex128)
        s = np.sin(theta_eff).astype(np.complex128)
        sigma_psi = np.einsum("xyab,bc->xyac", psi, self.sigma_b)
        return c[:, :, None, None] * psi - 1j * s[:, :, None, None] * sigma_psi

    def _apply_potential_half_step(self, psi: np.ndarray, t: float) -> np.ndarray:
        if self.cfg.spin_model == "spinning_wave":
            # Rotates the spin axis itself. This is a deliberate stress test and
            # generally changes Bell correlations because sigma_n(t) does not
            # commute with sigma_n(t').
            psi = self._apply_spinning_wave_unitary_A(psi, t)
            psi = self._apply_spinning_wave_unitary_B(psi, t)
        elif self.cfg.spin_model == "phase_flip_wave":
            # Keeps the analyzer axis fixed and only phase/sign-modulates the
            # SG coupling. This should preserve the Born spin probabilities
            # much better while still changing packet geometry.
            psi = self._apply_phase_flip_wave_unitary_A(psi, t)
            psi = self._apply_phase_flip_wave_unitary_B(psi, t)
        else:
            psi = self._apply_spin_unitary_A(psi)
            psi = self._apply_spin_unitary_B(psi)
        psi = self.damp_half[:, :, None, None] * psi
        return psi

    def step(self, psi: np.ndarray, t: float) -> np.ndarray:
        psi = self._apply_potential_half_step(psi, t)
        psi_k = np.fft.fft2(psi, axes=(0, 1))
        psi_k *= self.K_phase[:, :, None, None]
        psi = np.fft.ifft2(psi_k, axes=(0, 1))
        psi = self._apply_potential_half_step(psi, t + self.cfg.dt)
        return psi

    def evolve(self, psi0: np.ndarray, quiet: bool = False) -> dict:
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

                if (not quiet) and (frame_idx % max(1, cfg.print_every_frames)) == 0:
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
                psi = self.step(psi, n * cfg.dt)

        total_elapsed = time.perf_counter() - t_start
        if not quiet:
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
    return prob_dict_E(joint_spin_probs(psi, solver))


# ============================================================
# Measurement basis and detector model
# ============================================================

CHANNELS = ["++", "+-", "-+", "--"]
CHANNEL_COLORS = {
    "++": "tab:blue",
    "+-": "tab:orange",
    "-+": "tab:green",
    "--": "tab:red",
}


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


def build_peak_positions_from_components(
    comps: dict[str, np.ndarray],
    solver: TwoParticleSpin1DFastSolver,
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    peaks = {}
    for ch in CHANNELS:
        xa, xb, amp = component_peak_xy(comps[ch], solver.x)
        peaks[ch] = (xa, xb, amp)

    peak_pos = {ch: (float(peaks[ch][0]), float(peaks[ch][1])) for ch in CHANNELS}
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
    return peak_pos, diag


def build_voronoi_masks_from_peaks(
    solver: TwoParticleSpin1DFastSolver,
    peak_pos: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    dists = []
    for ch in CHANNELS:
        xa, xb = peak_pos[ch]
        d2 = (solver.XA - xa) ** 2 + (solver.XB - xb) ** 2
        dists.append(d2)
    labels = np.argmin(np.stack(dists, axis=0), axis=0)
    return {ch: labels == i for i, ch in enumerate(CHANNELS)}


def build_detector_responses_from_peaks(
    solver: TwoParticleSpin1DFastSolver,
    peak_pos: dict[str, tuple[float, float]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Finite detector response around each channel peak.

    This is the main change from the first PoC:
    detector_halfwidth now matters. The detector no longer integrates each
    channel over its entire Voronoi cell unless detector_halfwidth is made huge.
    """
    cfg = solver.cfg
    sigma = max(float(cfg.detector_halfwidth), 1e-12)
    masks = build_voronoi_masks_from_peaks(solver, peak_pos)
    responses: dict[str, np.ndarray] = {}

    for ch in CHANNELS:
        xa, xb = peak_pos[ch]
        d2 = (solver.XA - xa) ** 2 + (solver.XB - xb) ** 2
        if cfg.detector_response == "gaussian":
            resp = np.exp(-0.5 * d2 / (sigma * sigma))
        else:
            resp = (d2 <= sigma * sigma).astype(float)
        if cfg.detector_use_voronoi_gate:
            resp = resp * masks[ch].astype(float)
        responses[ch] = resp.astype(np.float64)

    return responses, masks


def debug_voronoi_partition(masks: dict[str, np.ndarray]) -> None:
    total = sum(masks[ch].astype(np.int32) for ch in CHANNELS)
    cover_all = bool(np.all(total == 1))
    overlap = bool(np.any(total > 1))
    print(f"[VORONOI CHECK] cover_all={cover_all}, overlap={overlap}")


def detector_click_probabilities_from_responses(
    psi: np.ndarray,
    solver: TwoParticleSpin1DFastSolver,
    responses: dict[str, np.ndarray],
) -> dict[str, float]:
    comps = channel_component_densities(psi, solver)
    dx2 = solver.dx * solver.dx

    raw = {ch: float(np.sum(comps[ch] * responses[ch]) * dx2) for ch in CHANNELS}
    total = sum(raw.values())
    if total > 0.0:
        cond = {ch: float(raw[ch] / total) for ch in CHANNELS}
    else:
        cond = {ch: 0.0 for ch in CHANNELS}

    return {
        "++": cond["++"],
        "+-": cond["+-"],
        "-+": cond["-+"],
        "--": cond["--"],
        "pp_raw": raw["++"],
        "pm_raw": raw["+-"],
        "mp_raw": raw["-+"],
        "mm_raw": raw["--"],
        "total_weight": float(total),
    }


def frame_detector_from_peaks(
    psi: np.ndarray,
    solver: TwoParticleSpin1DFastSolver,
    peak_pos: dict[str, tuple[float, float]],
    peak_diag: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    responses, masks = build_detector_responses_from_peaks(solver, peak_pos)
    det = detector_click_probabilities_from_responses(psi, solver, responses)
    proj = joint_spin_probs(psi, solver)
    basis = full_basis_component_probs(psi, solver)

    diag = dict(peak_diag or {})
    diag.update(
        {
            "err_det_vs_proj": float(sum((det[ch] - proj[ch]) ** 2 for ch in CHANNELS)),
            "err_det_vs_basis": float(sum((det[ch] - basis[ch]) ** 2 for ch in CHANNELS)),
            "W": float(det["total_weight"]),
            "E_det_cond": prob_dict_E(det),
            "E_proj": prob_dict_E(proj),
            "E_basis": prob_dict_E(basis),
        }
    )
    return det, responses, masks, diag


def frame_detector_dynamic(
    psi: np.ndarray,
    solver: TwoParticleSpin1DFastSolver,
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, tuple[float, float]], dict[str, float]]:
    comps = channel_component_densities(psi, solver)
    peak_pos, peak_diag = build_peak_positions_from_components(comps, solver)
    det, responses, masks, diag = frame_detector_from_peaks(psi, solver, peak_pos, peak_diag)
    return det, responses, masks, peak_pos, diag


def compute_detector_series(
    run: dict,
    solver: TwoParticleSpin1DFastSolver,
    quiet: bool = False,
) -> tuple[list[dict[str, float]], list[dict[str, np.ndarray]], list[dict[str, np.ndarray]], list[dict[str, tuple[float, float]]], list[dict[str, float]], int | None]:
    cfg = solver.cfg
    psi_frames = run["psi_frames"]

    calibration_idx: int | None = None

    if cfg.detector_mode == "dynamic":
        detector_series = []
        response_series = []
        masks_series = []
        peaks_series = []
        diag_series = []
        for psi in psi_frames:
            det, resp, masks, peaks, diag = frame_detector_dynamic(psi.astype(np.complex128), solver)
            detector_series.append(det)
            response_series.append(resp)
            masks_series.append(masks)
            peaks_series.append(peaks)
            diag_series.append(diag)
        return detector_series, response_series, masks_series, peaks_series, diag_series, calibration_idx

    if cfg.detector_mode == "fixed_final":
        calibration_idx = len(psi_frames) - 1
    elif cfg.detector_mode == "fixed_mid":
        calibration_idx = len(psi_frames) // 2
    elif cfg.detector_mode == "fixed_best":
        # Diagnostic dynamic pass only to choose a stable detector geometry.
        dyn_weights = []
        dyn_peaks = []
        dyn_peak_diags = []
        for psi in psi_frames:
            comps = channel_component_densities(psi.astype(np.complex128), solver)
            peaks, peak_diag = build_peak_positions_from_components(comps, solver)
            det, _, _, _ = frame_detector_from_peaks(psi.astype(np.complex128), solver, peaks, peak_diag)
            dyn_weights.append(det["total_weight"])
            dyn_peaks.append(peaks)
            dyn_peak_diags.append(peak_diag)
        calibration_idx = int(np.argmax(np.asarray(dyn_weights, dtype=float)))
        if not quiet:
            print(f"[DET] fixed_best calibration frame={calibration_idx}, W_dyn={dyn_weights[calibration_idx]:.6e}")

    psi_calib = psi_frames[int(calibration_idx)].astype(np.complex128)
    comps_calib = channel_component_densities(psi_calib, solver)
    fixed_peaks, fixed_peak_diag = build_peak_positions_from_components(comps_calib, solver)

    detector_series = []
    response_series = []
    masks_series = []
    peaks_series = []
    diag_series = []

    for psi in psi_frames:
        det, resp, masks, diag = frame_detector_from_peaks(
            psi.astype(np.complex128), solver, fixed_peaks, fixed_peak_diag
        )
        diag["calibration_idx"] = int(calibration_idx)
        detector_series.append(det)
        response_series.append(resp)
        masks_series.append(masks)
        peaks_series.append(fixed_peaks)
        diag_series.append(diag)

    return detector_series, response_series, masks_series, peaks_series, diag_series, calibration_idx


# ============================================================
# Click event model
# ============================================================

def choose_click_channel(det: dict[str, float]) -> str:
    vals = {ch: det[ch] for ch in CHANNELS}
    return max(vals, key=vals.get)


def sample_click_channel(det: dict[str, float], rng: np.random.Generator) -> str:
    p = np.array([max(det[ch], 0.0) for ch in CHANNELS], dtype=float)
    s = float(np.sum(p))
    if s <= 0.0:
        return choose_click_channel(det)
    p /= s
    return str(rng.choice(np.asarray(CHANNELS), p=p))


def find_click_event(
    times: np.ndarray,
    detector_series: list[dict[str, float]],
    psi_frames: np.ndarray,
    solver: TwoParticleSpin1DFastSolver,
    threshold: float,
) -> dict[str, float | str] | None:
    weights = np.array([d["total_weight"] for d in detector_series], dtype=float)
    if weights.size == 0:
        return None

    rng = np.random.default_rng(solver.cfg.rng_seed)

    if solver.cfg.click_model == "argmax":
        idx = int(np.argmax(weights))
        if detector_series[idx]["total_weight"] < threshold:
            return None
        clicked_channel = choose_click_channel(detector_series[idx])
    else:
        idx = -1
        clicked_channel = ""
        for i, det in enumerate(detector_series):
            W = float(det["total_weight"])
            if W < threshold:
                continue
            if len(times) > 1:
                if i == 0:
                    dt_frame = float(times[1] - times[0])
                else:
                    dt_frame = float(times[i] - times[i - 1])
            else:
                dt_frame = solver.cfg.dt * solver.cfg.save_every
            p_click = 1.0 - np.exp(-solver.cfg.click_hazard_rate * W * max(dt_frame, 1e-12))
            if rng.random() < p_click:
                idx = i
                clicked_channel = sample_click_channel(det, rng)
                break

        # For debugging it is useful to still return the best frame if the
        # hazard process happened not to fire, but label it clearly.
        if idx < 0:
            best = int(np.argmax(weights))
            if detector_series[best]["total_weight"] < threshold:
                return None
            idx = best
            clicked_channel = "no_hazard_fire_best_" + choose_click_channel(detector_series[best])

    det = detector_series[idx]
    psi_click = psi_frames[idx].astype(np.complex128)
    born = joint_spin_probs(psi_click, solver)
    comp = full_basis_component_probs(psi_click, solver)

    return {
        "frame_idx": int(idx),
        "time": float(times[idx]),
        "threshold": float(threshold),
        "click_model": solver.cfg.click_model,
        "clicked_channel": clicked_channel,
        "total_weight": float(det["total_weight"]),
        "det_pp": float(det["++"]),
        "det_pm": float(det["+-"]),
        "det_mp": float(det["-+"]),
        "det_mm": float(det["--"]),
        "det_pp_raw": float(det["pp_raw"]),
        "det_pm_raw": float(det["pm_raw"]),
        "det_mp_raw": float(det["mp_raw"]),
        "det_mm_raw": float(det["mm_raw"]),
        "E_det_cond": prob_dict_E(det),
        "born_pp": float(born["++"]),
        "born_pm": float(born["+-"]),
        "born_mp": float(born["-+"]),
        "born_mm": float(born["--"]),
        "E_born": prob_dict_E(born),
        "basis_pp": float(comp["++"]),
        "basis_pm": float(comp["+-"]),
        "basis_mp": float(comp["-+"]),
        "basis_mm": float(comp["--"]),
        "E_basis": prob_dict_E(comp),
    }


# ============================================================
# Plot helpers
# ============================================================

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


def detector_response_overlay(
    responses: dict[str, np.ndarray],
    alpha_scale: float = 0.35,
) -> np.ndarray:
    total_resp = safe_frame_normalize(sum(responses[ch] for ch in CHANNELS))
    overlay = np.zeros((*total_resp.shape, 4), dtype=float)
    overlay[..., 0] = 1.0
    overlay[..., 1] = 1.0
    overlay[..., 2] = 1.0
    overlay[..., 3] = alpha_scale * np.power(total_resp, 0.7)
    return overlay


def spinning_pocket_texture_overlay(
    joint_density: np.ndarray,
    masks: dict[str, np.ndarray],
    peaks: dict[str, tuple[float, float]],
    solver: TwoParticleSpin1DFastSolver,
    t: float,
    alpha_scale: float = 0.42,
) -> np.ndarray:
    cfg = solver.cfg
    rho_n = safe_frame_normalize(joint_density)
    phase_value = np.zeros_like(joint_density, dtype=float)
    alpha = np.zeros_like(joint_density, dtype=float)

    handedness = {
        "++": 1.0,
        "+-": -1.0,
        "-+": -1.0,
        "--": 1.0,
    }

    for ch in CHANNELS:
        xa, xb = peaks[ch]
        dx = solver.XA - xa
        dy = solver.XB - xb
        radius = np.sqrt(dx * dx + dy * dy)
        angle = np.arctan2(dy, dx)
        phase = handedness[ch] * angle + cfg.spin_wave_k * radius - cfg.spin_wave_omega * t
        mask = masks[ch]
        phase_value[mask] = np.mod(phase[mask] / (2.0 * np.pi), 1.0)
        alpha[mask] = alpha_scale * np.power(rho_n[mask], 0.55)

    return cyclic_texture_rgba(phase_value, alpha)


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
    errs_proj = []
    errs_basis = []
    e_det = []
    e_born = []
    e_basis = []

    for i, psi in enumerate(psi_frames):
        psi128 = psi.astype(np.complex128)
        jp = joint_spin_probs(psi128, solver)
        bp = full_basis_component_probs(psi128, solver)
        for ch in CHANNELS:
            born[ch].append(jp[ch])
            basis[ch].append(bp[ch])
        errs_proj.append(diag_series[i]["err_det_vs_proj"])
        errs_basis.append(diag_series[i]["err_det_vs_basis"])
        e_det.append(diag_series[i]["E_det_cond"])
        e_born.append(diag_series[i]["E_proj"])
        e_basis.append(diag_series[i]["E_basis"])

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
    ax.plot(times, det_arr["total_weight"], label="finite detector total weight", linewidth=2)
    ax.set_title(
        f"Detector total coincidence weight\nmode={solver.cfg.detector_mode}, response={solver.cfg.detector_response}, halfwidth={solver.cfg.detector_halfwidth}"
    )
    ax.set_xlabel("time")
    ax.set_ylabel("weight")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(times, errs_proj, label="sum sq. error(det vs Born)", linewidth=2)
    ax.plot(times, errs_basis, label="sum sq. error(det vs basis)", linewidth=2, linestyle=":")
    ax.plot(times, e_det, label="E_det_cond", linewidth=2)
    ax.plot(times, e_born, label="E_proj", linewidth=2, linestyle="--")
    ax.plot(times, e_basis, label="E_basis", linewidth=2, linestyle="-.")
    ax.set_title("Calibration diagnostics")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    fig.suptitle("Finite detector calibration summary", fontsize=16)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    print(f"[PLOT] saved {outpath}")


def make_click_summary_plot(outdir, solver, run, detector_series, click_event, masks_series, peaks_series, response_series):
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
    click_channel = str(click_event["clicked_channel"])

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
    overlay = joint_density_with_mask_overlay(joint, masks_series[idx], alpha_scale=0.14)
    ax_img.imshow(overlay, origin="lower", extent=extent, aspect="auto")
    ax_img.imshow(detector_response_overlay(response_series[idx], alpha_scale=0.45), origin="lower", extent=extent, aspect="auto")

    if solver.cfg.spin_model in {"spinning_wave", "phase_flip_wave"}:
        spin_overlay = spinning_pocket_texture_overlay(
            joint,
            masks_series[idx],
            peaks_series[idx],
            solver,
            t_click,
            alpha_scale=0.48,
        )
        ax_img.imshow(spin_overlay, origin="lower", extent=extent, aspect="auto")
    add_channel_peak_labels(ax_img, peaks_series[idx])
    cbar = fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
    cbar.set_label("frame-normalized density")

    info_text = (
        "CLICK FRAME\n"
        f"model={click_event['click_model']}\n"
        f"channel={click_channel}\n"
        f"E_det={click_event['E_det_cond']:.4f}, E_born={click_event['E_born']:.4f}\n"
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
        fontsize=10,
        color="black",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.2"),
    )
    ax_img.set_title("Joint density + detector response" + (" + spin-wave texture" if solver.cfg.spin_model in {"spinning_wave", "phase_flip_wave"} else ""))
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
        f"Detector vs Born click channels\nW={click_event['total_weight']:.3e} | {click_channel}"
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

    fig.suptitle(f"t={t_click:.2f} [{click_channel}]", fontsize=16)
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
    basis = {ch: [] for ch in CHANNELS}
    for psi in psi_frames:
        psi128 = psi.astype(np.complex128)
        jp = joint_spin_probs(psi128, solver)
        bp = full_basis_component_probs(psi128, solver)
        for ch in CHANNELS:
            born[ch].append(jp[ch])
            basis[ch].append(bp[ch])

    idx_click = int(click_event["frame_idx"])
    t_click = float(click_event["time"])

    fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True, constrained_layout=True)

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
    for ch in CHANNELS:
        ax.plot(times, basis[ch], color=CHANNEL_COLORS[ch], label=f"basis {ch}", linewidth=2)
    ax.axvline(t_click, color="k", linestyle="--", alpha=0.7)
    ax.set_title("Rotated-basis component masses")
    ax.set_ylabel("mass")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=4, fontsize=9)

    ax = axes[3]
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


def make_animation(outdir, solver, run, detector_series, click_event, masks_series, peaks_series, response_series):
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
    im_overlay = ax_img.imshow(
        joint_density_with_mask_overlay(rho0, masks_series[0], alpha_scale=0.14),
        origin="lower",
        extent=extent,
        aspect="auto",
        animated=True,
    )
    im_resp_overlay = ax_img.imshow(
        detector_response_overlay(response_series[0], alpha_scale=0.45),
        origin="lower",
        extent=extent,
        aspect="auto",
        animated=True,
    )
    im_spin_overlay = ax_img.imshow(
        spinning_pocket_texture_overlay(rho0, masks_series[0], peaks_series[0], solver, times[0], alpha_scale=0.48),
        origin="lower",
        extent=extent,
        aspect="auto",
        animated=True,
        visible=solver.cfg.spin_model in {"spinning_wave", "phase_flip_wave"},
    )

    peak_scatters = {}
    peak_texts = {}
    for ch in CHANNELS:
        xa, xb = peaks_series[0][ch]
        sc = ax_img.scatter([xa], [xb], s=30, c=CHANNEL_COLORS[ch], edgecolors="white", linewidths=0.7, zorder=4)
        peak_scatters[ch] = sc
        txt = ax_img.text(xa, xb, ch, color="white", fontsize=10, weight="bold", zorder=5)
        peak_texts[ch] = txt

    ax_img.set_title("Joint density + finite detector response")
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
        fontsize=10,
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

    ax_w.plot(times, weights, label="detector total weight", linewidth=2)
    ax_w.axhline(click_event["threshold"], linestyle="--", linewidth=1.2, label="click threshold")
    ax_w.axvline(click_time, linestyle="--", linewidth=1.2, label=f"click @ t={click_time:.2f}")
    current_dot, = ax_w.plot([times[0]], [weights[0]], marker="o", linestyle="None", markersize=5)
    ax_w.set_title("Detector coincidence weight")
    ax_w.set_xlabel("time")
    ax_w.set_ylabel("weight")
    ax_w.grid(True, alpha=0.25)
    ax_w.legend()

    frame_vline = ax_w.axvline(times[0], color="tab:gray", linestyle=":", linewidth=1.4, alpha=0.9)

    label_offsets = {
        "++": (0.6, 0.6),
        "+-": (0.6, -1.4),
        "-+": (-2.2, 0.6),
        "--": (-2.2, -1.4),
    }

    def update_peak_annotations(peaks):
        for ch in CHANNELS:
            xa, xb = peaks[ch]
            peak_scatters[ch].set_offsets(np.array([[xa, xb]]))
            dx, dy = label_offsets[ch]
            peak_texts[ch].set_position((xa + dx, xb + dy))

    def update(frame_idx: int):
        rho = joint_frames[frame_idx]
        im.set_data(safe_frame_normalize(rho))
        im_overlay.set_data(joint_density_with_mask_overlay(rho, masks_series[frame_idx], alpha_scale=0.14))
        im_resp_overlay.set_data(detector_response_overlay(response_series[frame_idx], alpha_scale=0.45))
        if solver.cfg.spin_model in {"spinning_wave", "phase_flip_wave"}:
            im_spin_overlay.set_data(
                spinning_pocket_texture_overlay(
                    rho,
                    masks_series[frame_idx],
                    peaks_series[frame_idx],
                    solver,
                    times[frame_idx],
                    alpha_scale=0.48,
                )
            )
        update_peak_annotations(peaks_series[frame_idx])

        det = detector_series[frame_idx]
        psi = run["psi_frames"][frame_idx].astype(np.complex128)
        born = joint_spin_probs(psi, solver)
        best_channel = choose_click_channel(det)

        for rect, ch in zip(bars, CHANNELS):
            rect.set_height(det[ch])

        born_scatter.set_offsets(np.column_stack([x, [born[ch] for ch in CHANNELS]]))

        info_text = (
            f"{'CLICK FRAME' if frame_idx == click_idx else 'FRAME'}\n"
            f"E_det={prob_dict_E(det):.4f}, E_born={prob_dict_E(born):.4f}\n"
            f"P++={det['++']:.3f}, P+-={det['+-']:.3f}\n"
            f"P-+={det['-+']:.3f}, P--={det['--']:.3f}"
        )
        info_box.set_text(info_text)

        ttl = f"t={times[frame_idx]:.2f}"
        if frame_idx == click_idx:
            ttl += f" [{click_event['clicked_channel']}]"
        elif frame_idx > click_idx:
            ttl += " [post-click]"
        fig.suptitle(ttl, fontsize=16)

        ax_bar.set_title(
            f"Detector vs Born click channels\nW={det['total_weight']:.3e}"
            + (f" | {best_channel}" if frame_idx == click_idx else "")
        )

        frame_vline.set_xdata([times[frame_idx], times[frame_idx]])
        current_dot.set_data([times[frame_idx]], [weights[frame_idx]])

        artists = [im, im_overlay, im_resp_overlay, im_spin_overlay, info_box, born_scatter, current_dot, frame_vline]
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
# Run helpers and CHSH
# ============================================================

def run_single_experiment(cfg: Config, outdir: Path, quiet: bool = False) -> dict:
    if not quiet:
        print("[START] building fast spin click-event Bell PoC")
        print(cfg)

    solver = TwoParticleSpin1DFastSolver(cfg)
    psi0 = solver.make_singlet_entangled_state()

    if not quiet:
        print("[INIT] initial singlet state built and normalized")
    run = solver.evolve(psi0, quiet=quiet)

    if not quiet:
        print("[DET] computing finite detector masks/responses ...")
    detector_series, response_series, masks_series, peaks_series, diag_series, calibration_idx = compute_detector_series(run, solver, quiet=quiet)

    mid_idx = len(run["psi_frames"]) // 2
    last_idx = len(run["psi_frames"]) - 1

    if not quiet:
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
                f"errProj={diag['err_det_vs_proj']:.3e}, "
                f"errBasis={diag['err_det_vs_basis']:.3e}, "
                f"W={diag['W']:.3e}, "
                f"E_det_cond={diag['E_det_cond']:.4f}, "
                f"E_proj={diag['E_proj']:.4f}, "
                f"E_basis={diag['E_basis']:.4f}"
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
    E_det_last = prob_dict_E(detector_series[-1])
    idx_best_W = int(np.argmax(np.array([d["total_weight"] for d in detector_series], dtype=float)))
    E_det_best = prob_dict_E(detector_series[idx_best_W])

    if not quiet:
        print(f"[RUN] frames={len(run['times'])}, final saved time={run['times'][-1]:.3f}")
        print(f"[RUN] total forward elapsed={run['elapsed_sec']:.2f}s")
        print(f"[RUN] norm min={np.min(run['norms']):.6e}, max={np.max(run['norms']):.6e}")
        print(f"[RUN] detector_mode={cfg.detector_mode}, calibration_idx={calibration_idx}")
        print(f"[RUN] final E_proj(a,b)={E_last:.6f}")
        print(f"[RUN] final E_det_cond(a,b)={E_det_last:.6f}")
        print(f"[RUN] best-W E_det_cond(a,b)={E_det_best:.6f} at frame {idx_best_W}")
        print(f"[RUN] ideal singlet E(a,b)={E_ideal:.6f}")

    if click_event is None:
        if not quiet:
            print("[CLICK] no click event found")
    else:
        idx_click = int(click_event["frame_idx"])
        det_click = detector_series[idx_click]
        peaks_click = peaks_series[idx_click]
        diag_click = diag_series[idx_click]

        if not quiet:
            print(
                "[CLICK DETECTOR FRAME] "
                f"++@({peaks_click['++'][0]:.3f},{peaks_click['++'][1]:.3f}), "
                f"+-@({peaks_click['+-'][0]:.3f},{peaks_click['+-'][1]:.3f}), "
                f"-+@({peaks_click['-+'][0]:.3f},{peaks_click['-+'][1]:.3f}), "
                f"--@({peaks_click['--'][0]:.3f},{peaks_click['--'][1]:.3f}), "
                f"errProj={diag_click['err_det_vs_proj']:.3e}, "
                f"errBasis={diag_click['err_det_vs_basis']:.3e}"
            )

            print(
                "[CLICK] "
                f"model={click_event['click_model']}, "
                f"channel={click_event['clicked_channel']}, "
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

    if not cfg.no_plots:
        make_detector_calibration_plot(outdir, solver, run, detector_series, diag_series)
        if click_event is not None:
            make_click_summary_plot(outdir, solver, run, detector_series, click_event, masks_series, peaks_series, response_series)
            make_click_channels_plot(outdir, solver, run, detector_series, click_event)

    if (not cfg.no_anim) and (not cfg.no_plots) and click_event is not None:
        make_animation(outdir, solver, run, detector_series, click_event, masks_series, peaks_series, response_series)

    return {
        "cfg": cfg,
        "solver": solver,
        "run": run,
        "detector_series": detector_series,
        "response_series": response_series,
        "masks_series": masks_series,
        "peaks_series": peaks_series,
        "diag_series": diag_series,
        "calibration_idx": calibration_idx,
        "click_event": click_event,
        "E_final_born": float(E_last),
        "E_final_det": float(E_det_last),
        "E_bestW_det": float(E_det_best),
        "idx_bestW": int(idx_best_W),
        "E_ideal": float(E_ideal),
    }


def run_chsh(cfg: Config, outdir: Path) -> dict:
    """
    Four-angle CHSH test.

    Uses convention:
        S = E(a,b) + E(a,b') + E(a',b) - E(a',b')
    with the standard singlet-friendly angles:
        a=0, a'=pi/2, b=pi/4, b'=-pi/4.

    Ideal singlet gives S=-2*sqrt(2), |S|≈2.828.
    """
    a = 0.0
    ap = np.pi / 2.0
    b = np.pi / 4.0
    bp = -np.pi / 4.0

    pairs = [
        ("ab", a, b, +1.0),
        ("abp", a, bp, +1.0),
        ("apb", ap, b, +1.0),
        ("apbp", ap, bp, -1.0),
    ]

    rows = []
    print("[CHSH] running four angle pairs")
    for name, ta, tb, sign in pairs:
        pair_outdir = outdir / f"chsh_{name}"
        pair_outdir.mkdir(parents=True, exist_ok=True)
        pair_cfg = replace(
            cfg,
            theta_a=float(ta),
            theta_b=float(tb),
            no_anim=True,
            no_plots=True,
        )
        print(f"[CHSH] pair={name}, theta_a={ta:.6f}, theta_b={tb:.6f}")
        result = run_single_experiment(pair_cfg, pair_outdir, quiet=True)

        click = result["click_event"]
        E_click_det = float(click["E_det_cond"]) if click is not None else float("nan")
        E_click_born = float(click["E_born"]) if click is not None else float("nan")

        row = {
            "name": name,
            "theta_a": float(ta),
            "theta_b": float(tb),
            "sign": float(sign),
            "E_final_born": result["E_final_born"],
            "E_final_det": result["E_final_det"],
            "E_bestW_det": result["E_bestW_det"],
            "E_click_det": E_click_det,
            "E_click_born": E_click_born,
            "E_ideal": result["E_ideal"],
        }
        rows.append(row)
        print(
            f"[CHSH] {name}: "
            f"E_ideal={row['E_ideal']:+.6f}, "
            f"E_final_born={row['E_final_born']:+.6f}, "
            f"E_bestW_det={row['E_bestW_det']:+.6f}, "
            f"E_click_det={row['E_click_det']:+.6f}"
        )

    def signed_sum(key: str) -> float:
        vals = [row["sign"] * row[key] for row in rows]
        if any(np.isnan(v) for v in vals):
            return float("nan")
        return float(sum(vals))

    S = {
        "S_ideal": signed_sum("E_ideal"),
        "S_final_born": signed_sum("E_final_born"),
        "S_final_det": signed_sum("E_final_det"),
        "S_bestW_det": signed_sum("E_bestW_det"),
        "S_click_det": signed_sum("E_click_det"),
        "S_click_born": signed_sum("E_click_born"),
    }

    print("[CHSH SUMMARY]")
    for key, val in S.items():
        print(f"  {key} = {val:+.6f}, |{key}| = {abs(val):.6f}")
    print("  classical local bound: |S| <= 2")
    print(f"  quantum singlet Tsirelson target: 2*sqrt(2) = {2.0 * np.sqrt(2.0):.6f}")

    # Save a tiny text summary for sweeps.
    summary_path = outdir / "chsh_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("CHSH summary\n")
        f.write("Convention: S = E(a,b) + E(a,b') + E(a',b) - E(a',b')\n")
        for row in rows:
            f.write(str(row) + "\n")
        f.write(str(S) + "\n")
    print(f"[CHSH] saved {summary_path}")

    return {"rows": rows, **S}


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast two-particle 1D spin click-event Bell/CHSH PoC")
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
    p.add_argument("--sg-region-halfwidth", type=float, default=30.0)
    p.add_argument("--sg-gradient-a", type=float, default=0.12)
    p.add_argument("--sg-gradient-b", type=float, default=0.12)
    p.add_argument("--spin-model", choices=["sg", "spinning_wave", "phase_flip_wave"], default="phase_flip_wave")
    p.add_argument("--spin-wave-k", type=float, default=0.45)
    p.add_argument("--spin-wave-omega", type=float, default=1.2)
    p.add_argument("--spin-wave-tilt", type=float, default=float(np.pi / 3.0))
    p.add_argument("--spin-wave-phase-a", type=float, default=0.0)
    p.add_argument("--spin-wave-phase-b", type=float, default=float(np.pi))

    p.add_argument("--detector-mode", choices=["dynamic", "fixed_final", "fixed_mid", "fixed_best"], default="fixed_best")
    p.add_argument("--detector-response", choices=["gaussian", "hard"], default="gaussian")
    p.add_argument("--detector-halfwidth", type=float, default=2.5)
    p.add_argument("--no-voronoi-gate", action="store_true")
    p.add_argument("--click-model", choices=["argmax", "hazard"], default="hazard")
    p.add_argument("--click-hazard-rate", type=float, default=12.0)
    p.add_argument("--click-weight-threshold", type=float, default=1e-3)
    p.add_argument("--rng-seed", type=int, default=12345)

    p.add_argument("--print-every-frames", type=int, default=20)
    p.add_argument("--no-anim", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--chsh", action="store_true", help="Run four-angle CHSH test instead of one plotted run")
    return p.parse_args()


def cfg_from_args(args: argparse.Namespace) -> Config:
    return Config(
        Nx=int(args.nx),
        n_steps=int(args.n_steps),
        dt=float(args.dt),
        save_every=int(args.save_every),
        k0=float(args.k0),
        sigma_cm=float(args.sigma_cm),
        sigma_rel=float(args.sigma_rel),
        theta_a=float(args.theta_a),
        theta_b=float(args.theta_b),
        sg_region_halfwidth=float(args.sg_region_halfwidth),
        sg_gradient_a=float(args.sg_gradient_a),
        sg_gradient_b=float(args.sg_gradient_b),
        spin_model=str(args.spin_model),
        spin_wave_k=float(args.spin_wave_k),
        spin_wave_omega=float(args.spin_wave_omega),
        spin_wave_tilt=float(args.spin_wave_tilt),
        spin_wave_phase_a=float(args.spin_wave_phase_a),
        spin_wave_phase_b=float(args.spin_wave_phase_b),
        detector_mode=str(args.detector_mode),
        detector_response=str(args.detector_response),
        detector_use_voronoi_gate=not bool(args.no_voronoi_gate),
        detector_halfwidth=float(args.detector_halfwidth),
        click_model=str(args.click_model),
        click_hazard_rate=float(args.click_hazard_rate),
        click_weight_threshold=float(args.click_weight_threshold),
        rng_seed=int(args.rng_seed),
        print_every_frames=int(args.print_every_frames),
        no_anim=bool(args.no_anim),
        no_plots=bool(args.no_plots),
    )


# ============================================================
# Main
# ============================================================

def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = cfg_from_args(args)

    if args.chsh:
        run_chsh(cfg, outdir)
    else:
        run_single_experiment(cfg, outdir, quiet=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
