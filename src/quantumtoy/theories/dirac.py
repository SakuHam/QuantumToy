from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from theories.base import TheoryModel, TheoryStepResult
from core.utils import normalize_unit


@dataclass
class DiracTheory(TheoryModel):
    grid: any
    potential: any

    m_mass: float = 1.0
    hbar: float = 1.0
    c_light: float = 1.0

    # Debug / safety flags
    debug_checks: bool = True
    debug_tol_unitarity: float = 1e-9
    debug_tol_projector: float = 1e-9
    debug_tol_velocity: float = 1e-9
    debug_tol_norm_growth: float = 1e-9

    # Optional richer debug
    debug_packet_stats: bool = True

    # -----------------------------------------------------
    # init
    # -----------------------------------------------------

    def __post_init__(self):
        self.m_mass = float(self.m_mass)
        self.hbar = float(self.hbar)
        self.c_light = float(self.c_light)

        self._assert(hasattr(self.grid, "dx"), "grid missing dx")
        self._assert(hasattr(self.grid, "dy"), "grid missing dy")
        self._assert(hasattr(self.grid, "Nx"), "grid missing Nx")
        self._assert(hasattr(self.grid, "Ny"), "grid missing Ny")
        self._assert(hasattr(self.grid, "X"), "grid missing X")
        self._assert(hasattr(self.grid, "Y"), "grid missing Y")

        self._assert(self.grid.dx > 0.0, "grid.dx must be positive")
        self._assert(self.grid.dy > 0.0, "grid.dy must be positive")
        self._assert(self.grid.Nx > 0, "grid.Nx must be positive")
        self._assert(self.grid.Ny > 0, "grid.Ny must be positive")
        self._assert(self.hbar > 0.0, "hbar must be positive")
        self._assert(self.c_light > 0.0, "c_light must be positive")
        self._assert(self.m_mass >= 0.0, "m_mass must be non-negative")

        self.grid.X = np.asarray(self.grid.X)
        self.grid.Y = np.asarray(self.grid.Y)

        self._assert(
            self.grid.X.shape == (self.grid.Ny, self.grid.Nx),
            f"grid.X shape {self.grid.X.shape} != {(self.grid.Ny, self.grid.Nx)}",
        )
        self._assert(
            self.grid.Y.shape == (self.grid.Ny, self.grid.Nx),
            f"grid.Y shape {self.grid.Y.shape} != {(self.grid.Ny, self.grid.Nx)}",
        )

        self.kx = 2.0 * np.pi * np.fft.fftfreq(self.grid.Nx, d=self.grid.dx)
        self.ky = 2.0 * np.pi * np.fft.fftfreq(self.grid.Ny, d=self.grid.dy)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing="xy")

        self._assert(self.KX.shape == (self.grid.Ny, self.grid.Nx), "KX shape mismatch")
        self._assert(self.KY.shape == (self.grid.Ny, self.grid.Nx), "KY shape mismatch")

        self.V_fwd = np.asarray(
            self.potential.V_real - 1j * self.potential.W,
            dtype=np.complex128,
        )
        self.V_adj = np.conjugate(self.V_fwd)

        self._assert(
            self.V_fwd.shape == (self.grid.Ny, self.grid.Nx),
            "Potential shape mismatch: expected (Ny, Nx)",
        )
        self._assert(
            self.V_adj.shape == (self.grid.Ny, self.grid.Nx),
            "Adjoint potential shape mismatch: expected (Ny, Nx)",
        )

        if self.debug_checks:
            self._debug_assert_finite_array("grid.X", self.grid.X)
            self._debug_assert_finite_array("grid.Y", self.grid.Y)
            self._debug_assert_finite_array("KX", self.KX)
            self._debug_assert_finite_array("KY", self.KY)
            self._debug_assert_finite_array("V_fwd", self.V_fwd)
            self._debug_assert_finite_array("V_adj", self.V_adj)
            self._debug_check_projector_once()
            self._debug_check_k_operator_once()

    # -----------------------------------------------------
    # generic asserts / debug
    # -----------------------------------------------------

    def _assert(self, cond: bool, msg: str):
        if not cond:
            raise AssertionError(msg)

    def _debug_warn(self, msg: str):
        if self.debug_checks:
            print(f"[DiracTheory DEBUG] {msg}")

    def _debug_info(self, msg: str):
        if self.debug_checks:
            print(f"[DiracTheory INFO] {msg}")

    def _debug_assert_finite_array(self, name: str, arr: np.ndarray):
        if not self.debug_checks:
            return
        arr = np.asarray(arr)
        if not np.all(np.isfinite(arr)):
            bad = np.size(arr) - np.count_nonzero(np.isfinite(arr))
            raise RuntimeError(f"{name} contains non-finite values ({bad} bad entries)")

    def _debug_assert_spinor_shape_full(self, name: str, psi: np.ndarray):
        if not self.debug_checks:
            return
        if psi.shape != (2, self.grid.Ny, self.grid.Nx):
            raise RuntimeError(
                f"{name} has shape {psi.shape}, expected (2, {self.grid.Ny}, {self.grid.Nx})"
            )

    def _assert_spinor_has_two_components(self, name: str, psi: np.ndarray):
        psi = np.asarray(psi)
        if psi.ndim != 3 or psi.shape[0] != 2:
            raise RuntimeError(
                f"{name} must have shape (2, H, W), got {psi.shape}"
            )

    def _assert_same_spatial_shape(self, name: str, a: np.ndarray, b: np.ndarray):
        if a.shape != b.shape:
            raise RuntimeError(f"{name}: shape mismatch {a.shape} != {b.shape}")

    def _debug_spinor_norm(self, psi1: np.ndarray, psi2: np.ndarray) -> float:
        rho = np.abs(psi1) ** 2 + np.abs(psi2) ** 2
        return float(np.sum(rho) * self.grid.dx * self.grid.dy)

    def _debug_check_spinor_full(self, name: str, psi: np.ndarray):
        if not self.debug_checks:
            return
        psi = np.asarray(psi)
        self._debug_assert_spinor_shape_full(name, psi)
        self._debug_assert_finite_array(f"{name}[0]", psi[0])
        self._debug_assert_finite_array(f"{name}[1]", psi[1])

        n = self._debug_spinor_norm(psi[0], psi[1])
        if not np.isfinite(n):
            raise RuntimeError(f"{name} norm is non-finite")
        if n < 0.0:
            raise RuntimeError(f"{name} norm is negative ({n})")

    def _debug_check_spinor_any(self, name: str, psi: np.ndarray):
        if not self.debug_checks:
            return
        psi = np.asarray(psi)
        self._assert_spinor_has_two_components(name, psi)
        self._debug_assert_finite_array(f"{name}[0]", psi[0])
        self._debug_assert_finite_array(f"{name}[1]", psi[1])

    def _debug_check_projector_once(self):
        P11, P12, P21, P22 = self._positive_energy_projector()

        sample_points = [
            (0, 0),
            (0, self.grid.Nx // 2),
            (self.grid.Ny // 2, 0),
            (self.grid.Ny // 2, self.grid.Nx // 2),
            (self.grid.Ny - 1, self.grid.Nx - 1),
        ]

        for iy, ix in sample_points:
            P = np.array(
                [
                    [P11[iy, ix], P12[iy, ix]],
                    [P21[iy, ix], P22[iy, ix]],
                ],
                dtype=np.complex128,
            )
            err = np.linalg.norm(P @ P - P)
            herm = np.linalg.norm(P.conj().T - P)
            if err > self.debug_tol_projector:
                self._debug_warn(
                    f"positive-energy projector idempotency error at ({iy},{ix}): {err:.3e}"
                )
            if herm > self.debug_tol_projector:
                self._debug_warn(
                    f"positive-energy projector Hermiticity error at ({iy},{ix}): {herm:.3e}"
                )

    def _debug_check_k_operator_once(self):
        dt = 0.123456789
        U11, U12, U21, U22 = self._dirac_k_operator(dt)

        sample_points = [
            (0, 0),
            (0, self.grid.Nx // 2),
            (self.grid.Ny // 2, 0),
            (self.grid.Ny // 2, self.grid.Nx // 2),
            (self.grid.Ny - 1, self.grid.Nx - 1),
        ]

        for iy, ix in sample_points:
            U = np.array(
                [
                    [U11[iy, ix], U12[iy, ix]],
                    [U21[iy, ix], U22[iy, ix]],
                ],
                dtype=np.complex128,
            )
            err = np.linalg.norm(U.conj().T @ U - np.eye(2))
            if err > self.debug_tol_unitarity:
                self._debug_warn(
                    f"k-operator unitarity error at ({iy},{ix}): {err:.3e}"
                )

    # -----------------------------------------------------
    # packet diagnostics
    # -----------------------------------------------------

    def expected_group_velocity(self, k0x: float, k0y: float = 0.0) -> tuple[float, float, float]:
        """
        Free-space Dirac group velocity estimate for momentum p = hbar * k.

        Returns:
            vx_est, vy_est, speed_est
        """
        px = self.hbar * float(k0x)
        py = self.hbar * float(k0y)
        p2 = px * px + py * py
        mc2 = self.m_mass * self.c_light**2
        E = float(np.sqrt((self.c_light**2) * p2 + mc2**2))

        vx = float((self.c_light**2) * px / (E + 1e-30))
        vy = float((self.c_light**2) * py / (E + 1e-30))
        sp = float(np.hypot(vx, vy))
        return vx, vy, sp

    def packet_summary(
        self,
        state_like: np.ndarray,
        X_like: np.ndarray | None = None,
        Y_like: np.ndarray | None = None,
    ) -> dict:
        """
        Summarize a full-grid or cropped Dirac spinor packet.

        Accepts:
            state_like shape (2, H, W)
            X_like, Y_like shape (H, W); defaults to full grid if omitted
        """
        psi = np.asarray(state_like, dtype=np.complex128)
        self._assert_spinor_has_two_components("state_like", psi)

        _, H, W = psi.shape

        if X_like is None or Y_like is None:
            self._assert(
                (H, W) == (self.grid.Ny, self.grid.Nx),
                "X_like/Y_like omitted but state_like is not full-grid sized",
            )
            X = self.grid.X
            Y = self.grid.Y
        else:
            X = np.asarray(X_like, dtype=float)
            Y = np.asarray(Y_like, dtype=float)
            self._assert(X.shape == (H, W), f"X_like shape {X.shape} != {(H, W)}")
            self._assert(Y.shape == (H, W), f"Y_like shape {Y.shape} != {(H, W)}")

        psi1, psi2 = psi
        rho = self._spinor_density(psi1, psi2).astype(float)

        mass = float(np.sum(rho) * self.grid.dx * self.grid.dy)
        if mass > 0.0:
            x_mean = float(np.sum(rho * X) * self.grid.dx * self.grid.dy / mass)
            y_mean = float(np.sum(rho * Y) * self.grid.dx * self.grid.dy / mass)
        else:
            x_mean = np.nan
            y_mean = np.nan

        iy, ix = np.unravel_index(int(np.argmax(rho)), rho.shape)
        x_peak = float(X[iy, ix])
        y_peak = float(Y[iy, ix])
        rho_max = float(np.max(rho))

        jx, jy, _ = self.current(psi)
        if mass > 0.0:
            jx_mean = float(np.sum(jx) * self.grid.dx * self.grid.dy / mass)
            jy_mean = float(np.sum(jy) * self.grid.dx * self.grid.dy / mass)
        else:
            jx_mean = np.nan
            jy_mean = np.nan

        return {
            "mass": mass,
            "x_mean": x_mean,
            "y_mean": y_mean,
            "x_peak": x_peak,
            "y_peak": y_peak,
            "rho_max": rho_max,
            "jx_mean": jx_mean,
            "jy_mean": jy_mean,
            "peak_index": (iy, ix),
        }

    def debug_packet_summary(
        self,
        label: str,
        state_like: np.ndarray,
        X_like: np.ndarray | None = None,
        Y_like: np.ndarray | None = None,
    ):
        if not self.debug_checks or not self.debug_packet_stats:
            return
        s = self.packet_summary(state_like, X_like=X_like, Y_like=Y_like)
        self._debug_info(
            f"{label}: "
            f"mass={s['mass']:.6e}, "
            f"x_mean={s['x_mean']:.4f}, y_mean={s['y_mean']:.4f}, "
            f"x_peak={s['x_peak']:.4f}, y_peak={s['y_peak']:.4f}, "
            f"rho_max={s['rho_max']:.6e}, "
            f"jx_mean={s['jx_mean']:.6e}, jy_mean={s['jy_mean']:.6e}"
        )

    # -----------------------------------------------------
    # small helpers
    # -----------------------------------------------------

    def _validate_positive_sigma(self, *vals):
        for v in vals:
            if float(v) <= 0.0:
                raise ValueError("All sigma values must be positive")

    def _validate_finite_dt(self, dt: float):
        dt = float(dt)
        if not np.isfinite(dt):
            raise ValueError("dt must be finite")
        return dt

    def _spinor_density(self, psi1: np.ndarray, psi2: np.ndarray) -> np.ndarray:
        self._assert_same_spatial_shape("_spinor_density", psi1, psi2)
        return np.abs(psi1) ** 2 + np.abs(psi2) ** 2

    def _normalize_spinor(self, psi1: np.ndarray, psi2: np.ndarray):
        rho = self._spinor_density(psi1, psi2)
        n2 = float(np.sum(rho) * self.grid.dx * self.grid.dy)
        if not np.isfinite(n2) or n2 <= 0.0:
            if self.debug_checks:
                self._debug_warn(f"_normalize_spinor got non-positive or non-finite norm^2: {n2}")
            return psi1.copy(), psi2.copy()

        n = np.sqrt(n2)
        psi1_n = psi1 / n
        psi2_n = psi2 / n

        if self.debug_checks:
            n_after = self._debug_spinor_norm(psi1_n, psi2_n)
            if abs(n_after - 1.0) > 1e-10:
                self._debug_warn(f"normalized spinor norm is {n_after:.12f}, expected ~1")

        return psi1_n, psi2_n

    # -----------------------------------------------------
    # free Dirac Hamiltonian in k-space
    # -----------------------------------------------------

    def _dirac_hamiltonian_k(self):
        """
        H = c (sigma_x px + sigma_y py) + m c^2 sigma_z
        with px = hbar * KX, py = hbar * KY.
        """
        px = self.hbar * self.KX
        py = self.hbar * self.KY

        mc2 = self.m_mass * self.c_light**2

        H11 = mc2
        H22 = -mc2
        H12 = self.c_light * (px - 1j * py)
        H21 = self.c_light * (px + 1j * py)

        E2 = mc2**2 + (self.c_light**2) * (px**2 + py**2)
        E = np.sqrt(np.maximum(E2, 0.0))

        if self.debug_checks:
            self._debug_assert_finite_array("px", px)
            self._debug_assert_finite_array("py", py)
            self._debug_assert_finite_array("E", E)

        return H11, H12, H21, H22, E, px, py

    def _positive_energy_eigenspinor(self):
        """
        Pointwise normalized positive-energy eigenspinor u_+(k):
            u_+ ~ [E + mc^2, c(px + i py)]^T
        """
        _, _, _, _, E, px, py = self._dirac_hamiltonian_k()
        mc2 = self.m_mass * self.c_light**2

        u1 = (E + mc2).astype(np.complex128)
        u2 = (self.c_light * (px + 1j * py)).astype(np.complex128)

        denom = np.sqrt(np.abs(u1) ** 2 + np.abs(u2) ** 2)
        denom = np.where(denom > 0.0, denom, 1.0)

        u1 /= denom
        u2 /= denom

        if self.debug_checks:
            chk = np.abs(u1) ** 2 + np.abs(u2) ** 2
            max_err = float(np.max(np.abs(chk - 1.0)))
            if max_err > 1e-10:
                self._debug_warn(f"positive-energy eigenspinor normalization max error: {max_err:.3e}")

        return u1, u2

    # -----------------------------------------------------
    # initialization
    # -----------------------------------------------------

    def initialize_state(self, state0):
        """
        Robust default Dirac initialization:
        project a scalar seed packet onto the positive-energy branch.
        """
        psi = self.initialize_projected_positive_energy_from_scalar(state0)

        if self.debug_checks:
            self._debug_check_spinor_full("initialize_state", psi)
            self.debug_packet_summary("initialize_state summary", psi)

        return psi

    def initialize_click_state(self, x_click, y_click, sigma_click):
        """
        Localized positive-energy Dirac click-state.
        """
        self._validate_positive_sigma(sigma_click)

        sigma = float(sigma_click)

        Xc = self.grid.X - x_click
        Yc = self.grid.Y - y_click

        phi = np.exp(-(Xc**2 + Yc**2) / (2.0 * sigma**2)).astype(np.complex128)
        phi, _ = normalize_unit(phi, self.grid.dx, self.grid.dy)

        if self.debug_checks:
            self._debug_assert_finite_array("click phi", phi)

        phi_k = np.fft.fft2(phi)

        px = self.hbar * self.KX
        py = self.hbar * self.KY
        p2 = px**2 + py**2

        p_cut = self.hbar * (2.5 / sigma)
        filt = np.exp(-0.5 * p2 / (p_cut**2))

        phi_k = phi_k * filt

        if self.debug_checks:
            self._debug_assert_finite_array("click phi_k", phi_k)
            self._debug_assert_finite_array("click filt", filt)

        u1, u2 = self._positive_energy_eigenspinor()

        psi1_k = phi_k * u1
        psi2_k = phi_k * u2

        psi1 = np.fft.ifft2(psi1_k)
        psi2 = np.fft.ifft2(psi2_k)

        psi1, psi2 = self._normalize_spinor(psi1, psi2)
        psi = np.stack([psi1, psi2], axis=0)

        if self.debug_checks:
            self._debug_check_spinor_full("initialize_click_state", psi)

        return psi

    def initialize_positive_energy_packet(
        self,
        x0: float,
        y0: float,
        sigma_x_k: float,
        sigma_y_k: float,
        k0x: float,
        k0y: float,
    ):
        """
        Build a mostly positive-energy Dirac wave packet in momentum space.
        """
        self._validate_positive_sigma(sigma_x_k, sigma_y_k)

        _, _, _, _, E, px, py = self._dirac_hamiltonian_k()
        mc2 = self.m_mass * self.c_light**2

        A = np.exp(
            -(
                ((self.KX - k0x) ** 2) / (2.0 * float(sigma_x_k) ** 2)
                + ((self.KY - k0y) ** 2) / (2.0 * float(sigma_y_k) ** 2)
            )
            - 1j * (self.KX * x0 + self.KY * y0)
        ).astype(np.complex128)

        u1 = (E + mc2).astype(np.complex128)
        u2 = (self.c_light * (px + 1j * py)).astype(np.complex128)

        spinor_norm = np.sqrt(np.abs(u1) ** 2 + np.abs(u2) ** 2)
        spinor_norm = np.where(spinor_norm > 0.0, spinor_norm, 1.0)

        u1 /= spinor_norm
        u2 /= spinor_norm

        psi1_k = A * u1
        psi2_k = A * u2

        psi1 = np.fft.ifft2(psi1_k)
        psi2 = np.fft.ifft2(psi2_k)

        psi1, psi2 = self._normalize_spinor(psi1, psi2)
        psi = np.stack([psi1, psi2], axis=0)

        if self.debug_checks:
            self._debug_check_spinor_full("initialize_positive_energy_packet", psi)

        return psi

    def _positive_energy_projector(self):
        """
        P_plus = (1/2) * (I + H/E)
        """
        H11, H12, H21, H22, E, _, _ = self._dirac_hamiltonian_k()

        E_safe = np.sqrt(np.maximum(E**2, 1e-30))

        P11 = 0.5 * (1.0 + H11 / E_safe)
        P22 = 0.5 * (1.0 + H22 / E_safe)
        P12 = 0.5 * (H12 / E_safe)
        P21 = 0.5 * (H21 / E_safe)

        return (
            np.asarray(P11, dtype=np.complex128),
            np.asarray(P12, dtype=np.complex128),
            np.asarray(P21, dtype=np.complex128),
            np.asarray(P22, dtype=np.complex128),
        )

    def initialize_projected_positive_energy_packet(
        self,
        x0: float,
        y0: float,
        sigma_x: float,
        sigma_y: float,
        k0x: float,
        k0y: float,
        spinor_up_weight: complex = 1.0 + 0.0j,
        spinor_down_weight: complex = 0.0 + 0.0j,
    ):
        """
        Build a real-space Gaussian wave packet and project it onto
        the positive-energy branch.
        """
        self._validate_positive_sigma(sigma_x, sigma_y)

        Xc = self.grid.X - x0
        Yc = self.grid.Y - y0

        env = np.exp(
            -(Xc**2) / (2.0 * float(sigma_x) ** 2)
            -(Yc**2) / (2.0 * float(sigma_y) ** 2)
        ).astype(np.complex128)

        phase = np.exp(
            1j * (k0x * self.grid.X + k0y * self.grid.Y)
        ).astype(np.complex128)

        phi = env * phase

        psi1 = complex(spinor_up_weight) * phi
        psi2 = complex(spinor_down_weight) * phi

        psi1_k = np.fft.fft2(psi1)
        psi2_k = np.fft.fft2(psi2)

        P11, P12, P21, P22 = self._positive_energy_projector()

        psi1_k_proj = P11 * psi1_k + P12 * psi2_k
        psi2_k_proj = P21 * psi1_k + P22 * psi2_k

        psi1_proj = np.fft.ifft2(psi1_k_proj)
        psi2_proj = np.fft.ifft2(psi2_k_proj)

        psi1_proj, psi2_proj = self._normalize_spinor(psi1_proj, psi2_proj)
        psi = np.stack([psi1_proj, psi2_proj], axis=0)

        if self.debug_checks:
            self._debug_check_spinor_full("initialize_projected_positive_energy_packet", psi)

        return psi

    def initialize_projected_positive_energy_from_scalar(
        self,
        state0: np.ndarray,
        spinor_up_weight: complex = 1.0 + 0.0j,
        spinor_down_weight: complex = 0.0 + 0.0j,
    ):
        """
        Take a scalar complex seed field in real space and project the
        corresponding 2-spinor onto the positive-energy branch.
        """
        phi = np.asarray(state0, dtype=np.complex128)

        if phi.shape != (self.grid.Ny, self.grid.Nx):
            raise ValueError("state0 must have shape (Ny, Nx)")

        if self.debug_checks:
            self._debug_assert_finite_array("state0", phi)

        psi1 = complex(spinor_up_weight) * phi
        psi2 = complex(spinor_down_weight) * phi

        psi1_k = np.fft.fft2(psi1)
        psi2_k = np.fft.fft2(psi2)

        P11, P12, P21, P22 = self._positive_energy_projector()

        psi1_k_proj = P11 * psi1_k + P12 * psi2_k
        psi2_k_proj = P21 * psi1_k + P22 * psi2_k

        psi1_proj = np.fft.ifft2(psi1_k_proj)
        psi2_proj = np.fft.ifft2(psi2_k_proj)

        psi1_proj, psi2_proj = self._normalize_spinor(psi1_proj, psi2_proj)
        psi = np.stack([psi1_proj, psi2_proj], axis=0)

        if self.debug_checks:
            self._debug_check_spinor_full("initialize_projected_positive_energy_from_scalar", psi)

        return psi

    # -----------------------------------------------------
    # momentum-space propagator
    # -----------------------------------------------------

    def _dirac_k_operator(self, dt: float):
        """
        Exact 2x2 momentum-space propagator:
            U = exp(-i H dt / hbar)
        """
        dt = self._validate_finite_dt(dt)

        H11, H12, H21, H22, E, _, _ = self._dirac_hamiltonian_k()

        theta = E * dt / self.hbar
        cos_t = np.cos(theta)

        E_safe = np.where(E > 1e-15, E, 1.0)
        sin_over_E = np.where(E > 1e-15, np.sin(theta) / E_safe, dt / self.hbar)

        U11 = cos_t - 1j * H11 * sin_over_E
        U22 = cos_t - 1j * H22 * sin_over_E
        U12 = -1j * H12 * sin_over_E
        U21 = -1j * H21 * sin_over_E

        return (
            np.asarray(U11, dtype=np.complex128),
            np.asarray(U12, dtype=np.complex128),
            np.asarray(U21, dtype=np.complex128),
            np.asarray(U22, dtype=np.complex128),
        )

    def _apply_dirac_k(self, psi, dt: float):
        psi = np.asarray(psi, dtype=np.complex128)
        if psi.shape != (2, self.grid.Ny, self.grid.Nx):
            raise ValueError("psi must have shape (2, Ny, Nx)")

        if self.debug_checks:
            self._debug_check_spinor_full("_apply_dirac_k input", psi)
            n_before = self._debug_spinor_norm(psi[0], psi[1])

        psi1_k = np.fft.fft2(psi[0])
        psi2_k = np.fft.fft2(psi[1])

        U11, U12, U21, U22 = self._dirac_k_operator(dt)

        psi1_k_new = U11 * psi1_k + U12 * psi2_k
        psi2_k_new = U21 * psi1_k + U22 * psi2_k

        psi1_new = np.fft.ifft2(psi1_k_new)
        psi2_new = np.fft.ifft2(psi2_k_new)

        psi_new = np.stack([psi1_new, psi2_new], axis=0)

        if self.debug_checks:
            self._debug_check_spinor_full("_apply_dirac_k output", psi_new)
            n_after = self._debug_spinor_norm(psi1_new, psi2_new)
            err = abs(n_after - n_before)
            if err > self.debug_tol_unitarity:
                self._debug_warn(
                    f"free k-step norm changed by {err:.3e} "
                    f"(before={n_before:.12f}, after={n_after:.12f})"
                )

        return psi_new

    # -----------------------------------------------------
    # stepping helpers
    # -----------------------------------------------------

    def _step_with_potential(self, state, dt: float, V_full):
        """
        Strang splitting:
            exp(-i V dt/2hbar) exp(-i T dt/hbar) exp(-i V dt/2hbar)
        """
        dt = self._validate_finite_dt(dt)

        state = np.asarray(state, dtype=np.complex128)
        if state.shape != (2, self.grid.Ny, self.grid.Nx):
            raise ValueError("state must have shape (2, Ny, Nx)")

        V_full = np.asarray(V_full, dtype=np.complex128)
        if V_full.shape != (self.grid.Ny, self.grid.Nx):
            raise ValueError("V_full must have shape (Ny, Nx)")

        if self.debug_checks:
            self._debug_check_spinor_full("_step_with_potential input", state)
            self._debug_assert_finite_array("V_full", V_full)
            n_before = self._debug_spinor_norm(state[0], state[1])

        psi1, psi2 = state

        P_half = np.exp(-1j * V_full * dt / (2.0 * self.hbar))

        if self.debug_checks:
            self._debug_assert_finite_array("P_half", P_half)

        psi1 = psi1 * P_half
        psi2 = psi2 * P_half

        psi = np.stack([psi1, psi2], axis=0)
        psi = self._apply_dirac_k(psi, dt)

        psi1, psi2 = psi
        psi1 = psi1 * P_half
        psi2 = psi2 * P_half

        psi_out = np.stack([psi1, psi2], axis=0)

        if self.debug_checks:
            self._debug_check_spinor_full("_step_with_potential output", psi_out)
            n_after = self._debug_spinor_norm(psi_out[0], psi_out[1])

            W_est = -np.imag(V_full)
            if dt > 0.0 and np.min(W_est) >= -1e-14:
                if n_after - n_before > self.debug_tol_norm_growth:
                    self._debug_warn(
                        f"forward step norm grew unexpectedly under nonnegative absorption: "
                        f"before={n_before:.12f}, after={n_after:.12f}"
                    )

        return psi_out

    # -----------------------------------------------------
    # stepping
    # -----------------------------------------------------

    def step_forward(self, state, dt: float):
        psi = self._step_with_potential(state, dt, self.V_fwd)
        return TheoryStepResult(state=psi, aux=None)

    def step_backward_adjoint(self, state, dt: float):
        """
        Adjoint-like backward evolution:
          - reversed dt in kinetic part
          - conjugated complex potential
        """
        psi = self._step_with_potential(state, -dt, self.V_adj)
        return TheoryStepResult(state=psi, aux=None)

    # -----------------------------------------------------
    # observables
    # -----------------------------------------------------

    def density(self, state):
        state = np.asarray(state, dtype=np.complex128)
        if state.shape != (2, self.grid.Ny, self.grid.Nx):
            raise ValueError("state must have shape (2, Ny, Nx)")

        psi1, psi2 = state
        rho = self._spinor_density(psi1, psi2).astype(float)

        if self.debug_checks:
            self._debug_assert_finite_array("density", rho)
            if np.min(rho) < -1e-14:
                self._debug_warn(f"density has negative minimum {np.min(rho):.3e}")

        return rho

    def current(self, state_vis):
        """
        2D Dirac probability current for

            H = c (sigma_x px + sigma_y py) + m c^2 sigma_z

        rho = psi^dagger psi

        jx = c * psi^dagger sigma_x psi = 2 c Re(conj(psi1) * psi2)
        jy = c * psi^dagger sigma_y psi = 2 c Im(conj(psi1) * psi2)

        Accepts both:
            - full-grid state: shape (2, Ny, Nx)
            - visible crop:    shape (2, H, W)
        """
        state_vis = np.asarray(state_vis, dtype=np.complex128)
        self._assert_spinor_has_two_components("state_vis", state_vis)

        psi1, psi2 = state_vis
        self._assert_same_spatial_shape("current spatial", psi1, psi2)

        rho = self._spinor_density(psi1, psi2).astype(float)
        overlap = np.conjugate(psi1) * psi2

        jx = (2.0 * self.c_light * np.real(overlap)).astype(float)
        jy = (2.0 * self.c_light * np.imag(overlap)).astype(float)

        if self.debug_checks:
            self._debug_assert_finite_array("jx", jx)
            self._debug_assert_finite_array("jy", jy)
            self._debug_assert_finite_array("rho(current)", rho)

            sp_local = np.zeros_like(rho, dtype=float)
            mask = rho > 1e-12
            sp_local[mask] = np.hypot(jx[mask] / rho[mask], jy[mask] / rho[mask])
            if np.any(mask):
                vmax = float(np.max(sp_local[mask]))
                if vmax > self.c_light + self.debug_tol_velocity:
                    self._debug_warn(
                        f"local |j/rho| exceeded c in current(): vmax={vmax:.12f}, c={self.c_light:.12f}"
                    )

        return jx, jy, rho

    def velocity(self, state_vis, eps_rho: float = 1e-10):
        """
        Relativistically constrained velocity from Dirac current:
            v = j / rho
        with numerical enforcement:
            |v| <= c_light
        """
        if eps_rho <= 0.0:
            raise ValueError("eps_rho must be positive")

        jx, jy, rho = self.current(state_vis)

        denom = np.maximum(rho, float(eps_rho))

        vx = jx / denom
        vy = jy / denom

        sp = np.hypot(vx, vy)

        if self.debug_checks:
            self._debug_assert_finite_array("vx pre-clamp", vx)
            self._debug_assert_finite_array("vy pre-clamp", vy)
            self._debug_assert_finite_array("speed pre-clamp", sp)
            sp_max = float(np.max(sp))
            if sp_max > self.c_light + self.debug_tol_velocity:
                self._debug_warn(
                    f"velocity exceeded c before clamp: max |v| = {sp_max:.12f}, c = {self.c_light:.12f}"
                )

        mask = sp > self.c_light
        if np.any(mask):
            scale = self.c_light / np.maximum(sp[mask], float(eps_rho))

            vx = vx.copy()
            vy = vy.copy()
            sp = sp.copy()

            vx[mask] *= scale
            vy[mask] *= scale
            sp[mask] = self.c_light

        if self.debug_checks:
            self._debug_assert_finite_array("vx", vx)
            self._debug_assert_finite_array("vy", vy)
            self._debug_assert_finite_array("speed", sp)
            sp_max_after = float(np.max(sp))
            if sp_max_after > self.c_light + self.debug_tol_velocity:
                self._debug_warn(
                    f"velocity still exceeded c after clamp: max |v| = {sp_max_after:.12f}"
                )

        return vx.astype(float), vy.astype(float), sp.astype(float)