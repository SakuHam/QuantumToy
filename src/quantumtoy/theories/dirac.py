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

    def __post_init__(self):
        # -----------------------------
        # Basic parameter checks
        # -----------------------------
        self.m_mass = float(self.m_mass)
        self.hbar = float(self.hbar)
        self.c_light = float(self.c_light)

        if self.grid.dx <= 0.0 or self.grid.dy <= 0.0:
            raise ValueError("grid.dx and grid.dy must be positive")
        if self.grid.Nx <= 0 or self.grid.Ny <= 0:
            raise ValueError("grid.Nx and grid.Ny must be positive")
        if self.hbar <= 0.0:
            raise ValueError("hbar must be positive")
        if self.c_light <= 0.0:
            raise ValueError("c_light must be positive")
        if self.m_mass < 0.0:
            raise ValueError("m_mass must be non-negative")

        # -----------------------------
        # FFT-compatible k-grid
        # X, Y are assumed shape (Ny, Nx)
        # so KX, KY must also be (Ny, Nx).
        # -----------------------------
        self.kx = 2.0 * np.pi * np.fft.fftfreq(self.grid.Nx, d=self.grid.dx)
        self.ky = 2.0 * np.pi * np.fft.fftfreq(self.grid.Ny, d=self.grid.dy)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing="xy")

        if self.KX.shape != (self.grid.Ny, self.grid.Nx):
            raise ValueError("KX shape mismatch")
        if self.KY.shape != (self.grid.Ny, self.grid.Nx):
            raise ValueError("KY shape mismatch")

        # -----------------------------
        # Complex scalar potential
        # Forward uses V_real - i W  => absorption for W > 0
        # Adjoint uses conjugate potential.
        # -----------------------------
        self.V_fwd = np.asarray(
            self.potential.V_real - 1j * self.potential.W,
            dtype=np.complex128,
        )
        self.V_adj = np.conjugate(self.V_fwd)

        if self.V_fwd.shape != (self.grid.Ny, self.grid.Nx):
            raise ValueError("Potential shape mismatch: expected (Ny, Nx)")
        if self.V_adj.shape != (self.grid.Ny, self.grid.Nx):
            raise ValueError("Adjoint potential shape mismatch: expected (Ny, Nx)")

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
    # debug helpers
    # -----------------------------------------------------

    def _debug_warn(self, msg: str):
        if self.debug_checks:
            print(f"[DiracTheory DEBUG] {msg}")

    def _debug_assert_finite_array(self, name: str, arr: np.ndarray):
        if not self.debug_checks:
            return
        arr = np.asarray(arr)
        if not np.all(np.isfinite(arr)):
            bad = np.size(arr) - np.count_nonzero(np.isfinite(arr))
            raise RuntimeError(f"{name} contains non-finite values ({bad} bad entries)")

    def _debug_assert_spinor_shape(self, name: str, psi: np.ndarray):
        if not self.debug_checks:
            return
        if psi.shape != (2, self.grid.Ny, self.grid.Nx):
            raise RuntimeError(
                f"{name} has shape {psi.shape}, expected (2, {self.grid.Ny}, {self.grid.Nx})"
            )

    def _debug_spinor_norm(self, psi1: np.ndarray, psi2: np.ndarray) -> float:
        rho = np.abs(psi1) ** 2 + np.abs(psi2) ** 2
        return float(np.sum(rho) * self.grid.dx * self.grid.dy)

    def _debug_check_spinor(self, name: str, psi: np.ndarray):
        if not self.debug_checks:
            return
        psi = np.asarray(psi)
        self._debug_assert_spinor_shape(name, psi)
        self._debug_assert_finite_array(f"{name}[0]", psi[0])
        self._debug_assert_finite_array(f"{name}[1]", psi[1])

        n = self._debug_spinor_norm(psi[0], psi[1])
        if not np.isfinite(n):
            raise RuntimeError(f"{name} norm is non-finite")
        if n < 0.0:
            raise RuntimeError(f"{name} norm is negative ({n})")

    def _debug_check_projector_once(self):
        P11, P12, P21, P22 = self._positive_energy_projector()

        # Test idempotency on a few representative points.
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
        # Check near-unitarity of free k-step at a small reference dt.
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

    def _dirac_hamiltonian_k(self):
        """
        Returns momentum-space Dirac Hamiltonian pieces for

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
        Pointwise normalized positive-energy eigenspinor u_+(k) for

            H = c (sigma_x px + sigma_y py) + m c^2 sigma_z

        A convenient choice:
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
        return self.initialize_projected_positive_energy_from_scalar(state0)

    def initialize_click_state(self, x_click, y_click, sigma_click):
        """
        Build a localized positive-energy Dirac click-state.

        The click is created from a localized scalar Gaussian envelope in x-space,
        optionally mild-low-pass filtered in k-space, then spinorized using the
        positive-energy eigenspinor of the free 2D Dirac Hamiltonian.

        This is a much more physical seed than raw [phi, 0].
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
            self._debug_check_spinor("initialize_click_state", psi)

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
                ((self.KX - k0x) ** 2) / (2.0 * float(sigma_x_k)**2)
                + ((self.KY - k0y) ** 2) / (2.0 * float(sigma_y_k)**2)
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
            self._debug_check_spinor("initialize_positive_energy_packet", psi)

        return psi

    def _positive_energy_projector(self):
        """
        Momentum-space positive-energy projector for

            H = c (sigma_x px + sigma_y py) + m c^2 sigma_z

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
        Build a real-space Gaussian wave packet and project it onto the
        positive-energy branch of the Dirac Hamiltonian.
        """
        self._validate_positive_sigma(sigma_x, sigma_y)

        Xc = self.grid.X - x0
        Yc = self.grid.Y - y0

        env = np.exp(
            -(Xc**2) / (2.0 * float(sigma_x)**2)
            -(Yc**2) / (2.0 * float(sigma_y)**2)
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
            self._debug_check_spinor("initialize_projected_positive_energy_packet", psi)

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
            self._debug_check_spinor("initialize_projected_positive_energy_from_scalar", psi)

        return psi

    # -----------------------------------------------------
    # momentum-space propagator
    # -----------------------------------------------------

    def _dirac_k_operator(self, dt: float):
        """
        Exact 2x2 momentum-space propagator for one time step dt:

            U = exp(-i H dt / hbar)

        where

            H = c (sigma_x px + sigma_y py) + m c^2 sigma_z
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
            self._debug_check_spinor("_apply_dirac_k input", psi)
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
            self._debug_check_spinor("_apply_dirac_k output", psi_new)
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

        Here V may be complex:
            V = V_real - i W

        so forward evolution is generally non-unitary when W != 0.
        """
        dt = self._validate_finite_dt(dt)

        state = np.asarray(state, dtype=np.complex128)
        if state.shape != (2, self.grid.Ny, self.grid.Nx):
            raise ValueError("state must have shape (2, Ny, Nx)")

        V_full = np.asarray(V_full, dtype=np.complex128)
        if V_full.shape != (self.grid.Ny, self.grid.Nx):
            raise ValueError("V_full must have shape (Ny, Nx)")

        if self.debug_checks:
            self._debug_check_spinor("_step_with_potential input", state)
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
            self._debug_check_spinor("_step_with_potential output", psi_out)
            n_after = self._debug_spinor_norm(psi_out[0], psi_out[1])

            # If this is a forward absorptive step, norm should not grow much
            # when W >= 0 everywhere.
            if np.all(np.real(-1j * V_full) <= 1e-14):
                # This condition is conservative and not especially informative,
                # so keep only a mild warning.
                pass

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
        return TheoryStepResult(
            state=psi,
            aux=None,
        )

    def step_backward_adjoint(self, state, dt: float):
        """
        Adjoint-like backward evolution.

        This is not a stable physical time-reversal when absorptive regions exist.
        It applies:
          - reversed dt in the kinetic part
          - conjugated complex potential

        For W > 0, this typically amplifies where forward evolution absorbed.
        """
        psi = self._step_with_potential(state, -dt, self.V_adj)
        return TheoryStepResult(
            state=psi,
            aux=None,
        )

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
        """
        state_vis = np.asarray(state_vis, dtype=np.complex128)
        expected = (2, self.grid.Ny, self.grid.Nx)
        if state_vis.shape != expected:
            raise ValueError(
                f"state_vis must have shape {expected}, got {state_vis.shape}"
            )

        psi1, psi2 = state_vis

        rho = self._spinor_density(psi1, psi2).astype(float)
        overlap = np.conjugate(psi1) * psi2

        jx = (2.0 * self.c_light * np.real(overlap)).astype(float)
        jy = (2.0 * self.c_light * np.imag(overlap)).astype(float)

        if self.debug_checks:
            self._debug_assert_finite_array("jx", jx)
            self._debug_assert_finite_array("jy", jy)
            self._debug_assert_finite_array("rho(current)", rho)

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