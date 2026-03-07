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

    def __post_init__(self):
        # Real-space grid comes from self.grid:
        #   X, Y shaped as (Ny, Nx)
        # so KX, KY must have same shape/order.
        self.kx = 2.0 * np.pi * np.fft.fftfreq(self.grid.Nx, d=self.grid.dx)
        self.ky = 2.0 * np.pi * np.fft.fftfreq(self.grid.Ny, d=self.grid.dy)

        self.KX, self.KY = np.meshgrid(self.kx, self.ky)

        self.V_fwd = self.potential.V_real - 1j * self.potential.W
        self.V_adj = np.conjugate(self.V_fwd)

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
        Backward-compatible simple click state [phi, 0].
        """
        Xc = self.grid.X - x_click
        Yc = self.grid.Y - y_click

        phi = np.exp(-(Xc**2 + Yc**2) / (2.0 * sigma_click**2)).astype(np.complex128)
        phi, _ = normalize_unit(phi, self.grid.dx, self.grid.dy)

        return np.stack([phi, np.zeros_like(phi)], axis=0)

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

        Parameters
        ----------
        x0, y0:
            Initial center in real space.
        sigma_x_k, sigma_y_k:
            Widths in momentum space (k-space Gaussians).
        k0x, k0y:
            Mean wavevector.

        Notes
        -----
        This is much better than initializing [phi, 0] in real space when
        you want a clean moving Dirac packet.
        """
        px = self.hbar * self.KX
        py = self.hbar * self.KY
        p2 = px**2 + py**2

        mc2 = self.m_mass * self.c_light**2
        E = np.sqrt(mc2**2 + (self.c_light**2) * p2)

        # k-space Gaussian envelope centered at (k0x, k0y)
        A = np.exp(
            -(
                ((self.KX - k0x) ** 2) / (2.0 * sigma_x_k**2)
                + ((self.KY - k0y) ** 2) / (2.0 * sigma_y_k**2)
            )
            - 1j * ((self.KX - k0x) * x0 + (self.KY - k0y) * y0)
        ).astype(np.complex128)

        # Positive-energy eigen-spinor for:
        # H = c (sigma_x px + sigma_y py) + mc^2 sigma_z
        #
        # One convenient choice:
        #   u_+(p) ~ [E + mc^2, c(px + i py)]^T
        u1 = (E + mc2).astype(np.complex128)
        u2 = (self.c_light * (px + 1j * py)).astype(np.complex128)

        spinor_norm = np.sqrt(np.abs(u1) ** 2 + np.abs(u2) ** 2 + 1e-30)
        u1 /= spinor_norm
        u2 /= spinor_norm

        psi1_k = A * u1
        psi2_k = A * u2

        psi1 = np.fft.ifft2(psi1_k)
        psi2 = np.fft.ifft2(psi2_k)

        rho = (np.abs(psi1) ** 2 + np.abs(psi2) ** 2).astype(float)
        n = float(np.sqrt(np.sum(rho) * self.grid.dx * self.grid.dy))
        if n > 0:
            psi1 /= n
            psi2 /= n

        return np.stack([psi1, psi2], axis=0)

    def _positive_energy_projector(self):
        """
        Momentum-space positive-energy projector for

            H = c (sigma_x px + sigma_y py) + m c^2 sigma_z

        P_plus = (1/2) * (I + H/E)

        Returns:
            P11, P12, P21, P22
        """
        px = self.hbar * self.KX
        py = self.hbar * self.KY

        mc2 = self.m_mass * self.c_light**2

        H11 = mc2
        H22 = -mc2
        H12 = self.c_light * (px - 1j * py)
        H21 = self.c_light * (px + 1j * py)

        E2 = (mc2**2 + (self.c_light**2) * (px**2 + py**2)).astype(float)
        E = np.sqrt(np.maximum(E2, 1e-30))

        P11 = 0.5 * (1.0 + H11 / E)
        P22 = 0.5 * (1.0 + H22 / E)
        P12 = 0.5 * (H12 / E)
        P21 = 0.5 * (H21 / E)

        return (
            P11.astype(np.complex128),
            P12.astype(np.complex128),
            P21.astype(np.complex128),
            P22.astype(np.complex128),
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

        Parameters
        ----------
        x0, y0:
            Initial packet center in real space.
        sigma_x, sigma_y:
            Real-space Gaussian widths.
        k0x, k0y:
            Mean wavevector of the packet.
        spinor_up_weight, spinor_down_weight:
            Initial raw 2-spinor weights before projection.
            Usually keep defaults first.

        Returns
        -------
        state : np.ndarray shape (2, Ny, Nx)
            Positive-energy projected Dirac spinor.
        """
        Xc = self.grid.X - x0
        Yc = self.grid.Y - y0

        # Real-space Gaussian envelope with plane-wave phase
        env = np.exp(
            -(Xc**2) / (2.0 * sigma_x**2)
            -(Yc**2) / (2.0 * sigma_y**2)
        ).astype(np.complex128)

        phase = np.exp(1j * (k0x * self.grid.X + k0y * self.grid.Y)).astype(np.complex128)

        phi = env * phase

        # Raw seed spinor in real space
        psi1 = spinor_up_weight * phi
        psi2 = spinor_down_weight * phi

        # FFT to momentum space
        psi1_k = np.fft.fft2(psi1)
        psi2_k = np.fft.fft2(psi2)

        # Project onto positive-energy branch
        P11, P12, P21, P22 = self._positive_energy_projector()

        psi1_k_proj = P11 * psi1_k + P12 * psi2_k
        psi2_k_proj = P21 * psi1_k + P22 * psi2_k

        # Back to real space
        psi1_proj = np.fft.ifft2(psi1_k_proj)
        psi2_proj = np.fft.ifft2(psi2_k_proj)

        # Normalize full spinor
        rho = (np.abs(psi1_proj) ** 2 + np.abs(psi2_proj) ** 2).astype(float)
        n = float(np.sqrt(np.sum(rho) * self.grid.dx * self.grid.dy))

        if n > 0:
            psi1_proj /= n
            psi2_proj /= n

        return np.stack([psi1_proj, psi2_proj], axis=0)

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
        phi = state0.astype(np.complex128)

        psi1 = spinor_up_weight * phi
        psi2 = spinor_down_weight * phi

        psi1_k = np.fft.fft2(psi1)
        psi2_k = np.fft.fft2(psi2)

        P11, P12, P21, P22 = self._positive_energy_projector()

        psi1_k_proj = P11 * psi1_k + P12 * psi2_k
        psi2_k_proj = P21 * psi1_k + P22 * psi2_k

        psi1_proj = np.fft.ifft2(psi1_k_proj)
        psi2_proj = np.fft.ifft2(psi2_k_proj)

        rho = (np.abs(psi1_proj) ** 2 + np.abs(psi2_proj) ** 2).astype(float)
        n = float(np.sqrt(np.sum(rho) * self.grid.dx * self.grid.dy))

        if n > 0:
            psi1_proj /= n
            psi2_proj /= n

        return np.stack([psi1_proj, psi2_proj], axis=0) 
       
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
        px = self.hbar * self.KX
        py = self.hbar * self.KY

        mc2 = self.m_mass * self.c_light**2

        H11 = mc2
        H22 = -mc2
        H12 = self.c_light * (px - 1j * py)
        H21 = self.c_light * (px + 1j * py)

        E2 = (mc2**2 + (self.c_light**2) * (px**2 + py**2)).astype(float)
        E = np.sqrt(np.maximum(E2, 0.0))

        theta = E * dt / self.hbar
        cos_t = np.cos(theta)

        # Stable handling of sin(theta)/E.
        # For nonzero mass this is normally safe anyway, but this form is
        # robust and avoids the old incorrect k=0 -> identity hack.
        E_safe = np.where(E > 1e-15, E, 1.0)
        sin_over_E = np.where(E > 1e-15, np.sin(theta) / E_safe, dt / self.hbar)

        U11 = cos_t - 1j * H11 * sin_over_E
        U22 = cos_t - 1j * H22 * sin_over_E
        U12 = -1j * H12 * sin_over_E
        U21 = -1j * H21 * sin_over_E

        return U11, U12, U21, U22

    def _apply_dirac_k(self, psi, dt: float):
        psi1_k = np.fft.fft2(psi[0])
        psi2_k = np.fft.fft2(psi[1])

        U11, U12, U21, U22 = self._dirac_k_operator(dt)

        psi1_k_new = U11 * psi1_k + U12 * psi2_k
        psi2_k_new = U21 * psi1_k + U22 * psi2_k

        psi1_new = np.fft.ifft2(psi1_k_new)
        psi2_new = np.fft.ifft2(psi2_k_new)

        return np.stack([psi1_new, psi2_new], axis=0)

    # -----------------------------------------------------
    # stepping helpers
    # -----------------------------------------------------

    def _step_with_potential(self, state, dt: float, V_full):
        """
        Strang splitting:
            exp(-i V dt/2hbar) exp(-i T dt/hbar) exp(-i V dt/2hbar)
        """
        psi1, psi2 = state

        P_half = np.exp(-1j * V_full * dt / (2.0 * self.hbar))

        psi1 = psi1 * P_half
        psi2 = psi2 * P_half

        psi = np.stack([psi1, psi2], axis=0)
        psi = self._apply_dirac_k(psi, dt)

        psi1, psi2 = psi
        psi1 = psi1 * P_half
        psi2 = psi2 * P_half

        return np.stack([psi1, psi2], axis=0)

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
        Adjoint-like backward evolution:
            - reverse dt in the kinetic propagation
            - use conjugated complex potential
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
        psi1, psi2 = state
        return (np.abs(psi1) ** 2 + np.abs(psi2) ** 2).astype(float)

    def current(self, state_vis):
        """
        2D Dirac probability current for:
            H = c (sigma_x px + sigma_y py) + m c^2 sigma_z

        rho = psi^dagger psi

        jx = c * psi^dagger sigma_x psi =  2 c Re(conj(psi1) * psi2)
        jy = c * psi^dagger sigma_y psi = -2 c Im(conj(psi1) * psi2)

        IMPORTANT:
        The minus sign in jy is essential for the standard sigma_y convention.
        """
        psi1, psi2 = state_vis

        rho = (np.abs(psi1) ** 2 + np.abs(psi2) ** 2).astype(float)
        overlap = np.conjugate(psi1) * psi2

        jx = (2.0 * self.c_light * np.real(overlap)).astype(float)
        jy = (-2.0 * self.c_light * np.imag(overlap)).astype(float)

        return jx, jy, rho

    def velocity(self, state_vis, eps_rho: float = 1e-10):
        """
        Relativistically constrained velocity from Dirac current:
            v = j / rho
        with numerical enforcement:
            |v| <= c_light
        """
        jx, jy, rho = self.current(state_vis)

        denom = np.maximum(rho, eps_rho)

        vx = jx / denom
        vy = jy / denom

        sp = np.hypot(vx, vy)

        mask = sp > self.c_light
        if np.any(mask):
            scale = self.c_light / np.maximum(sp[mask], eps_rho)

            vx = vx.copy()
            vy = vy.copy()
            sp = sp.copy()

            vx[mask] *= scale
            vy[mask] *= scale
            sp[mask] = self.c_light

        return vx.astype(float), vy.astype(float), sp.astype(float)