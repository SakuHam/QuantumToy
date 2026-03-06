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
        self.kx = 2.0 * np.pi * np.fft.fftfreq(self.grid.Nx, d=self.grid.dx)
        self.ky = 2.0 * np.pi * np.fft.fftfreq(self.grid.Ny, d=self.grid.dy)

        self.KX, self.KY = np.meshgrid(self.kx, self.ky)

        self.V_fwd = self.potential.V_real - 1j * self.potential.W
        self.V_adj = np.conjugate(self.V_fwd)

    # -----------------------------------------------------
    # initialization
    # -----------------------------------------------------

    def initialize_state(self, state0):
        psi1 = state0.astype(np.complex128)
        psi2 = np.zeros_like(psi1)

        psi1, _ = normalize_unit(psi1, self.grid.dx, self.grid.dy)

        return np.stack([psi1, psi2], axis=0)

    def initialize_click_state(self, x_click, y_click, sigma_click):
        Xc = self.grid.X - x_click
        Yc = self.grid.Y - y_click

        phi = np.exp(-(Xc**2 + Yc**2) / (2.0 * sigma_click**2)).astype(np.complex128)
        phi, _ = normalize_unit(phi, self.grid.dx, self.grid.dy)

        return np.stack([phi, np.zeros_like(phi)], axis=0)

    # -----------------------------------------------------
    # momentum-space propagator
    # -----------------------------------------------------

    def _dirac_k_operator(self, dt):
        px = self.hbar * self.KX
        py = self.hbar * self.KY

        mc2 = self.m_mass * self.c_light**2

        H11 = mc2
        H22 = -mc2
        H12 = self.c_light * (px - 1j * py)
        H21 = self.c_light * (px + 1j * py)

        E2 = (mc2**2 + np.abs(H12)**2).astype(float)
        E = np.sqrt(np.maximum(E2, 1e-30))

        theta = E * dt / self.hbar
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        U11 = cos_t - 1j * (H11 / E) * sin_t
        U22 = cos_t - 1j * (H22 / E) * sin_t
        U12 = -1j * (H12 / E) * sin_t
        U21 = -1j * (H21 / E) * sin_t

        # k = 0 special case safety
        mask_small = E < 1e-15
        if np.any(mask_small):
            U11 = U11.copy()
            U22 = U22.copy()
            U12 = U12.copy()
            U21 = U21.copy()

            U11[mask_small] = 1.0 + 0.0j
            U22[mask_small] = 1.0 + 0.0j
            U12[mask_small] = 0.0 + 0.0j
            U21[mask_small] = 0.0 + 0.0j

        return U11, U12, U21, U22

    def _apply_dirac_k(self, psi, dt):
        psi1_k = np.fft.fft2(psi[0])
        psi2_k = np.fft.fft2(psi[1])

        U11, U12, U21, U22 = self._dirac_k_operator(dt)

        psi1_k_new = U11 * psi1_k + U12 * psi2_k
        psi2_k_new = U21 * psi1_k + U22 * psi2_k

        psi1_new = np.fft.ifft2(psi1_k_new)
        psi2_new = np.fft.ifft2(psi2_k_new)

        return np.stack([psi1_new, psi2_new], axis=0)

    # -----------------------------------------------------
    # stepping
    # -----------------------------------------------------

    def step_forward(self, state, dt):
        psi1, psi2 = state

        P_half = np.exp(-1j * self.V_fwd * dt / (2.0 * self.hbar))

        psi1 = psi1 * P_half
        psi2 = psi2 * P_half

        psi = np.stack([psi1, psi2], axis=0)
        psi = self._apply_dirac_k(psi, dt)

        psi1, psi2 = psi
        psi1 = psi1 * P_half
        psi2 = psi2 * P_half

        return TheoryStepResult(
            state=np.stack([psi1, psi2], axis=0),
            aux=None,
        )

    def step_backward_adjoint(self, state, dt):
        return self.step_forward(state, -dt)

    # -----------------------------------------------------
    # observables
    # -----------------------------------------------------

    def density(self, state):
        psi1, psi2 = state
        return (np.abs(psi1) ** 2 + np.abs(psi2) ** 2).astype(float)

    def current(self, state_vis):
        """
        Relativistically correct 2D Dirac probability current:

            rho = psi^\dagger psi
            jx  = c * psi^\dagger sigma_x psi = 2 c Re(conj(psi1) * psi2)
            jy  = c * psi^\dagger sigma_y psi = 2 c Im(conj(psi1) * psi2)
        """
        psi1, psi2 = state_vis

        rho = (np.abs(psi1) ** 2 + np.abs(psi2) ** 2).astype(float)

        overlap = np.conjugate(psi1) * psi2

        jx = (2.0 * self.c_light * np.real(overlap)).astype(float)
        jy = (2.0 * self.c_light * np.imag(overlap)).astype(float)

        return jx, jy, rho
    
    def velocity(self, state_vis, eps_rho: float = 1e-10):
        """
        Relativistically constrained velocity from Dirac current.

        Uses:
            v = j / rho

        and enforces numerically:
            |v| <= c_light
        """
        jx, jy, rho = self.current(state_vis)

        denom = np.maximum(rho, eps_rho)

        vx = jx / denom
        vy = jy / denom

        sp = np.hypot(vx, vy)

        # Numerical safety: clamp speeds slightly exceeding c due to tiny rho / discretization
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