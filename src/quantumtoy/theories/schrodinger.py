from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from theories.base import TheoryModel, TheoryStepResult
from core.utils import normalize_unit


@dataclass
class SchrodingerTheory(TheoryModel):
    grid: any
    potential: any
    m_mass: float = 1.0
    hbar: float = 1.0

    def __post_init__(self):
        self.kx = 2.0 * np.pi * np.fft.fftfreq(self.grid.Nx, d=self.grid.dx)
        self.ky = 2.0 * np.pi * np.fft.fftfreq(self.grid.Ny, d=self.grid.dy)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky)
        self.K2 = self.KX**2 + self.KY**2

        self.V_fwd = self.potential.V_real - 1j * self.potential.W
        self.V_adj = np.conjugate(self.V_fwd)

    def initialize_state(self, state0: np.ndarray) -> np.ndarray:
        state0, _ = normalize_unit(state0, self.grid.dx, self.grid.dy)
        return state0.astype(np.complex128)

    def initialize_click_state(
        self,
        x_click: float,
        y_click: float,
        sigma_click: float,
    ) -> np.ndarray:
        Xc = self.grid.X - x_click
        Yc = self.grid.Y - y_click

        phi = np.exp(-(Xc**2 + Yc**2) / (2.0 * sigma_click**2)).astype(np.complex128)
        phi, _ = normalize_unit(phi, self.grid.dx, self.grid.dy)
        return phi

    def kinetic_phase(self, dt: float) -> np.ndarray:
        return np.exp(-1j * self.K2 * dt / (2.0 * self.m_mass))

    def potential_phase(self, V: np.ndarray, dt: float) -> np.ndarray:
        return np.exp(-1j * V * dt / self.hbar)

    def _step_field(
        self,
        field: np.ndarray,
        K_phase: np.ndarray,
        P_half: np.ndarray,
    ) -> np.ndarray:
        if not np.iscomplexobj(field):
            field = field.astype(np.complex128)

        field = field * P_half
        f_k = np.fft.fft2(field)
        f_k = f_k * K_phase
        field = np.fft.ifft2(f_k)
        field = field * P_half
        return field

    def step_forward(self, state: np.ndarray, dt: float) -> TheoryStepResult:
        P_half = self.potential_phase(self.V_fwd, dt / 2.0)
        K_phase = self.kinetic_phase(dt)
        new_state = self._step_field(state, K_phase, P_half)
        return TheoryStepResult(state=new_state, aux=None)

    def step_backward_adjoint(self, state: np.ndarray, dt: float) -> TheoryStepResult:
        P_half = self.potential_phase(self.V_adj, -dt / 2.0)
        K_phase = self.kinetic_phase(-dt)
        new_state = self._step_field(state, K_phase, P_half)
        return TheoryStepResult(state=new_state, aux=None)

    def density(self, state: np.ndarray) -> np.ndarray:
        return (np.abs(state) ** 2).astype(float)

    def current(self, state_vis: np.ndarray):
        dpsi_dx = (
            np.roll(state_vis, -1, axis=1) - np.roll(state_vis, 1, axis=1)
        ) / (2.0 * self.grid.dx)

        dpsi_dy = (
            np.roll(state_vis, -1, axis=0) - np.roll(state_vis, 1, axis=0)
        ) / (2.0 * self.grid.dy)

        rho = (np.abs(state_vis) ** 2).astype(float)

        jx = (
            (self.hbar / self.m_mass)
            * np.imag(np.conjugate(state_vis) * dpsi_dx)
        ).astype(float)

        jy = (
            (self.hbar / self.m_mass)
            * np.imag(np.conjugate(state_vis) * dpsi_dy)
        ).astype(float)

        return jx, jy, rho