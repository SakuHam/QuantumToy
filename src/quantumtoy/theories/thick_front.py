from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from theories.base import TheoryStepResult
from theories.schrodinger import SchrodingerTheory
from core.utils import normalize_unit


@dataclass
class ThickFrontTheory(SchrodingerTheory):
    """
    Thick reality front model.

    Idea:
    - Evolution is mostly Schrödinger
    - plus a mild coherence sharpening term
      that gradually collapses the wave packet
      toward locally coherent branches.
    """

    front_strength: float = 0.02
    front_sigma: float = 2.0

    # ============================================================
    # coherence operator
    # ============================================================

    def _coherence_filter(self, psi: np.ndarray) -> np.ndarray:
        """
        Local coherence sharpening.

        Uses Gaussian smoothing of phase alignment.
        """

        phase = psi / (np.abs(psi) + 1e-30)

        # gaussian kernel
        r = int(max(1, self.front_sigma))
        ax = np.arange(-r, r + 1)
        xx, yy = np.meshgrid(ax, ax)

        kernel = np.exp(-(xx**2 + yy**2) / (2 * self.front_sigma**2))
        kernel /= np.sum(kernel)

        phase_sm = np.real(
            np.fft.ifft2(
                np.fft.fft2(phase) * np.fft.fft2(kernel, s=phase.shape)
            )
        )

        coherence = np.abs(phase_sm)

        return psi * (1 + self.front_strength * coherence)

    # ============================================================
    # forward evolution
    # ============================================================

    def step_forward(self, state: np.ndarray, dt: float) -> TheoryStepResult:
        """
        Schrödinger step + front sharpening.
        """

        base = super().step_forward(state, dt)

        psi = base.state

        psi = self._coherence_filter(psi)

        psi, _ = normalize_unit(psi, self.grid.dx, self.grid.dy)

        return TheoryStepResult(state=psi, aux={"front": True})

    # ============================================================
    # backward evolution
    # ============================================================

    def step_backward_adjoint(self, state: np.ndarray, dt: float) -> TheoryStepResult:
        """
        Backward evolution uses pure Schrödinger adjoint.
        """

        base = super().step_backward_adjoint(state, dt)

        return TheoryStepResult(state=base.state, aux=None)

    # ============================================================
    # density
    # ============================================================

    def density(self, state: np.ndarray) -> np.ndarray:
        return (np.abs(state) ** 2).astype(float)

    # ============================================================
    # current
    # ============================================================

    def current(self, state_vis: np.ndarray):
        """
        Same current definition as Schrödinger.
        """

        dpsi_dx = (
            np.roll(state_vis, -1, axis=1)
            - np.roll(state_vis, 1, axis=1)
        ) / (2.0 * self.grid.dx)

        dpsi_dy = (
            np.roll(state_vis, -1, axis=0)
            - np.roll(state_vis, 1, axis=0)
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