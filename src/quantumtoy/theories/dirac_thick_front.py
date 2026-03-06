from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from theories.base import TheoryStepResult
from theories.dirac import DiracTheory


@dataclass
class DiracThickFrontTheory(DiracTheory):
    """
    Dirac + optimized thick-front theory.

    Forward step:
        Dirac split-operator step
        + local thick-front coherence sharpening on the 2-spinor

    Uses total spinor density:
        rho = |psi1|^2 + |psi2|^2
    """

    front_strength: float = 0.03
    front_misaligned_damp: float = 0.01
    front_diag_weight: float = 0.5
    front_density_weighted: bool = True
    front_phase_relax_strength: float = 0.0
    front_eps: float = 1e-12
    front_clip: float = 0.25

    # ============================================================
    # helpers
    # ============================================================

    def _neighbor_average_complex(self, z: np.ndarray) -> np.ndarray:
        z_xp = np.roll(z, -1, axis=1)
        z_xm = np.roll(z,  1, axis=1)
        z_yp = np.roll(z, -1, axis=0)
        z_ym = np.roll(z,  1, axis=0)

        z_d1 = np.roll(z_xp, -1, axis=0)
        z_d2 = np.roll(z_xp,  1, axis=0)
        z_d3 = np.roll(z_xm, -1, axis=0)
        z_d4 = np.roll(z_xm,  1, axis=0)

        axis_sum = z_xp + z_xm + z_yp + z_ym
        diag_sum = z_d1 + z_d2 + z_d3 + z_d4

        w_axis = 1.0
        w_diag = float(self.front_diag_weight)

        denom = 4.0 * w_axis + 4.0 * w_diag
        return (w_axis * axis_sum + w_diag * diag_sum) / max(denom, self.front_eps)

    def _spinor_phase_fields(self, psi1: np.ndarray, psi2: np.ndarray):
        """
        Build normalized local phase-like fields for each spinor component,
        using total density for stability.
        """
        rho = (np.abs(psi1) ** 2 + np.abs(psi2) ** 2).astype(float)
        amp = np.sqrt(np.maximum(rho, self.front_eps))

        u1 = psi1 / amp
        u2 = psi2 / amp

        return u1, u2, rho

    def _coherence_alignment_score(self, psi1: np.ndarray, psi2: np.ndarray):
        """
        Compute a shared alignment score for the full spinor.

        Returns:
            align_real : scalar field in roughly [-1, 1]
            rho        : total density
            u1_local   : local coherent phase suggestion for comp 1
            u2_local   : local coherent phase suggestion for comp 2
        """
        u1, u2, rho = self._spinor_phase_fields(psi1, psi2)

        u1_nei = self._neighbor_average_complex(u1)
        u2_nei = self._neighbor_average_complex(u2)

        if self.front_density_weighted:
            amp_nei = self._neighbor_average_complex(
                np.sqrt(np.maximum(rho, 0.0)).astype(np.complex128)
            ).real
            amp_nei = np.maximum(amp_nei, 0.0)

            weight = 1.0 + amp_nei
            u1_nei = u1_nei * weight
            u2_nei = u2_nei * weight

        u1_local = u1_nei / np.maximum(np.abs(u1_nei), self.front_eps)
        u2_local = u2_nei / np.maximum(np.abs(u2_nei), self.front_eps)

        align_1 = np.real(np.conjugate(u1) * u1_local)
        align_2 = np.real(np.conjugate(u2) * u2_local)

        norm1 = np.abs(u1) ** 2
        norm2 = np.abs(u2) ** 2
        denom = np.maximum(norm1 + norm2, self.front_eps)

        align_real = (align_1 * norm1 + align_2 * norm2) / denom

        return align_real.astype(float), rho.astype(float), u1_local, u2_local

    def _front_sharpen_spinor(
        self,
        psi1: np.ndarray,
        psi2: np.ndarray,
        dt: float,
    ):
        """
        Apply thick-front sharpening coherently to the full 2-spinor.
        """
        align_real, rho, u1_local, u2_local = self._coherence_alignment_score(psi1, psi2)

        gain = (
            self.front_strength * np.maximum(align_real, 0.0)
            - self.front_misaligned_damp * np.maximum(-align_real, 0.0)
        )

        rho_mean = float(np.mean(rho))
        if rho_mean > self.front_eps:
            rho_scale = rho / rho_mean
            gain = gain * np.sqrt(np.maximum(rho_scale, 0.0))

        gain = np.clip(gain * dt, -self.front_clip, self.front_clip)

        amp_factor = np.exp(gain)

        psi1_new = psi1 * amp_factor
        psi2_new = psi2 * amp_factor

        if self.front_phase_relax_strength > 0.0:
            alpha = float(np.clip(self.front_phase_relax_strength * dt, 0.0, 1.0))

            rho_new = (np.abs(psi1_new) ** 2 + np.abs(psi2_new) ** 2).astype(float)
            amp = np.sqrt(np.maximum(rho_new, self.front_eps))

            u1 = psi1_new / amp
            u2 = psi2_new / amp

            u1_mix = (1.0 - alpha) * u1 + alpha * u1_local
            u2_mix = (1.0 - alpha) * u2 + alpha * u2_local

            u1_mix /= np.maximum(np.abs(u1_mix), self.front_eps)
            u2_mix /= np.maximum(np.abs(u2_mix), self.front_eps)

            psi1_new = amp * u1_mix
            psi2_new = amp * u2_mix

        return psi1_new.astype(np.complex128), psi2_new.astype(np.complex128)

    def _normalize_spinor(self, psi1: np.ndarray, psi2: np.ndarray):
        rho = (np.abs(psi1) ** 2 + np.abs(psi2) ** 2).astype(float)
        n = float(np.sqrt(np.sum(rho) * self.grid.dx * self.grid.dy))

        if n <= 0:
            return psi1, psi2

        return psi1 / n, psi2 / n

    # ============================================================
    # API methods
    # ============================================================

    def step_forward(self, state, dt: float):
        """
        Dirac forward step + thick-front sharpening.
        """
        base = super().step_forward(state, dt)

        psi1, psi2 = base.state
        psi1, psi2 = self._front_sharpen_spinor(psi1, psi2, dt)
        psi1, psi2 = self._normalize_spinor(psi1, psi2)

        aux = dict(base.aux) if base.aux is not None else {}
        aux["thick_front"] = {
            "front_strength": float(self.front_strength),
            "front_misaligned_damp": float(self.front_misaligned_damp),
            "front_diag_weight": float(self.front_diag_weight),
            "front_phase_relax_strength": float(self.front_phase_relax_strength),
        }

        return TheoryStepResult(
            state=np.stack([psi1, psi2], axis=0),
            aux=aux,
        )

    def step_backward_adjoint(self, state, dt: float):
        """
        Keep backward evolution as plain Dirac adjoint evolution.
        """
        return super().step_backward_adjoint(state, dt)

    def density(self, state):
        psi1, psi2 = state
        return (np.abs(psi1) ** 2 + np.abs(psi2) ** 2).astype(float)

    def current(self, state_vis):
        """
        Relativistically correct 2D Dirac probability current.
        """
        psi1, psi2 = state_vis

        rho = (np.abs(psi1) ** 2 + np.abs(psi2) ** 2).astype(float)

        overlap = np.conjugate(psi1) * psi2

        jx = (2.0 * self.c_light * np.real(overlap)).astype(float)
        jy = (2.0 * self.c_light * np.imag(overlap)).astype(float)

        return jx, jy, rho