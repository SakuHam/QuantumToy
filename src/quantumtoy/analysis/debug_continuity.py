from __future__ import annotations
import numpy as np


def continuity_residual_from_state_frames(
    theory,
    state_vis_frames: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
):
    """
    Check continuity equation:

        ∂ρ/∂t + div(j) ≈ 0

    using forward simulation frames.

    Parameters
    ----------
    theory:
        theory object (DiracTheory or SchrödingerTheory)

    state_vis_frames:
        complex states on visible grid
        shape:
            Schr:  (Nt, Ny, Nx)
            Dirac: (Nt, 2, Ny, Nx)

    dx, dy:
        grid spacing

    dt:
        time step between frames
    """

    Nt = state_vis_frames.shape[0]

    rms_list = []
    max_list = []

    for i in range(1, Nt - 1):

        state_prev = state_vis_frames[i - 1]
        state_now  = state_vis_frames[i]
        state_next = state_vis_frames[i + 1]

        # density
        rho_prev = theory.density(state_prev)
        rho_now  = theory.density(state_now)
        rho_next = theory.density(state_next)

        # time derivative
        drho_dt = (rho_next - rho_prev) / (2.0 * dt)

        # current
        jx, jy, _ = theory.current(state_now)

        # divergence
        djx_dx = (np.roll(jx, -1, axis=1) - np.roll(jx, 1, axis=1)) / (2.0 * dx)
        djy_dy = (np.roll(jy, -1, axis=0) - np.roll(jy, 1, axis=0)) / (2.0 * dy)

        div_j = djx_dx + djy_dy

        residual = drho_dt + div_j

        rms = float(np.sqrt(np.mean(residual**2)))
        mx  = float(np.max(np.abs(residual)))

        rms_list.append(rms)
        max_list.append(mx)

    rms_mean = float(np.mean(rms_list))
    rms_max  = float(np.max(rms_list))
    abs_max  = float(np.max(max_list))

    return rms_mean, rms_max, abs_max