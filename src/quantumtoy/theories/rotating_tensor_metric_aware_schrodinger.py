from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from theories.tensor_metric_aware_schrodinger import TensorMetricAwareSchrodingerTheory


@dataclass
class RotatingTensorMetricAwareSchrodingerTheory(TensorMetricAwareSchrodingerTheory):
    """
    Orbit-capable extension of TensorMetricAwareSchrodingerTheory.

    Key features:
      - radial vs tangential metric separation
      - off-diagonal swirl (gxy)
      - optional centrifugal barrier
      - STABLE angular drift integrated into Hamiltonian
    """

    # -------------------------------
    # Metric shaping
    # -------------------------------
    tangential_boost: float = 0.8
    radial_suppression: float = 0.6
    orbit_sigma: float = 2.0

    # -------------------------------
    # Metric swirl (gxy)
    # -------------------------------
    rotation_strength: float = 0.15
    rotation_mode: str = "gaussian"
    rotation_handedness: float = 1.0
    clamp_gxy_ratio: float = 0.25

    # -------------------------------
    # Optional centrifugal barrier
    # -------------------------------
    use_centrifugal_barrier: bool = True
    centrifugal_strength: float = 0.5
    centrifugal_softening: float = 1.0

    # -------------------------------
    # NEW: angular drift (STABLE)
    # -------------------------------
    use_angular_drift: bool = True
    angular_drift_strength: float = 0.02
    angular_drift_sigma: float = 2.0
    angular_drift_mode: str = "gaussian"
    angular_drift_handedness: float = 1.0

    # ============================================================
    # Init
    # ============================================================

    def __post_init__(self):
        self.rotation_mode = str(self.rotation_mode).lower().strip()
        self.angular_drift_mode = str(self.angular_drift_mode).lower().strip()

        self._assert(self.orbit_sigma > 0.0, "orbit_sigma must be > 0")
        self._assert(self.angular_drift_sigma > 0.0, "angular_drift_sigma must be > 0")

        super().__post_init__()

    # ============================================================
    # Metric construction
    # ============================================================

    def _build_metric_fields(self):
        dx = self.grid.X - self.metric_center_x
        dy = self.grid.Y - self.metric_center_y

        r2 = dx * dx + dy * dy
        r = np.sqrt(r2 + self.metric_softening**2)

        alpha = 1.0 - (self.schwarzschild_radius / r)
        alpha = np.clip(alpha, self.min_lapse, 1.0)

        env = 1.0 - alpha

        inv_r = 1.0 / np.maximum(r, 1e-12)

        # radial / tangential basis
        ex_r_x = dx * inv_r
        ex_r_y = dy * inv_r
        ex_t_x = -dy * inv_r
        ex_t_y = dx * inv_r

        # anisotropic response
        g_r = alpha * (1.0 - self.radial_suppression * env)
        g_t = alpha * (1.0 + self.tangential_boost * env)

        g_r = np.maximum(g_r, 0.05 * self.min_lapse)
        g_t = np.maximum(g_t, 0.05 * self.min_lapse)

        # build inverse metric
        gxx_inv = g_r * ex_r_x * ex_r_x + g_t * ex_t_x * ex_t_x
        gyy_inv = g_r * ex_r_y * ex_r_y + g_t * ex_t_y * ex_t_y
        gxy_base = g_r * ex_r_x * ex_r_y + g_t * ex_t_x * ex_t_y

        # swirl
        if self.rotation_mode == "gaussian":
            rot_env = np.exp(-r2 / (2.0 * self.orbit_sigma**2))
        else:
            rot_env = 1.0 / (r2 + self.orbit_sigma**2)
            rot_env /= np.max(rot_env)

        gxy_rot = self.rotation_strength * self.rotation_handedness * rot_env
        gxy_inv = gxy_base + gxy_rot

        # clamp for stability
        if self.clamp_gxy_ratio > 0.0:
            ref = np.sqrt(np.maximum(gxx_inv * gyy_inv, 1e-12))
            lim = self.clamp_gxy_ratio * ref
            gxy_inv = np.clip(gxy_inv, -lim, lim)

        det_inv = np.maximum(gxx_inv * gyy_inv - gxy_inv * gxy_inv, 1e-8)
        sqrt_g = 1.0 / np.sqrt(det_inv)

        # potential
        V_metric = np.zeros_like(alpha)

        if self.use_centrifugal_barrier:
            V_metric += self.centrifugal_strength / (r2 + self.centrifugal_softening**2)

        # store
        self.r_metric = r
        self.alpha_metric = alpha
        self.gxx_inv = gxx_inv
        self.gyy_inv = gyy_inv
        self.gxy_inv = gxy_inv
        self.sqrt_g = sqrt_g
        self.V_metric = V_metric

    # ============================================================
    # Angular drift profile
    # ============================================================

    def _angular_drift_profile(self):
        dx = self.grid.X - self.metric_center_x
        dy = self.grid.Y - self.metric_center_y

        r2 = dx * dx + dy * dy

        if self.angular_drift_mode == "gaussian":
            env = np.exp(-r2 / (2.0 * self.angular_drift_sigma**2))
        else:
            env = 1.0 / (r2 + self.angular_drift_sigma**2)
            env /= np.max(env)

        return self.angular_drift_strength * self.angular_drift_handedness * env

    # ============================================================
    # Angular drift Hamiltonian
    # ============================================================

    def _angular_drift_hamiltonian_apply(self, psi: np.ndarray):
        if (not self.use_angular_drift) or (self.angular_drift_strength == 0.0):
            return np.zeros_like(psi, dtype=np.complex128)

        dx = self.grid.X - self.metric_center_x
        dy = self.grid.Y - self.metric_center_y

        dpsi_dx = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2.0 * self.grid.dx)
        dpsi_dy = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2.0 * self.grid.dy)

        lz_psi = dx * dpsi_dy - dy * dpsi_dx
        omega = self._angular_drift_profile()

        return -self.hbar * omega * lz_psi

    # ============================================================
    # FULL Hamiltonian (patched)
    # ============================================================

    def _hamiltonian_apply(self, psi: np.ndarray, V_real: np.ndarray, W: np.ndarray):
        # kinetic from tensor metric
        kinetic = -(self.hbar**2 / (2.0 * self.m_mass)) * self._metric_kinetic_apply(psi)

        # potential
        potential_term = (V_real + self.V_metric - 1j * W) * psi

        # NEW: angular drift
        drift = self._angular_drift_hamiltonian_apply(psi)

        Hpsi = kinetic + potential_term + drift

        if self.debug_checks:
            self._debug_assert_finite_array("Hpsi", Hpsi)

        return Hpsi