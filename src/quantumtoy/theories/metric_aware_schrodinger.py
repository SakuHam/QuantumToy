from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from theories.base import TheoryModel, TheoryStepResult
from core.utils import normalize_unit


@dataclass
class MetricAwareSchrodingerTheory(TheoryModel):
    """
    Metric-aware Schrödinger toy model.

    Evolves:
        i hbar dpsi/dt = H psi

    with Hamiltonian

        H psi =
            -(hbar^2 / 2m) div( a(x,y) grad psi )
            + (V_real + V_metric - i W) psi

    where a(x,y) is a positive spatial coefficient derived from a
    Schwarzschild-like "lapse" field around a center (x_bh, y_bh).

    Notes
    -----
    - This is NOT full GR quantum dynamics.
    - It is a useful curved-space-inspired toy model.
    - Uses periodic finite differences via np.roll, so CAP is still useful.
    - Time stepping is explicit RK4; dt usually needs to be smaller than
      with FFT split-operator Schrödinger.
    """

    grid: any
    potential: any

    m_mass: float = 1.0
    hbar: float = 1.0

    # "Black hole" / metric parameters
    metric_center_x: float = 0.0
    metric_center_y: float = 0.0
    schwarzschild_radius: float = 1.0
    metric_softening: float = 0.25
    min_lapse: float = 0.08

    # How the metric enters the kinetic operator
    # "lapse"      -> a = alpha
    # "lapse_sq"   -> a = alpha^2
    # "inv_spatial"-> a = 1 / g_spatial ~ alpha
    metric_mode: str = "lapse"

    # Optional extra redshift-like scalar potential
    use_metric_potential: bool = True
    metric_potential_strength: float = 2.0

    # Time integration
    integrator: str = "rk4"   # "euler", "rk2", "rk4"

    # Debug
    debug_checks: bool = True
    debug_packet_stats: bool = True
    debug_tol_norm_growth: float = 1e-7

    # -----------------------------------------------------
    # init
    # -----------------------------------------------------

    def __post_init__(self):
        self.m_mass = float(self.m_mass)
        self.hbar = float(self.hbar)

        self.metric_center_x = float(self.metric_center_x)
        self.metric_center_y = float(self.metric_center_y)
        self.schwarzschild_radius = float(self.schwarzschild_radius)
        self.metric_softening = float(self.metric_softening)
        self.min_lapse = float(self.min_lapse)

        self.metric_mode = str(self.metric_mode).lower().strip()
        self.integrator = str(self.integrator).lower().strip()

        self.use_metric_potential = bool(self.use_metric_potential)
        self.metric_potential_strength = float(self.metric_potential_strength)

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
        self._assert(self.m_mass > 0.0, "m_mass must be positive")
        self._assert(self.metric_softening > 0.0, "metric_softening must be positive")
        self._assert(self.schwarzschild_radius >= 0.0, "schwarzschild_radius must be non-negative")
        self._assert(0.0 < self.min_lapse <= 1.0, "min_lapse must be in (0, 1]")
        self._assert(
            self.metric_mode in ("lapse", "lapse_sq", "inv_spatial"),
            f"invalid metric_mode={self.metric_mode!r}",
        )
        self._assert(
            self.integrator in ("euler", "rk2", "rk4"),
            f"invalid integrator={self.integrator!r}",
        )

        self.grid.X = np.asarray(self.grid.X, dtype=float)
        self.grid.Y = np.asarray(self.grid.Y, dtype=float)

        self._assert(
            self.grid.X.shape == (self.grid.Ny, self.grid.Nx),
            f"grid.X shape {self.grid.X.shape} != {(self.grid.Ny, self.grid.Nx)}",
        )
        self._assert(
            self.grid.Y.shape == (self.grid.Ny, self.grid.Nx),
            f"grid.Y shape {self.grid.Y.shape} != {(self.grid.Ny, self.grid.Nx)}",
        )

        self.V_real = np.asarray(self.potential.V_real, dtype=float)
        self.W = np.asarray(self.potential.W, dtype=float)

        self._assert(
            self.V_real.shape == (self.grid.Ny, self.grid.Nx),
            "potential.V_real shape mismatch",
        )
        self._assert(
            self.W.shape == (self.grid.Ny, self.grid.Nx),
            "potential.W shape mismatch",
        )

        self._build_metric_fields()

        if self.debug_checks:
            self._debug_assert_finite_array("grid.X", self.grid.X)
            self._debug_assert_finite_array("grid.Y", self.grid.Y)
            self._debug_assert_finite_array("V_real", self.V_real)
            self._debug_assert_finite_array("W", self.W)
            self._debug_assert_finite_array("r_metric", self.r_metric)
            self._debug_assert_finite_array("alpha_metric", self.alpha_metric)
            self._debug_assert_finite_array("a_metric", self.a_metric)
            self._debug_assert_finite_array("V_metric", self.V_metric)

    # -----------------------------------------------------
    # debug / asserts
    # -----------------------------------------------------

    def _assert(self, cond: bool, msg: str):
        if not cond:
            raise AssertionError(msg)

    def _debug_warn(self, msg: str):
        if self.debug_checks:
            print(f"[MetricAwareSchrodingerTheory DEBUG] {msg}")

    def _debug_info(self, msg: str):
        if self.debug_checks:
            print(f"[MetricAwareSchrodingerTheory INFO] {msg}")

    def _debug_assert_finite_array(self, name: str, arr: np.ndarray):
        if not self.debug_checks:
            return
        arr = np.asarray(arr)
        if not np.all(np.isfinite(arr)):
            bad = np.size(arr) - np.count_nonzero(np.isfinite(arr))
            raise RuntimeError(f"{name} contains non-finite values ({bad} bad entries)")

    # -----------------------------------------------------
    # metric fields
    # -----------------------------------------------------

    def _build_metric_fields(self):
        dx = self.grid.X - self.metric_center_x
        dy = self.grid.Y - self.metric_center_y
        r = np.sqrt(dx * dx + dy * dy + self.metric_softening**2)

        # Schwarzschild-like lapse:
        # alpha ~ 1 - r_s / r, clipped positive
        alpha = 1.0 - (self.schwarzschild_radius / r)
        alpha = np.clip(alpha, self.min_lapse, 1.0)

        if self.metric_mode == "lapse":
            a = alpha
        elif self.metric_mode == "lapse_sq":
            a = alpha**2
        elif self.metric_mode == "inv_spatial":
            a = alpha
        else:
            raise AssertionError(f"Unhandled metric_mode={self.metric_mode!r}")

        # Optional extra scalar potential that deepens near center.
        if self.use_metric_potential:
            # zero far away, negative near the center
            V_metric = -self.metric_potential_strength * (1.0 - alpha)
        else:
            V_metric = np.zeros_like(alpha, dtype=float)

        self.r_metric = r
        self.alpha_metric = alpha
        self.a_metric = a
        self.V_metric = V_metric

    # -----------------------------------------------------
    # initialization
    # -----------------------------------------------------

    def initialize_state(self, state0):
        psi = np.asarray(state0, dtype=np.complex128)
        if psi.shape != (self.grid.Ny, self.grid.Nx):
            raise ValueError(
                f"state0 must have shape {(self.grid.Ny, self.grid.Nx)}, got {psi.shape}"
            )

        psi, _ = normalize_unit(psi, self.grid.dx, self.grid.dy)

        if self.debug_checks:
            self._debug_assert_finite_array("initialize_state psi", psi)
            self.debug_packet_summary("initialize_state summary", psi)

        return psi

    def initialize_click_state(self, x_click, y_click, sigma_click):
        sigma_click = float(sigma_click)
        if sigma_click <= 0.0:
            raise ValueError("sigma_click must be positive")

        Xc = self.grid.X - float(x_click)
        Yc = self.grid.Y - float(y_click)

        psi = np.exp(-(Xc**2 + Yc**2) / (2.0 * sigma_click**2)).astype(np.complex128)
        psi, _ = normalize_unit(psi, self.grid.dx, self.grid.dy)

        if self.debug_checks:
            self._debug_assert_finite_array("initialize_click_state psi", psi)

        return psi

    # -----------------------------------------------------
    # finite-difference helpers
    # -----------------------------------------------------

    def _grad_x(self, psi: np.ndarray) -> np.ndarray:
        return (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2.0 * self.grid.dx)

    def _grad_y(self, psi: np.ndarray) -> np.ndarray:
        return (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2.0 * self.grid.dy)

    def _div(self, Fx: np.ndarray, Fy: np.ndarray) -> np.ndarray:
        dFx_dx = (np.roll(Fx, -1, axis=1) - np.roll(Fx, 1, axis=1)) / (2.0 * self.grid.dx)
        dFy_dy = (np.roll(Fy, -1, axis=0) - np.roll(Fy, 1, axis=0)) / (2.0 * self.grid.dy)
        return dFx_dx + dFy_dy

    def _metric_kinetic_apply(self, psi: np.ndarray) -> np.ndarray:
        """
        Compute:
            div( a(x,y) grad psi )
        """
        gx = self._grad_x(psi)
        gy = self._grad_y(psi)

        Fx = self.a_metric * gx
        Fy = self.a_metric * gy

        out = self._div(Fx, Fy)

        if self.debug_checks:
            self._debug_assert_finite_array("metric kinetic apply", out)

        return out

    # -----------------------------------------------------
    # Hamiltonian / rhs
    # -----------------------------------------------------

    def _hamiltonian_apply(self, psi: np.ndarray, V_real: np.ndarray, W: np.ndarray) -> np.ndarray:
        kinetic = -(self.hbar**2 / (2.0 * self.m_mass)) * self._metric_kinetic_apply(psi)
        potential_term = (V_real + self.V_metric - 1j * W) * psi
        Hpsi = kinetic + potential_term

        if self.debug_checks:
            self._debug_assert_finite_array("Hpsi", Hpsi)

        return Hpsi

    def _rhs(self, psi: np.ndarray, V_real: np.ndarray, W: np.ndarray) -> np.ndarray:
        Hpsi = self._hamiltonian_apply(psi, V_real=V_real, W=W)
        rhs = (-1j / self.hbar) * Hpsi

        if self.debug_checks:
            self._debug_assert_finite_array("rhs", rhs)

        return rhs

    # -----------------------------------------------------
    # stepping
    # -----------------------------------------------------

    def _step_with_fields(self, psi: np.ndarray, dt: float, V_real: np.ndarray, W: np.ndarray) -> np.ndarray:
        dt = float(dt)
        if not np.isfinite(dt):
            raise ValueError("dt must be finite")

        psi = np.asarray(psi, dtype=np.complex128)
        if psi.shape != (self.grid.Ny, self.grid.Nx):
            raise ValueError(f"psi shape {psi.shape} != {(self.grid.Ny, self.grid.Nx)}")

        if self.debug_checks:
            self._debug_assert_finite_array("step input psi", psi)
            n_before = float(np.sum(np.abs(psi) ** 2) * self.grid.dx * self.grid.dy)

        if self.integrator == "euler":
            psi_new = psi + dt * self._rhs(psi, V_real, W)

        elif self.integrator == "rk2":
            k1 = self._rhs(psi, V_real, W)
            k2 = self._rhs(psi + 0.5 * dt * k1, V_real, W)
            psi_new = psi + dt * k2

        elif self.integrator == "rk4":
            k1 = self._rhs(psi, V_real, W)
            k2 = self._rhs(psi + 0.5 * dt * k1, V_real, W)
            k3 = self._rhs(psi + 0.5 * dt * k2, V_real, W)
            k4 = self._rhs(psi + dt * k3, V_real, W)
            psi_new = psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        else:
            raise AssertionError(f"Unhandled integrator={self.integrator!r}")

        if self.debug_checks:
            self._debug_assert_finite_array("step output psi", psi_new)
            n_after = float(np.sum(np.abs(psi_new) ** 2) * self.grid.dx * self.grid.dy)

            # Nonnegative W should not create huge norm growth.
            if np.min(W) >= -1e-14:
                if n_after - n_before > self.debug_tol_norm_growth:
                    self._debug_warn(
                        "norm grew during forward step under nonnegative absorption: "
                        f"before={n_before:.12f}, after={n_after:.12f}"
                    )

        return psi_new

    def step_forward(self, state, dt: float):
        psi = self._step_with_fields(state, dt, self.V_real, self.W)
        return TheoryStepResult(state=psi, aux=None)

    def step_backward_adjoint(self, state, dt: float):
        # Adjoint-like backward evolution:
        # use conjugate complex potential => V_real + iW
        psi = self._step_with_fields(state, -float(dt), self.V_real, -self.W)
        return TheoryStepResult(state=psi, aux=None)

    # -----------------------------------------------------
    # observables
    # -----------------------------------------------------

    def density(self, state):
        psi = np.asarray(state, dtype=np.complex128)
        rho = (np.abs(psi) ** 2).astype(float)

        if self.debug_checks:
            self._debug_assert_finite_array("density", rho)

        return rho

    def current(self, state_vis):
        """
        Metric-aware current approximation:
            j = (hbar/m) * a(x,y) * Im(conj(psi) grad psi)
        """
        psi = np.asarray(state_vis, dtype=np.complex128)

        if psi.ndim != 2:
            raise ValueError(f"state_vis must be 2D, got ndim={psi.ndim}")

        # Need the matching a-field for current, depending on whether we got full
        # grid or visible crop.
        if psi.shape == (self.grid.Ny, self.grid.Nx):
            a_here = self.a_metric
        else:
            a_here = self.a_metric[self.grid.ys, self.grid.xs]
            if psi.shape != a_here.shape:
                raise ValueError(
                    f"state_vis shape {psi.shape} incompatible with visible metric shape {a_here.shape}"
                )

        gx = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2.0 * self.grid.dx)
        gy = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2.0 * self.grid.dy)

        rho = (np.abs(psi) ** 2).astype(float)
        jx = ((self.hbar / self.m_mass) * a_here * np.imag(np.conjugate(psi) * gx)).astype(float)
        jy = ((self.hbar / self.m_mass) * a_here * np.imag(np.conjugate(psi) * gy)).astype(float)

        if self.debug_checks:
            self._debug_assert_finite_array("jx", jx)
            self._debug_assert_finite_array("jy", jy)
            self._debug_assert_finite_array("rho(current)", rho)

        return jx, jy, rho

    def velocity(self, state_vis, eps_rho: float = 1e-10):
        if eps_rho <= 0.0:
            raise ValueError("eps_rho must be positive")

        jx, jy, rho = self.current(state_vis)
        denom = np.maximum(rho, float(eps_rho))

        vx = jx / denom
        vy = jy / denom
        sp = np.hypot(vx, vy)

        if self.debug_checks:
            self._debug_assert_finite_array("vx", vx)
            self._debug_assert_finite_array("vy", vy)
            self._debug_assert_finite_array("speed", sp)

        return vx.astype(float), vy.astype(float), sp.astype(float)

    # -----------------------------------------------------
    # diagnostics
    # -----------------------------------------------------

    def packet_summary(
        self,
        state_like: np.ndarray,
        X_like: np.ndarray | None = None,
        Y_like: np.ndarray | None = None,
    ) -> dict:
        psi = np.asarray(state_like, dtype=np.complex128)

        if X_like is None or Y_like is None:
            self._assert(
                psi.shape == (self.grid.Ny, self.grid.Nx),
                "X_like/Y_like omitted but state_like is not full-grid sized",
            )
            X = self.grid.X
            Y = self.grid.Y
        else:
            X = np.asarray(X_like, dtype=float)
            Y = np.asarray(Y_like, dtype=float)
            self._assert(X.shape == psi.shape, f"X_like shape {X.shape} != {psi.shape}")
            self._assert(Y.shape == psi.shape, f"Y_like shape {Y.shape} != {psi.shape}")

        rho = self.density(psi)
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