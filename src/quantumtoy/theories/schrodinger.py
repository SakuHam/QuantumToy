from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from theories.base import TheoryModel, TheoryStepResult
from core.utils import normalize_unit


# ============================================================
# Validation helpers
# ============================================================

def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def _assert_finite_scalar(x, name: str):
    _assert(np.isscalar(x), f"{name} must be a scalar, got type={type(x)}")
    xf = float(x)
    _assert(np.isfinite(xf), f"{name} must be finite, got {x}")
    return xf


def _assert_positive_scalar(x, name: str):
    xf = _assert_finite_scalar(x, name)
    _assert(xf > 0.0, f"{name} must be > 0, got {x}")
    return xf


def _assert_complex_array(arr: np.ndarray, name: str):
    _assert(isinstance(arr, np.ndarray), f"{name} must be np.ndarray")
    _assert(arr.ndim == 2, f"{name} must be 2D, got ndim={arr.ndim}")
    _assert(np.all(np.isfinite(arr.real)), f"{name}.real contains non-finite values")
    _assert(np.all(np.isfinite(arr.imag)), f"{name}.imag contains non-finite values")


def _assert_real_array(arr: np.ndarray, name: str):
    _assert(isinstance(arr, np.ndarray), f"{name} must be np.ndarray")
    _assert(arr.ndim == 2, f"{name} must be 2D, got ndim={arr.ndim}")
    _assert(np.all(np.isfinite(arr)), f"{name} contains non-finite values")


def _assert_shape(arr: np.ndarray, shape: tuple[int, ...], name: str):
    _assert(arr.shape == shape, f"{name} shape {arr.shape} != expected {shape}")


def _assert_same_shape(a: np.ndarray, b: np.ndarray, aname: str, bname: str):
    _assert(a.shape == b.shape, f"{aname}.shape {a.shape} != {bname}.shape {b.shape}")


# ============================================================
# Schrödinger theory
# ============================================================

@dataclass
class SchrodingerTheory(TheoryModel):
    grid: any
    potential: any
    m_mass: float = 1.0
    hbar: float = 1.0

    def __post_init__(self):
        # ----------------------------------------------------
        # Validate scalar parameters
        # ----------------------------------------------------
        self.m_mass = _assert_positive_scalar(self.m_mass, "m_mass")
        self.hbar = _assert_positive_scalar(self.hbar, "hbar")

        # ----------------------------------------------------
        # Validate grid
        # ----------------------------------------------------
        for attr in ("Nx", "Ny", "dx", "dy", "X", "Y", "xs", "ys", "n_visible_x", "n_visible_y"):
            _assert(hasattr(self.grid, attr), f"grid missing required attribute '{attr}'")

        _assert(isinstance(self.grid.Nx, int) and self.grid.Nx > 0, f"grid.Nx invalid: {self.grid.Nx}")
        _assert(isinstance(self.grid.Ny, int) and self.grid.Ny > 0, f"grid.Ny invalid: {self.grid.Ny}")
        _assert_positive_scalar(self.grid.dx, "grid.dx")
        _assert_positive_scalar(self.grid.dy, "grid.dy")

        _assert(isinstance(self.grid.X, np.ndarray), "grid.X must be np.ndarray")
        _assert(isinstance(self.grid.Y, np.ndarray), "grid.Y must be np.ndarray")
        _assert_shape(self.grid.X, (self.grid.Ny, self.grid.Nx), "grid.X")
        _assert_shape(self.grid.Y, (self.grid.Ny, self.grid.Nx), "grid.Y")
        _assert(np.all(np.isfinite(self.grid.X)), "grid.X contains non-finite values")
        _assert(np.all(np.isfinite(self.grid.Y)), "grid.Y contains non-finite values")

        # ----------------------------------------------------
        # Validate potential
        # ----------------------------------------------------
        for attr in ("V_real", "W"):
            _assert(hasattr(self.potential, attr), f"potential missing required attribute '{attr}'")

        _assert(isinstance(self.potential.V_real, np.ndarray), "potential.V_real must be np.ndarray")
        _assert(isinstance(self.potential.W, np.ndarray), "potential.W must be np.ndarray")
        _assert_shape(self.potential.V_real, (self.grid.Ny, self.grid.Nx), "potential.V_real")
        _assert_shape(self.potential.W, (self.grid.Ny, self.grid.Nx), "potential.W")
        _assert(np.all(np.isfinite(self.potential.V_real)), "potential.V_real contains non-finite values")
        _assert(np.all(np.isfinite(self.potential.W)), "potential.W contains non-finite values")
        _assert(np.all(self.potential.W >= -1e-14), "potential.W contains significantly negative values")

        # ----------------------------------------------------
        # Fourier grids
        # ----------------------------------------------------
        self.kx = 2.0 * np.pi * np.fft.fftfreq(self.grid.Nx, d=self.grid.dx)
        self.ky = 2.0 * np.pi * np.fft.fftfreq(self.grid.Ny, d=self.grid.dy)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky)
        self.K2 = self.KX**2 + self.KY**2

        _assert_shape(self.kx, (self.grid.Nx,), "kx")
        _assert_shape(self.ky, (self.grid.Ny,), "ky")
        _assert_shape(self.KX, (self.grid.Ny, self.grid.Nx), "KX")
        _assert_shape(self.KY, (self.grid.Ny, self.grid.Nx), "KY")
        _assert_shape(self.K2, (self.grid.Ny, self.grid.Nx), "K2")
        _assert(np.all(np.isfinite(self.kx)), "kx contains non-finite values")
        _assert(np.all(np.isfinite(self.ky)), "ky contains non-finite values")
        _assert(np.all(np.isfinite(self.K2)), "K2 contains non-finite values")
        _assert(np.all(self.K2 >= 0.0), "K2 contains negative values")

        # ----------------------------------------------------
        # Forward / adjoint potentials
        # ----------------------------------------------------
        self.V_fwd = self.potential.V_real - 1j * self.potential.W
        self.V_adj = np.conjugate(self.V_fwd)

        _assert_complex_array(self.V_fwd, "V_fwd")
        _assert_complex_array(self.V_adj, "V_adj")
        _assert_same_shape(self.V_fwd, self.V_adj, "V_fwd", "V_adj")

        # sanity: adjoint must flip imaginary sign only
        _assert(
            np.allclose(self.V_adj, self.potential.V_real + 1j * self.potential.W),
            "V_adj must equal conjugate(V_fwd) = V_real + iW",
        )

    # --------------------------------------------------------
    # State initialization
    # --------------------------------------------------------
    def initialize_state(self, state0: np.ndarray) -> np.ndarray:
        _assert(isinstance(state0, np.ndarray), "state0 must be np.ndarray")
        _assert_shape(state0, (self.grid.Ny, self.grid.Nx), "state0")
        _assert(np.all(np.isfinite(state0.real)), "state0.real contains non-finite values")
        _assert(np.all(np.isfinite(state0.imag)), "state0.imag contains non-finite values")

        state0, n0 = normalize_unit(state0, self.grid.dx, self.grid.dy)
        _assert(np.isfinite(float(n0)), f"normalize_unit returned non-finite norm {n0}")

        out = state0.astype(np.complex128)
        _assert_complex_array(out, "initialized state")

        rho = self.density(out)
        prob = float(np.sum(rho) * self.grid.dx * self.grid.dy)
        _assert(np.isfinite(prob), "initialized state probability is non-finite")
        _assert(np.isclose(prob, 1.0, atol=1e-10), f"initialized state norm must be 1, got {prob}")

        return out

    def initialize_click_state(
        self,
        x_click: float,
        y_click: float,
        sigma_click: float,
    ) -> np.ndarray:
        _assert_finite_scalar(x_click, "x_click")
        _assert_finite_scalar(y_click, "y_click")
        _assert_positive_scalar(sigma_click, "sigma_click")

        Xc = self.grid.X - x_click
        Yc = self.grid.Y - y_click

        _assert_shape(Xc, (self.grid.Ny, self.grid.Nx), "Xc")
        _assert_shape(Yc, (self.grid.Ny, self.grid.Nx), "Yc")
        _assert(np.all(np.isfinite(Xc)), "Xc contains non-finite values")
        _assert(np.all(np.isfinite(Yc)), "Yc contains non-finite values")

        phi = np.exp(-(Xc**2 + Yc**2) / (2.0 * sigma_click**2)).astype(np.complex128)
        _assert_complex_array(phi, "raw click state")

        phi, n0 = normalize_unit(phi, self.grid.dx, self.grid.dy)
        _assert(np.isfinite(float(n0)), f"click-state normalize_unit returned non-finite norm {n0}")

        phi = phi.astype(np.complex128)
        _assert_complex_array(phi, "normalized click state")

        rho = self.density(phi)
        prob = float(np.sum(rho) * self.grid.dx * self.grid.dy)
        _assert(np.isfinite(prob), "click-state probability is non-finite")
        _assert(np.isclose(prob, 1.0, atol=1e-10), f"click-state norm must be 1, got {prob}")

        return phi

    # --------------------------------------------------------
    # Phases
    # --------------------------------------------------------
    def kinetic_phase(self, dt: float) -> np.ndarray:
        _assert_finite_scalar(dt, "dt")
        phase = np.exp(-1j * self.K2 * dt / (2.0 * self.m_mass))
        _assert_complex_array(phase, "kinetic_phase")
        _assert_shape(phase, (self.grid.Ny, self.grid.Nx), "kinetic_phase")
        _assert(np.all(np.isfinite(phase.real)), "kinetic_phase.real non-finite")
        _assert(np.all(np.isfinite(phase.imag)), "kinetic_phase.imag non-finite")
        return phase

    def potential_phase(self, V: np.ndarray, dt: float) -> np.ndarray:
        _assert_finite_scalar(dt, "dt")
        _assert_complex_array(V.astype(np.complex128), "V")
        _assert_shape(V, (self.grid.Ny, self.grid.Nx), "V")
        phase = np.exp(-1j * V * dt / self.hbar)
        _assert_complex_array(phase, "potential_phase")
        _assert_shape(phase, (self.grid.Ny, self.grid.Nx), "potential_phase")
        return phase

    # --------------------------------------------------------
    # Core split-step
    # --------------------------------------------------------
    def _step_field(
        self,
        field: np.ndarray,
        K_phase: np.ndarray,
        P_half: np.ndarray,
    ) -> np.ndarray:
        _assert(isinstance(field, np.ndarray), "field must be np.ndarray")
        _assert_shape(field, (self.grid.Ny, self.grid.Nx), "field")
        _assert_shape(K_phase, (self.grid.Ny, self.grid.Nx), "K_phase")
        _assert_shape(P_half, (self.grid.Ny, self.grid.Nx), "P_half")
        _assert(np.all(np.isfinite(field.real)), "field.real contains non-finite values")
        _assert(np.all(np.isfinite(field.imag)), "field.imag contains non-finite values")
        _assert(np.all(np.isfinite(K_phase.real)), "K_phase.real contains non-finite values")
        _assert(np.all(np.isfinite(K_phase.imag)), "K_phase.imag contains non-finite values")
        _assert(np.all(np.isfinite(P_half.real)), "P_half.real contains non-finite values")
        _assert(np.all(np.isfinite(P_half.imag)), "P_half.imag contains non-finite values")

        if not np.iscomplexobj(field):
            field = field.astype(np.complex128)
        else:
            field = field.astype(np.complex128, copy=False)

        field = field * P_half
        _assert_complex_array(field, "field after first half-potential")

        f_k = np.fft.fft2(field)
        _assert_complex_array(f_k, "fft(field)")

        f_k = f_k * K_phase
        _assert_complex_array(f_k, "fft(field)*K_phase")

        field = np.fft.ifft2(f_k)
        _assert_complex_array(field, "ifft(...)")

        field = field * P_half
        _assert_complex_array(field, "field after second half-potential")

        return field

    # --------------------------------------------------------
    # Time evolution
    # --------------------------------------------------------
    def step_forward(self, state: np.ndarray, dt: float) -> TheoryStepResult:
        _assert_finite_scalar(dt, "dt")
        _assert_shape(state, (self.grid.Ny, self.grid.Nx), "state")
        _assert(np.all(np.isfinite(state.real)), "state.real contains non-finite values")
        _assert(np.all(np.isfinite(state.imag)), "state.imag contains non-finite values")

        P_half = self.potential_phase(self.V_fwd, dt / 2.0)
        K_phase = self.kinetic_phase(dt)
        new_state = self._step_field(state, K_phase, P_half)

        _assert_complex_array(new_state, "new_state(step_forward)")
        return TheoryStepResult(state=new_state, aux=None)

    def step_backward_adjoint(self, state: np.ndarray, dt: float) -> TheoryStepResult:
        _assert_finite_scalar(dt, "dt")
        _assert_shape(state, (self.grid.Ny, self.grid.Nx), "state")
        _assert(np.all(np.isfinite(state.real)), "state.real contains non-finite values")
        _assert(np.all(np.isfinite(state.imag)), "state.imag contains non-finite values")

        # Adjoint of forward step:
        # forward:  P(V_fwd, +dt/2) -> K(+dt) -> P(V_fwd, +dt/2)
        # adjoint:  P(V_adj, -dt/2) -> K(-dt) -> P(V_adj, -dt/2)
        P_half = self.potential_phase(self.V_adj, -dt / 2.0)
        K_phase = self.kinetic_phase(-dt)
        new_state = self._step_field(state, K_phase, P_half)

        _assert_complex_array(new_state, "new_state(step_backward_adjoint)")
        return TheoryStepResult(state=new_state, aux=None)

    # --------------------------------------------------------
    # Observables
    # --------------------------------------------------------
    def density(self, state: np.ndarray) -> np.ndarray:
        _assert(isinstance(state, np.ndarray), "state must be np.ndarray")
        _assert(state.ndim == 2, f"state must be 2D, got ndim={state.ndim}")
        _assert(np.all(np.isfinite(state.real)), "state.real contains non-finite values")
        _assert(np.all(np.isfinite(state.imag)), "state.imag contains non-finite values")

        rho = (np.abs(state) ** 2).astype(float)
        _assert_real_array(rho, "rho")
        _assert(np.all(rho >= -1e-14), "rho contains significantly negative values")
        return rho

    def current(self, state_vis: np.ndarray):
        """
        Current density on a 2D scalar field.

        NOTE:
        This uses centered finite differences with np.roll, i.e. periodic wrapping.
        If state_vis is only the visible crop (not the full padded grid), this can
        create wraparound artifacts at the visible-window boundaries.
        """
        _assert(isinstance(state_vis, np.ndarray), "state_vis must be np.ndarray")
        _assert(state_vis.ndim == 2, f"state_vis must be 2D, got ndim={state_vis.ndim}")
        _assert(np.all(np.isfinite(state_vis.real)), "state_vis.real contains non-finite values")
        _assert(np.all(np.isfinite(state_vis.imag)), "state_vis.imag contains non-finite values")

        Nyv, Nxv = state_vis.shape
        _assert(Nyv >= 3 and Nxv >= 3,
                f"state_vis too small for centered differences, got shape={state_vis.shape}")

        dpsi_dx = (
            np.roll(state_vis, -1, axis=1) - np.roll(state_vis, 1, axis=1)
        ) / (2.0 * self.grid.dx)

        dpsi_dy = (
            np.roll(state_vis, -1, axis=0) - np.roll(state_vis, 1, axis=0)
        ) / (2.0 * self.grid.dy)

        _assert_complex_array(dpsi_dx, "dpsi_dx")
        _assert_complex_array(dpsi_dy, "dpsi_dy")

        rho = (np.abs(state_vis) ** 2).astype(float)
        _assert_real_array(rho, "rho(current)")
        _assert(np.all(rho >= -1e-14), "rho(current) contains significantly negative values")

        jx = (
            (self.hbar / self.m_mass)
            * np.imag(np.conjugate(state_vis) * dpsi_dx)
        ).astype(float)

        jy = (
            (self.hbar / self.m_mass)
            * np.imag(np.conjugate(state_vis) * dpsi_dy)
        ).astype(float)

        _assert_real_array(jx, "jx")
        _assert_real_array(jy, "jy")
        _assert_shape(jx, state_vis.shape, "jx")
        _assert_shape(jy, state_vis.shape, "jy")
        _assert_shape(rho, state_vis.shape, "rho(current)")

        return jx, jy, rho