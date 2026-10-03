"""Forward quantum trajectories for the single-selection history instrument.

States have shape (Ny,Nx) or (shots,Ny,Nx). Every live trajectory is
conditionally normalized using dx*dy; absorbed trajectories are exactly zero.
The density/current are averages over ALL preparations, including these zeros.
No nonlinear guidance or classical-memory feedback is added to the Hamiltonian.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import log_ndtr, ndtri_exp

from theories.base import TheoryStepResult
from theories.schrodinger import SchrodingerTheory


@dataclass
class HistoryInstrumentTheory(SchrodingerTheory):
    sigma_t: float = .2
    front_time: float = .76
    selection_strength: float = 1.0
    pointer_y: float = 0.0
    absorption_rate: float = 4.0
    detector_x: float = 1.5
    detector_width: float = .3
    y_bins: int = 16
    propagation_step: float = .005
    rng_seed: int = 731

    def __post_init__(self):
        super().__post_init__()
        values = [self.sigma_t, self.front_time, self.selection_strength, self.pointer_y,
                  self.absorption_rate, self.detector_x, self.detector_width, self.propagation_step]
        if (not np.all(np.isfinite(values)) or min(self.sigma_t, self.detector_width, self.propagation_step) <= 0
                or min(self.selection_strength, self.absorption_rate) < 0):
            raise ValueError('Invalid history instrument scales')
        if (isinstance(self.y_bins, bool) or not isinstance(self.y_bins, (int, np.integer))
                or not 1 <= self.y_bins <= self.grid.Ny):
            raise ValueError('y_bins must be an integer between 1 and Ny')
        if isinstance(self.rng_seed, bool) or not isinstance(self.rng_seed, (int, np.integer)) or self.rng_seed < 0:
            raise ValueError('rng_seed must be a nonnegative integer')
        self._upper = self.grid.Y >= self.pointer_y
        if not np.any(self._upper) or np.all(self._upper):
            raise ValueError('The pointer must partition the grid')
        self._detector_rate = self.absorption_rate*np.exp(
            -.5*((self.grid.X-self.detector_x)/self.detector_width)**2)
        # W is an amplitude absorber. Its probability-loss rate is 2W/hbar.
        self._escape_rate = 2*np.maximum(self.potential.W, 0)/self.hbar
        self._row_bins = np.empty(self.grid.Ny, dtype=int)
        for j, rows in enumerate(np.array_split(np.arange(self.grid.Ny), self.y_bins)):
            self._row_bins[rows] = j
        self._phases = {}
        self._initialized = False

    def initialize_state(self, state0):
        state = np.asarray(state0, dtype=complex).copy()
        if state.ndim not in (2, 3) or state.shape[-2:] != self.grid.X.shape or not np.all(np.isfinite(state)):
            raise ValueError('Expected finite scalar wavefunction(s) on the theory grid')
        norm = np.sum(np.abs(state)**2, axis=(-2, -1))*self.grid.dx*self.grid.dy
        if np.any(norm <= 0):
            raise ValueError('Every preparation must have nonzero norm')
        state /= np.sqrt(norm)[..., None, None]
        self._shape = state.shape
        self.shots = 1 if state.ndim == 2 else state.shape[0]
        if self.shots < 1:
            raise ValueError('At least one preparation is required')
        clock_seed, jump_seed = np.random.SeedSequence(self.rng_seed).spawn(2)
        clock_rng = np.random.default_rng(clock_seed)
        self._rng = np.random.default_rng(jump_seed)
        q = -np.expm1(-self.selection_strength)
        fires = clock_rng.random(self.shots) < q
        log_tail = log_ndtr(self.front_time/self.sigma_t)
        u = np.maximum(clock_rng.random(self.shots), np.finfo(float).tiny)
        times = self.front_time-self.sigma_t*ndtri_exp(log_tail+np.log(u))
        self.selection_times = np.where(fires, np.maximum(times, 0), np.inf)
        self.selected_half = np.full(self.shots, -1, int)
        self.alive = np.ones(self.shots, bool)
        self.detected = np.zeros(self.shots, bool)
        self.escaped = np.zeros(self.shots, bool)
        self.events = []
        self.time = 0.0
        self._initialized = True
        return state

    def _unitary(self, states, duration):
        full = int(np.floor(duration/self.propagation_step+1e-12))
        remainder = duration-full*self.propagation_step
        for step in [self.propagation_step]*full + ([remainder] if remainder > 1e-14 else []):
            key = round(step, 15)
            if key not in self._phases:
                self._phases[key] = (
                    np.exp(-.5j*self.potential.V_real*step/self.hbar),
                    np.exp(-.5j*self.hbar*self.K2*step/self.m_mass))
            P, K = self._phases[key]
            states = P*np.fft.ifft2(K*np.fft.fft2(P*states, axes=(-2, -1)), axes=(-2, -1))
        return states

    def step_forward(self, state, dt):
        if not self._initialized:
            raise RuntimeError('initialize_state must be called before stepping')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        state = np.asarray(state, dtype=complex)
        if state.shape != self._shape or not np.all(np.isfinite(state)):
            raise ValueError('State shape or values changed unexpectedly')
        batch = state.reshape(self.shots, self.grid.Ny, self.grid.Nx).copy()
        area = self.grid.dx*self.grid.dy
        norms = np.sum(np.abs(batch)**2, axis=(1, 2))*area
        if not np.allclose(norms, self.alive.astype(float), atol=1e-9, rtol=0):
            raise ValueError('Live trajectories must have unit norm and terminal trajectories zero norm')
        indices = np.flatnonzero(self.alive)
        events = []
        midpoint = self.time+dt/2
        if indices.size:
            live = self._unitary(batch[indices], dt/2)
            due = (self.selected_half[indices] < 0) & (self.selection_times[indices] < self.time+dt)
            for local in np.flatnonzero(due):
                trajectory = int(indices[local])
                p_upper = float(np.sum(np.abs(live[local][self._upper])**2)*area)
                upper = self._rng.random() < np.clip(p_upper, 0, 1)
                mask = self._upper if upper else ~self._upper
                live[local] *= mask
                live[local] /= np.sqrt(np.sum(np.abs(live[local])**2)*area)
                self.selected_half[trajectory] = 1 if upper else 0
                events.append(dict(type='selection', trajectory=trajectory, time=midpoint,
                                   half='upper' if upper else 'lower'))
            live = self._unitary(live, dt/2)
            total_rate = self._detector_rate+self._escape_rate
            absorption = -np.expm1(-total_rate*dt)
            density = np.abs(live)**2*area
            jump = density*absorption
            jump_mass = jump.sum(axis=(1, 2))
            jumps = self._rng.random(indices.size) < np.clip(jump_mass, 0, 1)
            for local in np.flatnonzero(jumps):
                trajectory = int(indices[local])
                probabilities = jump[local].ravel()
                site = min(int(np.searchsorted(np.cumsum(probabilities), self._rng.random()*probabilities.sum(), side='right')), probabilities.size-1)
                row, col = np.unravel_index(site, self.grid.X.shape)
                detected = self._rng.random() < self._detector_rate[row, col]/total_rate[row, col]
                self.alive[trajectory] = False
                self.detected[trajectory] = detected
                self.escaped[trajectory] = not detected
                events.append(dict(type='detection' if detected else 'escape', trajectory=trajectory,
                    time=midpoint, x=float(self.grid.X[row, col]), y=float(self.grid.Y[row, col]),
                    y_bin=int(self._row_bins[row])))
            live *= np.exp(-total_rate*dt/2)
            live[jumps] = 0
            remaining = ~jumps
            if np.any(remaining):
                live[remaining] /= np.sqrt(np.sum(np.abs(live[remaining])**2, axis=(1, 2))*area)[:, None, None]
            batch[indices] = live
        self.time += dt
        self.events.extend(events)
        return TheoryStepResult(batch.reshape(self._shape), dict(
            time=self.time, events=events, surviving=float(self.alive.mean()),
            detected=float(self.detected.mean()), escaped=float(self.escaped.mean()),
            selected=float(np.mean(self.selected_half >= 0))))

    def density(self, state):
        rho = np.abs(state)**2
        return rho if rho.ndim == 2 else rho.mean(axis=0)

    def current(self, state_vis):
        state = np.asarray(state_vis)
        dx = (np.roll(state, -1, axis=-1)-np.roll(state, 1, axis=-1))/(2*self.grid.dx)
        dy = (np.roll(state, -1, axis=-2)-np.roll(state, 1, axis=-2))/(2*self.grid.dy)
        jx = self.hbar/self.m_mass*np.imag(state.conj()*dx)
        jy = self.hbar/self.m_mass*np.imag(state.conj()*dy)
        if state.ndim == 3:
            jx, jy = jx.mean(axis=0), jy.mean(axis=0)
        return jx, jy, self.density(state)

    def step_backward_adjoint(self, state, dt):
        raise NotImplementedError('History trajectories require a register-aware adjoint instrument; use the forward history runner')

    def initialize_click_state(self, x_click, y_click, sigma_click):
        raise NotImplementedError('History detection is generated by absorbing jumps, not an imposed terminal click')
