import unittest
from types import SimpleNamespace

import numpy as np

from config import AppConfig
from core.grid import build_grid
from core.potentials import build_potential
from main import ClickResolution, ForwardRunResult, QuantumSimulationApp, SimulationSetup
from analysis.ridge import compute_detector_anchored_ridge, contiguous_finite_track_slice
from theories.schrodinger import SchrodingerTheory


class _Theory:
    def channel_densities(self, state):
        # All joint weight is in +-, retaining a genuine joint outcome rather
        # than constructing it from separately sampled marginals.
        z = np.zeros(state.shape[:2], dtype=float)
        d = np.sum(np.abs(state) ** 2, axis=(-2, -1))
        return {"++": z, "+-": d, "-+": z, "--": z}


class ConditionalClickTests(unittest.TestCase):
    def test_detector_anchored_track_is_local_ordered_and_obstacle_safe(self):
        nt, ny, nx = 6, 9, 9
        density = np.zeros((nt, ny, nx), dtype=float)
        # Chronological path bends around, rather than through, the obstacle.
        path = [(4, 1), (4, 2), (3, 3), (2, 4), (2, 5), (2, 6)]
        for i, (iy, ix) in enumerate(path):
            density[i, iy, ix] = 1.0
        blocked = np.zeros((ny, nx), dtype=bool)
        blocked[4, 3:6] = True
        current_x = np.ones_like(density)
        current_y = np.zeros_like(density)
        axis = np.arange(nx, dtype=float)

        rx, ry, _, diag = compute_detector_anchored_ridge(
            density,
            axis,
            axis,
            click_frame_idx=nt - 1,
            click_x=6.0,
            click_y=2.0,
            forbidden_mask=blocked,
            current_x_frames=current_x,
            current_y_frames=current_y,
            radius_px=2,
        )
        self.assertEqual(diag["endpoint_distance_px"], 0.0)
        self.assertLessEqual(diag["max_jump_px"], 2.0)
        self.assertEqual(diag["obstacle_intersections"], 0)
        self.assertTrue(diag["times_monotonic"])
        finite = np.isfinite(rx) & np.isfinite(ry)
        self.assertFalse(np.any(blocked[ry[finite].astype(int), rx[finite].astype(int)]))

    def test_terminated_tracks_are_not_concatenated(self):
        xs = np.array([0.0, 1.0, np.nan, 7.0, 8.0])
        ys = np.array([0.0, 0.0, np.nan, 1.0, 1.0])
        segment = contiguous_finite_track_slice(xs, ys, 4)
        self.assertEqual((segment.start, segment.stop), (3, 5))

    def _potential(self, *, obstacle=False, absorption=0.0):
        cfg = AppConfig(
            N_VISIBLE_X=16,
            N_VISIBLE_Y=8,
            PAD_FACTOR=1,
            barrier_thickness=3.0,
            USE_SIMPLE_BARRIER=obstacle,
            simple_barrier_center_x=2.0,
            simple_barrier_center_y=2.0,
            simple_barrier_half_height=1.2,
            simple_barrier_absorption=absorption,
        )
        grid = build_grid(
            visible_lx=cfg.VISIBLE_LX,
            visible_ly=cfg.VISIBLE_LY,
            n_visible_x=cfg.N_VISIBLE_X,
            n_visible_y=cfg.N_VISIBLE_Y,
            pad_factor=cfg.PAD_FACTOR,
        )
        return cfg, grid, build_potential(grid, cfg)

    def test_verification_geometries_and_shared_adjoint(self):
        # A/D: standard baseline geometry, usable with TRF or Schrödinger.
        _, _, no_obstacle = self._potential(obstacle=False)
        self.assertNotIn("simple_barrier", [c.name for c in no_obstacle.components])

        # B: finite obstacle permits diffraction and has no absorbing part.
        cfg_b, grid_b, finite = self._potential(obstacle=True, absorption=0.0)
        self.assertIn("simple_barrier", [c.name for c in finite.components])
        finite_simple = next(c for c in finite.components if c.name == "simple_barrier")
        self.assertAlmostEqual(float(np.max(finite_simple.W)), 0.0)

        # C: the same full-channel geometry can be made absorbing.
        _, _, absorbing = self._potential(obstacle=True, absorption=40.0)
        absorbing_simple = next(c for c in absorbing.components if c.name == "simple_barrier")
        self.assertGreater(float(np.max(absorbing_simple.W)), 0.0)

        # The actual forward potential and its adjoint share every component.
        theory = SchrodingerTheory(grid_b, finite, cfg_b.m_mass, cfg_b.hbar)
        self.assertTrue(np.allclose(theory.V_adj, np.conjugate(theory.V_fwd)))
        rng = np.random.default_rng(3)
        a = rng.normal(size=(grid_b.Ny, grid_b.Nx)) + 1j * rng.normal(size=(grid_b.Ny, grid_b.Nx))
        b = rng.normal(size=(grid_b.Ny, grid_b.Nx)) + 1j * rng.normal(size=(grid_b.Ny, grid_b.Nx))
        ub = theory.step_forward(b, 1e-3).state
        udag_a = theory.step_backward_adjoint(a, 1e-3).state
        self.assertTrue(np.allclose(np.vdot(a, ub), np.vdot(udag_a, b), rtol=1e-10, atol=1e-10))

    def test_closed_upper_region_cannot_be_sampled(self):
        cfg = AppConfig(
            CLICK_RNG_SEED=7,
            TRF_SIGMA_T=0.25,
            K_JITTER=1,
            save_every=1,
            dt=1.0,
        )
        x = np.array([[10.0, 10.0], [10.0, 10.0]])
        y = np.array([[1.0, 1.0], [-1.0, -1.0]])
        grid = SimpleNamespace(X_vis=x, Y_vis=y, dx=1.0, dy=1.0)
        potential = SimpleNamespace(screen_mask_vis=np.ones((2, 2), dtype=bool))
        theory = _Theory()
        setup = SimulationSetup(cfg, grid, potential, theory, None, False)

        state = np.zeros((2, 2, 2, 2), dtype=np.complex128)
        state[1, :, 0, 1] = 1.0  # lower detector only
        frames = np.stack([state, state])
        forward = ForwardRunResult(
            frames_density=np.sum(np.abs(frames) ** 2, axis=(-2, -1)),
            visible_intensity_frames=None,
            latent_intensity_frames=None,
            state_vis_frames=frames,
            posthoc_gamma_like_frames=None,
            times=np.array([0.0, 1.0]),
            norms=np.ones(2),
            detector_diags=[],
            detector_clicked=False,
            det_result_final=None,
            actual_last_step=1,
            batch_runtime=SimpleNamespace(),
        )
        click = ClickResolution(1, 1.0, 10.0, -1.0, np.ones(2), False, "+-")
        # An effect field with zero upper support models a fully closed upper
        # channel after obstacle-aware backward propagation.
        phi = np.zeros_like(frames)
        phi[:, 1, :, 0, 1] = 1.0

        result = QuantumSimulationApp(cfg).refine_click_from_conditional_trf(
            setup, forward, click, phi
        )
        self.assertEqual(result.coincidence_channel, "+-")
        self.assertLess(result.y_click, 0.0)
        self.assertAlmostEqual(result.conditional_upper_probability, 0.0)
        self.assertAlmostEqual(result.conditional_lower_probability, 1.0)
        self.assertGreater(result.conditional_bin_probability, 0.0)


if __name__ == "__main__":
    unittest.main()
