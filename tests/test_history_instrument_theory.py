"""Cross-check the streaming Theory against the independent branch reference."""
import unittest
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from core.grid import build_grid
from core.history_runner import simulate_history, sample_memory_reads, digitize_history_events
from core.simulation_types import PotentialSpec
from theories.history_instrument import HistoryInstrumentTheory
from theories.registry import build_theory
from analysis.spatial_effect_measurement import SpatialEffectExperiment, initial_spatial_state, spatial_effect_potential
from analysis.history_dynamics import HistoryDynamicsConfig, build_history_dynamics_basis, evaluate_history_dynamics
from analysis.temporal_history_profile import TemporalHistoryEnvelope
from analysis.memory_banks import MemoryBankConfig, evaluate_memory_banks


def fixture(**kwargs):
    experiment = SpatialEffectExperiment(nx=8, ny=8, y_bins=4, reference_time=0)
    grid = build_grid(experiment.lx, experiment.ly, 8, 8, 1)
    zeros, mask = np.zeros_like(grid.X), np.zeros_like(grid.X, dtype=bool)
    potential = PotentialSpec(spatial_effect_potential(experiment), zeros,
                             mask, mask, mask, mask, mask)
    theory = HistoryInstrumentTheory(grid, potential, sigma_t=.2, front_time=.4,
        detector_width=experiment.detector_width, y_bins=4, rng_seed=31, **kwargs)
    packet = initial_spatial_state(experiment).reshape(8, 8)
    return experiment, theory, packet


class HistoryInstrumentTests(unittest.TestCase):
    def test_streaming_distribution_matches_exact_branch_ensemble(self):
        exp, theory, packet = fixture()
        steps, dt, count = 12, .1, 12000
        state = theory.initialize_state(np.broadcast_to(packet, (count, 8, 8)))
        for _ in range(steps):
            state = theory.step_forward(state, dt).state
        basis = build_history_dynamics_basis(exp, HistoryDynamicsConfig(duration=steps*dt, steps=steps))
        exact = evaluate_history_dynamics(basis, TemporalHistoryEnvelope(sigma_t=.2), front_time=.4)
        clicks = np.zeros_like(exact.ideal_joint)
        for event in theory.events:
            if event['type'] == 'detection':
                clicks[min(int(event['time']/dt), steps-1), event['y_bin']] += 1/count
        # Binomial sampling bounds on each unconditional outcome, with a small
        # absolute floor for rare categories. No click-conditioned normalization.
        error = np.abs(clicks-exact.ideal_joint)
        self.assertTrue(np.all(error < 6*np.sqrt(exact.ideal_joint*(1-exact.ideal_joint)/count)+2/count))
        self.assertLess(abs(theory.alive.mean()-exact.surviving_probability), .015)
        density = theory.density(state)*theory.grid.dx*theory.grid.dy
        self.assertLess(np.abs(density.ravel()-exact.density).sum(), .035)
        self.assertAlmostEqual(theory.alive.mean()+theory.detected.mean()+theory.escaped.mean(), 1)
        assert_allclose((np.abs(state)**2).sum(axis=(1, 2))*theory.grid.dx*theory.grid.dy, theory.alive, atol=1e-12)

    def test_unitary_null_and_unsupported_backward(self):
        exp, theory, packet = fixture(selection_strength=0, absorption_rate=0)
        from analysis.spatial_effect_measurement import _propagate_states
        state = theory.initialize_state(packet)
        final = theory.step_forward(state, .1).state
        exact = _propagate_states(exp, state.ravel(), .1).reshape(8, 8)
        assert_allclose(final, exact, atol=1e-14)
        self.assertEqual(theory.events, [])
        with self.assertRaises(NotImplementedError):
            theory.step_backward_adjoint(final, .1)

    def test_terminal_trajectory_cannot_be_selected_or_detected_again(self):
        _, theory, packet = fixture(absorption_rate=1e6)
        theory = replace(theory, detector_width=100, front_time=10)
        state = theory.initialize_state(np.broadcast_to(packet, (30, 8, 8)))
        state = theory.step_forward(state, .1).state
        self.assertTrue(np.all(theory.detected))
        events = list(theory.events)
        state = theory.step_forward(state, 20).state
        assert_allclose(state, 0)
        self.assertEqual(theory.events, events)

    def test_cap_loss_is_separate_from_detection(self):
        _, theory, packet = fixture(selection_strength=0, absorption_rate=0)
        potential = replace(theory.potential, W=np.full_like(theory.grid.X, .7))
        theory = replace(theory, potential=potential)
        state = theory.initialize_state(np.broadcast_to(packet, (4000, 8, 8)))
        theory.step_forward(state, .5)
        self.assertFalse(np.any(theory.detected))
        self.assertLess(abs(theory.escaped.mean()-(1-np.exp(-.7))), .025)

    def test_runner_reproducibility_and_readout_independence(self):
        _, theory, packet = fixture()
        a = simulate_history(theory, packet, dt=.1, steps=6, trajectories=48)
        b = simulate_history(theory, packet, dt=.1, steps=6, trajectories=48,
            memory=MemoryBankConfig(0, 0, read_count=4, read_spacing=.2))
        assert_allclose(a.densities, b.densities, atol=0, rtol=0)
        self.assertEqual(a.events, b.events)
        self.assertEqual(a.records, b.records)
        self.assertEqual(len(b.reads), 4)
        self.assertGreater(b.reads[-1]['time'], b.times[-1])
        self.assertEqual(b.reads[-1]['recovered'], 0)
        self.assertTrue(all(abs(d['surviving']+d['detected']+d['escaped']-1) < 1e-12 for d in a.diagnostics))

    def test_example_replay_is_the_actual_recorded_quantum_trajectory(self):
        _, theory, packet = fixture(absorption_rate=20)
        run = simulate_history(theory, packet, dt=.1, steps=12, trajectories=24)
        indices = run.parameters['example_indices']
        state = theory.initialize_state(np.broadcast_to(packet, (24, 8, 8)))
        for k in range(13):
            assert_allclose(run.example_densities[k], np.abs(state[indices])**2, atol=1e-7)
            if k < 12:
                state = theory.step_forward(state, .1).state
        if np.any(theory.detected):
            self.assertTrue(theory.detected[indices[0]])
            assert_allclose(run.example_densities[-1, 0], 0)

    def test_sampled_memories_match_analytic_law_and_never_revive(self):
        records = [dict(trajectory=i, created=0., timestamp=0., dark=False, y_bin=0) for i in range(20000)]
        env = TemporalHistoryEnvelope(retention_time=0, fade_time=1, fade_power=1)
        for mode in ['independent', 'shared']:
            memory = MemoryBankConfig(1, 3, .6, mode, 4, .2, .7)
            reads = sample_memory_reads(records, env, memory, acquisition_end=.5, read_wait=0,
                                        rng=np.random.default_rng(91))
            exact = evaluate_memory_banks([[1]], [0], env, readout_time=.5, config=memory)
            for key, probability in [('last', exact.delayed_joint.sum()), ('logged', exact.any_read_joint.sum()),
                                      ('recovered', exact.either_joint.sum())]:
                self.assertLess(abs(reads[-1][key]/len(records)-probability), .015)
            for a, b in zip(reads, reads[1:]):
                self.assertTrue(set(a['logged_ids']) <= set(b['logged_ids']))
            perfect = sample_memory_reads(records, env, replace(memory, read_efficiency=1),
                acquisition_end=.5, read_wait=0, rng=np.random.default_rng(91))
            for a, b in zip(perfect, perfect[1:]):
                self.assertTrue(set(b['last_ids']) <= set(a['last_ids']))

    def test_registry_and_validation(self):
        _, theory, packet = fixture()
        cfg = SimpleNamespace(THEORY_NAME='history_instrument', m_mass=1., hbar=1., HISTORY_Y_BINS=4)
        self.assertIsInstance(build_theory(cfg, theory.grid, theory.potential), HistoryInstrumentTheory)
        with self.assertRaises(ValueError):
            replace(theory, sigma_t=-1)
        with self.assertRaises(ValueError):
            theory.step_forward(theory.initialize_state(packet), 0)

    def test_digitizer_keeps_creation_time_and_accounts_for_gate_rejection(self):
        from scipy.special import ndtr
        from analysis.spatial_detector_inference import DetectorResponse
        count = 6000
        events = [dict(type='detection', trajectory=i, time=.05 if i % 2 else .95, y_bin=1)
                  for i in range(count)]
        response = DetectorResponse(efficiency=.7, dark_probability=.1, timing_jitter=.2)
        records = digitize_history_events(events, shots=count, edges=np.linspace(0, 1, 11),
            y_bins=4, response=response, rng=np.random.default_rng(251))
        accepted = .7*(ndtr((1-.05)/.2)-ndtr(-.05/.2))
        expected = accepted+(1-accepted)*.1
        self.assertLess(abs(len(records)/count-expected), .025)
        self.assertEqual(len({r['trajectory'] for r in records}), len(records))
        self.assertTrue(all(0 <= r['timestamp'] < 1 for r in records))
        self.assertTrue(all(r['created'] == events[r['trajectory']]['time'] for r in records if not r['dark']))
        self.assertTrue(any(r['timestamp'] != r['created'] for r in records if not r['dark']))

    def test_app_routes_to_history_export_without_legacy_detector_or_backward(self):
        from config import AppConfig
        from main import QuantumSimulationApp
        with self.assertRaisesRegex(ValueError, 'USE_SCREEN_CAP'):
            QuantumSimulationApp(AppConfig(THEORY_NAME='history_instrument', USE_SCREEN_CAP=True)).build_setup()
        with tempfile.TemporaryDirectory() as directory:
            cfg = AppConfig(THEORY_NAME='history_instrument', dt=.1, n_steps=3,
                save_every=1, HISTORY_TRAJECTORIES=8, HISTORY_OUTPUT=str(Path(directory)/'run.html'),
                VISIBLE_LX=10., VISIBLE_LY=8., N_VISIBLE_X=16, N_VISIBLE_Y=16,
                PAD_FACTOR=1, x0=-2.5, k0x=3.)
            cfg.DEBUG_FREE_CASE = True
            app = QuantumSimulationApp(cfg)
            with patch.object(app, 'run_forward', side_effect=AssertionError('legacy path called')), \
                 patch('main.build_detector', side_effect=AssertionError('legacy detector constructed')):
                runs = app.run()
            self.assertEqual(len(runs), 2)
            for suffix in ['.html', '.json', '.npz']:
                self.assertTrue(Path(cfg.HISTORY_OUTPUT).with_suffix(suffix).exists())


if __name__ == '__main__':
    unittest.main()
