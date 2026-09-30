"""Probability, back-action, erasure, and convergence of the history instrument."""
import unittest
from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose

from analysis.history_dynamics import (
    HistoryDynamicsConfig, build_history_dynamics_basis, evaluate_history_dynamics,
    read_history_detector, detector_step_diagonals, sample_history_wavefunction,
)
from analysis.temporal_history_profile import (
    TemporalHistoryEnvelope, selection_time_weights, record_survival,
)
from analysis.spatial_effect_measurement import (
    SpatialEffectExperiment, _unitary_matrix, initial_spatial_state, _coordinates,
)
from analysis.spatial_detector_inference import DetectorResponse


class HistoryDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = SpatialEffectExperiment(nx=8, ny=8, y_bins=4, reference_time=0)
        cls.config = HistoryDynamicsConfig(duration=1.2, steps=12)
        cls.basis = build_history_dynamics_basis(cls.experiment, cls.config)
        cls.envelope = TemporalHistoryEnvelope(sigma_t=.2, retention_time=.1, fade_time=.4)

    def test_clock_retains_beyond_window_mass_and_null(self):
        for center in [-10, 0, .6, 10]:
            weights = selection_time_weights(self.basis.edges, self.envelope, front_time=center)
            self.assertGreaterEqual(weights.min(), 0)
            self.assertAlmostEqual(weights.sum(), 1)
        late = selection_time_weights(self.basis.edges, self.envelope, front_time=10)
        self.assertAlmostEqual(late[-1], 1)
        off = selection_time_weights(self.basis.edges, self.envelope, front_time=.6, strength=0)
        assert_allclose(off, np.r_[np.zeros(12), 1])

    def test_detector_kraus_completeness(self):
        no, effects = detector_step_diagonals(self.experiment, self.config, .1)
        assert_allclose(no**2+effects.sum(axis=0), 1, atol=1e-15)
        self.assertTrue(np.all(effects >= 0))
        totals = self.basis.clicks.sum(axis=(1, 2)) + (np.abs(self.basis.final_kets)**2).sum(axis=(1, 2))
        assert_allclose(totals, 1, atol=1e-13)

    def test_full_density_matches_independent_one_step_operator_calculation(self):
        config = replace(self.config, steps=1)
        basis = build_history_dynamics_basis(self.experiment, config)
        result = evaluate_history_dynamics(basis, self.envelope, front_time=.6)
        U = _unitary_matrix(self.experiment, .6)
        no, click = detector_step_diagonals(self.experiment, config, 1.2)
        psi = initial_spatial_state(self.experiment)
        _, y = _coordinates(self.experiment)
        P = np.diag((y.ravel() >= 0).astype(float))
        p = result.weights[0]
        rho_mid = np.outer(U@psi, (U@psi).conj())
        rho_selected = (1-p)*rho_mid + p*(P@rho_mid@P+(np.eye(64)-P)@rho_mid@(np.eye(64)-P))
        rho_pre = U@rho_selected@U.conj().T
        rho_final = no[:, None]*rho_pre*no[None, :]
        ensemble = np.einsum('c,cbx,cby->xy', result.weights, basis.final_kets, basis.final_kets.conj())
        assert_allclose(ensemble, rho_final, atol=1e-14)
        assert_allclose(result.ideal_joint[0], click@rho_pre.diagonal().real, atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(ensemble).min(), -1e-14)

    def test_selection_changes_coherences_and_detector_statistics(self):
        on = evaluate_history_dynamics(self.basis, self.envelope, front_time=.2, strength=5)
        off = evaluate_history_dynamics(self.basis, self.envelope, front_time=.2, strength=0)
        self.assertGreater(np.abs(on.density-off.density).sum(), .01)
        self.assertGreater(np.abs(on.ideal_joint-off.ideal_joint).sum(), .001)
        self.assertAlmostEqual(on.surviving_probability+on.detected_probability, 1)

    def test_record_loss_never_reopens_or_renormalizes_wavefunction(self):
        fast = self.envelope
        slow = replace(fast, fade_time=40)
        a = evaluate_history_dynamics(self.basis, fast, front_time=.4)
        b = evaluate_history_dynamics(self.basis, slow, front_time=.4)
        assert_allclose(a.density, b.density, atol=0)
        assert_allclose(a.ideal_joint, b.ideal_joint, atol=0)
        self.assertLess(a.accessible_selection, b.accessible_selection)
        self.assertAlmostEqual(a.accessible_selection+a.erased_selection, a.selected_probability)
        response = DetectorResponse(timing_jitter=.02)
        fa = read_history_detector(self.basis, a, fast, response=response)
        fb = read_history_detector(self.basis, b, slow, response=response)
        self.assertLess(fa.joint.sum(), fb.joint.sum())
        assert_allclose(fa.before_erasure_joint, fb.before_erasure_joint)
        for law in [fa, fb]:
            self.assertAlmostEqual(law.probabilities.sum(), 1)
            self.assertGreaterEqual(law.probabilities.min(), 0)

    def test_future_or_negative_memory_age_is_rejected(self):
        with self.assertRaises(ValueError):
            record_survival(-1, self.envelope)
        with self.assertRaises(ValueError):
            evaluate_history_dynamics(self.basis, self.envelope, front_time=.4, readout_time=.5)

    def test_no_selection_no_detector_is_unitary_schrodinger_limit(self):
        basis = build_history_dynamics_basis(self.experiment, replace(self.config, absorption_rate=0))
        result = evaluate_history_dynamics(basis, self.envelope, front_time=.6, strength=0)
        reference = _unitary_matrix(self.experiment, self.config.duration) @ initial_spatial_state(self.experiment)
        assert_allclose(result.density, np.abs(reference)**2, atol=1e-13)
        self.assertEqual(result.detected_probability, 0)

    def test_absorbed_particle_cannot_later_make_a_selection_record(self):
        basis = build_history_dynamics_basis(self.experiment, replace(self.config, absorption_rate=40))
        result = evaluate_history_dynamics(basis, self.envelope, front_time=.9, strength=4)
        self.assertLess(result.selected_probability, result.weights[:-1].sum())

    def test_trajectory_sampling_matches_survival_and_keeps_selected_branch(self):
        result = evaluate_history_dynamics(self.basis, self.envelope, front_time=.4, strength=2)
        rng = np.random.default_rng(148)
        mass = np.zeros_like(result.density)
        n = 12000
        survived = 0
        for _ in range(n):
            draw = sample_history_wavefunction(self.basis, result, rng)
            if draw is not None:
                case, branch, psi = draw
                self.assertAlmostEqual(np.linalg.norm(psi), 1)
                self.assertEqual(branch is None, case == self.config.steps)
                mass += np.abs(psi)**2/n
                survived += 1
        self.assertAlmostEqual(survived/n, result.surviving_probability, delta=.018)
        self.assertLess(np.abs(mass-result.density).sum(), .035)

    def test_time_refinement_converges_on_shared_detection_bins(self):
        experiment = replace(self.experiment, nx=12, ny=12)
        laws = []
        for steps in [24, 48, 96]:
            basis = build_history_dynamics_basis(experiment, replace(self.config, steps=steps))
            result = evaluate_history_dynamics(basis, self.envelope, front_time=.6)
            binned = result.ideal_joint.reshape(24, steps//24, 4).sum(axis=1)
            laws.append(np.r_[binned.ravel(), result.surviving_probability])
        coarse_error = np.abs(laws[0]-laws[2]).sum()/2
        fine_error = np.abs(laws[1]-laws[2]).sum()/2
        self.assertLess(fine_error, .6*coarse_error)


if __name__ == '__main__':
    unittest.main()
