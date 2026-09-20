"""Operator-level tests for the compact spatial detector instrument."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from analysis.spatial_effect_measurement import (
    SpatialEffectExperiment,
    build_spatial_effect_instrument,
    fit_spatial_effect_joint_response,
    fit_spatial_effect_response,
    initial_spatial_state,
    run_spatial_effect_measurement,
    spatial_effect_evolution,
    spatial_effect_fisher_information,
    spatial_effect_convergence,
    temporal_delays_and_weights,
    terminal_detector_effects,
)


class SpatialEffectMeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = SpatialEffectExperiment()
        cls.small = SpatialEffectExperiment(
            nx=6, ny=4, y_bins=2, delay_step=0.2, horizon_sigmas=2)

    def test_terminal_detector_is_positive_and_complete_with_no_click(self):
        labels, effects = terminal_detector_effects(self.small)
        self.assertEqual(labels, ("y_bin_0", "y_bin_1", "no_click"))
        assert_allclose(
            np.sum(effects, axis=0), np.eye(self.small.nx * self.small.ny),
            atol=1e-14)
        for effect in effects:
            self.assertGreaterEqual(np.linalg.eigvalsh(effect).min(), -1e-14)

    def test_temporally_mixed_full_effects_are_physical_and_phase_invariant(self):
        delays, weights = temporal_delays_and_weights(0.4, 0.2, 2)
        self.assertAlmostEqual(float(np.sum(weights)), 1)
        reference = build_spatial_effect_instrument(self.small, 0.4)
        shifted = build_spatial_effect_instrument(
            self.small, 0.4,
            component_phases=np.linspace(0.2, 2.4, delays.size))
        assert_allclose(reference.effects, shifted.effects, atol=2e-15)
        assert_allclose(
            np.sum(reference.effects, axis=0),
            np.eye(self.small.nx * self.small.ny), atol=1e-14)
        for effect in reference.effects:
            self.assertGreaterEqual(np.linalg.eigvalsh(effect).min(), -1e-13)
        # The backward effects retain nonlocal complex coherences.
        off_diagonal = reference.effects.copy()
        indices = np.arange(off_diagonal.shape[-1])
        off_diagonal[:, indices, indices] = 0
        self.assertGreater(np.max(np.abs(off_diagonal)), 1e-5)
        self.assertGreater(np.max(np.abs(off_diagonal.imag)), 1e-5)

    def test_fast_complete_law_matches_full_operator_evaluation(self):
        instrument = build_spatial_effect_instrument(self.small, 0.4)
        state = initial_spatial_state(self.small)
        operator_probabilities = np.einsum(
            "i,rij,j->r", state.conj(), instrument.effects, state).real
        run = run_spatial_effect_measurement(self.small, 0.4)
        assert_allclose(run.probabilities, operator_probabilities, atol=2e-15)
        self.assertAlmostEqual(float(np.sum(run.probabilities)), 1)
        self.assertAlmostEqual(
            float(np.sum(run.conditional_y_distribution)), 1)
        self.assertAlmostEqual(
            run.click_probability, float(np.sum(run.probabilities[:-1])))

    def test_lambda_zero_is_an_exact_sigma_independent_null(self):
        narrow = run_spatial_effect_measurement(
            self.experiment, 0.2, lambda_strength=0)
        wide = run_spatial_effect_measurement(
            self.experiment, 1.2, lambda_strength=0)
        self.assertTrue(np.array_equal(narrow.probabilities, wide.probabilities))
        narrow_instrument = build_spatial_effect_instrument(
            self.small, 0.2, lambda_strength=0)
        wide_instrument = build_spatial_effect_instrument(
            self.small, 1.2, lambda_strength=0)
        self.assertTrue(np.array_equal(
            narrow_instrument.effects, wide_instrument.effects))

    def test_visualization_frames_reproduce_complete_measurement(self):
        evolution = spatial_effect_evolution(self.experiment, 0.6)
        run = run_spatial_effect_measurement(self.experiment, 0.6)
        self.assertEqual(
            evolution.densities.shape,
            (evolution.delays.size, self.experiment.ny, self.experiment.nx))
        assert_allclose(
            np.sum(evolution.densities, axis=(1, 2)), 1, atol=2e-15)
        assert_allclose(
            np.sum(evolution.cumulative_probabilities, axis=1), 1,
            atol=2e-15)
        assert_allclose(
            evolution.cumulative_probabilities[-1], run.probabilities,
            atol=2e-15)

    def test_complete_law_recovers_injected_width(self):
        profile = fit_spatial_effect_response(
            self.experiment, 0.6, [0.3, 0.45, 0.6, 0.8, 1.0])
        self.assertEqual(profile.best_sigma_t, 0.6)
        self.assertEqual(profile.kl_divergence[2], 0)
        self.assertGreater(profile.kl_divergence[[0, 1, 3, 4]].min(), 0)
        self.assertGreater(profile.signal_total_variation, 0.05)
        self.assertEqual(profile.lambda_zero_max_error, 0)

    def test_joint_profile_recovers_injection_and_exposes_degeneracy(self):
        sigmas = [0.3, 0.45, 0.6, 0.8, 1.0]
        lambdas = [0, 0.5, 1.0, 1.5, 2.0]
        profile = fit_spatial_effect_joint_response(
            self.experiment, 0.6, 1.0, sigmas, lambdas)
        self.assertEqual(profile.expected_deviance.shape, (5, 5))
        self.assertEqual(profile.best_sigma_t, 0.6)
        self.assertEqual(profile.best_lambda_strength, 1.0)
        self.assertEqual(profile.expected_deviance[2, 2], 0)
        # At lambda=0 every sigma gives the same exact null law.
        assert_allclose(
            profile.expected_deviance[0],
            np.full(5, profile.expected_deviance[0, 0]), atol=1e-10)
        # The profile ridge trades a larger width for a smaller coupling.
        self.assertGreater(
            profile.profiled_lambda_strength[0],
            profile.profiled_lambda_strength[-1])
        self.assertGreater(profile.fisher_condition_number, 100)
        self.assertGreater(profile.fisher_score_correlation, 0.9)
        self.assertLess(profile.local_parameter_correlation, -0.9)
        self.assertTrue(np.all(profile.local_standard_errors > 0))
        self.assertGreaterEqual(profile.fisher_eigenvalues[0], 0)

    def test_quantum_null_has_no_local_sigma_information(self):
        fisher = spatial_effect_fisher_information(
            self.experiment, 0.6, 0, shots=100_000)
        self.assertEqual(fisher[0, 0], 0)
        self.assertEqual(fisher[0, 1], 0)
        self.assertGreater(fisher[1, 1], 0)

    def test_delay_grid_spatial_grid_and_horizon_converge(self):
        convergence = spatial_effect_convergence(self.experiment, 0.6)
        self.assertLess(convergence["delay_step_half"], 5e-5)
        self.assertLess(convergence["grid_5_over_4"], 0.002)
        self.assertLess(convergence["horizon_plus_one_sigma"], 5e-6)

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            SpatialEffectExperiment(y_bins=17)
        with self.assertRaises(ValueError):
            run_spatial_effect_measurement(self.experiment, 0)
        with self.assertRaises(ValueError):
            run_spatial_effect_measurement(
                self.experiment, 0.6, lambda_strength=-1)
        with self.assertRaises(ValueError):
            fit_spatial_effect_response(self.experiment, 0.6, [0.6])
        with self.assertRaises(ValueError):
            fit_spatial_effect_joint_response(
                self.experiment, 0.6, 1, [0.4, 0.6], [1])


if __name__ == "__main__":
    unittest.main()
