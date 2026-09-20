"""End-to-end observable response of the spatial measurement-guided model."""

import unittest
from dataclasses import replace

import numpy as np

from analysis.spatial_response import (
    SpatialResponseExperiment,
    fit_cross_grid_response,
    fit_spatial_response,
    fixed_normalization_scales,
    run_spatial_detector,
    spatial_convergence,
)


class SpatialResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = SpatialResponseExperiment()
        cls.profile = fit_spatial_response(
            cls.experiment, 0.1, [0.06, 0.1, 0.15], shots=100_000)

    def test_exact_distribution_recovers_injected_sigma(self):
        self.assertEqual(self.profile.best_sigma_t, 0.1)
        self.assertAlmostEqual(
            float(np.sum(self.profile.target_complete_distribution)), 1.0)
        self.assertAlmostEqual(
            float(np.sum(self.profile.target_y_distribution)), 1.0)
        self.assertGreater(self.profile.detector_mass, 0)
        self.assertGreater(self.profile.baseline_total_variation, 1e-4)
        self.assertEqual(self.profile.kl_divergence[1], 0)
        self.assertGreater(self.profile.kl_divergence[[0, 2]].min(), 0)

    def test_zero_response_is_exact_worldline_null(self):
        self.assertEqual(self.profile.null_state_max_error, 0)
        self.assertEqual(self.profile.null_distribution_max_error, 0)

    def test_dt_grid_and_horizon_convergence_are_reported(self):
        convergence = spatial_convergence(self.experiment, 0.1)
        self.assertLess(convergence["dt_half"], 0.005)
        self.assertLess(convergence["horizon_plus_one_sigma"], 0.005)
        self.assertLess(convergence["grid_5_over_4"], 0.05)
        self.assertLess(convergence["dt_half_excess"], 0.001)
        self.assertLess(convergence["horizon_plus_one_sigma_excess"], 0.001)
        self.assertLess(convergence["grid_5_over_4_excess"], 0.001)

    def test_cross_grid_excess_response_recovers_width(self):
        recovery = fit_cross_grid_response(
            self.experiment, 0.1, [0.08, 0.1, 0.15])
        self.assertEqual(recovery.reference_grid, (80, 60))
        self.assertEqual(recovery.fit_grid, (64, 48))
        self.assertEqual(recovery.best_sigma_t, 0.1)

    def test_fixed_and_absent_normalizations_are_complete(self):
        fixed_experiment = replace(
            self.experiment, normalization_mode="fixed")
        scales = fixed_normalization_scales(fixed_experiment)
        self.assertGreater(scales.effect, 0)
        self.assertGreater(scales.overlap, 0)
        for mode in ["fixed", "none"]:
            run = run_spatial_detector(
                replace(self.experiment, normalization_mode=mode), 0.1)
            self.assertAlmostEqual(float(np.sum(run.complete_distribution)), 1)
            self.assertGreater(run.detector_mass, 0)

    def test_invalid_design_and_profile_are_rejected(self):
        with self.assertRaises(ValueError):
            SpatialResponseExperiment(dt=0)
        with self.assertRaises(ValueError):
            SpatialResponseExperiment(normalization_mode="unknown")
        with self.assertRaises(ValueError):
            fit_spatial_response(self.experiment, 0.1, [0.1])


if __name__ == "__main__":
    unittest.main()
