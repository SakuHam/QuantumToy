"""Tests for resolved and coherent temporal-register benchmarks."""

import unittest
from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose

from analysis.spatial_detector_inference import (
    DetectorResponse,
    realistic_spatial_probabilities,
)
from analysis.spatial_effect_measurement import double_slit_effect_experiment
from analysis.spatial_temporal_observation import (
    fixed_half_gaussian_weights,
    fixed_temporal_delays,
    temporal_observation_fisher_information,
    temporal_observation_probabilities,
)


class SpatialTemporalObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = replace(
            double_slit_effect_experiment(nx=8, ny=8),
            y_bins=4, reference_time=0.3, delay_step=0.1,
            propagation_step=0.02, horizon_sigmas=2)
        cls.response = DetectorResponse(0.88, 0.012, 0.45, 0.05)
        cls.horizon = 0.4

    def test_fixed_weights_are_normalized_on_one_outcome_grid(self):
        delays = fixed_temporal_delays(self.experiment, self.horizon)
        self.assertEqual(delays.shape, (5,))
        narrow = fixed_half_gaussian_weights(0.15, delays)
        wide = fixed_half_gaussian_weights(0.25, delays)
        self.assertAlmostEqual(float(np.sum(narrow)), 1)
        self.assertAlmostEqual(float(np.sum(wide)), 1)
        self.assertGreater(wide[-1], narrow[-1])

    def test_stronger_observations_are_complete_and_share_y_marginal(self):
        integrated = temporal_observation_probabilities(
            self.experiment, 0.2, 1, self.response, strategy="integrated",
            delay_horizon=self.horizon)
        assert_allclose(
            integrated,
            realistic_spatial_probabilities(
                self.experiment, 0.2, 1, self.response), atol=2e-14)
        for strategy, phase, category_blur in (
                ("resolved", 0, 0), ("resolved", 0, 2.0),
                ("coherent", 0.07, 0)):
            law = temporal_observation_probabilities(
                self.experiment, 0.2, 1, self.response, strategy=strategy,
                delay_horizon=self.horizon, phase=phase,
                category_blur_sigma_bins=category_blur)
            self.assertAlmostEqual(float(np.sum(law)), 1)
            self.assertGreaterEqual(float(np.min(law)), 0)
            marginal = np.concatenate([
                law[:-1].reshape(-1, self.experiment.y_bins).sum(axis=0),
                law[-1:],
            ])
            assert_allclose(marginal, integrated, atol=2e-14)
            if strategy == "resolved":
                self.assertEqual(
                    law.size,
                    fixed_temporal_delays(self.experiment, self.horizon).size
                    * self.experiment.y_bins + 1)

    def test_null_is_sigma_independent_for_every_readout(self):
        for strategy in ("integrated", "resolved", "coherent"):
            first = temporal_observation_probabilities(
                self.experiment, 0.15, 0, self.response, strategy=strategy,
                delay_horizon=self.horizon, phase=0.03)
            second = temporal_observation_probabilities(
                self.experiment, 0.25, 0, self.response, strategy=strategy,
                delay_horizon=self.horizon, phase=0.03)
            assert_allclose(first, second, atol=2e-14)

    def test_local_fisher_matrices_are_positive(self):
        for strategy in ("integrated", "resolved", "coherent"):
            information = temporal_observation_fisher_information(
                self.experiment, 0.2, 1, self.response, strategy=strategy,
                delay_horizon=self.horizon, phase=0.03)
            self.assertTrue(np.all(np.linalg.eigvalsh(information) > 0))


if __name__ == "__main__":
    unittest.main()
