"""Tests for the finite-resolution joint spatial/arrival-time instrument."""

import unittest
from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose

from analysis.spatial_arrival_time import (
    ArrivalTimeResponse,
    arrival_time_bin_probabilities,
    joint_spatial_arrival_fisher_information,
    joint_spatial_arrival_probabilities,
    marginalize_joint_arrival,
)
from analysis.spatial_detector_inference import DetectorResponse
from analysis.spatial_effect_measurement import double_slit_effect_experiment


class SpatialArrivalTimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = replace(
            double_slit_effect_experiment(nx=8, ny=8),
            y_bins=4, reference_time=0.3, delay_step=0.1,
            propagation_step=0.02, horizon_sigmas=2)
        cls.response = DetectorResponse(0.88, 0.012, 0.45, 0.05)
        cls.timing = ArrivalTimeResponse(
            source_pulse_sigma=0.03, clock_offset=0.01,
            window_start=-0.2, window_stop=0.7, time_bins=18)
        cls.horizon = 0.4

    def test_timestamp_bins_retain_finite_gate_loss(self):
        central = arrival_time_bin_probabilities(0.2, self.timing, 0.05)
        late = arrival_time_bin_probabilities(1.0, self.timing, 0.05)
        self.assertEqual(central.shape, (self.timing.time_bins,))
        self.assertGreater(float(np.sum(central)), 0.999)
        self.assertLess(float(np.sum(late)), 1e-6)

    def test_joint_law_is_complete_with_no_click_and_marginals(self):
        law = joint_spatial_arrival_probabilities(
            self.experiment, 0.2, 1, self.response, self.timing,
            delay_horizon=self.horizon)
        self.assertEqual(
            law.size,
            self.experiment.y_bins * self.timing.time_bins + 1)
        self.assertAlmostEqual(float(np.sum(law)), 1)
        self.assertGreaterEqual(float(np.min(law)), 0)
        spatial, temporal = marginalize_joint_arrival(
            law, self.experiment.y_bins, self.timing.time_bins)
        self.assertAlmostEqual(float(np.sum(spatial)), 1)
        self.assertAlmostEqual(float(np.sum(temporal)), 1)
        self.assertAlmostEqual(float(spatial[-1]), float(temporal[-1]))

    def test_response_null_is_exactly_sigma_independent(self):
        narrow = joint_spatial_arrival_probabilities(
            self.experiment, 0.15, 0, self.response, self.timing,
            delay_horizon=self.horizon)
        wide = joint_spatial_arrival_probabilities(
            self.experiment, 0.3, 0, self.response, self.timing,
            delay_horizon=self.horizon)
        assert_allclose(narrow, wide, atol=2e-14)

    def test_timing_widths_require_independent_calibration(self):
        names = (
            "sigma_t", "lambda_strength", "source_pulse_sigma",
            "detector_jitter", "clock_offset",
        )
        information = joint_spatial_arrival_fisher_information(
            self.experiment, 0.2, 1, self.response, self.timing,
            delay_horizon=self.horizon, parameter_names=names)
        eigenvalues = np.linalg.eigvalsh(information)
        self.assertGreater(eigenvalues[-1], 0)
        # Source width and detector jitter enter only through their quadrature
        # sum, leaving one locally unidentifiable direction without controls.
        self.assertLess(abs(float(eigenvalues[0])), 1e-9 * eigenvalues[-1])
        identified = information[np.ix_([0, 1, 2, 4], [0, 1, 2, 4])]
        self.assertTrue(np.all(np.linalg.eigvalsh(identified) > 0))


if __name__ == "__main__":
    unittest.main()
