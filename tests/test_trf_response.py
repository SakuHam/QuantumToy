"""Independent temporal-width measurement and held-out alpha tests."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from analysis.record_environment import RecordExperiment, simulate_record_formation
from analysis.trf_response import (
    HALF_RISE_FACTOR,
    ResponseSetting,
    excess_dephasing,
    fit_heldout_width,
    fit_response,
    half_rise_time,
    response_probabilities,
    simulate_response_counts,
)


class TrfTemporalResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = RecordExperiment()
        times = np.arange(0, 12.001, 0.02)
        runs = [simulate_record_formation(cls.experiment, g, times)
                for g in [0.5, 1.0, 2.0]]
        cls.tau = {run.g: run.stabilization.latency for run in runs}
        fractions = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.5, 4.0]
        cls.settings = tuple(
            ResponseSetting(g, fraction * cls.tau[g], phase, 10_000)
            for g in cls.tau for fraction in fractions
            for phase in [0.0, np.pi / 2]
        )
        cls.counts = simulate_response_counts(
            cls.experiment, cls.settings,
            lambda_strength=0.2,
            sigma_by_g=cls.tau,
            ordinary_dephasing_rate=0.02,
            rng=np.random.default_rng(8),
        )

    def test_half_rise_time_removes_amplitude(self):
        sigma_t, strength = 3.2, 0.7
        time = half_rise_time(sigma_t)
        self.assertAlmostEqual(time, HALF_RISE_FACTOR * sigma_t)
        self.assertAlmostEqual(excess_dephasing(time, strength, sigma_t),
                               strength / 2)

    def test_phase_scan_probabilities_are_complete(self):
        probabilities = response_probabilities(
            self.experiment, self.settings[:4],
            lambda_strength=0.2,
            sigma_by_g=self.tau,
            ordinary_dephasing_rate=0.02)
        for probability in probabilities:
            self.assertGreaterEqual(probability.min(), 0)
            self.assertAlmostEqual(probability.sum(), 1)

    def test_universal_response_recovers_lambda_and_alpha(self):
        fit = fit_response(
            self.experiment, self.settings, self.counts,
            tau_by_g=self.tau, mode="universal_alpha",
            gamma_mean=0.02, gamma_sigma=0.002)
        self.assertTrue(fit.success)
        self.assertAlmostEqual(fit.lambda_strength, 0.2, delta=0.04)
        self.assertAlmostEqual(fit.parameters[1], 1.0, delta=0.15)
        self.assertAlmostEqual(fit.ordinary_dephasing_rate, 0.02, delta=0.004)
        self.assertIsNotNone(fit.standard_errors)

    def test_free_widths_recover_shared_alpha_without_enforcing_it(self):
        fit = fit_response(
            self.experiment, self.settings, self.counts,
            tau_by_g=self.tau, mode="free_widths",
            gamma_mean=0.02, gamma_sigma=0.002)
        self.assertTrue(fit.success)
        alpha = np.array(list(fit.alpha_by_g(self.tau).values()))
        assert_allclose(alpha, 1, atol=0.15)

    def test_calibration_predicts_heldout_widths(self):
        calibration_indices = [i for i, setting in enumerate(self.settings)
                               if setting.g == 1.0]
        calibration = fit_response(
            self.experiment,
            [self.settings[i] for i in calibration_indices],
            [self.counts[i] for i in calibration_indices],
            tau_by_g=self.tau, mode="universal_alpha",
            gamma_mean=0.02, gamma_sigma=0.002)
        alpha = calibration.parameters[1]
        for g in [0.5, 2.0]:
            indices = [i for i, setting in enumerate(self.settings) if setting.g == g]
            heldout = fit_heldout_width(
                self.experiment,
                [self.settings[i] for i in indices],
                [self.counts[i] for i in indices],
                lambda_strength=calibration.lambda_strength,
                ordinary_dephasing_rate=calibration.ordinary_dephasing_rate,
                predicted_sigma_t=alpha * self.tau[g])
            self.assertTrue(heldout.success)
            self.assertAlmostEqual(heldout.sigma_t / self.tau[g], 1, delta=0.2)
            self.assertLess(heldout.prediction_likelihood_ratio, 6.64)

    def test_invalid_width_and_setting_are_rejected(self):
        with self.assertRaises(ValueError):
            excess_dephasing([0, 1], 0.2, 0)
        with self.assertRaises(ValueError):
            ResponseSetting(0, 1, 0, 10)


if __name__ == "__main__":
    unittest.main()
