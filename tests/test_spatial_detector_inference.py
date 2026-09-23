"""Tests for calibrated detector response and robust spatial inference."""

import unittest
from dataclasses import asdict, replace
from unittest.mock import patch
import json

import numpy as np
from numpy.testing import assert_allclose

from analysis.spatial_detector_inference import (
    DetectorResponse,
    apply_detector_response,
    bootstrap_spatial_detection_threshold,
    build_spatial_detector_inference_library,
    compare_spatial_detector_grids,
    detector_response_matrix,
    fit_detector_calibration,
    fit_spatial_detector_counts,
    monte_carlo_spatial_detector_recovery,
    realistic_spatial_probabilities,
    select_robust_double_slit_detector_setting,
    simulate_realistic_counts,
)
from analysis.spatial_effect_measurement import double_slit_effect_experiment


class SpatialDetectorInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = double_slit_effect_experiment(nx=24, ny=24)
        cls.response = DetectorResponse(0.88, 0.012, 0.45, 0.05)
        cls.calibration = tuple(replace(
            cls.experiment, potential_mode="free", detector_x=0,
            detector_width=0.3, reference_time=time)
            for time in (0.55, 0.7, 0.85, 1.0, 1.15))
        cls.second = replace(
            cls.experiment, detector_x=0, detector_width=0.5,
            reference_time=13 / 30)
        cls.library = build_spatial_detector_inference_library(
            cls.calibration, (cls.experiment, cls.second),
            calibration_sigma_t=0.2, calibration_lambda_strength=0,
            candidate_sigmas=[0.15, 0.2, 0.25],
            candidate_lambdas=[0, 0.5, 1.0],
            candidate_timing_jitters=[0.025, 0.05, 0.075])

    def test_detector_channel_is_positive_complete_and_non_postselected(self):
        matrix = detector_response_matrix(
            self.experiment.y_bins, self.response)
        self.assertGreaterEqual(float(np.min(matrix)), 0)
        assert_allclose(np.sum(matrix, axis=0), 1, atol=2e-15)
        ideal = np.zeros(self.experiment.y_bins + 1)
        ideal[-1] = 1
        observed = apply_detector_response(ideal, self.response)
        self.assertAlmostEqual(np.sum(observed[:-1]), 0.012)
        self.assertAlmostEqual(observed[-1], 0.988)

        law = realistic_spatial_probabilities(
            self.experiment, 0.2, 1.0, self.response)
        self.assertAlmostEqual(float(np.sum(law)), 1)
        self.assertTrue(np.all(law >= 0))

    def test_free_controls_calibrate_detector_without_test_data(self):
        counts = simulate_realistic_counts(
            self.calibration, 0.2, 0, self.response, 100_000,
            rng=np.random.default_rng(2))
        fit = fit_detector_calibration(
            self.calibration, counts, 0.2, 0,
            candidate_timing_jitters=[0.025, 0.05, 0.075])
        self.assertTrue(fit.success)
        self.assertEqual(fit.response.timing_jitter, 0.05)
        self.assertAlmostEqual(fit.response.efficiency, 0.88, delta=0.015)
        self.assertAlmostEqual(
            fit.response.dark_probability, 0.012, delta=0.002)
        self.assertAlmostEqual(fit.response.blur_sigma_bins, 0.45, delta=0.06)
        self.assertTrue(np.all(fit.response_standard_errors > 0))

    def test_robust_design_uses_worst_case_parameter_point(self):
        design = select_robust_double_slit_detector_setting(
            replace(self.experiment, nx=20, ny=20), self.response,
            parameter_sigmas=[0.15, 0.25], parameter_lambdas=[0.6, 1.4],
            candidate_detector_x=[0.0, 0.5],
            candidate_detector_width=[0.35, 0.5],
            candidate_time_offset=[-0.4, -0.2])
        self.assertEqual(
            design.best_index,
            int(np.argmax(design.worst_log_determinant_gain)))
        self.assertEqual(design.best_detector_x, 0.0)
        self.assertEqual(design.best_detector_width, 0.5)
        self.assertLess(design.worst_sigma_error_ratio[design.best_index], 1)
        self.assertLess(design.worst_lambda_error_ratio[design.best_index], 1)

    def test_joint_calibration_and_test_fit_recovers_injection(self):
        rng = np.random.default_rng(8)
        calibration_counts = simulate_realistic_counts(
            self.calibration, 0.2, 0, self.response, 100_000, rng=rng)
        test_counts = simulate_realistic_counts(
            (self.experiment, self.second), 0.2, 1.0, self.response,
            [50_000, 50_000], rng=rng)
        fit = fit_spatial_detector_counts(
            self.library, calibration_counts, test_counts)
        self.assertTrue(fit.success)
        self.assertEqual(fit.best_sigma_t, 0.2)
        self.assertEqual(fit.best_lambda_strength, 1.0)
        self.assertTrue(fit.detected)
        self.assertTrue(fit.joint_confidence_region[2, 1])
        self.assertEqual(fit.profile_negative_log_likelihood.shape, (3, 3))
        self.assertTrue(fit.diagnostics.best.success)
        self.assertTrue(fit.diagnostics.null.success)
        self.assertEqual(fit.diagnostics.profile_fits, 27)
        self.assertEqual(fit.diagnostics.best_boundaries["lambda_strength"], "upper")
        self.assertEqual(fit.diagnostics.best_boundaries["sigma_t"], "interior")
        json.dumps(asdict(fit.diagnostics))

    def test_nested_grids_use_same_counts_and_match_common_profiles(self):
        refined = self.library
        coarse = replace(
            refined, candidate_sigmas=refined.candidate_sigmas[[0, 2]],
            candidate_lambdas=refined.candidate_lambdas[[0, 2]],
            test_ideal_probabilities=refined.test_ideal_probabilities[:, [0, 2]][:, :, [0, 2]])
        rng = np.random.default_rng(11)
        control = simulate_realistic_counts(self.calibration, 0.2, 0, self.response, 30_000, rng=rng)
        test = simulate_realistic_counts((self.experiment, self.second), 0.2, 1,
                                        self.response, [15_000, 15_000], rng=rng)
        saved = tuple(x.copy() for x in control + test)
        first, second, difference = compare_spatial_detector_grids(coarse, refined, control, test)
        self.assertLessEqual(second.negative_log_likelihood, first.negative_log_likelihood + 1e-7)
        self.assertAlmostEqual(difference["delta_null_nll"], 0, places=7)
        self.assertLess(difference["max_abs_common_profile_nll_difference"], 1e-7)
        self.assertFalse(difference["nested_minimum_violation"])
        for old, new in zip(saved, control + test):
            assert_allclose(old, new, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "identical physics"):
            compare_spatial_detector_grids(coarse, replace(refined, calibration_sigma_t=0.3), control, test)
        with self.assertRaisesRegex(ValueError, "every coarse grid point"):
            compare_spatial_detector_grids(refined, coarse, control, test)

    def test_nonwinning_optimizer_failure_is_visible(self):
        from scipy.optimize import OptimizeResult
        count = 0
        def fake_fit(ideal, counts, jitter, initial):
            nonlocal count
            count += 1
            failed = count == 2  # First profile point fails; a later one wins.
            return OptimizeResult(x=np.array([0.88, 0.012, 0.45]),
                                  fun=100 if failed else 0,
                                  success=not failed, status=2 if failed else 0,
                                  message="forced failure" if failed else "converged", nit=1, nfev=4)
        counts = tuple(np.ones(self.experiment.y_bins + 1, dtype=int) for _ in range(5))
        with patch("analysis.spatial_detector_inference._fit_response", side_effect=fake_fit):
            fit = fit_spatial_detector_counts(self.library, counts, counts[:2])
        self.assertTrue(fit.success)
        self.assertEqual(fit.diagnostics.failed_profile_fits, 1)
        self.assertEqual(fit.diagnostics.failed_profile_status_counts, {"2: forced failure": 1})
        self.assertFalse(fit.diagnostics.sigma_identified_at_best)

    def test_small_monte_carlo_reports_false_positives_and_coverage(self):
        threshold = bootstrap_spatial_detection_threshold(
            self.library, self.response,
            calibration_shots=10_000, test_shots=[5_000, 5_000],
            repetitions=3, seed=3)
        self.assertGreaterEqual(threshold.threshold, 0)
        self.assertEqual(threshold.likelihood_ratios.shape, (3,))
        self.assertEqual(len(threshold.replicate_records), 3)
        assert_allclose(threshold.likelihood_ratios,
                        [r.likelihood_ratio for r in threshold.replicate_records])
        summaries = monte_carlo_spatial_detector_recovery(
            self.library, self.response, [(0.2, 0), (0.2, 1.0)],
            calibration_shots=30_000, test_shots=[15_000, 15_000],
            repetitions=3, seed=4,
            detection_threshold=threshold.threshold)
        self.assertEqual(len(summaries), 2)
        for summary in summaries:
            self.assertEqual(summary.successful_fits, 3)
            self.assertTrue(0 <= summary.detection_rate <= 1)
            self.assertTrue(0 <= summary.joint_coverage <= 1)
            self.assertEqual(summary.sigma_quantiles.shape, (3,))
            self.assertEqual(summary.lambda_quantiles.shape, (3,))
            self.assertEqual(summary.likelihood_ratio_quantiles.shape, (3,))
            self.assertEqual(len(summary.replicate_records), 3)
            self.assertEqual(len({r.seed for r in summary.replicate_records}), 3)
            self.assertAlmostEqual(summary.detection_rate,
                                   np.mean([r.detected for r in summary.replicate_records]))
            json.dumps([asdict(r) for r in summary.replicate_records])
        self.assertLessEqual(summaries[0].detection_rate, 1 / 3)
        self.assertEqual(summaries[1].detection_rate, 1)


if __name__ == "__main__":
    unittest.main()
