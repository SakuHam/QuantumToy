"""Design optimization and multinomial recovery for the TRF candidate."""

import unittest

import numpy as np

from analysis.record_environment import RecordExperiment, simulate_record_formation
from analysis.trf_inference import (
    covariance_from_information,
    design_prior_information,
    fit_candidate_counts,
    monte_carlo_recovery,
    optimize_lambda_design,
    per_shot_design_fisher_matrices,
    simulate_multinomial_counts,
)


class TrfInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = RecordExperiment()
        times = np.arange(0, 12.001, 0.04)
        runs = [simulate_record_formation(cls.experiment, g, times)
                for g in [0.5, 1.0, 2.0]]
        cls.candidates = tuple(
            (run.g, time, run.stabilization.latency)
            for run in runs for time in [0.25, 0.5, 1.0, 2.0, 4.0]
        )
        cls.design = optimize_lambda_design(
            cls.experiment, cls.candidates,
            total_shots=12_000, shot_block=1_000,
            lambda_strength=0.2,
            ordinary_dephasing_sigma=0.005,
        )

    def test_design_uses_exact_budget_and_improves_uniform_allocation(self):
        self.assertEqual(sum(row.shots for row in self.design.allocations), 12_000)
        self.assertGreater(len(self.design.allocations), 0)
        self.assertTrue(np.isfinite(self.design.lambda_standard_error))
        matrices = per_shot_design_fisher_matrices(
            self.experiment, self.candidates,
            lambda_strength=0.2, ordinary_dephasing_rate=0.02)
        prior = design_prior_information(
            g_scale_sigma=0.01, probe_strength_sigma=0.01,
            ordinary_dephasing_sigma=0.005)
        # Compare against the closest equal allocation with the same total.
        uniform = prior + (12_000 / len(matrices)) * matrices.sum(axis=0)
        uniform_covariance = covariance_from_information(uniform)
        self.assertLessEqual(self.design.lambda_standard_error,
                             np.sqrt(uniform_covariance[0, 0]) + 1e-12)

    def test_multinomial_counts_retain_every_shot(self):
        counts = simulate_multinomial_counts(
            self.experiment, self.design.allocations,
            [0.2, 1.0, self.experiment.probe_strength, 0.02],
            rng=np.random.default_rng(4))
        self.assertEqual(len(counts), len(self.design.allocations))
        for allocation, observed in zip(self.design.allocations, counts):
            self.assertEqual(observed.shape, (2, 16, 3))
            self.assertEqual(observed.sum(), allocation.shots)

    def test_fit_profiles_nuisances_and_clock_scale(self):
        counts = simulate_multinomial_counts(
            self.experiment, self.design.allocations,
            [0.2, 1.0, self.experiment.probe_strength, 0.02],
            rng=np.random.default_rng(7))
        fit = fit_candidate_counts(
            self.experiment, self.design.allocations, counts,
            tau_scales=(0.8, 1.0, 1.2))
        self.assertTrue(fit.success)
        self.assertIn(fit.tau_scale, (0.8, 1.0, 1.2))
        self.assertGreaterEqual(fit.parameters[0], 0)
        self.assertGreaterEqual(fit.likelihood_ratio, 0)

    def test_small_monte_carlo_reports_null_and_signal_samples(self):
        summaries = monte_carlo_recovery(
            self.experiment, self.design.allocations,
            true_lambda_values=[0, 0.2], repetitions=3, seed=9,
            tau_scales=(0.8, 1.0, 1.2))
        self.assertEqual(len(summaries), 2)
        for summary in summaries:
            self.assertEqual(summary.repetitions, 3)
            self.assertEqual(summary.successful_fits, 3)
            self.assertTrue(0 <= summary.detection_rate <= 1)
            self.assertEqual(summary.quantiles.shape, (3,))
            self.assertEqual(sum(summary.selected_tau_scale_counts.values()), 3)

    def test_invalid_design_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            optimize_lambda_design(
                self.experiment, self.candidates,
                total_shots=12_500, shot_block=1_000,
                lambda_strength=0.2)


if __name__ == "__main__":
    unittest.main()
