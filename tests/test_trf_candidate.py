"""Admissibility and identifiability tests for the minimal TRF candidate."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from analysis.record_environment import RecordExperiment, record_snapshot, simulate_record_formation
from analysis.trf_candidate import (
    HalfGaussianDephasing,
    candidate_joint,
    dephase_record_states,
    detector_visibility,
    expected_fisher_information,
)


class MinimalTrfCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = RecordExperiment()
        times = np.arange(0, 12.001, 0.02)
        cls.runs = [simulate_record_formation(cls.experiment, g, times)
                    for g in [0.5, 1.0, 2.0]]

    def test_half_gaussian_weight_and_quantum_limit(self):
        candidate = HalfGaussianDephasing(lambda_strength=0, alpha=1)
        times = np.array([0, 1, 10], dtype=float)
        weights = candidate.cumulative_weight(times, tau_stab=2)
        self.assertEqual(weights[0], 0)
        self.assertTrue(np.all(np.diff(weights) > 0))
        self.assertLess(weights[-1], 1)
        assert_allclose(candidate.coherence_factor(times, 2), 1, atol=1e-15)

    def test_zero_latency_is_an_immediate_closed_front_convention(self):
        candidate = HalfGaussianDephasing(lambda_strength=0.3)
        assert_allclose(candidate.cumulative_weight([0, 0.1, 1], 0), [0, 1, 1])

    def test_dephasing_preserves_trace_and_populations(self):
        vectors = np.array([[1, 1j], [2, -1]], dtype=complex) / np.sqrt(2)
        original = dephase_record_states(vectors, 1)
        damped = dephase_record_states(vectors, 0.2)
        assert_allclose(np.diagonal(damped, axis1=-2, axis2=-1),
                        np.diagonal(original, axis1=-2, axis2=-1), atol=1e-15)
        assert_allclose(damped[..., 0, 1], 0.2 * original[..., 0, 1], atol=1e-15)
        assert_allclose(np.trace(damped, axis1=-2, axis2=-1),
                        np.trace(original, axis1=-2, axis2=-1), atol=1e-15)
        self.assertTrue(np.all(np.linalg.eigvalsh(damped) >= -1e-14))

    def test_lambda_zero_exactly_recovers_quantum_joint(self):
        for run in self.runs:
            time = 1.3
            reference = record_snapshot(self.experiment, run.g, time).joint
            candidate = candidate_joint(
                self.experiment, run.g, time, run.stabilization.latency,
                HalfGaussianDephasing(0))
            assert_allclose(candidate, reference, atol=1e-14)

    def test_candidate_is_normalized_positive_and_preserves_earlier_records(self):
        for run in self.runs:
            tau = run.stabilization.latency
            reference = candidate_joint(
                self.experiment, run.g, 1.0, tau, HalfGaussianDephasing(0))
            for strength in [0.1, 0.5, 2.0]:
                with self.subTest(g=run.g, strength=strength):
                    candidate = candidate_joint(
                        self.experiment, run.g, 1.0, tau,
                        HalfGaussianDephasing(strength))
                    self.assertGreaterEqual(candidate.min(), 0)
                    self.assertAlmostEqual(candidate.sum(), 1)
                    assert_allclose(candidate.sum(axis=2), reference.sum(axis=2),
                                    atol=1e-14)

    def test_candidate_changes_a_declared_terminal_observable(self):
        run = self.runs[0]
        values = [detector_visibility(candidate_joint(
            self.experiment, run.g, 1.0, run.stabilization.latency,
            HalfGaussianDephasing(strength))) for strength in [0, 0.5, 1.0]]
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])

    def test_multiple_times_and_couplings_identify_lambda_with_nuisances(self):
        observations = [
            (run.g, time, run.stabilization.latency)
            for run in self.runs for time in [0.5, 1.0, 2.0, 3.0, 4.0]
        ]
        result = expected_fisher_information(
            self.experiment, observations, lambda_strength=0.2,
            ordinary_dephasing_rate=0.02, shots_per_observation=10_000)
        self.assertEqual(result.rank, 4)
        self.assertTrue(np.all(np.isfinite(result.standard_errors)))
        self.assertGreater(result.standard_errors[0], 0)
        self.assertLess(result.standard_errors[0], 0.2)
        # TRF dephasing and ordinary dephasing remain strongly correlated.
        self.assertGreater(abs(result.correlation[0, 3]), 0.8)

    def test_one_time_cannot_separate_two_dephasing_sources(self):
        run = self.runs[0]
        result = expected_fisher_information(
            self.experiment,
            [(run.g, 1.0, run.stabilization.latency)],
            lambda_strength=0.2,
            ordinary_dephasing_rate=0.02,
            shots_per_observation=10_000,
        )
        self.assertLess(result.rank, len(result.parameter_names))
        self.assertTrue(np.isinf(result.condition_number))

    def test_invalid_candidate_values_and_unresolved_clock_are_rejected(self):
        for kwargs in [dict(lambda_strength=-1), dict(alpha=0)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HalfGaussianDephasing(**kwargs)
        with self.assertRaisesRegex(ValueError, "stabilization latency"):
            candidate_joint(self.experiment, 1, 1, None,
                            HalfGaussianDephasing(0.2))


if __name__ == "__main__":
    unittest.main()
