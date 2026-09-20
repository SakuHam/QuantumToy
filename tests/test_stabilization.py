"""Clock mechanics, not a physical derivation of record redundancy or sigma_T."""

import unittest

import numpy as np

from analysis.stabilization import sampled_stabilization


class StabilizationTests(unittest.TestCase):
    def test_section_8_monotone_rate_crossing_and_sampling_convergence(self):
        delta, t0, hold = 0.1, 7.0, 0.5
        for gamma in [0.5, 1.0, 2.0]:
            exact_latency = np.log(1 / delta) / gamma
            for dt in [0.1, 0.02, 0.005]:
                with self.subTest(gamma=gamma, dt=dt):
                    times = t0 + np.arange(0, 8 + dt / 2, dt)
                    information = 1 - np.exp(-gamma * (times - t0))
                    stable = information >= 1 - delta
                    result = sampled_stabilization(times, stable, hold_time=hold)
                    self.assertIsNotNone(result)
                    self.assertGreaterEqual(result.latency, exact_latency - 1e-12)
                    self.assertLessEqual(result.latency - exact_latency, dt + 1e-12)
                    self.assertGreaterEqual(result.confirmed_at - result.onset, hold)
                    self.assertLessEqual(result.confirmed_at - result.onset,
                                         hold + dt + 1e-12)

    def test_already_stable_record_has_zero_remaining_latency(self):
        result = sampled_stabilization([10, 11, 12], [True, True, True], hold_time=2)
        self.assertEqual((result.onset, result.confirmed_at, result.latency), (10, 12, 0))

    def test_short_crossing_does_not_close_the_front(self):
        result = sampled_stabilization(np.arange(8),
                                       [False, True, True, False, True, True, True, True],
                                       hold_time=3)
        self.assertEqual((result.onset, result.confirmed_at, result.latency), (4, 7, 4))

    def test_failure_at_hold_endpoint_restarts_confirmation(self):
        result = sampled_stabilization(np.arange(6),
                                       [True, True, False, True, True, True], hold_time=2)
        self.assertEqual(result.onset, 3)
        self.assertEqual(result.confirmed_at, 5)

    def test_incomplete_or_absent_stabilization_is_unresolved(self):
        for stable in [[False, False, False], [False, True, True], [True, True, True]]:
            with self.subTest(stable=stable):
                self.assertIsNone(sampled_stabilization([0, 1, 2], stable, hold_time=3))

    def test_all_predeclared_criteria_must_hold(self):
        redundancy_ok = np.array([False, True, True, True, True, True])
        coherence_ok = np.array([False, False, False, True, True, True])
        result = sampled_stabilization(np.arange(6), redundancy_ok & coherence_ok,
                                       hold_time=2)
        self.assertEqual(result.onset, 3)

    def test_irregular_sampling_uses_elapsed_time_and_event_origin(self):
        result = sampled_stabilization([20, 20.1, 20.2, 21.7, 22.0],
                                       [False, True, True, True, True], hold_time=1)
        self.assertAlmostEqual(result.latency, 0.1)
        self.assertEqual(result.confirmed_at, 21.7)

    def test_zero_hold_time_confirms_first_passing_sample(self):
        result = sampled_stabilization([5, 6], [False, True], hold_time=0)
        self.assertEqual((result.onset, result.confirmed_at, result.latency), (6, 6, 1))


if __name__ == "__main__":
    unittest.main()
