"""Threshold and sampling diagnostics for the physical record clock."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from analysis.record_environment import RecordExperiment, simulate_record_formation
from analysis.record_sensitivity import (
    build_reference_runs,
    profile_clock_convention_widths,
    reevaluate_stabilization,
    threshold_grid,
    time_sampling_convergence,
)


class RecordSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.experiment = RecordExperiment()
        self.times = np.arange(0, 12.001, 0.02)
        self.run = simulate_record_formation(self.experiment, 1.0, self.times)

    def test_default_threshold_re_evaluation_matches_original_clock(self):
        window = reevaluate_stabilization(
            self.run,
            information_deficit=self.experiment.information_deficit,
            required_copies=self.experiment.required_copies,
            coherence_tolerance=self.experiment.coherence_tolerance,
            hold_time=self.experiment.hold_time,
        )
        self.assertEqual(window, self.run.stabilization)

    def test_stricter_information_and_copy_requirements_delay_onset(self):
        loose = reevaluate_stabilization(
            self.run, information_deficit=0.2, required_copies=2,
            coherence_tolerance=0.1, hold_time=0.5)
        strict = reevaluate_stabilization(
            self.run, information_deficit=0.05, required_copies=4,
            coherence_tolerance=0.01, hold_time=0.5)
        self.assertIsNotNone(loose)
        self.assertIsNotNone(strict)
        self.assertLess(loose.latency, strict.latency)

    def test_hold_time_changes_confirmation_but_not_monotone_onset(self):
        short = reevaluate_stabilization(
            self.run, information_deficit=0.1, required_copies=3,
            coherence_tolerance=0.05, hold_time=0.25)
        long = reevaluate_stabilization(
            self.run, information_deficit=0.1, required_copies=3,
            coherence_tolerance=0.05, hold_time=1.0)
        self.assertEqual(short.latency, long.latency)
        self.assertGreater(long.confirmed_at, short.confirmed_at)

    def test_threshold_grid_has_declared_cartesian_size(self):
        runs = build_reference_runs(self.experiment, [0.5, 1.0],
                                    duration=12, dt=0.04)
        rows = threshold_grid(
            runs,
            information_deficits=[0.05, 0.1],
            required_copies=[2, 3, 4],
            coherence_tolerances=[0.01, 0.05],
            hold_times=[0.25, 0.5],
        )
        self.assertEqual(len(rows), 2 * 2 * 3 * 2 * 2)
        self.assertTrue(all(row.stabilization is not None for row in rows))

    def test_shared_convention_profile_recalibrates_only_at_control(self):
        runs = build_reference_runs(
            self.experiment, [0.5, 1.0, 2.0], duration=12, dt=0.02)
        rows = threshold_grid(
            runs, information_deficits=[0.05, 0.2],
            required_copies=[2, 4], coherence_tolerances=[0.01, 0.1],
            hold_times=[0.5])
        profile = profile_clock_convention_widths(
            rows, g_values=[0.5, 1.0, 2.0],
            locked_alpha=0.2 / 2.76, calibration_g=1.0,
            calibration_sigma_t=0.2)
        self.assertEqual(len(profile.rows), 8)
        self.assertEqual(profile.unresolved_conventions, 0)
        for row in profile.rows:
            self.assertAlmostEqual(row.calibration_profiled_widths[1], 0.2)
            self.assertAlmostEqual(
                row.locked_alpha_widths[0],
                row.locked_alpha * row.latencies[0])
            self.assertAlmostEqual(
                row.calibration_profiled_widths[0], 0.4, delta=0.002)
            self.assertAlmostEqual(
                row.calibration_profiled_widths[2], 0.1, delta=0.001)

    def test_time_sampling_converges_with_exact_joint_normalization(self):
        results = time_sampling_convergence(
            self.experiment, [0.5, 1.0, 2.0], duration=12,
            time_steps=[0.08, 0.04, 0.02])
        self.assertTrue(all(row.max_joint_normalization_error < 5e-15
                            for row in results))
        for g in [0.5, 1.0, 2.0]:
            rows = sorted((row for row in results if row.g == g),
                          key=lambda row: row.dt)
            reference = rows[0].latency
            for row in rows[1:]:
                self.assertLessEqual(abs(row.latency - reference), row.dt + 1e-12)
        finest = [row.latency for row in results if row.dt == 0.02]
        assert_allclose(np.asarray(finest) * [0.5, 1.0, 2.0],
                        finest[1], atol=0.02)

    def test_invalid_thresholds_are_rejected(self):
        invalid = [
            dict(information_deficit=0, required_copies=3,
                 coherence_tolerance=0.05, hold_time=0.5),
            dict(information_deficit=0.1, required_copies=5,
                 coherence_tolerance=0.05, hold_time=0.5),
            dict(information_deficit=0.1, required_copies=3,
                 coherence_tolerance=-1, hold_time=0.5),
            dict(information_deficit=0.1, required_copies=3,
                 coherence_tolerance=0.05, hold_time=-1),
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                reevaluate_stabilization(self.run, **kwargs)


if __name__ == "__main__":
    unittest.main()
