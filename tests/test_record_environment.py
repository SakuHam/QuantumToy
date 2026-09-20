"""Physical weak-probe and environment-record model for TRF-IT v0.2."""

import unittest
from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose

from analysis.record_environment import (
    RecordExperiment,
    detector_kraus,
    fragment_states,
    initial_state,
    probe_environment_state,
    probe_kraus,
    record_snapshot,
    simulate_record_formation,
)


class RecordEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.experiment = RecordExperiment()

    def test_preparation_and_instruments_are_normalized_and_complete(self):
        self.assertAlmostEqual(np.vdot(initial_state(self.experiment),
                                       initial_state(self.experiment)).real, 1)
        for operators in [probe_kraus(self.experiment),
                          detector_kraus(self.experiment)]:
            completeness = sum(k.conj().T @ k for k in operators)
            assert_allclose(completeness, np.eye(2), atol=1e-14)

    def test_conditional_fragment_unitaries_preserve_norm_and_have_known_overlap(self):
        states = fragment_states(self.experiment, g=0.7, time=1.3)
        assert_allclose(np.sum(np.abs(states) ** 2, axis=-1), 1, atol=1e-14)
        theta = np.pi / 2 * (1 - np.exp(
            -0.7 * np.asarray(self.experiment.fragment_rates) * 1.3))
        overlap = np.einsum("ki,ki->k", states[:, 0].conj(), states[:, 1])
        assert_allclose(overlap, np.cos(theta), atol=1e-14)

    def test_probe_branches_keep_physical_outcome_probabilities(self):
        branches = probe_environment_state(self.experiment, g=0.8, time=1.1)
        probabilities = np.sum(np.abs(branches) ** 2, axis=(1, 2))
        assert_allclose(probabilities, [0.5, 0.5], atol=1e-14)
        self.assertAlmostEqual(probabilities.sum(), 1)

    def test_complete_joint_record_includes_no_click_and_conditional_state(self):
        snapshot = record_snapshot(self.experiment, g=1.0, time=1.0)
        self.assertEqual(snapshot.joint.shape, (2, 16, 3))
        self.assertAlmostEqual(snapshot.joint.sum(), 1)
        self.assertGreater(snapshot.joint[..., 2].sum(), 0)
        index = np.unravel_index(np.argmax(snapshot.joint), snapshot.joint.shape)
        conditional = snapshot.conditional_system_state(*index)
        self.assertAlmostEqual(np.trace(conditional).real, 1)
        assert_allclose(conditional, conditional.conj().T, atol=1e-14)

        perfect = replace(self.experiment, detector_efficiencies=(1.0, 1.0))
        impossible = record_snapshot(perfect, g=1.0, time=1.0)
        with self.assertRaisesRegex(ValueError, "zero-probability"):
            impossible.conditional_system_state(0, 0, 2)

    def test_later_detector_setting_cannot_change_earlier_record_marginal(self):
        reference = record_snapshot(self.experiment, g=0.9, time=0.8)
        marginal = reference.joint.sum(axis=2)
        for angle, phase in [(0, 0), (0.7, 0.4), (np.pi, np.pi / 2)]:
            changed = replace(self.experiment, detector_angle=angle,
                              detector_phase=phase)
            with self.subTest(angle=angle, phase=phase):
                assert_allclose(record_snapshot(changed, 0.9, 0.8).joint.sum(axis=2),
                                marginal, atol=1e-14)

    def test_unread_probe_and_environment_give_analytic_coherence(self):
        strength = self.experiment.probe_strength
        for g, time in [(0, 0), (0.4, 0.7), (1.2, 1.8)]:
            with self.subTest(g=g, time=time):
                theta = np.pi / 2 * (1 - np.exp(
                    -g * np.asarray(self.experiment.fragment_rates) * time))
                expected = np.sqrt(1 - strength ** 2) * np.prod(np.cos(theta))
                snapshot = record_snapshot(self.experiment, g, time)
                self.assertAlmostEqual(snapshot.coherence, expected, places=13)
                self.assertAlmostEqual(np.trace(snapshot.system_density).real, 1)

    def test_fixed_fragment_readout_is_bounded_by_holevo_information(self):
        for time in np.linspace(0, 5, 11):
            snapshot = record_snapshot(self.experiment, g=1.0, time=float(time))
            self.assertTrue(np.all(snapshot.readout_information >= -1e-14))
            self.assertTrue(np.all(snapshot.readout_information
                                   <= snapshot.holevo_information + 1e-13))
            self.assertLessEqual(snapshot.all_fragments_information,
                                 self.experiment.branch_entropy + 1e-13)

    def test_blank_and_orthogonal_memories_have_expected_information(self):
        blank = record_snapshot(self.experiment, g=0, time=10)
        assert_allclose(blank.readout_information, 0, atol=1e-13)
        assert_allclose(blank.holevo_information, 0, atol=1e-13)
        self.assertAlmostEqual(blank.all_fragments_information, 0, places=13)
        self.assertEqual(blank.redundancy, 0)

        saturated = record_snapshot(self.experiment, g=10, time=10)
        assert_allclose(saturated.readout_information,
                        self.experiment.branch_entropy, atol=1e-13)
        assert_allclose(saturated.holevo_information,
                        self.experiment.branch_entropy, atol=1e-13)
        self.assertAlmostEqual(saturated.all_fragments_information,
                               self.experiment.branch_entropy, places=13)
        self.assertEqual(saturated.redundancy, 4)

    def test_redundancy_counts_predeclared_physical_singletons(self):
        experiment = replace(self.experiment, fragment_rates=(1, 1, 0, 0),
                             required_copies=2)
        snapshot = record_snapshot(experiment, g=10, time=10)
        self.assertEqual(snapshot.redundancy, 2)
        self.assertTrue(snapshot.stable)
        assert_allclose(snapshot.readout_information[2:], 0, atol=1e-13)

    def test_clock_uses_shared_thresholds_and_coupling_controls_latency(self):
        times = np.arange(0, 8.001, 0.01)
        latencies = []
        for g in [0.5, 1.0, 2.0]:
            run = simulate_record_formation(self.experiment, g, times)
            self.assertIsNotNone(run.stabilization)
            latencies.append(run.stabilization.latency)
            self.assertGreaterEqual(run.stabilization.confirmed_at
                                    - run.stabilization.onset,
                                    self.experiment.hold_time)
        self.assertGreater(latencies[0], latencies[1])
        self.assertGreater(latencies[1], latencies[2])
        # The model depends on g*t, so onset scales as 1/g up to one time bin.
        assert_allclose(np.array(latencies) * np.array([0.5, 1, 2]),
                        latencies[1], atol=0.02)

    def test_zero_coupling_and_short_observation_remain_unresolved(self):
        times = np.linspace(0, 2, 21)
        self.assertIsNone(simulate_record_formation(self.experiment, 0, times).stabilization)
        self.assertIsNone(simulate_record_formation(self.experiment, 0.2, times).stabilization)

    def test_input_validation_fixes_nontrivial_ensemble_and_fragments(self):
        invalid = [
            dict(path_probability=0),
            dict(probe_strength=1.1),
            dict(fragment_rates=()),
            dict(fragment_rates=(1, -1)),
            dict(required_copies=5),
            dict(detector_efficiencies=(1.1, 0.5)),
            dict(information_deficit=0),
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RecordExperiment(**kwargs)
        with self.assertRaises(ValueError):
            simulate_record_formation(self.experiment, 1, [0, 1, 0.5])


if __name__ == "__main__":
    unittest.main()
