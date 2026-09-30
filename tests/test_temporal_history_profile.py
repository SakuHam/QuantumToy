"""Tests for the future/front/past phenomenological envelope."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from analysis.temporal_history_profile import (
    TemporalHistoryEnvelope,
    temporal_history_profile,
)


class TemporalHistoryProfileTests(unittest.TestCase):
    def test_selection_is_monotone_while_the_record_eventually_fades(self):
        ages = np.linspace(-8.0, 40.0, 1001)
        result = temporal_history_profile(
            ages,
            TemporalHistoryEnvelope(
                sigma_t=1.0, retention_time=1.0,
                fade_time=4.0, fade_power=1.5),
        )
        self.assertTrue(np.all(np.diff(result.selection) >= 0.0))
        self.assertGreater(result.selection[-1], 1.0 - 1e-12)
        self.assertLess(result.accessible_record[-1], 1e-6)

    def test_front_is_maximal_at_half_selection(self):
        result = temporal_history_profile(
            np.array([-20.0, 0.0, 20.0]),
            TemporalHistoryEnvelope(sigma_t=1.0),
        )
        assert_allclose(result.selection, [0.0, 0.5, 1.0], atol=1e-12)
        assert_allclose(result.present_front, [0.0, 1.0, 0.0], atol=1e-12)
        assert_allclose(result.future_openness, 1.0 - result.selection)

    def test_fading_never_reopens_the_selected_history(self):
        envelope = TemporalHistoryEnvelope(
            sigma_t=0.2, retention_time=0.4,
            fade_time=0.5, fade_power=2.0)
        result = temporal_history_profile(np.array([1.0, 3.0]), envelope)
        self.assertGreater(result.accessible_record[0], result.accessible_record[1])
        self.assertGreaterEqual(result.selection[1], result.selection[0])

    def test_invalid_envelope_is_rejected(self):
        for kwargs in (
            {"sigma_t": 0.0},
            {"retention_time": -1.0},
            {"fade_time": 0.0},
            {"fade_power": 0.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TemporalHistoryEnvelope(**kwargs)


if __name__ == "__main__":
    unittest.main()
