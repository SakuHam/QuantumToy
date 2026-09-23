"""Tests for the dimensionful closure of the spatial TRF benchmark."""

import unittest

import numpy as np
from scipy.constants import hbar, m_e, m_n

from analysis.trf_physical_scale import (
    LOCKED_REFERENCE_ALPHA,
    SchrodingerPhysicalScale,
    bridge_record_and_spatial_clocks,
    direct_delay_prediction,
    scale_from_packet_velocity,
    scale_from_slit_separation,
)


class TrfPhysicalScaleTests(unittest.TestCase):
    def test_schrodinger_units_follow_nondimensional_equation(self):
        scale = SchrodingerPhysicalScale(m_n, 2e-6)
        self.assertAlmostEqual(scale.time_unit_s, m_n * (2e-6) ** 2 / hbar)
        self.assertAlmostEqual(
            scale.energy_unit_j * scale.time_unit_s, hbar)
        self.assertAlmostEqual(
            scale.velocity_unit_m_per_s * scale.time_unit_s,
            scale.length_unit_m)

    def test_slit_and_velocity_parameterizations_are_inverse(self):
        slit_scale = scale_from_slit_separation(m_e, 1e-6)
        packet_velocity = 3 * slit_scale.velocity_unit_m_per_s
        velocity_scale = scale_from_packet_velocity(
            m_e, packet_velocity, packet_kx=3)
        self.assertAlmostEqual(
            slit_scale.length_unit_m, velocity_scale.length_unit_m)
        self.assertAlmostEqual(slit_scale.time_unit_s, velocity_scale.time_unit_s)

    def test_length_rescaling_changes_time_quadratically(self):
        first = SchrodingerPhysicalScale(m_n, 1e-6)
        second = SchrodingerPhysicalScale(m_n, 5e-6)
        self.assertAlmostEqual(second.time_unit_s / first.time_unit_s, 25)

    def test_direct_delay_prediction_has_kappa_one_closure_and_moments(self):
        scale = SchrodingerPhysicalScale(m_n, 1e-6)
        prediction = direct_delay_prediction(scale)
        self.assertEqual(prediction.delay_scale, 1)
        self.assertAlmostEqual(prediction.sigma_t_s, 0.2 * scale.time_unit_s)
        self.assertAlmostEqual(
            prediction.responding_mean_delay_s,
            0.2 * np.sqrt(2 / np.pi) * scale.time_unit_s)
        self.assertAlmostEqual(
            prediction.response_fraction, 1 - np.exp(-1))
        self.assertLess(
            prediction.ensemble_mean_delay_s,
            prediction.responding_mean_delay_s)

    def test_record_clock_requires_an_explicit_alpha_bridge(self):
        scale = SchrodingerPhysicalScale(m_n, 1e-6)
        bridge = bridge_record_and_spatial_clocks(
            scale, 1 / scale.time_unit_s)
        self.assertAlmostEqual(bridge.dimensionless_rate, 1)
        self.assertAlmostEqual(
            bridge.stabilization_latency_s, 2.76 * scale.time_unit_s)
        self.assertAlmostEqual(
            bridge.alpha_required_for_spatial_sigma, 0.2 / 2.76)
        self.assertAlmostEqual(LOCKED_REFERENCE_ALPHA, 0.2 / 2.76)
        self.assertGreater(
            bridge.sigma_from_alpha_s, 10 * 0.2 * scale.time_unit_s)


if __name__ == "__main__":
    unittest.main()
