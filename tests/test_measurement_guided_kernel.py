"""Physical-time kernel and configuration wiring for measurement guidance."""

import math
import unittest

import numpy as np
from numpy.testing import assert_allclose

from config import AppConfig
from core.grid import build_grid
from core.potentials import build_potential
from theories.registry import build_theory
from theories.thick_front_measurement_guided import ThickFrontMeasurementGuidedTheory


class MeasurementGuidedKernelTests(unittest.TestCase):
    def _build(self, theory_name="thick_front_measurement_guided", **overrides):
        values = {
            "THEORY_NAME": theory_name,
            "N_VISIBLE_X": 12,
            "N_VISIBLE_Y": 8,
            "PAD_FACTOR": 1,
            "TRF_SIGMA_T": 0.4,
            "TRF_MEASUREMENT_RESPONSE_STRENGTH": 0.7,
            "TRF_MEASUREMENT_BACK_STRIDE": 3,
            "TRF_MEASUREMENT_BACK_HORIZON_SIGMAS": 5.0,
        }
        values.update(overrides)
        cfg = AppConfig(**values)
        grid = build_grid(
            visible_lx=cfg.VISIBLE_LX,
            visible_ly=cfg.VISIBLE_LY,
            n_visible_x=cfg.N_VISIBLE_X,
            n_visible_y=cfg.N_VISIBLE_Y,
            pad_factor=cfg.PAD_FACTOR,
        )
        return build_theory(cfg, grid, build_potential(grid, cfg))

    def test_registry_uses_one_physical_sigma_t(self):
        theory = self._build()
        self.assertIsInstance(theory, ThickFrontMeasurementGuidedTheory)
        self.assertEqual(theory.measurement_sigma_t, 0.4)
        self.assertEqual(theory.measurement_response_strength, 0.7)
        self.assertEqual(theory.measurement_back_stride, 3)
        self.assertEqual(theory.measurement_back_horizon_sigmas, 5.0)

        alias = self._build("thick_front_measured_guided")
        self.assertIsInstance(alias, ThickFrontMeasurementGuidedTheory)
        self.assertEqual(alias.measurement_sigma_t, 0.4)

    def test_dt_and_stride_only_set_physical_sampling_interval(self):
        fine_stride = self._build(TRF_MEASUREMENT_BACK_STRIDE=1)
        coarse_stride = self._build(TRF_MEASUREMENT_BACK_STRIDE=2)
        delays_a, weights_a = fine_stride._measurement_time_kernel(0.01)
        delays_b, weights_b = coarse_stride._measurement_time_kernel(0.005)
        assert_allclose(delays_a, delays_b, rtol=0, atol=1e-15)
        assert_allclose(weights_a, weights_b, rtol=0, atol=1e-15)

    def test_propagated_effect_field_is_step_size_stable(self):
        coarse = self._build(
            TRF_SIGMA_T=0.04,
            TRF_MEASUREMENT_BACK_STRIDE=1,
            TRF_MEASUREMENT_BACK_HORIZON_SIGMAS=4.0,
        )
        fine = self._build(
            TRF_SIGMA_T=0.04,
            TRF_MEASUREMENT_BACK_STRIDE=2,
            TRF_MEASUREMENT_BACK_HORIZON_SIGMAS=4.0,
        )
        coarse.measurement_debug_print = False
        fine.measurement_debug_print = False
        state = np.exp(
            -((coarse.grid.X - 7.0) ** 2 + coarse.grid.Y ** 2)
            / (2 * 1.5 ** 2)
        ).astype(np.complex128)
        state /= np.sqrt(
            np.sum(np.abs(state) ** 2) * coarse.grid.dx * coarse.grid.dy
        )
        effect_coarse = coarse._build_measurement_effect_mix(state, 0.01)
        effect_fine = fine._build_measurement_effect_mix(state, 0.005)
        assert_allclose(effect_coarse, effect_fine, rtol=1e-7, atol=1e-9)

    def test_time_step_refinement_converges_at_fixed_sigma_t(self):
        theory = self._build(
            TRF_SIGMA_T=0.25,
            TRF_MEASUREMENT_BACK_STRIDE=1,
            TRF_MEASUREMENT_BACK_HORIZON_SIGMAS=4.0,
        )
        target_second_moment = theory.measurement_sigma_t ** 2
        z = theory.measurement_back_horizon_sigmas
        target_second_moment *= 1 - (
            np.sqrt(2 / np.pi) * z * np.exp(-0.5 * z ** 2)
            / math.erf(z / np.sqrt(2))
        )
        errors = []
        for dt in [0.02, 0.01, 0.005]:
            delays, weights = theory._measurement_time_kernel(dt)
            errors.append(abs(np.sum(weights * delays ** 2) - target_second_moment))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertLess(errors[-1] / target_second_moment, 0.01)

    def test_horizon_converges_by_four_sigma(self):
        four_sigma = self._build(
            TRF_SIGMA_T=0.25,
            TRF_MEASUREMENT_BACK_STRIDE=1,
            TRF_MEASUREMENT_BACK_HORIZON_SIGMAS=4.0,
        )
        five_sigma = self._build(
            TRF_SIGMA_T=0.25,
            TRF_MEASUREMENT_BACK_STRIDE=1,
            TRF_MEASUREMENT_BACK_HORIZON_SIGMAS=5.0,
        )
        delays_4, weights_4 = four_sigma._measurement_time_kernel(0.0025)
        delays_5, weights_5 = five_sigma._measurement_time_kernel(0.0025)
        moment_4 = float(np.sum(weights_4 * delays_4 ** 2))
        moment_5 = float(np.sum(weights_5 * delays_5 ** 2))
        self.assertLess(abs(moment_4 - moment_5) / moment_5, 0.002)

    def test_invalid_kernel_configuration_is_rejected(self):
        with self.assertRaises(AssertionError):
            self._build(TRF_SIGMA_T=0)
        with self.assertRaises(AssertionError):
            self._build(TRF_MEASUREMENT_RESPONSE_STRENGTH=-1)
        with self.assertRaises(AssertionError):
            self._build(TRF_MEASUREMENT_BACK_STRIDE=0)
        with self.assertRaises(AssertionError):
            self._build(TRF_MEASUREMENT_BACK_HORIZON_SIGMAS=0)
        theory = self._build()
        theory.measurement_effect_normalization_scale = 0
        with self.assertRaises(ValueError):
            theory.__post_init__()


if __name__ == "__main__":
    unittest.main()
