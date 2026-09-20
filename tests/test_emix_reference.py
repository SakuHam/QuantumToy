"""Effect mixtures versus geometric evidence and coherent amplitude sums."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from analysis.emix import (
    build_Emix_density_from_phi_tau,
    build_Emix_from_phi_tau,
    make_emix_density,
)
from analysis.quantum_reference import mix_record_effects, rank_one_posterior


class EffectMixtureTests(unittest.TestCase):
    def test_section_5_arithmetic_and_geometric_results(self):
        likelihoods = np.array([[0.9, 0.2], [0.1, 0.8]])
        mixed = mix_record_effects([np.diag(e) for e in likelihoods], [0.5, 0.5])
        assert_allclose(np.diag(mixed), [0.5, 0.5], atol=1e-14)
        arithmetic = rank_one_posterior([0.5, 0.5], np.diag(mixed).real)
        geometric = rank_one_posterior([0.5, 0.5],
                                       np.exp(np.mean(np.log(likelihoods), axis=0)))
        assert_allclose(arithmetic, [0.5, 0.5], atol=1e-14)
        assert_allclose(geometric, [3 / 7, 4 / 7], atol=1e-14)

    def test_component_posteriors_need_record_probability_weights(self):
        prior = np.array([0.8, 0.2])
        likelihoods = np.array([[0.9, 0.2], [0.1, 0.8]])
        weights = np.array([0.5, 0.5])
        posteriors = np.array([rank_one_posterior(prior, e) for e in likelihoods])
        record_weights = weights * (likelihoods @ prior)
        record_weights /= record_weights.sum()
        correct = rank_one_posterior(prior, weights @ likelihoods)
        assert_allclose(record_weights @ posteriors, correct, atol=1e-14)
        assert_allclose(correct, prior, atol=1e-14)
        self.assertGreater(np.max(np.abs(weights @ posteriors - correct)), 0.1)

    def test_full_effect_mixture_is_invariant_to_component_global_phases(self):
        vectors = np.array([[1, 1j], [2, 1]], dtype=complex)
        vectors /= np.linalg.norm(vectors, axis=1)[:, None]
        shifted = vectors * np.exp(1j * np.array([0.3, 2.2]))[:, None]
        effects = [np.outer(v, v.conj()) for v in vectors]
        shifted_effects = [np.outer(v, v.conj()) for v in shifted]
        mixed = mix_record_effects(effects, [0.25, 0.75])
        assert_allclose(mixed, mix_record_effects(shifted_effects, [0.25, 0.75]),
                        atol=1e-14)
        # Off-diagonal entries are retained, not just the scalar density.
        self.assertGreater(abs(mixed[0, 1]), 0.1)

    def test_existing_density_mix_is_phase_invariant_but_amplitude_mix_is_not(self):
        scalar = np.sqrt(np.array([[0.9, 0.2], [0.1, 0.8], [0.4, 0.6]]))[:, None, :]
        # Exercise scalar, spinor and entangled storage layouts in existing code.
        spin = np.stack([scalar / np.sqrt(2), 1j * scalar / np.sqrt(2)], axis=1)
        entangled = np.zeros((3, 1, 2, 2, 2), dtype=complex)
        entangled[..., 0, 1] = scalar / np.sqrt(2)
        entangled[..., 1, 0] = -scalar / np.sqrt(2)
        kwargs = dict(times=np.arange(3.0), t_det=1.0, sigmaT=1.0,
                      tau_step=1.0, K_JITTER=3)
        weights = np.exp(-0.5 * np.array([1, 0, 1]))
        weights /= weights.sum()
        reference = mix_record_effects(
            [np.diag(row) for row in np.abs(scalar[:, 0]) ** 2], weights)
        for frames in [scalar, spin, entangled]:
            with self.subTest(shape=frames.shape):
                phases = np.exp(1j * np.array([0, np.pi, 0.7]))
                shifted = frames * phases.reshape((3,) + (1,) * (frames.ndim - 1))
                density = build_Emix_density_from_phi_tau(frames, **kwargs)
                changed_density = build_Emix_density_from_phi_tau(shifted, **kwargs)
                assert_allclose(density, changed_density, atol=1e-14)
                assert_allclose(density[0, 0], np.diag(reference).real, atol=1e-14)
                coherent = make_emix_density(build_Emix_from_phi_tau(frames, **kwargs))
                changed_coherent = make_emix_density(
                    build_Emix_from_phi_tau(shifted, **kwargs))
                self.assertGreater(np.max(np.abs(coherent - changed_coherent)), 0.01)

    def test_floor_converges_and_has_an_explicit_error_bound(self):
        prior, likelihood = np.array([0.8, 0.2]), np.array([0.0, 0.6])
        exact = rank_one_posterior(prior, likelihood)
        assert_allclose(exact, [0, 1], atol=1e-14)
        previous_error = np.inf
        for epsilon in [1e-2, 1e-4, 1e-6, 1e-8]:
            floored = rank_one_posterior(prior, likelihood, epsilon=epsilon)
            error = np.sum(np.abs(floored - exact))
            # p_epsilon is a convex mixture of exact posterior and prior.
            bound = 2 * epsilon / (prior @ likelihood + epsilon)
            self.assertLessEqual(error, bound + 1e-14)
            self.assertLess(error, previous_error)
            previous_error = error
        assert_allclose(rank_one_posterior(prior, 7 * likelihood, reference_scale=7,
                                           epsilon=0.01),
                        rank_one_posterior(prior, likelihood, epsilon=0.01), atol=1e-14)

    def test_floor_does_not_create_a_physically_possible_record(self):
        for epsilon in [0, 1e-6, 1]:
            with self.subTest(epsilon=epsilon), self.assertRaisesRegex(
                    ValueError, "zero-probability"):
                rank_one_posterior([1, 0], [0, 1], epsilon=epsilon)

    def test_effect_mixture_rejects_invalid_prior_weights(self):
        for weights in [[0.3, 0.3], [-0.1, 1.1], [np.nan, 0.5]]:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                mix_record_effects([np.eye(2), np.eye(2)], weights)


if __name__ == "__main__":
    unittest.main()
