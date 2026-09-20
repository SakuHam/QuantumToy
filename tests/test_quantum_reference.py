"""Analytic operational regressions from TRF-IT v0.2 (September 2026)."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from analysis.quantum_reference import (
    condition_on_record,
    joint_record_probabilities,
    rank_one_posterior,
)


IDENTITY = np.eye(2)
P0, P1 = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])
Z_INSTRUMENT = [[P0], [P1]]


def projector(vector):
    vector = np.asarray(vector, dtype=complex)
    return np.outer(vector, vector.conj())


def basis_effects(angle, phase=0.0):
    ket = [np.cos(angle / 2), np.exp(1j * phase) * np.sin(angle / 2)]
    effect = projector(ket)
    return [effect, IDENTITY - effect]


class QuantumReferenceTests(unittest.TestCase):
    def setUp(self):
        self.rho = projector(np.array([2, 1]) / np.sqrt(5))
        self.effects = [self.rho, IDENTITY - self.rho]

    def test_section_4_joint_table_and_same_experiment_recovery(self):
        joint = joint_record_probabilities(self.rho, Z_INSTRUMENT, self.effects)
        # Rows are intermediate a=0,1; columns are final R=+,-.
        assert_allclose(joint, np.array([[16, 4], [1, 4]]) / 25, atol=1e-14)
        assert_allclose(joint.sum(axis=0), [17 / 25, 8 / 25], atol=1e-14)
        posteriors = np.array([condition_on_record(joint, r) for r in range(2)])
        assert_allclose(posteriors, [[16 / 17, 1 / 17], [0.5, 0.5]], atol=1e-14)
        assert_allclose(joint.sum(axis=0) @ posteriors, [0.8, 0.2], atol=1e-14)

    def test_unread_instrument_differs_from_absent_instrument(self):
        measured = joint_record_probabilities(self.rho, Z_INSTRUMENT, self.effects)
        absent = joint_record_probabilities(self.rho, [[IDENTITY]], self.effects)
        unread = joint_record_probabilities(self.rho, [[P0, P1]], self.effects)
        assert_allclose(unread[0], measured.sum(axis=0), atol=1e-14)
        assert_allclose(absent[0], [1, 0], atol=1e-14)
        # Deliberately incompatible final weights give the paper's false failure.
        posteriors = np.array([condition_on_record(measured, r) for r in range(2)])
        wrong_recovery = absent[0] @ posteriors
        assert_allclose(wrong_recovery, [16 / 17, 1 / 17], atol=1e-14)
        self.assertGreater(np.max(np.abs(wrong_recovery - [0.8, 0.2])), 0.1)

    def test_full_outcomes_include_state_dependent_no_click(self):
        effects = [0.9 * P0, 0.2 * P1, 0.1 * P0 + 0.8 * P1]
        joint = joint_record_probabilities(self.rho, Z_INSTRUMENT, effects)
        assert_allclose(joint, [[0.72, 0, 0.08], [0, 0.04, 0.16]], atol=1e-14)
        self.assertAlmostEqual(joint.sum(), 1)
        assert_allclose(joint.sum(axis=1), [0.8, 0.2], atol=1e-14)
        surviving = joint[:, :2].sum(axis=1) / joint[:, :2].sum()
        self.assertGreater(np.max(np.abs(surviving - [0.8, 0.2])), 0.1)
        with self.assertRaisesRegex(ValueError, "complete"):
            joint_record_probabilities(self.rho, Z_INSTRUMENT, effects[:2])

    def test_loss_instrument_requires_its_missing_outcome(self):
        click = np.diag(np.sqrt([0.9, 0.2]))
        loss = np.diag(np.sqrt([0.1, 0.8]))
        joint = joint_record_probabilities(self.rho, [[click], [loss]], [P0, P1])
        assert_allclose(joint, [[0.72, 0.04], [0.08, 0.16]], atol=1e-14)
        with self.assertRaisesRegex(ValueError, "complete"):
            joint_record_probabilities(self.rho, [[click]], [P0, P1])

    def test_impossible_record_is_not_conditioned_but_rare_record_is(self):
        joint = joint_record_probabilities(P0, Z_INSTRUMENT, [P0, P1])
        with self.assertRaisesRegex(ValueError, "zero-probability"):
            condition_on_record(joint, 1)
        rare = np.array([[1 - 1e-16, 1e-16], [0, 0]])
        assert_allclose(condition_on_record(rare, 1), [1, 0])

    def test_rank_one_rule_matches_operator_conditioning(self):
        for effect in self.effects:
            joint = joint_record_probabilities(self.rho, Z_INSTRUMENT,
                                               [effect, IDENTITY - effect])
            scalar = rank_one_posterior(np.diag(self.rho).real, np.diag(effect).real)
            assert_allclose(scalar, condition_on_record(joint, 0), atol=1e-14)

    def test_weak_probe_retains_coherence_and_includes_back_action(self):
        plus = projector(np.array([1, 1]) / np.sqrt(2))
        weak = [[np.diag(np.sqrt([0.8, 0.2]))],
                [np.diag(np.sqrt([0.2, 0.8]))]]
        effects = [plus, IDENTITY - plus]
        joint = joint_record_probabilities(plus, weak, effects)
        assert_allclose(joint, [[0.45, 0.05], [0.45, 0.05]], atol=1e-14)
        absent = joint_record_probabilities(plus, [[IDENTITY]], effects)
        strong = joint_record_probabilities(plus, Z_INSTRUMENT, effects)
        assert_allclose(absent.sum(axis=0), [1, 0], atol=1e-14)
        assert_allclose(strong.sum(axis=0), [0.5, 0.5], atol=1e-14)

    def test_dissipative_channel_matches_backward_adjoint_effect(self):
        damping = 0.3
        channel = [np.diag([1, np.sqrt(1 - damping)]),
                   np.array([[0, np.sqrt(damping)], [0, 0]])]
        joint = joint_record_probabilities(self.rho, Z_INSTRUMENT, [P0, P1],
                                           channel=channel)
        assert_allclose(joint, [[0.8, 0], [0.06, 0.14]], atol=1e-14)
        # Independently propagate a complex-basis effect backward by the adjoint.
        effects = basis_effects(0.8, 0.6)
        backward = [sum(k.conj().T @ e @ k for k in channel) for e in effects]
        forward_joint = joint_record_probabilities(self.rho, [[IDENTITY]], effects,
                                                   channel=channel)
        adjoint_joint = joint_record_probabilities(self.rho, [[IDENTITY]], backward)
        assert_allclose(forward_joint, adjoint_joint, atol=1e-14)
        with self.assertRaisesRegex(ValueError, "complete"):
            joint_record_probabilities(self.rho, Z_INSTRUMENT, [P0, P1],
                                       channel=channel[:1])

    def test_later_setting_does_not_change_recorded_weak_probe_marginal(self):
        weak = [[np.diag(np.sqrt([0.8, 0.2]))],
                [np.diag(np.sqrt([0.2, 0.8]))]]
        for angle in [0, 0.7, np.pi / 2, np.pi]:
            for phase in [0, 0.6, np.pi / 2]:
                with self.subTest(angle=angle, phase=phase):
                    joint = joint_record_probabilities(self.rho, weak,
                                                       basis_effects(angle, phase))
                    assert_allclose(joint.sum(axis=1), [0.68, 0.32], atol=1e-14)

    def test_singlet_local_marginals_and_quantum_correlations(self):
        singlet = projector(np.array([0, 1, -1, 0]) / np.sqrt(2))
        signs = np.array([1, -1])
        for u in [0, 0.7, np.pi / 2]:
            for v in [0, 0.4, np.pi]:
                with self.subTest(u=u, v=v):
                    alice = [[np.kron(e, IDENTITY)] for e in basis_effects(u)]
                    bob = [np.kron(IDENTITY, e) for e in basis_effects(v)]
                    joint = joint_record_probabilities(singlet, alice, bob)
                    assert_allclose(joint.sum(axis=0), [0.5, 0.5], atol=1e-14)
                    assert_allclose(joint.sum(axis=1), [0.5, 0.5], atol=1e-14)
                    self.assertAlmostEqual(signs @ joint @ signs, -np.cos(u - v))

    def test_invalid_states_and_nonpositive_effects_are_rejected(self):
        for state in [np.diag([1.1, -0.1]), 2 * self.rho,
                      np.array([[0.5, 1j], [0, 0.5]])]:
            with self.subTest(state=state), self.assertRaises(ValueError):
                joint_record_probabilities(state, Z_INSTRUMENT, [P0, P1])
        with self.assertRaisesRegex(ValueError, "positive"):
            joint_record_probabilities(self.rho, Z_INSTRUMENT,
                                       [1.1 * P0, IDENTITY - 1.1 * P0])


if __name__ == "__main__":
    unittest.main()
