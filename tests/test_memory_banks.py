"""Record redundancy must not create quantum events or circumvent shared loss."""
from dataclasses import replace
import unittest

import numpy as np
from numpy.testing import assert_allclose

from analysis.memory_banks import MemoryBankConfig, evaluate_memory_banks
from analysis.temporal_history_profile import TemporalHistoryEnvelope


class MemoryBankTests(unittest.TestCase):
    def setUp(self):
        self.env = TemporalHistoryEnvelope(retention_time=0, fade_time=1, fade_power=1)
        self.joint = np.array([[.1, .2], [.05, .15]])
        self.times = np.array([0., 1.])

    def law(self, config=MemoryBankConfig(), readout=2):
        return evaluate_memory_banks(self.joint, self.times, self.env,
                                    readout_time=readout, config=config)

    def test_zero_banks_and_perfect_archive(self):
        absent = self.law(MemoryBankConfig(0, 0))
        assert_allclose(absent.probabilities, [0, 0, 0, .5, .5])
        perfect = self.law(MemoryBankConfig(1, 0, 1))
        assert_allclose(perfect.either_joint, self.joint)
        self.assertEqual(perfect.unrecovered_record, 0)

    def test_independent_enumeration_of_copy_outcomes(self):
        # Enumerate all 16 outcomes of two reference and two delayed bits.
        expected = np.zeros(5)
        for time, mass in zip(self.times, self.joint.sum(axis=1)):
            p = [.7, .7, np.exp(time-2), np.exp(time-2)]
            for code in range(16):
                bits = [(code >> k) & 1 for k in range(4)]
                weight = mass*np.prod([q if bit else 1-q for bit, q in zip(bits, p)])
                ref, delayed = any(bits[:2]), any(bits[2:])
                index = 0 if ref and delayed else 1 if ref else 2 if delayed else 3
                expected[index] += weight
        expected[-1] = .5
        assert_allclose(self.law(MemoryBankConfig(2, 2, .7)).probabilities, expected)

    def test_shared_loss_cannot_be_overcome_with_extra_copies(self):
        one = self.law(MemoryBankConfig(0, 1, loss_mode="shared"))
        many = self.law(MemoryBankConfig(0, 16, loss_mode="shared"))
        assert_allclose(one.probabilities, many.probabilities)
        self.assertAlmostEqual(many.expected_readable_copies, 16*one.expected_readable_copies)
        independent = self.law(MemoryBankConfig(0, 16))
        self.assertGreater(independent.either_joint.sum(), many.either_joint.sum())

    def test_monotone_counts_delay_and_birth_age(self):
        old = self.joint.copy()
        previous = 0
        for n in [0, 1, 2, 8, 64]:
            law = self.law(MemoryBankConfig(0, n))
            self.assertGreaterEqual(law.either_joint.sum(), previous)
            self.assertLessEqual(law.either_joint.sum(), self.joint.sum())
            self.assertAlmostEqual(law.probabilities.sum(), 1)
            previous = law.either_joint.sum()
        early, late = self.law(readout=2), self.law(readout=4)
        self.assertTrue(np.all(early.delayed_joint >= late.delayed_joint))
        assert_allclose(early.reference_joint, late.reference_joint)
        # Newer records survive better; averaging the ages first is incorrect.
        ratio = early.delayed_joint/self.joint
        self.assertTrue(np.all(ratio[1] > ratio[0]))
        assert_allclose(self.joint, old)

    def test_limits_and_validation(self):
        for p in [0, 1]:
            for n in [0, 1, 64]:
                law = self.law(MemoryBankConfig(n, n, p))
                self.assertTrue(np.all(law.probabilities >= 0))
                self.assertAlmostEqual(law.probabilities.sum(), 1)
        for kwargs in [dict(reference_copies=-1), dict(delayed_copies=.5),
                       dict(delayed_copies=True), dict(reference_survival=np.nan),
                       dict(loss_mode="unknown")]:
            with self.assertRaises(ValueError):
                MemoryBankConfig(**kwargs)
        with self.assertRaises(ValueError):
            self.law(readout=.5)
        with self.assertRaises(ValueError):
            evaluate_memory_banks(3*self.joint, self.times, self.env, readout_time=2)

    def test_repeated_perfect_reads_do_not_restore_or_refresh_records(self):
        for mode in ["independent", "shared"]:
            config = MemoryBankConfig(0, 3, loss_mode=mode)
            one = self.law(config)
            many = self.law(replace(config, read_count=8, read_spacing=.5))
            assert_allclose(many.any_read_joint, one.any_read_joint)
            self.assertTrue(np.all(many.delayed_joint < one.delayed_joint))
            self.assertTrue(np.all(many.any_read_joint >= many.delayed_joint))

    def test_retries_match_explicit_persistent_lifetime_enumeration(self):
        # One copy: lifetime categories and all three Bernoulli read outcomes.
        eta, spacing = .4, .3
        expected = np.zeros_like(self.joint)
        for i, birth in enumerate(self.times):
            R = np.exp(-(2+np.arange(3)*spacing-birth))
            lifetime = np.r_[1-R[0], R[:-1]-R[1:], R[-1]]
            success = 0
            for live_reads in range(4):
                for bits in range(8):
                    outcomes = [(bits >> k) & 1 for k in range(3)]
                    weight = np.prod([eta if bit else 1-eta for bit in outcomes])
                    success += lifetime[live_reads]*weight*any(outcomes[:live_reads])
            expected[i] = self.joint[i]*success
        cfg = MemoryBankConfig(0, 1, read_count=3, read_spacing=spacing, read_efficiency=eta)
        law = self.law(cfg)
        assert_allclose(law.any_read_joint, expected)
        self.assertGreater(law.any_read_joint.sum(), self.law(replace(cfg, read_count=1)).any_read_joint.sum())
        self.assertAlmostEqual(law.probabilities.sum(), 1)

    def test_shared_lifetime_retries_and_zero_read_efficiency(self):
        cfg = MemoryBankConfig(0, 4, loss_mode="shared", read_count=3, read_efficiency=.2)
        # Simultaneous reads: one shared lifetime, twelve independent read attempts.
        law = self.law(cfg)
        expected = self.joint*np.exp(self.times[:, None]-2)*(1-.8**12)
        assert_allclose(law.any_read_joint, expected)
        unread = self.law(replace(cfg, read_efficiency=0))
        assert_allclose(unread.probabilities, [0, 0, 0, .5, .5])


if __name__ == '__main__':
    unittest.main()
