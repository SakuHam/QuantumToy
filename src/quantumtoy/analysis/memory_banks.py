"""Classical copies of one detector record, read after a variable delay.

Copies are written when a classical record is formed, before erasure. They
are not extra quantum measurements. Redundancy only helps against independent
copy loss; the shared mode erases all delayed copies of an event together.
"""
from dataclasses import dataclass

import numpy as np

from analysis.temporal_history_profile import record_survival


@dataclass(frozen=True)
class MemoryBankConfig:
    reference_copies: int = 1
    delayed_copies: int = 1
    reference_survival: float = .995
    loss_mode: str = "independent"
    read_count: int = 1
    read_spacing: float = 0.0
    read_efficiency: float = 1.0

    def __post_init__(self):
        for count in (self.reference_copies, self.delayed_copies):
            if (isinstance(count, (bool, np.bool_))
                    or not isinstance(count, (int, np.integer)) or not 0 <= count <= 64):
                raise ValueError("Memory counts must be integers from 0 to 64")
        if not np.isfinite(self.reference_survival) or not 0 <= self.reference_survival <= 1:
            raise ValueError("Reference survival must lie in [0, 1]")
        if self.loss_mode not in ("independent", "shared"):
            raise ValueError("loss_mode must be independent or shared")
        if (isinstance(self.read_count, (bool, np.bool_))
                or not isinstance(self.read_count, (int, np.integer)) or not 1 <= self.read_count <= 64):
            raise ValueError("read_count must be an integer from 1 to 64")
        if not np.isfinite(self.read_spacing) or self.read_spacing < 0:
            raise ValueError("read_spacing must be finite and nonnegative")
        if not np.isfinite(self.read_efficiency) or not 0 <= self.read_efficiency <= 1:
            raise ValueError("read_efficiency must lie in [0, 1]")


@dataclass
class MemoryBankLaw:
    reference_joint: np.ndarray
    delayed_joint: np.ndarray
    any_read_joint: np.ndarray
    either_joint: np.ndarray
    both_joint: np.ndarray
    unrecovered_record: float
    no_record: float
    expected_readable_copies: float

    @property
    def probabilities(self):
        """Disjoint outcomes per preparation, counting every event once."""
        return np.array([
            self.both_joint.sum(),
            (self.reference_joint-self.both_joint).sum(),
            (self.any_read_joint-self.both_joint).sum(),
            self.unrecovered_record,
            self.no_record,
        ])


def evaluate_memory_banks(record_joint, creation_times, envelope, *, readout_time,
                          config=MemoryBankConfig()):
    """Unique readable events, including loss and preparations without records.

    Input is an unconditional [true creation time, outcome] record law, before
    memory erasure. Dark records belong in it too. Timestamp noise must not be
    substituted for the true creation time when evaluating lifetime.

    readout_time is the FIRST read; subsequent reads have config.read_spacing.
    Reading is nondestructive and never refreshes lifetime. Retrieval fails
    independently with probability 1-read_efficiency on each surviving copy
    and read. Successful results are retained in an ideal external read log.
    delayed_joint describes the last read; any_read_joint describes that log.

    Reference copies retain each record with fixed probability r, independently
    of one another and of the delayed bank; this idealized stable archive has
    no ageing over the studied delay range. Delayed copies each have survival
    R(age). Their losses are independent conditional on age, or fully shared
    within that bank. Neither mode changes the upstream detector or state.
    """
    joint = np.asarray(record_joint, dtype=float)
    times = np.asarray(creation_times, dtype=float)
    if (joint.ndim != 2 or 0 in joint.shape or times.shape != (joint.shape[0],)
            or not np.all(np.isfinite(joint)) or np.any(joint < 0)
            or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)
            or not np.isfinite(readout_time) or np.any(times > readout_time)):
        raise ValueError("Expected finite nonnegative records and ordered past creation times")
    mass = float(joint.sum())
    if mass > 1:
        raise ValueError("Record probability cannot exceed one; do not pass normalized counts")
    read_times = readout_time + np.arange(config.read_count)*config.read_spacing
    survival = record_survival(read_times[:, None]-times[None, :], envelope)
    reference = 1-(1-config.reference_survival)**config.reference_copies
    eta = config.read_efficiency
    if config.delayed_copies == 0:
        delayed = np.zeros_like(times)
        any_read = np.zeros_like(times)
    elif config.loss_mode == "shared":
        per_read = 1-(1-eta)**config.delayed_copies
        delayed = survival[-1]*per_read
        any_read = np.sum(survival*(per_read*(1-per_read)**np.arange(config.read_count))[:, None], axis=0)
    else:
        delayed = 1-(1-eta*survival[-1])**config.delayed_copies
        first_success = np.sum(survival*(eta*(1-eta)**np.arange(config.read_count))[:, None], axis=0)
        any_read = 1-(1-first_success)**config.delayed_copies
    ref_joint = joint*reference
    delayed_joint = joint*delayed[:, None]
    any_joint = joint*any_read[:, None]
    both = any_joint*reference
    either = ref_joint + any_joint*(1-reference)
    lost = float(np.sum(joint*(1-reference)*(1-any_read[:, None])))
    readable_copies = float(np.sum(joint*(
        config.reference_copies*config.reference_survival
        + config.delayed_copies*eta*survival[-1, :, None])))
    return MemoryBankLaw(ref_joint, delayed_joint, any_joint, either, both, lost,
                         1-mass, readable_copies)
