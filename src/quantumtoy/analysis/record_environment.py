"""Exact monitored two-path experiment with physical qubit memory fragments.

Protocol: prepare S; record a weak Z probe; couple S to initially blank
memories; read every memory in X and S with a complete terminal instrument.
Each time sample describes a separately prepared run stopped at that time.
See paper/record_environment_model.md for the Hamiltonian and conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from analysis.stabilization import StabilizationWindow, sampled_stabilization


HADAMARD = np.array([[1, 1], [1, -1]], dtype=float) / np.sqrt(2)


@dataclass(frozen=True)
class RecordExperiment:
    """Fixed preparation, instruments, fragments, and clock thresholds.

    fragment_rates are fixed relative pulse rates, one per physical memory.
    The sweep multiplies these by g (inverse chosen time unit). Redundancy
    counts qualifying singleton memories, with a fixed X readout restriction.
    """

    path_probability: float = 0.5
    preparation_phase: float = 0.0
    probe_strength: float = 0.4
    fragment_rates: tuple[float, ...] = (1.0, 0.85, 0.70, 0.55)
    detector_angle: float = np.pi / 2
    detector_phase: float = 0.0
    detector_efficiencies: tuple[float, float] = (0.9, 0.8)
    information_deficit: float = 0.1
    required_copies: int = 3
    coherence_tolerance: float = 0.05
    hold_time: float = 0.5

    def __post_init__(self):
        # Freeze sequences too, so a caller cannot change the protocol mid-run.
        object.__setattr__(self, "fragment_rates", tuple(self.fragment_rates))
        object.__setattr__(self, "detector_efficiencies", tuple(self.detector_efficiencies))
        scalars = [self.path_probability, self.preparation_phase, self.probe_strength,
                   self.detector_angle, self.detector_phase, self.information_deficit,
                   self.coherence_tolerance, self.hold_time]
        if not np.all(np.isfinite(scalars)):
            raise ValueError("Experiment parameters must be finite")
        if not 0 < self.path_probability < 1:
            raise ValueError("A nontrivial branch ensemble requires 0 < path_probability < 1")
        if not 0 <= self.probe_strength <= 1:
            raise ValueError("probe_strength must lie in [0, 1]")
        if not 1 <= len(self.fragment_rates) <= 8:
            raise ValueError("Use between 1 and 8 explicit memory fragments")
        if not np.all(np.isfinite(self.fragment_rates)) or min(self.fragment_rates) < 0:
            raise ValueError("fragment_rates must be finite and nonnegative")
        if (len(self.detector_efficiencies) != 2
                or not np.all(np.isfinite(self.detector_efficiencies))
                or min(self.detector_efficiencies) < 0 or max(self.detector_efficiencies) > 1):
            raise ValueError("Two detector efficiencies in [0, 1] are required")
        if not 0 < self.information_deficit < 1:
            raise ValueError("information_deficit must lie strictly between 0 and 1")
        if (not isinstance(self.required_copies, (int, np.integer))
                or not 1 <= self.required_copies <= len(self.fragment_rates)):
            raise ValueError("required_copies must be an integer within the fragment count")
        if not 0 <= self.coherence_tolerance <= 1 or self.hold_time < 0:
            raise ValueError("Invalid coherence tolerance or hold time")

    @property
    def branch_probabilities(self):
        return np.array([self.path_probability, 1 - self.path_probability])

    @property
    def branch_entropy(self):
        return _entropy(self.branch_probabilities)


def _entropy(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    positive = probabilities[probabilities > 0]
    return float(-np.sum(positive * np.log(positive)))


def _mutual_information(joint):
    marginal_product = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
    supported = joint > 0
    return max(0.0, float(np.sum(joint[supported] * np.log(
        joint[supported] / marginal_product[supported]))))


def initial_state(experiment):
    """Pure system preparation in the fixed path (Z) basis."""
    return np.sqrt(experiment.branch_probabilities) * np.array(
        [1, np.exp(1j * experiment.preparation_phase)])


def probe_kraus(experiment):
    """Outcome +,- operators: M_a = sqrt((I + a kappa Z)/2)."""
    strength = experiment.probe_strength
    return np.array([np.diag(np.sqrt([1 + strength, 1 - strength]) / np.sqrt(2)),
                     np.diag(np.sqrt([1 - strength, 1 + strength]) / np.sqrt(2))])


def detector_kraus(experiment):
    """Terminal +,-,no_click Kraus operators, including their state update."""
    angle, phase = experiment.detector_angle, experiment.detector_phase
    plus = np.array([np.cos(angle / 2), np.exp(1j * phase) * np.sin(angle / 2)])
    p_plus = np.outer(plus, plus.conj())
    p_minus = np.eye(2) - p_plus
    eta_plus, eta_minus = experiment.detector_efficiencies
    return np.array([np.sqrt(eta_plus) * p_plus, np.sqrt(eta_minus) * p_minus,
                     np.sqrt(1 - eta_plus) * p_plus + np.sqrt(1 - eta_minus) * p_minus])


def fragment_states(experiment, g, time):
    """Conditional pure kets indexed [fragment, branch, memory_basis].

    theta_k(t) = pi/2 (1 - exp(-g r_k t)). The controlled Ry(+/-theta)
    pulse starts at zero and asymptotically stores the path in orthogonal
    X states. A fragment with zero rate never records anything.
    """
    if not np.isfinite(g) or g < 0 or not np.isfinite(time) or time < 0:
        raise ValueError("g and elapsed time must be finite and nonnegative")
    theta = -np.pi / 2 * np.expm1(-g * np.asarray(experiment.fragment_rates) * time)
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.stack([np.stack([c, s], axis=-1), np.stack([c, -s], axis=-1)], axis=1)


def _environment_kets(fragments):
    return np.array([_tensor_product(fragments[:, branch]) for branch in range(2)])


def _tensor_product(vectors):
    result = np.array([1.0])
    for vector in vectors:
        result = np.kron(result, vector)
    return result


def _probe_environment_state(experiment, fragments):
    environment = _environment_kets(fragments)
    system = np.array([m @ initial_state(experiment) for m in probe_kraus(experiment)])
    return system[:, :, None] * environment[None, :, :]


def probe_environment_state(experiment, g, time):
    """Unnormalized pure branches [probe a, system path, environment basis].

    The squared norm of branch a is its probe probability. Summing |Psi_a>
    <Psi_a| retains the unread physical probe's back-action.
    """
    return _probe_environment_state(experiment, fragment_states(experiment, g, time))


@lru_cache(maxsize=8)
def _environment_readout(number_of_fragments):
    return _tensor_product([HADAMARD] * number_of_fragments)


def pre_detector_record_vectors(experiment, g, time, *, readout_basis="X"):
    """Unnormalized system vectors indexed [probe, fragment_record, path].

    The fragment_record axis is the fixed-width binary X-readout record. Its
    squared vector norms form the complete earlier-record marginal p(a, Y).
    """
    if readout_basis == "X":
        readout = _environment_readout(len(experiment.fragment_rates))
    elif readout_basis == "Z":
        readout = np.eye(2 ** len(experiment.fragment_rates))
    else:
        raise ValueError("readout_basis must be 'X' or 'Z'")
    fragments = fragment_states(experiment, g, time)
    branches = _probe_environment_state(experiment, fragments)
    return np.transpose(
        branches @ readout.T,
        (0, 2, 1),
    )


@dataclass
class RecordSnapshot:
    # R is (fragment_record, detector_outcome); outcome signs are +,-.
    joint: np.ndarray  # [probe, fragment_record, detector(+,-,no_click)]
    system_density: np.ndarray  # unread probe, before terminal measurements
    terminal_system_vectors: np.ndarray  # [probe, fragment_record, detector, path]
    readout_information: np.ndarray  # I(B:Y_k), nats, for each fixed X readout
    holevo_information: np.ndarray  # chi(B:F_k), upper bounds, nats
    all_fragments_information: float  # I(B:Y_1,...,Y_N), not sum_k I(B:Y_k)
    coherence: float  # l1 coherence = 2 abs(rho_01) in the fixed path basis
    redundancy: int  # count of qualifying, physically fixed singleton fragments
    stable: bool

    def conditional_system_state(self, probe, fragment_record, detector):
        """Normalized post-measurement system state for one possible full record.

        The memory states after readout are the corresponding X eigenstates.
        Their tensor product with this state specifies the complete conditional
        quantum state. A zero-probability record has no conditional state.
        """
        vector = self.terminal_system_vectors[probe, fragment_record, detector]
        probability = self.joint[probe, fragment_record, detector]
        if probability <= 0:
            raise ValueError("Cannot condition on a zero-probability record")
        return np.outer(vector, vector.conj()) / probability


def record_snapshot(experiment, g, time):
    """Compute exact probabilities and record metrics at one stopping time."""
    fragments = fragment_states(experiment, g, time)
    branches = _probe_environment_state(experiment, fragments)
    rho = np.einsum("abe,ace->bc", branches, branches.conj())
    # Project each physical memory onto X. Tensor ordering is F0,F1,... .
    readout = branches @ _environment_readout(len(fragments)).T
    terminal = np.einsum("dij,ajr->ardi", detector_kraus(experiment), readout)
    joint = np.sum(np.abs(terminal) ** 2, axis=-1)

    prior = experiment.branch_probabilities
    information, holevo = [], []
    for states in fragments:
        likelihood = np.abs(states @ HADAMARD.T) ** 2
        information.append(_mutual_information(prior[:, None] * likelihood))
        average = np.einsum("b,bi,bj->ij", prior, states, states.conj())
        # Conditional fragment states are pure; their entropy term is zero.
        holevo.append(_entropy(np.maximum(np.linalg.eigvalsh(average), 0)))
    env_readout = _environment_kets(fragments) @ _environment_readout(len(fragments)).T
    all_information = _mutual_information(prior[:, None] * np.abs(env_readout) ** 2)
    information = np.asarray(information)
    redundancy = int(np.count_nonzero(
        information >= (1 - experiment.information_deficit) * experiment.branch_entropy))
    coherence = float(2 * abs(rho[0, 1]))
    stable = redundancy >= experiment.required_copies and coherence <= experiment.coherence_tolerance
    return RecordSnapshot(joint, rho, terminal, information, np.asarray(holevo),
                          all_information, coherence, redundancy, stable)


@dataclass
class RecordRun:
    experiment: RecordExperiment
    g: float
    times: np.ndarray
    joint: np.ndarray
    system_density: np.ndarray
    readout_information: np.ndarray
    holevo_information: np.ndarray
    all_fragments_information: np.ndarray
    coherence: np.ndarray
    redundancy: np.ndarray
    stable: np.ndarray
    stabilization: StabilizationWindow | None


def simulate_record_formation(experiment, g, times):
    """Sample exact states; dt controls the clock resolution, not evolution error."""
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times))
            or times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError("times must start at event time zero and increase strictly")
    snapshots = [record_snapshot(experiment, g, float(t)) for t in times]
    arrays = {name: np.array([getattr(s, name) for s in snapshots]) for name in (
        "joint", "system_density", "readout_information", "holevo_information",
        "all_fragments_information", "coherence", "redundancy", "stable")}
    window = sampled_stabilization(times, arrays["stable"], hold_time=experiment.hold_time)
    return RecordRun(experiment, float(g), times.copy(), **arrays, stabilization=window)
