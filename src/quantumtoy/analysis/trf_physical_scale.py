"""Dimensionful closure for the synthetic spatial TRF timing model.

The spatial solver uses a nondimensional Schrödinger equation.  Choosing one
physical length unit L0 and a particle mass fixes its time, velocity, and
energy units.  A separate direct-delay closure then identifies the latent
delay in ``U(t + tau)`` with an additive detector-clock delay.  That closure
predicts kappa_t = 1; it is an extra falsifiable model assumption, not a value
derived by TRF-IT v0.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import hbar


REFERENCE_SPATIAL_SIGMA_T = 0.2
REFERENCE_STABILIZATION_LATENCY = 2.76
LOCKED_REFERENCE_ALPHA = (
    REFERENCE_SPATIAL_SIGMA_T / REFERENCE_STABILIZATION_LATENCY)


@dataclass(frozen=True)
class SchrodingerPhysicalScale:
    """Physical units corresponding to one nondimensional simulation unit."""

    particle_mass_kg: float
    length_unit_m: float

    def __post_init__(self):
        values = (self.particle_mass_kg, self.length_unit_m)
        if not np.all(np.isfinite(values)) or min(values) <= 0:
            raise ValueError("mass and length unit must be finite and positive")

    @property
    def time_unit_s(self):
        """T0 = m L0^2 / hbar."""
        return self.particle_mass_kg * self.length_unit_m ** 2 / hbar

    @property
    def velocity_unit_m_per_s(self):
        return self.length_unit_m / self.time_unit_s

    @property
    def energy_unit_j(self):
        return hbar / self.time_unit_s

    def physical_time(self, dimensionless_time):
        return np.asarray(dimensionless_time) * self.time_unit_s

    def physical_length(self, dimensionless_length):
        return np.asarray(dimensionless_length) * self.length_unit_m

    def physical_velocity(self, dimensionless_velocity):
        return np.asarray(dimensionless_velocity) * self.velocity_unit_m_per_s


def scale_from_slit_separation(
        particle_mass_kg, slit_separation_m, *,
        dimensionless_slit_separation=1.6):
    """Fix L0 by matching the simulated center-to-center slit separation."""
    values = (float(slit_separation_m), float(dimensionless_slit_separation))
    if not np.all(np.isfinite(values)) or min(values) <= 0:
        raise ValueError("slit separations must be finite and positive")
    return SchrodingerPhysicalScale(
        float(particle_mass_kg), values[0] / values[1])


def scale_from_packet_velocity(
        particle_mass_kg, packet_velocity_m_per_s, *, packet_kx=3.0):
    """Fix L0 from v = hbar kx / (m L0) for the simulated packet."""
    mass = float(particle_mass_kg)
    velocity = float(packet_velocity_m_per_s)
    packet_kx = float(packet_kx)
    if (not np.all(np.isfinite((mass, velocity, packet_kx)))
            or min(mass, velocity, packet_kx) <= 0):
        raise ValueError("mass, packet velocity, and packet_kx must be positive")
    return SchrodingerPhysicalScale(
        mass, hbar * packet_kx / (mass * velocity))


@dataclass(frozen=True)
class DirectDelayPrediction:
    """Dimensionful moments of the minimal direct arrival-delay closure."""

    time_unit_s: float
    sigma_t_s: float
    delay_scale: float
    response_fraction: float
    responding_mean_delay_s: float
    responding_standard_deviation_s: float
    ensemble_mean_delay_s: float
    ensemble_standard_deviation_s: float
    source_pulse_sigma_s: float
    detector_jitter_s: float
    time_bin_width_s: float
    reference_time_s: float
    classical_arrival_time_s: float


def direct_delay_prediction(
        scale, *, sigma_t=0.2, lambda_strength=1.0,
        delay_scale=1.0, source_pulse_sigma=0.03,
        detector_jitter=0.05, time_bin_width=0.05,
        reference_time=0.9, packet_x=-2.5, detector_x=1.5,
        packet_kx=3.0, dimensionless_hbar=1.0, dimensionless_mass=1.0):
    """Map the current benchmark to seconds under an explicit closure.

    The response branch has a half-normal delay with scale ``sigma_t``.  The
    full ensemble also contains a zero-delay branch with weight exp(-lambda).
    ``delay_scale=1`` is the minimal direct-delay prediction because physical
    recorded and latent delays are both multiplied by the same T0.
    """
    if not isinstance(scale, SchrodingerPhysicalScale):
        raise TypeError("scale must be a SchrodingerPhysicalScale")
    values = np.asarray([
        sigma_t, lambda_strength, delay_scale, source_pulse_sigma,
        detector_jitter, time_bin_width, reference_time, packet_kx,
        dimensionless_hbar, dimensionless_mass,
    ], dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values[:1] <= 0):
        raise ValueError("prediction parameters must be finite and sigma_t positive")
    if (lambda_strength < 0 or delay_scale < 0 or source_pulse_sigma < 0
            or detector_jitter < 0 or time_bin_width <= 0
            or reference_time < 0 or packet_kx <= 0
            or dimensionless_hbar <= 0 or dimensionless_mass <= 0):
        raise ValueError("prediction scales must be in their physical domains")
    response_fraction = -np.expm1(-lambda_strength)
    responding_mean = delay_scale * sigma_t * np.sqrt(2 / np.pi)
    responding_variance = (delay_scale * sigma_t) ** 2 * (1 - 2 / np.pi)
    ensemble_mean = response_fraction * responding_mean
    ensemble_second_moment = response_fraction * (delay_scale * sigma_t) ** 2
    ensemble_variance = max(0.0, ensemble_second_moment - ensemble_mean ** 2)
    velocity = dimensionless_hbar * packet_kx / dimensionless_mass
    classical_arrival = (detector_x - packet_x) / velocity
    unit = scale.time_unit_s
    return DirectDelayPrediction(
        time_unit_s=unit,
        sigma_t_s=sigma_t * unit,
        delay_scale=float(delay_scale),
        response_fraction=float(response_fraction),
        responding_mean_delay_s=responding_mean * unit,
        responding_standard_deviation_s=np.sqrt(responding_variance) * unit,
        ensemble_mean_delay_s=ensemble_mean * unit,
        ensemble_standard_deviation_s=np.sqrt(ensemble_variance) * unit,
        source_pulse_sigma_s=source_pulse_sigma * unit,
        detector_jitter_s=detector_jitter * unit,
        time_bin_width_s=time_bin_width * unit,
        reference_time_s=reference_time * unit,
        classical_arrival_time_s=classical_arrival * unit,
    )


@dataclass(frozen=True)
class ClockBridge:
    """Consistency relation between record and spatial simulation clocks."""

    physical_rate_per_s: float
    schrodinger_time_unit_s: float
    dimensionless_rate: float
    stabilization_latency_s: float
    sigma_from_alpha_s: float
    alpha_required_for_spatial_sigma: float


def bridge_record_and_spatial_clocks(
        scale, physical_record_rate_per_s, *, dimensionless_g=1.0,
        stabilization_latency=2.76, alpha=1.0,
        spatial_sigma_t=0.2):
    """Compare a calibrated record rate with the spatial Schrödinger clock.

    In the record model ``g * t`` is dimensionless.  A physical base rate G
    therefore corresponds to dimensionless ``g = G T0``.  The default record
    and spatial fixtures do not share a clock automatically: at the baseline
    record latency 2.76, reproducing spatial sigma 0.2 requires
    alpha = 0.2 / 2.76 rather than the response fixture's alpha = 1.
    """
    if not isinstance(scale, SchrodingerPhysicalScale):
        raise TypeError("scale must be a SchrodingerPhysicalScale")
    values = np.asarray([
        physical_record_rate_per_s, dimensionless_g,
        stabilization_latency, alpha, spatial_sigma_t,
    ], dtype=float)
    if np.any(~np.isfinite(values)) or min(values) <= 0:
        raise ValueError("clock bridge values must be finite and positive")
    dimensionless_rate = physical_record_rate_per_s * scale.time_unit_s
    # The latency scales inversely with g in the analytic record fixture.
    latency_simulation_units = stabilization_latency * dimensionless_g / dimensionless_rate
    latency_s = latency_simulation_units * scale.time_unit_s
    return ClockBridge(
        physical_rate_per_s=float(physical_record_rate_per_s),
        schrodinger_time_unit_s=scale.time_unit_s,
        dimensionless_rate=float(dimensionless_rate),
        stabilization_latency_s=float(latency_s),
        sigma_from_alpha_s=float(alpha * latency_s),
        alpha_required_for_spatial_sigma=float(
            spatial_sigma_t * scale.time_unit_s / latency_s),
    )
