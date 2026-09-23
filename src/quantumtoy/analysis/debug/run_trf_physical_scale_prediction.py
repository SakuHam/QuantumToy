"""Derive physical clock scales and direct-delay predictions for the TRF model."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import atomic_mass, electron_volt, hbar, m_e, m_n
from scipy.special import erf

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.trf_physical_scale import (
    LOCKED_REFERENCE_ALPHA,
    REFERENCE_SPATIAL_SIGMA_T,
    REFERENCE_STABILIZATION_LATENCY,
    bridge_record_and_spatial_clocks,
    direct_delay_prediction,
    scale_from_packet_velocity,
    scale_from_slit_separation,
)


PARTICLES = {
    "electron": m_e,
    "neutron": m_n,
    "c60": 720 * atomic_mass,
}
DIMENSIONLESS_SLIT_SEPARATION = 1.6
PACKET_KX = 3.0
SIGMA_T = REFERENCE_SPATIAL_SIGMA_T
LAMBDA_STRENGTH = 1.0


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _scenario(name, particle, scale, construction, input_value):
    mass = PARTICLES[particle]
    prediction = direct_delay_prediction(scale)
    record_prediction = direct_delay_prediction(
        scale, sigma_t=REFERENCE_STABILIZATION_LATENCY)
    slit_separation = DIMENSIONLESS_SLIT_SEPARATION * scale.length_unit_m
    velocity = PACKET_KX * scale.velocity_unit_m_per_s
    kinetic_energy = 0.5 * mass * velocity ** 2
    bridge = bridge_record_and_spatial_clocks(
        scale, 1 / scale.time_unit_s)
    return {
        "name": name,
        "particle": particle,
        "particle_mass_kg": mass,
        "construction": construction,
        "input_value": input_value,
        "length_unit_m": scale.length_unit_m,
        "slit_separation_m": slit_separation,
        "packet_velocity_m_per_s": velocity,
        "packet_kinetic_energy_j": kinetic_energy,
        "packet_kinetic_energy_ev": kinetic_energy / electron_volt,
        "time_unit_s": scale.time_unit_s,
        "energy_unit_j": scale.energy_unit_j,
        "barrier_height_j": 35 * scale.energy_unit_j,
        "prediction": asdict(prediction),
        "record_clock_alpha1_prediction": asdict(record_prediction),
        "record_clock_if_G_equals_1_over_T0": asdict(bridge),
    }


def _format_time(seconds):
    for scale, suffix in (
            (1, "s"), (1e-3, "ms"), (1e-6, "us"),
            (1e-9, "ns"), (1e-12, "ps"), (1e-15, "fs")):
        if seconds >= scale:
            return f"{seconds / scale:.4g} {suffix}"
    return f"{seconds:.4g} s"


def _plot(scenarios, output):
    separations = np.logspace(-9, -3, 240)
    colors = {"electron": "#2563eb", "neutron": "#f97316", "c60": "#16a34a"}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)

    for particle, mass in PARTICLES.items():
        length_units = separations / DIMENSIONLESS_SLIT_SEPARATION
        time_units = mass * length_units ** 2 / hbar
        axes[0, 0].loglog(
            separations, SIGMA_T * time_units,
            color=colors[particle], label=particle)
        axes[0, 1].loglog(
            separations, PACKET_KX * length_units / time_units,
            color=colors[particle], label=particle)
    for scenario in scenarios:
        axes[0, 0].scatter(
            scenario["slit_separation_m"],
            scenario["prediction"]["sigma_t_s"],
            color=colors[scenario["particle"]], edgecolor="black", zorder=4)
        axes[0, 1].scatter(
            scenario["slit_separation_m"], scenario["packet_velocity_m_per_s"],
            color=colors[scenario["particle"]], edgecolor="black", zorder=4)
    axes[0, 0].set(
        xlabel="physical slit separation (m)", ylabel="predicted sigma_T (s)",
        title="Clock scale from T0 = m L0^2 / hbar")
    axes[0, 1].set(
        xlabel="physical slit separation (m)", ylabel="packet velocity (m/s)",
        title="Geometry fixes beam speed in the locked model")
    for ax in axes[0]:
        ax.grid(which="both", alpha=0.22)
        ax.legend()

    worked = next(row for row in scenarios if row["name"] == "neutron_10um_slits")
    prediction = worked["prediction"]
    sigma = prediction["sigma_t_s"]
    q = prediction["response_fraction"]
    times = np.linspace(0, 4 * sigma, 300)
    responding_pdf = np.sqrt(2 / np.pi) / sigma * np.exp(
        -0.5 * (times / sigma) ** 2)
    axes[1, 0].plot(times * 1e6, responding_pdf / 1e6, color="#7c3aed",
                    label="response-branch density")
    axes[1, 0].fill_between(
        times * 1e6, q * responding_pdf / 1e6,
        color="#a78bfa", alpha=0.35, label="ensemble continuous weight")
    axes[1, 0].axvline(
        prediction["ensemble_mean_delay_s"] * 1e6,
        color="#dc2626", linestyle="--", label="ensemble mean")
    axes[1, 0].set(
        xlabel="extra arrival delay (microseconds)",
        ylabel="probability density (1/microsecond)",
        title="Worked prediction: neutron, 10 micrometer slits")
    axes[1, 0].text(
        0.98, 0.95,
        f"zero-delay weight = {1-q:.3f}\n"
        f"sigma_T = {_format_time(sigma)}\n"
        f"mean = {_format_time(prediction['ensemble_mean_delay_s'])}",
        transform=axes[1, 0].transAxes, ha="right", va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9})
    axes[1, 0].grid(alpha=0.22)
    axes[1, 0].legend(fontsize=8)

    names = ["spatial sigma", "record sigma\n(alpha=1)",
             "record alpha needed"]
    spatial_sigma = SIGMA_T
    record_latency = REFERENCE_STABILIZATION_LATENCY
    alpha_needed = LOCKED_REFERENCE_ALPHA
    axes[1, 1].bar(
        np.arange(3), [spatial_sigma, record_latency, alpha_needed],
        color=["#2563eb", "#f97316", "#16a34a"])
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(
        xticks=np.arange(3), xticklabels=names,
        ylabel="dimensionless value (log scale)",
        title="The record and spatial clocks need an alpha bridge")
    axes[1, 1].grid(axis="y", which="both", alpha=0.22)
    axes[1, 1].text(
        0.98, 0.95,
        "sigma_T = 0.2\ntau_stab(g=1) = 2.76\n"
        f"alpha required = {alpha_needed:.5f}",
        transform=axes[1, 1].transAxes, ha="right", va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9})

    fig.suptitle(
        "Dimensionful TRF direct-delay closure and its unresolved bridge",
        fontsize=15, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-output", type=Path,
        default=Path("trf_physical_scale_prediction.json"))
    parser.add_argument(
        "--plot-output", type=Path,
        default=Path("trf_physical_scale_prediction.png"))
    args = parser.parse_args()

    scenarios = [
        _scenario(
            "electron_1um_slits", "electron",
            scale_from_slit_separation(m_e, 1e-6),
            "fixed_slit_separation_m", 1e-6),
        _scenario(
            "neutron_10um_slits", "neutron",
            scale_from_slit_separation(m_n, 10e-6),
            "fixed_slit_separation_m", 10e-6),
        _scenario(
            "c60_100nm_slits", "c60",
            scale_from_slit_separation(720 * atomic_mass, 100e-9),
            "fixed_slit_separation_m", 100e-9),
        _scenario(
            "electron_1e6mps", "electron",
            scale_from_packet_velocity(m_e, 1e6),
            "fixed_packet_velocity_m_per_s", 1e6),
    ]
    payload = {
        "status": "conditional_prediction_under_direct_delay_closure",
        "trf_it_v0p2_document_audit": {
            "section_6_equation_12": (
                "Defines a normalized half-Gaussian K_sigma(tau), sigma_T>0, "
                "with K carrying inverse-time units; it does not set sigma_T."),
            "section_8_equation_17": (
                "Defines event-anchored tau_stab from the onset of a qualifying "
                "record-stability interval."),
            "section_8_equation_18": (
                "Postulates sigma_T=alpha*tau_stab and states that dimensionless "
                "alpha is fixed by an independent convention or calibration, not "
                "derived as a universal constant."),
            "section_10": (
                "States that v0.2 does not yet supply a distinct observable TRF "
                "prediction and requires a complete measurable joint model."),
            "section_11": (
                "States explicitly that the cited information bounds do not "
                "derive a TRF timescale or kernel."),
            "arrival_timestamp_map": (
                "No eventwise map from latent tau to detector arrival timestamp "
                "is specified; kappa_t=1 is therefore an added direct-delay closure."),
        },
        "derivation": {
            "spatial_units": "x=L0*x_tilde; t=(m L0^2/hbar)*t_tilde",
            "time_unit": "T0=m L0^2/hbar",
            "velocity": "v=hbar*kx/(m L0)",
            "direct_delay_closure": "delta_t_recorded=T0*tau_tilde, hence kappa_t=1",
            "width_prediction": "sigma_T_physical=0.2*T0",
            "response_fraction": "q=1-exp(-lambda), lambda=1",
            "ensemble_mean_delay": "q*kappa_t*sqrt(2/pi)*sigma_T_physical",
        },
        "scope": (
            "T0 follows from nondimensionalizing the implemented Schrodinger "
            "equation. Kappa_t=1 follows only after adding the direct-delay "
            "closure. TRF-IT v0.2 does not select L0, particle mass, alpha, or "
            "the direct-delay closure."),
        "locked_dimensionless_values": {
            "slit_separation": DIMENSIONLESS_SLIT_SEPARATION,
            "packet_kx": PACKET_KX,
            "sigma_t": SIGMA_T,
            "lambda_strength": LAMBDA_STRENGTH,
            "delay_scale_kappa_t": 1.0,
            "source_pulse_sigma": 0.03,
            "detector_jitter": 0.05,
            "time_bin_width": 0.05,
            "reference_time": 0.9,
            "record_stabilization_latency_g1": REFERENCE_STABILIZATION_LATENCY,
            "response_fixture_alpha": 1.0,
            "alpha_required_to_match_spatial_sigma": LOCKED_REFERENCE_ALPHA,
        },
        "unit_time_gate_mass_without_instrument_blur": {
            "definition": "P(latent delay <= 1*T0), including exp(-lambda) zero-delay branch",
            "spatial_sigma_0p2": float(
                np.exp(-1) + (1 - np.exp(-1))
                * erf(1 / (np.sqrt(2) * 0.2))),
            "record_alpha1_sigma_2p76": float(
                np.exp(-1) + (1 - np.exp(-1))
                * erf(1 / (np.sqrt(2) * REFERENCE_STABILIZATION_LATENCY))),
        },
        "scenarios": scenarios,
        "decisive_falsifiable_prediction": (
            "After subtracting the calibrated standard time of flight, the "
            "response branch has a nonnegative half-normal delay and the full "
            "ensemble has zero-delay weight exp(-1). The current spatial fixture "
            "sets its width to 0.2*m*L0^2/hbar. Combining the record fixture's "
            "tau_stab=2.76 with its alpha=1 convention instead predicts "
            "2.76*m*L0^2/hbar. Both statements require the direct-delay closure."),
        "unresolved_theory_requirement": (
            "A TRF derivation must choose the direct-delay closure or replace it, "
            "and must predict alpha linking sigma_T to the operational record "
            "stabilization time. The current fixtures require alpha=0.2/2.76, "
            "whereas the earlier response fixture assumed alpha=1."),
        "theory_prediction_status": (
            "No parameter-free numerical TRF prediction exists yet. T0 is derived "
            "from the implemented Schrodinger scaling, kappa_t=1 is the added "
            "direct-delay closure, and the remaining dimensionless width requires "
            "a predicted or independently calibrated alpha."),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, default=_json_value, indent=2) + "\n")
    _plot(scenarios, args.plot_output)

    print("Direct-delay closure: kappa_t = 1")
    print("sigma_T(physical) = 0.2 * m * L0^2 / hbar")
    print(f"record/spatial alpha required = {LOCKED_REFERENCE_ALPHA:.8f}")
    for row in scenarios:
        prediction = row["prediction"]
        print(
            f"{row['name']:<22} L0={row['length_unit_m']:.6g} m "
            f"v={row['packet_velocity_m_per_s']:.6g} m/s "
            f"T0={_format_time(row['time_unit_s'])} "
            f"sigma={_format_time(prediction['sigma_t_s'])} "
            f"sigma(alpha=1)={_format_time(row['record_clock_alpha1_prediction']['sigma_t_s'])} "
            f"mean={_format_time(prediction['ensemble_mean_delay_s'])} "
            f"jitter={_format_time(prediction['detector_jitter_s'])}")
    print(f"results={args.json_output}; plot={args.plot_output}")


if __name__ == "__main__":
    main()
