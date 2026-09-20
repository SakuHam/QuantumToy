"""Threshold, sampling, and minimal-TRF identifiability report."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.record_environment import RecordExperiment
from analysis.record_sensitivity import (
    build_reference_runs,
    threshold_grid,
    time_sampling_convergence,
)
from analysis.trf_candidate import (
    HalfGaussianDephasing,
    candidate_joint,
    detector_visibility,
    expected_fisher_information,
)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _window_dict(window):
    return None if window is None else asdict(window)


def main():
    parser = argparse.ArgumentParser(
        description="TRF record-clock sensitivity and candidate identifiability study")
    parser.add_argument("--g", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--dt", type=float, default=0.02,
                        help="Threshold-grid time sampling")
    parser.add_argument("--convergence-dt", nargs="+", type=float,
                        default=[0.08, 0.04, 0.02])
    parser.add_argument("--lambda-strength", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Fixed sigma_T/tau_stab convention")
    parser.add_argument("--ordinary-dephasing-rate", type=float, default=0.02)
    parser.add_argument("--shots", type=int, default=10_000,
                        help="Expected shots per (g,time) Fisher setting")
    parser.add_argument("--observable-time", type=float, default=0.5,
                        help="Fixed laboratory delay for the scalar contrast")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if (any(not np.isfinite(g) or g <= 0 for g in args.g)
            or not np.isfinite(args.duration) or args.duration <= 0
            or not np.isfinite(args.dt) or args.dt <= 0
            or not np.isfinite(args.observable_time)
            or not 0 <= args.observable_time <= args.duration):
        parser.error("g, duration, and dt must be positive and observable-time must fit the run")

    experiment = RecordExperiment()
    runs = build_reference_runs(
        experiment, args.g, duration=args.duration, dt=args.dt)
    unresolved = [run.g for run in runs if run.stabilization is None]
    if unresolved:
        parser.error(f"Baseline stabilization unresolved for g={unresolved}; increase duration")

    rows = threshold_grid(
        runs,
        information_deficits=[0.05, 0.1, 0.2],
        required_copies=[2, 3, 4],
        coherence_tolerances=[0.01, 0.05, 0.1],
        hold_times=[0.25, 0.5, 1.0],
    )
    sampling = time_sampling_convergence(
        experiment, args.g, duration=args.duration,
        time_steps=args.convergence_dt)

    print("Threshold sensitivity (81 shared conventions per coupling)")
    print("g       resolved  unresolved  tau_min   tau_max")
    sensitivity_summary = []
    for g in args.g:
        selected = [row for row in rows if row.g == g]
        latencies = [row.stabilization.latency for row in selected
                     if row.stabilization is not None]
        summary = {
            "g": g,
            "resolved": len(latencies),
            "unresolved": len(selected) - len(latencies),
            "latency_min": None if not latencies else min(latencies),
            "latency_max": None if not latencies else max(latencies),
        }
        sensitivity_summary.append(summary)
        lower = "-" if not latencies else f"{min(latencies):.6g}"
        upper = "-" if not latencies else f"{max(latencies):.6g}"
        print(f"{g:<7g} {len(latencies):<9d} {len(selected)-len(latencies):<11d} "
              f"{lower:<9} {upper}")

    print("\nTime-sampling convergence (analytic states; clock sampling only)")
    print("g       dt       tau_stab  max_norm_error")
    for row in sampling:
        latency = "unresolved" if row.latency is None else f"{row.latency:.6g}"
        print(f"{row.g:<7g} {row.dt:<8g} {latency:<9} "
              f"{row.max_joint_normalization_error:.3g}")

    observations = [
        (run.g, time, run.stabilization.latency)
        for run in runs for time in [0.5, 1.0, 2.0, 3.0, 4.0]
        if time <= args.duration
    ]
    fisher = expected_fisher_information(
        experiment,
        observations,
        lambda_strength=args.lambda_strength,
        alpha=args.alpha,
        ordinary_dephasing_rate=args.ordinary_dephasing_rate,
        shots_per_observation=args.shots,
    )
    print("\nMinimal candidate expected identifiability")
    print(f"rank={fisher.rank}/{len(fisher.parameter_names)}  "
          f"condition={fisher.condition_number:.6g}")
    for name, error in zip(fisher.parameter_names, fisher.standard_errors):
        print(f"  sigma({name}) = {error:.6g}")
    lambda_index = fisher.parameter_names.index("lambda_strength")
    noise_index = fisher.parameter_names.index("ordinary_dephasing_rate")
    print("  corr(lambda_strength, ordinary_dephasing_rate) = "
          f"{fisher.correlation[lambda_index, noise_index]:.6g}")

    print(f"\nDeclared observable at fixed t={args.observable_time:g}")
    print("g       V_Q          V_theta      delta_V")
    observable_rows = []
    for run in runs:
        tau = run.stabilization.latency
        time = args.observable_time
        reference = detector_visibility(candidate_joint(
            experiment, run.g, time, tau, HalfGaussianDephasing(0, args.alpha)))
        candidate_value = detector_visibility(candidate_joint(
            experiment, run.g, time, tau,
            HalfGaussianDephasing(args.lambda_strength, args.alpha)))
        row = {"g": run.g, "time": time, "quantum": reference,
               "candidate": candidate_value, "difference": candidate_value-reference}
        observable_rows.append(row)
        print(f"{run.g:<7g} {reference:<12.6g} {candidate_value:<12.6g} "
              f"{candidate_value-reference:.6g}")

    if args.json_output:
        payload = {
            "status": "synthetic_design_not_empirical_fit",
            "experiment": asdict(experiment),
            "candidate": {"lambda_strength": args.lambda_strength,
                          "alpha": args.alpha,
                          "ordinary_dephasing_rate": args.ordinary_dephasing_rate,
                          "shots_per_observation": args.shots},
            "baseline": [{"g": run.g,
                          "stabilization": _window_dict(run.stabilization)}
                         for run in runs],
            "sensitivity_summary": sensitivity_summary,
            "threshold_rows": [{**asdict(row),
                                "stabilization": _window_dict(row.stabilization)}
                               for row in rows],
            "sampling": [asdict(row) for row in sampling],
            "fisher": asdict(fisher),
            "observable": observable_rows,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, default=_json_value, indent=2) + "\n",
            encoding="utf-8")
        print(f"\nWrote {args.json_output}")


if __name__ == "__main__":
    main()
