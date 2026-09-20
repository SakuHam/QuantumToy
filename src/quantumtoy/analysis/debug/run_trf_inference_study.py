"""Optimize and Monte Carlo-test the minimal TRF measurement design."""

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
from analysis.record_sensitivity import build_reference_runs, threshold_grid
from analysis.trf_inference import monte_carlo_recovery, optimize_lambda_design


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(
        description="TRF design optimization and full-record Monte Carlo recovery")
    parser.add_argument("--total-shots", type=int, default=150_000)
    parser.add_argument("--shot-block", type=int, default=5_000)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--lambda-signal", type=float, default=0.2)
    parser.add_argument("--ordinary-dephasing-rate", type=float, default=0.02)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if (args.total_shots <= 0 or args.shot_block <= 0
            or args.total_shots % args.shot_block or args.repetitions <= 0):
        parser.error("Use positive repetitions and a total-shots multiple of shot-block")

    experiment = RecordExperiment()
    runs = build_reference_runs(
        experiment, [0.5, 1.0, 2.0], duration=12, dt=0.02)
    threshold_rows = threshold_grid(
        runs,
        information_deficits=[0.05, 0.1, 0.2],
        required_copies=[2, 3, 4],
        coherence_tolerances=[0.01, 0.05, 0.1],
        hold_times=[0.25, 0.5, 1.0],
    )
    central = next(run for run in runs if run.g == 1.0)
    central_latencies = [row.stabilization.latency for row in threshold_rows
                         if row.g == 1.0 and row.stabilization is not None]
    tau_scales = (
        min(central_latencies) / central.stabilization.latency,
        1.0,
        max(central_latencies) / central.stabilization.latency,
    )
    candidate_times = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
    observations = tuple(
        (run.g, time, run.stabilization.latency)
        for run in runs for time in candidate_times
    )

    calibration_scenarios = [0.002, 0.005, 0.02]
    designs = []
    print("Design sensitivity to independent ordinary-dephasing calibration")
    print("(Fisher errors profile a continuous clock scale; recovery below profiles 3 conventions)")
    print("sigma_gamma  sigma_lambda  worst_sigma_lambda  corr_lam_gamma  corr_lam_clock")
    for noise_sigma in calibration_scenarios:
        design = optimize_lambda_design(
            experiment,
            observations,
            total_shots=args.total_shots,
            shot_block=args.shot_block,
            lambda_strength=args.lambda_signal,
            ordinary_dephasing_rate=args.ordinary_dephasing_rate,
            ordinary_dephasing_sigma=noise_sigma,
            tau_scales=tau_scales,
        )
        designs.append((noise_sigma, design))
        print(f"{noise_sigma:<12g} {design.lambda_standard_error:<13.6g} "
              f"{design.worst_case_lambda_standard_error:<19.6g} "
              f"{design.lambda_noise_correlation:<15.6g} "
              f"{design.lambda_clock_correlation:.6g}")

    selected = next(design for sigma, design in designs if sigma == 0.005)
    print("\nSelected robust allocation (sigma_gamma=0.005)")
    print("g       time    tau_stab  shots")
    for row in selected.allocations:
        print(f"{row.g:<7g} {row.time:<7g} {row.tau_stab:<9.6g} {row.shots}")
    print("tau convention scales:", ", ".join(f"{value:.6g}" for value in tau_scales))

    recovery = monte_carlo_recovery(
        experiment,
        selected.allocations,
        true_lambda_values=[0.0, args.lambda_signal],
        repetitions=args.repetitions,
        seed=args.seed,
        true_ordinary_dephasing_rate=args.ordinary_dephasing_rate,
        tau_scales=tau_scales,
        ordinary_dephasing_sigma=0.005,
    )
    print("\nMonte Carlo full-record recovery")
    print("lambda_true  detected  mean_hat  bias       RMSE       q05/q50/q95")
    for row in recovery:
        quantiles = "/".join(f"{value:.4g}" for value in row.quantiles)
        print(f"{row.true_lambda:<12g} {row.detection_rate:<9.3f} "
              f"{row.mean_lambda:<9.5g} {row.bias:<10.5g} {row.rmse:<10.5g} {quantiles}")
        print("  selected tau scales:", row.selected_tau_scale_counts)

    if args.json_output:
        payload = {
            "status": "synthetic_design_not_empirical_evidence",
            "settings": vars(args) | {"json_output": None},
            "experiment": asdict(experiment),
            "tau_scales": tau_scales,
            "calibration_designs": [
                {"ordinary_dephasing_sigma": sigma, "design": asdict(design)}
                for sigma, design in designs
            ],
            "selected_noise_sigma": 0.005,
            "recovery": [asdict(row) for row in recovery],
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, default=_json_value, indent=2) + "\n",
            encoding="utf-8")
        print(f"\nWrote {args.json_output}")


if __name__ == "__main__":
    main()
