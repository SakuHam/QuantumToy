"""Run the compact complete spatial effect-instrument benchmark."""

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

from analysis.spatial_effect_measurement import (
    SpatialEffectExperiment,
    fit_spatial_effect_response,
    spatial_effect_convergence,
)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile sigma_T with a complete compact spatial POVM")
    parser.add_argument("--true-sigma", type=float, default=0.6)
    parser.add_argument(
        "--candidate-sigmas", type=float, nargs="+",
        default=[0.3, 0.45, 0.6, 0.8, 1.0])
    parser.add_argument("--lambda-strength", type=float, default=1.0)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--delay-step", type=float, default=0.1)
    parser.add_argument("--horizon-sigmas", type=float, default=4.0)
    parser.add_argument("--skip-convergence", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    experiment = SpatialEffectExperiment(
        nx=args.nx,
        ny=args.ny,
        delay_step=args.delay_step,
        horizon_sigmas=args.horizon_sigmas,
    )
    profile = fit_spatial_effect_response(
        experiment, args.true_sigma, args.candidate_sigmas,
        lambda_strength=args.lambda_strength, shots=args.shots)
    convergence = ({} if args.skip_convergence else spatial_effect_convergence(
        experiment, args.true_sigma, lambda_strength=args.lambda_strength))

    print("Complete spatial effect-instrument profile")
    print(f"  injected sigma_T       = {profile.true_sigma_t:.6g}")
    print(f"  recovered sigma_T      = {profile.best_sigma_t:.6g}")
    print(f"  declared lambda        = {profile.lambda_strength:.6g}")
    print(f"  TV(signal, null)       = {profile.signal_total_variation:.6g}")
    print(f"  lambda=0 max error     = {profile.lambda_zero_max_error:.3e}")
    print(f"  complete probability   = {np.sum(profile.target_probabilities):.12g}")
    print("\nCandidate profile")
    print("sigma_T       KL             expected_deviance")
    for sigma, kl, deviance in zip(
            profile.candidate_sigmas, profile.kl_divergence,
            profile.expected_deviance):
        print(f"{sigma:<13.6g} {kl:<14.6g} {deviance:.6g}")
    if convergence:
        print("\nConvergence: complete-law total-variation distance")
        for name, value in convergence.items():
            print(f"  {name:<24} {value:.6g}")

    if args.json_output:
        payload = {
            "status": "synthetic_complete_effect_instrument_not_empirical_measurement",
            "experiment": asdict(experiment),
            "profile": asdict(profile),
            "convergence_total_variation": convergence,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, default=_json_value, indent=2) + "\n",
            encoding="utf-8")
        print(f"\nWrote {args.json_output}")


if __name__ == "__main__":
    main()
