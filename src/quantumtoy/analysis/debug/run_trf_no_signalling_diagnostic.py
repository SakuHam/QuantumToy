from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from theories.thick_front_entanglement import (
    run_trf_history_velocity_matrix,
    run_trf_no_signalling_diagnostic,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Minimal TRF no-signalling diagnostic for entangled spin settings."
    )
    parser.add_argument("--lambda-signal", type=float, default=0.0)
    parser.add_argument(
        "--history-velocity-mode",
        choices=["constant", "past_superluminal", "symmetric"],
        default="constant",
    )
    parser.add_argument("--history-v-future-factor", type=float, default=0.5)
    parser.add_argument("--history-v-present-factor", type=float, default=1.0)
    parser.add_argument("--history-v-past-factor", type=float, default=2.0)
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run A-D comparison: constant/safe, past_superluminal/safe, constant/forbidden, past_superluminal/forbidden.",
    )
    parser.add_argument(
        "--lambda-signal-forbidden",
        type=float,
        default=0.4,
        help="Forbidden lambda used for matrix cases C and D.",
    )
    parser.add_argument("--a0", type=float, default=0.0, help="Alice setting for ensemble 0, radians")
    parser.add_argument(
        "--a1",
        type=float,
        default=float(np.pi / 3.0),
        help="Alice setting for ensemble 1, radians",
    )
    parser.add_argument(
        "--bob",
        type=float,
        default=float(np.pi / 5.0),
        help="Fixed Bob setting, radians",
    )
    parser.add_argument("--n-trials", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    if args.matrix:
        result = run_trf_history_velocity_matrix(
            lambda_signal_safe=float(args.lambda_signal),
            lambda_signal_forbidden=float(args.lambda_signal_forbidden),
            history_v_future_factor=float(args.history_v_future_factor),
            history_v_present_factor=float(args.history_v_present_factor),
            history_v_past_factor=float(args.history_v_past_factor),
            alice_setting_a0=float(args.a0),
            alice_setting_a1=float(args.a1),
            bob_setting=float(args.bob),
            n_trials=int(args.n_trials),
            rng_seed=int(args.seed),
        )

        print("TRF history-velocity no-signalling matrix")
        for name, case in result["cases"].items():
            hist = case["history_locking"]
            print(
                f"  {name}: "
                f"mode={hist['history_velocity_mode']}, "
                f"lambda={case['lambda_signal']:.6g}, "
                f"effective_lambda={case['effective_lambda_signal']:.6g}, "
                f"Delta_signal_exact={case['Delta_signal_exact']:.6f}, "
                f"past/future={hist['past_future_bias_ratio']:.3f}, "
                f"avg_spread={hist['average_trf_spread_radius']:.3f}"
            )
        print()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    result = run_trf_no_signalling_diagnostic(
        lambda_signal=float(args.lambda_signal),
        history_velocity_mode=str(args.history_velocity_mode),
        history_v_future_factor=float(args.history_v_future_factor),
        history_v_present_factor=float(args.history_v_present_factor),
        history_v_past_factor=float(args.history_v_past_factor),
        alice_setting_a0=float(args.a0),
        alice_setting_a1=float(args.a1),
        bob_setting=float(args.bob),
        n_trials=int(args.n_trials),
        rng_seed=int(args.seed),
    )

    print("TRF no-signalling diagnostic")
    print(f"  lambda_signal              : {result['lambda_signal']:.6g}")
    print(f"  effective_lambda_signal    : {result['effective_lambda_signal']:.6g}")
    print(f"  history_velocity_mode      : {result['history_locking']['history_velocity_mode']}")
    print(f"  avg TRF spread radius      : {result['history_locking']['average_trf_spread_radius']:.6f}")
    print(f"  past/future bias ratio     : {result['history_locking']['past_future_bias_ratio']:.6f}")
    print(f"  Alice settings             : a0={result['alice_setting_a0']:.6f}, a1={result['alice_setting_a1']:.6f}")
    print(f"  Bob setting                : {result['bob_setting']:.6f}")
    print(f"  n_trials                   : {result['n_trials']}")
    print(f"  P_B_plus_given_a0          : {result['P_B_plus_given_a0']:.6f}")
    print(f"  P_B_plus_given_a1          : {result['P_B_plus_given_a1']:.6f}")
    print(f"  Delta_signal               : {result['Delta_signal']:.6f}")
    print(f"  P_B_plus_given_a0_exact    : {result['P_B_plus_given_a0_exact']:.6f}")
    print(f"  P_B_plus_given_a1_exact    : {result['P_B_plus_given_a1_exact']:.6f}")
    print(f"  Delta_signal_exact         : {result['Delta_signal_exact']:.6f}")
    print()
    print(json.dumps(result, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
