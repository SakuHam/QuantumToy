from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

MAIN_PATH = ROOT_DIR / "main.py"
if not MAIN_PATH.exists():
    raise RuntimeError(
        f"main.py not found at expected location:\n"
        f"  {MAIN_PATH}\n"
        f"ROOT_DIR was resolved to:\n"
        f"  {ROOT_DIR}\n"
        f"Check project layout."
    )

OUT_ROOT = ROOT_DIR / "trend_runs"


COMMON_ENV = {
    "BATCH_FAST_MODE": "True",
    "ENABLE_FLUX_BATCH_SAMPLER": "True",
    "BREAK_ON_DETECTOR_CLICK": "True",
    "ENABLE_BOHMIAN_OVERLAY": "False",
}


def run_case(case_name: str, overrides: dict[str, str], python_bin: str = "python3") -> None:
    print("\n" + "=" * 60)
    print(f"RUN CASE: {case_name}")
    print("=" * 60)

    case_dir = OUT_ROOT / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(COMMON_ENV)
    env.update(overrides)
    env["OUTPUT_PREFIX"] = str(case_dir / case_name)

    cmd = [python_bin, "main.py"]

    print("[ENV OVERRIDES]")
    for k in sorted({**COMMON_ENV, **overrides, "OUTPUT_PREFIX": env["OUTPUT_PREFIX"]}):
        print(f"  {k}={env[k]}")

    case_prefix = case_dir / case_name

    stale_files = [
        case_prefix.with_name(case_prefix.name + "_flux_summary.json"),
        case_prefix.with_name(case_prefix.name + "_pseudo_clicks.json"),
        case_prefix.with_name(case_prefix.name + "_pseudo_clicks.jsonl"),
    ]

    for path in stale_files:
        if path.exists():
            path.unlink()
            print(f"[REMOVE STALE] {path}")
            
    subprocess.run(
        cmd,
        cwd=ROOT_DIR,
        env=env,
        check=True,
    )

    print(f"[DONE] {case_name}")


def main() -> int:
    python_bin = sys.executable or "python3"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------
    # Sweep 1: screen position
    # Expectation:
    #   larger SCREEN_CENTER_X -> larger fringe spacing
    # ----------------------------------------------------
    for L in [8.0, 10.0, 12.0, 14.0]:
        case_name = f"screen_x_{str(L).replace('.', 'p')}"
        run_case(
            case_name,
            overrides={
                "SCREEN_CENTER_X": str(L),
                "DETECTOR_GATE_CENTER_X": str(L),
            },
            python_bin=python_bin,
        )

    # ----------------------------------------------------
    # Sweep 2: packet momentum k0x
    # Expectation:
    #   larger K0X -> smaller wavelength -> smaller fringe spacing
    # ----------------------------------------------------
    for K in [4.0, 5.0, 6.0, 7.0]:
        case_name = f"k0x_{str(K).replace('.', 'p')}"
        run_case(
            case_name,
            overrides={
                "K0X": str(K),
            },
            python_bin=python_bin,
        )

    # ----------------------------------------------------
    # Sweep 3: slit separation via slit_center_offset
    # Note:
    #   mask separation d ~= 2 * slit_center_offset
    # Expectation:
    #   larger offset -> larger d -> smaller fringe spacing
    # ----------------------------------------------------
    for off in [1.5, 2.0, 2.5, 3.0]:
        case_name = f"slit_offset_{str(off).replace('.', 'p')}"
        run_case(
            case_name,
            overrides={
                "SLIT_CENTER_OFFSET": str(off),
            },
            python_bin=python_bin,
        )

    print("\nAll sweeps complete.")
    print(f"Outputs saved under: {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())