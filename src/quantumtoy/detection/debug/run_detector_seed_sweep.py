from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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

OUT_ROOT = ROOT_DIR / "sweep_runs" / "detector_seed_sweep"

# ------------------------------------------------------------
# Default parallelism
# ------------------------------------------------------------

DEFAULT_MAX_WORKERS = 5

# ------------------------------------------------------------
# Common environment overrides
# ------------------------------------------------------------

COMMON_ENV: dict[str, str] = {
    "BATCH_FAST_MODE": "True",
    "BREAK_ON_DETECTOR_CLICK": "True",
    "ENABLE_BOHMIAN_OVERLAY": "False",
    "SAVE_MP4": "False",
    "THEORY_NAME": "schrodinger",
    "DETECTOR_NAME": "emergent",
}

# ------------------------------------------------------------
# Optional fixed overrides for the specific experiment
# ------------------------------------------------------------

EXPERIMENT_ENV: dict[str, str] = {
    # e.g.
    # "SIGMA_T": "1.2",
    # "DETECTOR_DRIVE_MODE": "rho",
}


@dataclass(frozen=True)
class SweepCase:
    case_name: str
    overrides: dict[str, str]


def safe_case_name(s: str) -> str:
    return s.replace(".", "p").replace("-", "m")


def build_cases() -> list[SweepCase]:
    seeds = list(range(1001, 1501))

    cases: list[SweepCase] = []
    for seed in seeds:
        case_name = f"seed_{safe_case_name(str(seed))}"
        overrides = {
            "DETECTOR_NOISE_SEED": str(seed),
        }
        overrides.update(EXPERIMENT_ENV)
        cases.append(SweepCase(case_name=case_name, overrides=overrides))

    return cases


def case_dir_for(case: SweepCase) -> Path:
    return OUT_ROOT / case.case_name


def case_prefix_for(case: SweepCase) -> Path:
    return case_dir_for(case) / case.case_name


def case_summary_path(case: SweepCase) -> Path:
    return case_dir_for(case) / f"{case.case_name}_summary.json"


def case_log_path(case: SweepCase) -> Path:
    return case_dir_for(case) / f"{case.case_name}.log"


def remove_stale_outputs(case_prefix: Path) -> list[str]:
    stale_files = [
        case_prefix.with_suffix(".npz"),
        case_prefix.with_suffix(".json"),
        case_prefix.with_name(case_prefix.name + "_meta.json"),
        case_prefix.with_name(case_prefix.name + "_flux_summary.json"),
        case_prefix.with_name(case_prefix.name + "_pseudo_clicks.json"),
        case_prefix.with_name(case_prefix.name + "_pseudo_clicks.jsonl"),
        case_prefix.with_name(case_prefix.name + "_summary.json"),
        case_prefix.with_name(case_prefix.name + "_detector.json"),
        case_prefix.with_name(case_prefix.name + ".mp4"),
        case_prefix.with_name(case_prefix.name + ".log"),
    ]

    removed: list[str] = []
    for path in stale_files:
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def parse_click_info_from_log(log_text: str) -> dict[str, object]:
    result: dict[str, object] = {
        "clicked": False,
        "click_step": None,
        "click_time": None,
        "click_x": None,
        "click_y": None,
    }

    marker = "[DETECTOR] clicked at step="
    idx = log_text.find(marker)
    if idx < 0:
        return result

    line = log_text[idx:].splitlines()[0].strip()
    result["clicked"] = True

    try:
        parts = [p.strip() for p in line.split(",")]
        for p in parts:
            if "step=" in p:
                result["click_step"] = int(p.split("step=")[-1].strip())
            elif p.startswith("t="):
                result["click_time"] = float(p.split("=", 1)[1].strip())
            elif p.startswith("x="):
                result["click_x"] = float(p.split("=", 1)[1].strip())
            elif p.startswith("y="):
                result["click_y"] = float(p.split("=", 1)[1].strip())
    except Exception:
        result["parse_error"] = line

    return result


def read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_successful_summary(data: dict[str, object] | None) -> bool:
    if not data:
        return False
    return int(data.get("returncode", 999999)) == 0


def build_skip_result(case: SweepCase, existing_summary: dict[str, object]) -> dict[str, object]:
    result = dict(existing_summary)
    result["case_name"] = case.case_name
    result["skipped"] = True
    result["skip_reason"] = "existing_successful_summary"
    return result


def rebuild_summary_jsonl_from_case_summaries(cases: list[SweepCase], summary_jsonl: Path) -> int:
    count = 0
    with open(summary_jsonl, "w", encoding="utf-8") as out:
        for case in cases:
            data = read_json(case_summary_path(case))
            if data is None:
                continue
            out.write(json.dumps(data, ensure_ascii=False) + "\n")
            count += 1
    return count


def run_case(case: SweepCase, python_bin: str, force_rerun: bool) -> dict[str, object]:
    start = time.time()

    case_dir = case_dir_for(case)
    case_dir.mkdir(parents=True, exist_ok=True)

    case_prefix = case_prefix_for(case)
    log_path = case_log_path(case)
    summary_path = case_summary_path(case)

    existing_summary = read_json(summary_path)
    if (not force_rerun) and is_successful_summary(existing_summary):
        return build_skip_result(case, existing_summary)

    env = os.environ.copy()
    env.update(COMMON_ENV)
    env.update(case.overrides)
    env["OUTPUT_PREFIX"] = str(case_prefix)

    removed = remove_stale_outputs(case_prefix)

    # Jitter to reduce simultaneous memory spikes
    time.sleep(float(np.random.uniform(0.0, 2.0)))

    cmd = [python_bin, "main.py"]

    header_lines = [
        "=" * 80,
        f"RUN CASE: {case.case_name}",
        "=" * 80,
        "[ENV OVERRIDES]",
    ]
    all_env_to_print = {**COMMON_ENV, **case.overrides, "OUTPUT_PREFIX": env["OUTPUT_PREFIX"]}
    for k in sorted(all_env_to_print):
        header_lines.append(f"  {k}={all_env_to_print[k]}")
    if removed:
        header_lines.append("[REMOVED STALE]")
        for p in removed:
            header_lines.append(f"  {p}")
    header_lines.append("[CMD]")
    header_lines.append(f"  {' '.join(cmd)}")
    header_lines.append("")

    header = "\n".join(header_lines) + "\n"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.flush()

        proc = subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    elapsed = time.time() - start

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    click_info = parse_click_info_from_log(log_text)

    result = {
        "case_name": case.case_name,
        "returncode": int(proc.returncode),
        "elapsed_sec": elapsed,
        "log_path": str(log_path),
        "output_prefix": str(case_prefix),
        "overrides": case.overrides,
        "skipped": False,
        "likely_oom": int(proc.returncode) == -9,
        **click_info,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def append_jsonl(path: Path, data: dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--case-prefix",
        default=None,
        help="Only run cases whose case_name starts with this prefix",
    )
    args = parser.parse_args()

    python_bin = sys.executable or "python3"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    cases = build_cases()
    if args.case_prefix:
        cases = [c for c in cases if c.case_name.startswith(args.case_prefix)]

    if not cases:
        print("No cases to run.")
        return 0

    summary_jsonl = OUT_ROOT / "summary.jsonl"

    rebuilt_count = rebuild_summary_jsonl_from_case_summaries(cases, summary_jsonl)

    print("=" * 80)
    print("Detector seed sweep")
    print("=" * 80)
    print(f"ROOT_DIR      : {ROOT_DIR}")
    print(f"MAIN_PATH     : {MAIN_PATH}")
    print(f"OUT_ROOT      : {OUT_ROOT}")
    print(f"MAX_WORKERS   : {args.max_workers}")
    print(f"N_CASES       : {len(cases)}")
    print(f"FORCE_RERUN   : {args.force_rerun}")
    print(f"CASE_PREFIX   : {args.case_prefix}")
    print(f"REBUILT_JSONL : {rebuilt_count}")
    print("")

    completed = 0
    failed = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        future_to_case = {
            ex.submit(run_case, case, python_bin, args.force_rerun): case
            for case in cases
        }

        for fut in as_completed(future_to_case):
            case = future_to_case[fut]

            try:
                result = fut.result()
            except Exception as e:
                failed += 1
                err = {
                    "case_name": case.case_name,
                    "returncode": None,
                    "elapsed_sec": None,
                    "clicked": False,
                    "error": repr(e),
                    "overrides": case.overrides,
                    "skipped": False,
                    "likely_oom": False,
                }
                append_jsonl(summary_jsonl, err)
                print(f"[FAIL ] {case.case_name}: {e}")
                continue

            # Update summary.jsonl with the newest status for this session too
            append_jsonl(summary_jsonl, result)

            if result.get("skipped", False):
                skipped += 1
                click_part = (
                    f"clicked={result.get('clicked')} "
                    f"t={result.get('click_time')} "
                    f"x={result.get('click_x')} "
                    f"y={result.get('click_y')}"
                )
                print(
                    f"[SKIP ] {case.case_name} | "
                    f"rc={result.get('returncode')} | "
                    f"{click_part}"
                )
                continue

            completed += 1

            if result["returncode"] != 0:
                failed += 1
                status = "FAIL"
            else:
                status = "DONE"

            click_part = (
                f"clicked={result.get('clicked')} "
                f"t={result.get('click_time')} "
                f"x={result.get('click_x')} "
                f"y={result.get('click_y')}"
            )

            oom_part = " | likely_oom=True" if result.get("likely_oom") else ""

            print(
                f"[{status:4}] {case.case_name} | "
                f"rc={result['returncode']} | "
                f"{result['elapsed_sec']:.2f}s | "
                f"{click_part}{oom_part}"
            )

    print("")
    print("=" * 80)
    print("Sweep complete")
    print("=" * 80)
    print(f"Completed new runs : {completed}")
    print(f"Skipped existing   : {skipped}")
    print(f"Failed             : {failed}")
    print(f"Summary            : {summary_jsonl}")
    print(f"Outputs            : {OUT_ROOT}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())