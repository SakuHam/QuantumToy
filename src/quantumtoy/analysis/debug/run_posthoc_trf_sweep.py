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

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from file.run_io import load_run_bundle  # noqa: E402


OUT_ROOT = ROOT_DIR / "sweep_runs" / "posthoc_trf_seed_sweep"

# ------------------------------------------------------------
# Default parallelism
# ------------------------------------------------------------

DEFAULT_MAX_WORKERS = 5

# ------------------------------------------------------------
# Common environment overrides
# ------------------------------------------------------------

COMMON_ENV: dict[str, str] = {
    "BATCH_FAST_MODE": "False",
    "BREAK_ON_DETECTOR_CLICK": "True",
    "ENABLE_BOHMIAN_OVERLAY": "False",
    "SAVE_MP4": "False",
    "SAVE_COMPLEX_STATE_FRAMES": "True",
    "THEORY_NAME": "schrodinger",
    "DETECTOR_NAME": "emergent",
}

# ------------------------------------------------------------
# Optional fixed overrides for the specific experiment
# ------------------------------------------------------------

EXPERIMENT_ENV: dict[str, str] = {
    # Examples:
    # "POSTHOC_TRF_CORRIDOR_Y_SIGMA": "1.3",
    # "POSTHOC_TRF_CORRIDOR_X_WEIGHT_POWER": "2.5",
    # "POSTHOC_USE_WORLDLINE": "True",
}


@dataclass(frozen=True)
class SweepCase:
    case_name: str
    overrides: dict[str, str]


def safe_case_name(s: str) -> str:
    return s.replace(".", "p").replace("-", "m")


def build_cases() -> list[SweepCase]:
    seeds = list(range(2001, 2051))

    cases: list[SweepCase] = []
    for seed in seeds:
        case_name = f"seed_{safe_case_name(str(seed))}"
        overrides = {
            "DETECTOR_NOISE_SEED": str(seed),
            "CLICK_RNG_SEED": str(seed),
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


def append_jsonl(path: Path, data: dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


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


def _safe_get(d: dict | None, key: str, default=None):
    if d is None:
        return default
    return d.get(key, default)


def extract_posthoc_summary_from_bundle(npz_path: Path) -> dict[str, object]:
    bundle = load_run_bundle(npz_path)

    x_click = bundle.get("x_click")
    y_click = bundle.get("y_click")
    t_det = bundle.get("t_det")
    idx_det = bundle.get("idx_det")
    detector_clicked = bundle.get("detector_clicked")

    if y_click is None:
        click_side = None
    else:
        click_side = "upper" if float(y_click) > 0.0 else "lower"

    trf = bundle.get("posthoc_trf_info")
    wl = bundle.get("posthoc_worldline_info")

    trf_valid = _safe_get(trf, "valid", None)
    trf_chosen_side = _safe_get(trf, "chosen_side", None)

    wl_used = _safe_get(wl, "used", False)
    wl_seed_side = _safe_get(wl, "seed_side", None)
    wl_seed_x = _safe_get(wl, "seed_x", None)
    wl_seed_y = _safe_get(wl, "seed_y", None)

    posthoc_matches_click = (
        bool(trf_valid)
        and (click_side is not None)
        and (trf_chosen_side is not None)
        and (str(trf_chosen_side) == str(click_side))
    )

    worldline_seed_matches_click = (
        bool(wl_used)
        and (click_side is not None)
        and (wl_seed_side is not None)
        and (str(wl_seed_side) == str(click_side))
    )

    posthoc_base_rho = bundle.get("posthoc_base_rho")
    posthoc_selected_rho = bundle.get("posthoc_selected_rho")

    result = {
        "clicked": bool(detector_clicked) if detector_clicked is not None else None,
        "click_time": None if t_det is None else float(t_det),
        "click_idx": None if idx_det is None else int(idx_det),
        "click_x": None if x_click is None else float(x_click),
        "click_y": None if y_click is None else float(y_click),
        "click_side": click_side,

        "posthoc_trf_valid": None if trf_valid is None else bool(trf_valid),
        "posthoc_trf_ref_idx": _safe_get(trf, "ref_idx", None),
        "posthoc_trf_ref_time": _safe_get(trf, "ref_time", None),
        "posthoc_trf_chosen_side": trf_chosen_side,
        "posthoc_trf_upper_evidence": _safe_get(trf, "upper_evidence", None),
        "posthoc_trf_lower_evidence": _safe_get(trf, "lower_evidence", None),
        "posthoc_trf_total_evidence": _safe_get(trf, "total_evidence", None),
        "posthoc_trf_abs_margin": _safe_get(trf, "abs_margin", None),
        "posthoc_trf_rel_margin": _safe_get(trf, "rel_margin", None),
        "posthoc_trf_ratio": _safe_get(trf, "ratio", None),
        "posthoc_trf_dominance": _safe_get(trf, "dominance", None),
        "posthoc_trf_adaptive_score": _safe_get(trf, "adaptive_score", None),

        "posthoc_worldline_used": bool(wl_used),
        "posthoc_worldline_seed_side": wl_seed_side,
        "posthoc_worldline_seed_x": wl_seed_x,
        "posthoc_worldline_seed_y": wl_seed_y,

        "posthoc_matches_click": bool(posthoc_matches_click),
        "worldline_seed_matches_click": bool(worldline_seed_matches_click),

        "has_posthoc_base_rho": bool(posthoc_base_rho is not None),
        "has_posthoc_selected_rho": bool(posthoc_selected_rho is not None),
    }

    return result


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

    result = {
        "case_name": case.case_name,
        "returncode": int(proc.returncode),
        "elapsed_sec": elapsed,
        "log_path": str(log_path),
        "output_prefix": str(case_prefix),
        "overrides": case.overrides,
        "skipped": False,
        "likely_oom": int(proc.returncode) == -9,
    }

    npz_path = case_prefix.with_suffix(".npz")
    if proc.returncode == 0 and npz_path.exists():
        try:
            result.update(extract_posthoc_summary_from_bundle(npz_path))
        except Exception as e:
            result["bundle_parse_error"] = repr(e)
    else:
        result.update(
            {
                "clicked": False,
                "click_time": None,
                "click_idx": None,
                "click_x": None,
                "click_y": None,
                "click_side": None,
                "posthoc_trf_valid": None,
                "posthoc_trf_ref_idx": None,
                "posthoc_trf_ref_time": None,
                "posthoc_trf_chosen_side": None,
                "posthoc_trf_upper_evidence": None,
                "posthoc_trf_lower_evidence": None,
                "posthoc_trf_total_evidence": None,
                "posthoc_trf_abs_margin": None,
                "posthoc_trf_rel_margin": None,
                "posthoc_trf_ratio": None,
                "posthoc_trf_dominance": None,
                "posthoc_trf_adaptive_score": None,
                "posthoc_worldline_used": False,
                "posthoc_worldline_seed_side": None,
                "posthoc_worldline_seed_x": None,
                "posthoc_worldline_seed_y": None,
                "posthoc_matches_click": False,
                "worldline_seed_matches_click": False,
                "has_posthoc_base_rho": False,
                "has_posthoc_selected_rho": False,
            }
        )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


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
    print("Posthoc TRF seed sweep")
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
                    "click_time": None,
                    "click_idx": None,
                    "click_x": None,
                    "click_y": None,
                    "click_side": None,
                    "posthoc_trf_valid": None,
                    "posthoc_trf_chosen_side": None,
                    "posthoc_matches_click": False,
                    "worldline_seed_matches_click": False,
                    "error": repr(e),
                    "overrides": case.overrides,
                    "skipped": False,
                    "likely_oom": False,
                }
                append_jsonl(summary_jsonl, err)
                print(f"[FAIL ] {case.case_name}: {e}")
                continue

            append_jsonl(summary_jsonl, result)

            if result.get("skipped", False):
                skipped += 1
                print(
                    f"[SKIP ] {case.case_name} | "
                    f"rc={result.get('returncode')} | "
                    f"click={result.get('click_side')} | "
                    f"trf={result.get('posthoc_trf_chosen_side')} | "
                    f"match={result.get('posthoc_matches_click')}"
                )
                continue

            completed += 1

            if result["returncode"] != 0:
                failed += 1
                status = "FAIL"
            else:
                status = "DONE"

            oom_part = " | likely_oom=True" if result.get("likely_oom") else ""

            print(
                f"[{status:4}] {case.case_name} | "
                f"rc={result['returncode']} | "
                f"{result['elapsed_sec']:.2f}s | "
                f"click={result.get('click_side')} | "
                f"trf={result.get('posthoc_trf_chosen_side')} | "
                f"valid={result.get('posthoc_trf_valid')} | "
                f"match={result.get('posthoc_matches_click')} | "
                f"wl_match={result.get('worldline_seed_matches_click')}"
                f"{oom_part}"
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