from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
WORKER_PATH = ROOT_DIR / "worldline_hybrid_single.py"

if not WORKER_PATH.exists():
    raise RuntimeError(f"Worker script not found: {WORKER_PATH}")

OUT_ROOT = ROOT_DIR / "sweep_runs" / "worldline_seed_sweep"
DEFAULT_MAX_WORKERS = 10


@dataclass(frozen=True)
class SweepCase:
    case_name: str
    seed: int


def build_cases(seed_start: int, seed_end: int) -> list[SweepCase]:
    return [SweepCase(case_name=f"seed_{seed}", seed=seed) for seed in range(seed_start, seed_end + 1)]


def case_dir_for(case: SweepCase) -> Path:
    return OUT_ROOT / case.case_name


def case_summary_path(case: SweepCase) -> Path:
    return case_dir_for(case) / f"{case.case_name}_summary.json"


def case_log_path(case: SweepCase) -> Path:
    return case_dir_for(case) / f"{case.case_name}.log"


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_successful_summary(data: dict | None) -> bool:
    if not data:
        return False
    return int(data.get("returncode", 999999)) == 0


def append_jsonl(path: Path, data: dict) -> None:
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


def format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:
        return "n/a"

    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def progress_line(
    total: int,
    started: int,
    finished: int,
    running: int,
    t0: float,
) -> str:
    elapsed = max(0.0, time.time() - t0)
    pct = 100.0 * finished / total if total > 0 else 0.0

    eta = None
    if finished > 0 and finished < total:
        avg_per_finished = elapsed / finished
        remaining = total - finished
        eta = avg_per_finished * remaining

    return (
        f"[PROGRESS] finished={finished}/{total} ({pct:.1f}%) | "
        f"started={started}/{total} | "
        f"running={running} | "
        f"elapsed={format_eta(elapsed)} | "
        f"eta≈{format_eta(eta)}"
    )


def run_case(case: SweepCase, python_bin: str, force_rerun: bool) -> dict:
    start = time.time()

    case_dir = case_dir_for(case)
    case_dir.mkdir(parents=True, exist_ok=True)

    summary_path = case_summary_path(case)
    log_path = case_log_path(case)

    existing_summary = read_json(summary_path)
    if (not force_rerun) and is_successful_summary(existing_summary):
        result = dict(existing_summary)
        result["skipped"] = True
        result["skip_reason"] = "existing_successful_summary"
        return result

    cmd = [
        python_bin,
        str(WORKER_PATH),
        "--seed", str(case.seed),
        "--out-dir", str(case_dir),
        "--case-name", case.case_name,
    ]

    header = "\n".join([
        "=" * 80,
        f"RUN CASE: {case.case_name}",
        "=" * 80,
        f"SEED={case.seed}",
        "[CMD]",
        "  " + " ".join(cmd),
        "",
    ])

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.flush()

    with open(log_path, "a", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    elapsed = time.time() - start

    data = read_json(summary_path)
    if data is None:
        data = {
            "case_name": case.case_name,
            "seed": case.seed,
            "returncode": int(proc.returncode),
            "elapsed_sec": float(elapsed),
            "clicked": False,
            "error": "summary_json_missing",
        }
    else:
        data["elapsed_sec_launcher"] = float(elapsed)
        data["returncode"] = int(proc.returncode)

    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-end", type=int, required=True)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--case-prefix", default=None)
    args = parser.parse_args()

    python_bin = sys.executable or "python3"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    cases = build_cases(args.seed_start, args.seed_end)
    if args.case_prefix:
        cases = [c for c in cases if c.case_name.startswith(args.case_prefix)]

    if not cases:
        print("No cases to run.")
        return 0

    summary_jsonl = OUT_ROOT / "summary.jsonl"
    rebuilt_count = rebuild_summary_jsonl_from_case_summaries(cases, summary_jsonl)

    print("=" * 100)
    print("Worldline seed sweep")
    print("=" * 100)
    print(f"ROOT_DIR      : {ROOT_DIR}")
    print(f"WORKER_PATH   : {WORKER_PATH}")
    print(f"OUT_ROOT      : {OUT_ROOT}")
    print(f"MAX_WORKERS   : {args.max_workers}")
    print(f"N_CASES       : {len(cases)}")
    print(f"FORCE_RERUN   : {args.force_rerun}")
    print(f"REBUILT_JSONL : {rebuilt_count}")
    print("")

    total = len(cases)
    t0 = time.time()

    completed = 0
    failed = 0
    skipped = 0
    started = 0

    pending_cases = list(cases)
    running: dict = {}

    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        while pending_cases or running:
            while pending_cases and len(running) < args.max_workers:
                case = pending_cases.pop(0)
                fut = ex.submit(run_case, case, python_bin, args.force_rerun)
                running[fut] = case
                started += 1

                print(
                    f"[START] {case.case_name} | "
                    f"seed={case.seed} | "
                    f"slot={len(running)}/{args.max_workers} | "
                    f"started={started}/{total}"
                )

            if not running:
                break

            done, _ = wait(running.keys(), return_when=FIRST_COMPLETED)

            for fut in done:
                case = running.pop(fut)

                try:
                    result = fut.result()
                except Exception as e:
                    failed += 1
                    completed += 1
                    err = {
                        "case_name": case.case_name,
                        "seed": case.seed,
                        "returncode": None,
                        "elapsed_sec": None,
                        "clicked": False,
                        "error": repr(e),
                        "skipped": False,
                    }
                    append_jsonl(summary_jsonl, err)
                    print(f"[FAIL ] {case.case_name} | seed={case.seed} | error={e}")
                    print(progress_line(total, started, completed, len(running), t0))
                    continue

                append_jsonl(summary_jsonl, result)
                completed += 1

                if result.get("skipped", False):
                    skipped += 1
                    print(
                        f"[SKIP ] {case.case_name} | "
                        f"seed={case.seed} | "
                        f"click={result.get('click_side')}"
                    )
                    print(progress_line(total, started, completed, len(running), t0))
                    continue

                if int(result.get("returncode", 1)) != 0:
                    failed += 1
                    status = "FAIL"
                else:
                    status = "DONE"

                print(
                    f"[{status:4}] {case.case_name} | "
                    f"seed={case.seed} | "
                    f"click={result.get('click_side')} | "
                    f"dyn_init={result.get('dynamic_init_side')} | "
                    f"dyn_final={result.get('dynamic_final_side')} | "
                    f"posthoc={result.get('posthoc_chosen_side')} | "
                    f"{result.get('elapsed_sec', 0.0):.2f}s"
                )
                print(progress_line(total, started, completed, len(running), t0))

    print("")
    print("=" * 100)
    print("Sweep complete")
    print("=" * 100)
    print(f"Completed new runs : {completed}")
    print(f"Skipped existing   : {skipped}")
    print(f"Failed             : {failed}")
    print(f"Summary            : {summary_jsonl}")
    print(f"Outputs            : {OUT_ROOT}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())