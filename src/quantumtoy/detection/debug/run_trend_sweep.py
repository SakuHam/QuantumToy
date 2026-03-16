from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]   # .../src/quantumtoy
CONFIG_PATH = ROOT / "config.py"
OUTPUT_DIR = ROOT / "trend_runs"


def replace_config_value(text: str, key: str, value_repr: str) -> str:
    pattern = rf"^({re.escape(key)}\s*=\s*).*$"
    repl = rf"\1{value_repr}"
    new_text, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
    if n == 0:
        raise RuntimeError(f"Config key not found: {key}")
    return new_text


def run_one(case_name: str, updates: dict[str, str]):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original = CONFIG_PATH.read_text(encoding="utf-8")
    patched = original

    # give each run a unique output prefix
    run_prefix = f"trend_runs/{case_name}"
    updates = dict(updates)
    updates["OUTPUT_PREFIX"] = repr(run_prefix)

    for key, value_repr in updates.items():
        patched = replace_config_value(patched, key, value_repr)

    CONFIG_PATH.write_text(patched, encoding="utf-8")

    try:
        print(f"\n=== RUN {case_name} ===")
        print("updates:", updates)

        subprocess.run(
            ["python3", "main.py"],
            cwd=ROOT,
            check=True,
        )

        # collect key outputs
        expected_files = [
            ROOT / f"{run_prefix}_flux_summary.json",
            ROOT / f"{run_prefix}_pseudo_clicks.json",
            ROOT / f"{run_prefix}_pseudo_clicks.jsonl",
        ]

        case_dir = OUTPUT_DIR / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        for src in expected_files:
            if src.exists():
                dst = case_dir / src.name
                shutil.copy2(src, dst)
                print(f"[COPIED] {src.name} -> {dst}")
            else:
                print(f"[WARN] missing expected output: {src}")

    finally:
        CONFIG_PATH.write_text(original, encoding="utf-8")


def main():
    # Example 1: sweep screen position L
    screen_values = [8.0, 10.0, 12.0, 14.0]

    for screen_x in screen_values:
        case = f"L_{screen_x:.1f}".replace(".", "p")
        run_one(
            case_name=case,
            updates={
                "BATCH_FAST_MODE": "True",
                "ENABLE_FLUX_BATCH_SAMPLER": "True",
                "BREAK_ON_DETECTOR_CLICK": "True",
                "SCREEN_CENTER_X": str(screen_x),  # change if your config key is lowercase
            },
        )


if __name__ == "__main__":
    main()