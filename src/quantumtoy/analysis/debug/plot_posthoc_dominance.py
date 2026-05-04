from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

SUMMARY = Path("sweep_runs/posthoc_trf_velocity_sweep/summary.jsonl")


def load_latest_per_case(path: Path) -> list[dict]:
    latest_by_case: dict[str, dict] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            case_name = row.get("case_name")
            if case_name is None:
                continue
            latest_by_case[case_name] = row

    return list(latest_by_case.values())


def safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def main():
    print("Reading:", SUMMARY)
    print("Exists:", SUMMARY.exists())

    if not SUMMARY.exists():
        raise FileNotFoundError(f"Summary file not found: {SUMMARY}")

    rows = load_latest_per_case(SUMMARY)

    # Per-seed data
    data_dom = defaultdict(list)
    data_side = defaultdict(list)
    data_ratio = defaultdict(list)
    data_rel_margin = defaultdict(list)
    data_ev_ratio = defaultdict(list)
    data_upper_ev = defaultdict(list)
    data_lower_ev = defaultdict(list)
    data_ref_time = defaultdict(list)
    data_delta_ev = defaultdict(list)

    for row in rows:
        if int(row.get("returncode", 999999)) != 0:
            continue
        if bool(row.get("skipped", False)):
            continue

        seed = row.get("base_case_name", row.get("case_name"))
        k0x = safe_float(row.get("k0x_value"))
        dom = safe_float(row.get("posthoc_trf_dominance"))
        side = row.get("posthoc_trf_chosen_side")

        ratio = safe_float(row.get("posthoc_trf_ratio"))
        rel_margin = safe_float(row.get("posthoc_trf_rel_margin"))
        upper_ev = safe_float(row.get("posthoc_trf_upper_evidence"))
        lower_ev = safe_float(row.get("posthoc_trf_lower_evidence"))
        ref_time = safe_float(row.get("posthoc_trf_ref_time"))

        if seed is None or k0x is None or dom is None or side is None:
            continue

        side_val = 1 if side == "upper" else -1

        data_dom[seed].append((k0x, dom))
        data_side[seed].append((k0x, side_val))

        if ratio is not None:
            data_ratio[seed].append((k0x, ratio))

        if rel_margin is not None:
            data_rel_margin[seed].append((k0x, rel_margin))

        if upper_ev is not None:
            data_upper_ev[seed].append((k0x, upper_ev))

        if lower_ev is not None:
            data_lower_ev[seed].append((k0x, lower_ev))

        if upper_ev is not None and lower_ev is not None:
            ev_ratio = upper_ev / max(lower_ev, 1e-30)
            data_ev_ratio[seed].append((k0x, ev_ratio))
            if data_delta_ev is not None:
                data_delta_ev[seed].append((k0x, upper_ev-lower_ev))

        if ref_time is not None:
            data_ref_time[seed].append((k0x, ref_time))

    print("Collected seeds:", list(data_dom.keys()))
    for seed, pts in sorted(data_dom.items()):
        print(seed, sorted(pts)[:5])

    # ---------------------------------------------------------
    # 1) Dominance vs k0x
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for seed, pts in sorted(data_dom.items()):
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=seed)

    plt.xlabel("k0x")
    plt.ylabel("dominance")
    plt.title("Dominance vs velocity")
    if data_dom:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("dominance_vs_k0x.png", dpi=150)
    plt.show()

    # ---------------------------------------------------------
    # 2) Branch vs k0x
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 4))
    for seed, pts in sorted(data_side.items()):
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", linestyle="--", label=seed)

    plt.yticks([-1, 1], ["lower", "upper"])
    plt.xlabel("k0x")
    plt.ylabel("chosen side")
    plt.title("Branch vs velocity")
    if data_side:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("branch_vs_k0x.png", dpi=150)
    plt.show()

    # ---------------------------------------------------------
    # 3) Ratio vs k0x
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for seed, pts in sorted(data_ratio.items()):
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=seed)

    plt.xlabel("k0x")
    plt.ylabel("posthoc_trf_ratio")
    plt.title("TRF evidence ratio vs velocity")
    if data_ratio:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ratio_vs_k0x.png", dpi=150)
    plt.show()

    # ---------------------------------------------------------
    # 4) Relative margin vs k0x
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for seed, pts in sorted(data_rel_margin.items()):
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=seed)

    plt.xlabel("k0x")
    plt.ylabel("posthoc_trf_rel_margin")
    plt.title("TRF relative margin vs velocity")
    if data_rel_margin:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("rel_margin_vs_k0x.png", dpi=150)
    plt.show()

    # ---------------------------------------------------------
    # 5) upper_ev / lower_ev vs k0x
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for seed, pts in sorted(data_ev_ratio.items()):
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=seed)

    plt.xlabel("k0x")
    plt.ylabel("upper_evidence / lower_evidence")
    plt.title("Upper/Lower evidence ratio vs velocity")
    if data_ev_ratio:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("upper_lower_ratio_vs_k0x.png", dpi=150)
    plt.show()

    # ---------------------------------------------------------
    # 6) Upper evidence vs k0x
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for seed, pts in sorted(data_upper_ev.items()):
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=f"{seed} upper")

    plt.xlabel("k0x")
    plt.ylabel("posthoc_trf_upper_evidence")
    plt.title("Upper evidence vs velocity")
    if data_upper_ev:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("upper_evidence_vs_k0x.png", dpi=150)
    plt.show()

    # ---------------------------------------------------------
    # 7) Lower evidence vs k0x
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for seed, pts in sorted(data_lower_ev.items()):
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=f"{seed} lower")

    plt.xlabel("k0x")
    plt.ylabel("posthoc_trf_lower_evidence")
    plt.title("Lower evidence vs velocity")
    if data_lower_ev:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lower_evidence_vs_k0x.png", dpi=150)
    plt.show()

    # ---------------------------------------------------------
    # 8) Delta evidence vs k0x
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for seed, pts in sorted(data_delta_ev.items()):
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=f"{seed} delta")

    plt.xlabel("k0x")
    plt.ylabel("delta evidence")
    plt.title("Delta evidence vs velocity")
    if data_delta_ev:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("delta_evidence_vs_k0x.png", dpi=150)
    plt.show()

    # ---------------------------------------------------------
    # 9) Ref time vs velocity
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for seed, pts in sorted(data_ref_time.items()):
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=f"{seed}")

    plt.xlabel("k0x")
    plt.ylabel("posthoc_trf_ref_time")
    plt.title("Ref time vs velocity")
    if data_ref_time:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ref_time_vs_k0x.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()