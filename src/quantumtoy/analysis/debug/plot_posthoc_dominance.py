import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

SUMMARY = Path("sweep_runs/posthoc_trf_velocity_sweep/summary.jsonl")

data = defaultdict(list)

with open(SUMMARY, "r") as f:
    for line in f:
        row = json.loads(line)
        if row.get("returncode") != 0:
            continue
        
        seed = row["case_name"].split("__")[0]
        k0x = row.get("k0x")
        dom = row.get("dominance")
        side = row.get("trf_side")

        if k0x is None or dom is None:
            continue

        side_val = 1 if side == "upper" else -1

        data[seed].append((k0x, dom, side_val))

# --- plot dom vs k0x ---
plt.figure()

for seed, pts in data.items():
    pts.sort()
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    plt.plot(xs, ys, marker="o", label=seed)

plt.xlabel("k0x")
plt.ylabel("dominance")
plt.title("Dominance vs velocity")
plt.legend()
plt.grid()

plt.show()


# --- plot side vs k0x ---
plt.figure()

for seed, pts in data.items():
    pts.sort()
    xs = [p[0] for p in pts]
    ys = [p[2] for p in pts]
    plt.plot(xs, ys, marker="o", linestyle="--", label=seed)

plt.yticks([-1, 1], ["lower", "upper"])
plt.xlabel("k0x")
plt.ylabel("chosen side")
plt.title("Branch vs velocity")
plt.legend()
plt.grid()

plt.show()