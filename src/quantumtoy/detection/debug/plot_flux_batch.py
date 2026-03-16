from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_flux_summary(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_clicks_json(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["clicks"]


def load_clicks_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_safe(arr: np.ndarray) -> np.ndarray:
    s = float(np.sum(arr))
    if s <= 0.0:
        return np.zeros_like(arr)
    return arr / s


def plot_flux_y_distribution(summary: dict):
    y = np.asarray(summary["y_coords"], dtype=float)
    flux_y = np.asarray(summary["flux_y_accum"], dtype=float)
    py = normalize_safe(flux_y)

    plt.figure(figsize=(9, 5))
    plt.plot(y, py)
    plt.xlabel("y")
    plt.ylabel("P(y)")
    plt.title("Accumulated positive flux distribution along detector screen")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_flux_x_distribution(summary: dict):
    x = np.asarray(summary["x_coords"], dtype=float)
    flux_x = np.asarray(summary["flux_x_accum"], dtype=float)
    px = normalize_safe(flux_x)

    plt.figure(figsize=(9, 5))
    plt.plot(x, px)
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.title("Accumulated positive flux distribution along x")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_click_hist_y(clicks: list[dict], bins: int = 80):
    y = np.asarray([float(c["y"]) for c in clicks], dtype=float)

    plt.figure(figsize=(9, 5))
    plt.hist(y, bins=bins, density=True)
    plt.xlabel("y click")
    plt.ylabel("Density")
    plt.title("Pseudo-click histogram along y")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_click_hist_x(clicks: list[dict], bins: int = 80):
    x = np.asarray([float(c["x"]) for c in clicks], dtype=float)

    plt.figure(figsize=(9, 5))
    plt.hist(x, bins=bins, density=True)
    plt.xlabel("x click")
    plt.ylabel("Density")
    plt.title("Pseudo-click histogram along x")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_click_scatter(clicks: list[dict], max_points: int = 5000):
    pts = clicks
    if len(pts) > max_points:
        rng = np.random.default_rng(1234)
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = [pts[i] for i in idx]

    x = np.asarray([float(c["x"]) for c in pts], dtype=float)
    y = np.asarray([float(c["y"]) for c in pts], dtype=float)

    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, s=5, alpha=0.5)
    plt.xlabel("x click")
    plt.ylabel("y click")
    plt.title("Pseudo-click scatter")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_flux_vs_clicks(summary: dict, clicks: list[dict], bins: int = 80):
    y = np.asarray(summary["y_coords"], dtype=float)
    flux_y = np.asarray(summary["flux_y_accum"], dtype=float)
    py = normalize_safe(flux_y)

    y_click = np.asarray([float(c["y"]) for c in clicks], dtype=float)

    plt.figure(figsize=(9, 5))
    plt.plot(y, py, label="Flux-derived P(y)")
    plt.hist(y_click, bins=bins, density=True, alpha=0.45, label="Pseudo-clicks")
    plt.xlabel("y")
    plt.ylabel("Density")
    plt.title("Flux distribution vs pseudo-click histogram")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Plot flux summary and pseudo-click data.")
    parser.add_argument("--summary", type=str, help="Path to *_flux_summary.json")
    parser.add_argument("--clicks-json", type=str, help="Path to *_pseudo_clicks.json")
    parser.add_argument("--clicks-jsonl", type=str, help="Path to *_pseudo_clicks.jsonl")
    parser.add_argument("--bins", type=int, default=80)
    args = parser.parse_args()

    summary = None
    clicks = None

    if args.summary:
        summary = load_flux_summary(args.summary)

    if args.clicks_json:
        clicks = load_clicks_json(args.clicks_json)
    elif args.clicks_jsonl:
        clicks = load_clicks_jsonl(args.clicks_jsonl)

    if summary is not None:
        plot_flux_y_distribution(summary)
        plot_flux_x_distribution(summary)

    if clicks is not None:
        plot_click_hist_y(clicks, bins=args.bins)
        plot_click_hist_x(clicks, bins=args.bins)
        plot_click_scatter(clicks)

    if summary is not None and clicks is not None:
        plot_flux_vs_clicks(summary, clicks, bins=args.bins)

    if summary is None and clicks is None:
        raise SystemExit("Give at least --summary or --clicks-json / --clicks-jsonl")

    plt.show()


if __name__ == "__main__":
    main()