"""Animate the unresolved delay components of the spatial effect instrument."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import PowerNorm


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.spatial_effect_measurement import (
    SpatialEffectExperiment,
    double_slit_effect_experiment,
    spatial_effect_evolution,
    spatial_effect_potential,
)


def _bin_boundaries(y, count):
    rows = np.array_split(np.arange(y.size), count)
    return [0.5 * (y[left[-1]] + y[right[0]])
            for left, right in zip(rows[:-1], rows[1:])]


def build_animation(experiment, sigma_t, lambda_strength, fps):
    evolution = spatial_effect_evolution(
        experiment, sigma_t, lambda_strength=lambda_strength)
    plt.style.use("dark_background")
    figure_height = 9.0 if experiment.y_bins > 8 else 7.2
    figure = plt.figure(figsize=(13.5, figure_height), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.42, 1), height_ratios=(1, 1))
    density_ax = figure.add_subplot(grid[:, 0])
    kernel_ax = figure.add_subplot(grid[0, 1])
    probability_ax = figure.add_subplot(grid[1, 1])

    dx = experiment.lx / experiment.nx
    dy = experiment.ly / experiment.ny
    extent = (
        evolution.x[0] - dx / 2, evolution.x[-1] + dx / 2,
        evolution.y[0] - dy / 2, evolution.y[-1] + dy / 2,
    )
    maximum_density = float(np.max(evolution.densities))
    density_image = density_ax.imshow(
        evolution.densities[0], origin="lower", extent=extent,
        cmap="magma", interpolation="bilinear", aspect="equal",
        norm=PowerNorm(gamma=0.5, vmin=0, vmax=maximum_density),
    )
    colorbar = figure.colorbar(density_image, ax=density_ax, fraction=0.046)
    colorbar.set_label(r"Probability density $|\psi_\tau(x,y)|^2$")
    potential = spatial_effect_potential(experiment)
    if np.max(potential) > 0:
        density_ax.contour(
            potential, levels=[0.1 * float(np.max(potential))],
            origin="lower", extent=extent, colors=["#f8fafc"],
            linewidths=[1.2], alpha=0.9)
    density_ax.axvspan(
        experiment.detector_x - experiment.detector_width,
        experiment.detector_x + experiment.detector_width,
        color="#43c6db", alpha=0.12, label="detector gate width")
    density_ax.axvline(
        experiment.detector_x, color="#67e8f9", linewidth=1.6,
        linestyle="--")
    for boundary in _bin_boundaries(evolution.y, experiment.y_bins):
        density_ax.axhline(
            boundary, color="white", linewidth=0.8, alpha=0.35)
    density_ax.set(
        title="Unitary component on the compact spatial lattice",
        xlabel="x", ylabel="y", xlim=extent[:2], ylim=extent[2:])
    density_ax.legend(loc="upper left", framealpha=0.75)
    delay_text = density_ax.text(
        0.98, 0.97, "", transform=density_ax.transAxes,
        ha="right", va="top", fontsize=12,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#111827",
              "edgecolor": "#67e8f9", "alpha": 0.88})

    bar_width = (evolution.delays[1] - evolution.delays[0]) * 0.78
    kernel_bars = kernel_ax.bar(
        evolution.delays, evolution.delay_weights, width=bar_width,
        color="#38bdf8", alpha=0.25, edgecolor="#7dd3fc", linewidth=0.6)
    current_weight, = kernel_ax.plot(
        [evolution.delays[0]], [evolution.delay_weights[0]], marker="o",
        markersize=8, color="#fbbf24", linestyle="none")
    current_delay = kernel_ax.axvline(
        evolution.delays[0], color="#fbbf24", linewidth=1.2, alpha=0.8)
    kernel_ax.set(
        title=rf"Half-Gaussian delay mixture, $\sigma_T={sigma_t:g}$",
        xlabel=r"Unresolved delay $\tau$", ylabel="Discrete prior weight",
        xlim=(-0.04 * evolution.delays[-1], 1.04 * evolution.delays[-1]),
        ylim=(0, 1.2 * float(np.max(evolution.delay_weights))))
    kernel_ax.grid(alpha=0.18)
    accumulated_text = kernel_ax.text(
        0.98, 0.92, "", transform=kernel_ax.transAxes,
        ha="right", va="top", color="#bae6fd")

    positions = np.arange(len(evolution.labels))
    current_bars = probability_ax.barh(
        positions - 0.18, evolution.component_probabilities[0], height=0.32,
        color="#64748b", alpha=0.7, label="current delay component")
    cumulative_bars = probability_ax.barh(
        positions + 0.18, evolution.cumulative_probabilities[0], height=0.32,
        color="#22d3ee", alpha=0.9,
        label=rf"cumulative complete law, $\lambda={lambda_strength:g}$")
    probability_ax.set(
        title="All outcomes retained",
        xlabel="Probability", yticks=positions,
        yticklabels=[label.replace("_", " ") for label in evolution.labels],
        xlim=(0, 1.0))
    if experiment.y_bins > 8:
        probability_ax.tick_params(axis="y", labelsize=7)
    probability_ax.invert_yaxis()
    probability_ax.grid(axis="x", alpha=0.18)
    probability_ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1,
        framealpha=0.75, fontsize=8)
    value_texts = [probability_ax.text(
        0.99, position + 0.18, "", ha="right", va="center",
        color="white", fontsize=9)
        for position in positions]

    geometry_name = ("double-slit" if experiment.potential_mode == "double_slit"
                     else "free-particle")
    figure.suptitle(
        f"Complete {geometry_name} effect instrument: temporal alternatives and detector law",
        fontsize=16, fontweight="bold")
    figure.text(
        0.5, 0.006,
        "Frames are unresolved delay alternatives in one POVM mixture, not a particle trajectory.",
        ha="center", va="bottom", color="#cbd5e1", fontsize=10)

    cumulative_weight = np.cumsum(evolution.delay_weights)

    def update(frame_index):
        density_image.set_data(evolution.densities[frame_index])
        delay = evolution.delays[frame_index]
        weight = evolution.delay_weights[frame_index]
        delay_text.set_text(
            f"$t={experiment.reference_time + delay:.2f}$, "
            f"$\\tau={delay:.2f}$\n"
            f"component norm $={np.sum(evolution.densities[frame_index]):.6f}$")
        current_weight.set_data([delay], [weight])
        current_delay.set_xdata([delay, delay])
        accumulated_text.set_text(
            f"included prior mass = {cumulative_weight[frame_index]:.4f}")
        for index, patch in enumerate(kernel_bars):
            patch.set_alpha(0.9 if index <= frame_index else 0.2)
        for patch, value in zip(
                current_bars, evolution.component_probabilities[frame_index]):
            patch.set_width(value)
        for patch, text_artist, value in zip(
                cumulative_bars, value_texts,
                evolution.cumulative_probabilities[frame_index]):
            patch.set_width(value)
            text_artist.set_text(f"{value:.4f}")
        return (
            density_image, current_weight, current_delay, delay_text,
            accumulated_text, *kernel_bars, *current_bars, *cumulative_bars,
            *value_texts,
        )

    hold_frames = max(1, int(1.5 * fps))
    frame_indices = list(range(evolution.delays.size)) + [
        evolution.delays.size - 1] * hold_frames
    animation = FuncAnimation(
        figure, update, frames=frame_indices,
        interval=1000 / fps, blit=False)
    update(evolution.delays.size - 1)
    return figure, animation, evolution


def main():
    parser = argparse.ArgumentParser(
        description="Animate the complete spatial effect-instrument mixture")
    parser.add_argument(
        "--geometry", choices=["free", "double_slit"], default="free")
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--lambda-strength", type=float, default=1.0)
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--delay-step", type=float)
    parser.add_argument("--horizon-sigmas", type=float)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument(
        "--output-mp4", type=Path,
        default=Path("spatial_effect_measurement.mp4"))
    parser.add_argument("--snapshot-output", type=Path)
    args = parser.parse_args()
    if args.fps <= 0 or args.dpi <= 0:
        parser.error("--fps and --dpi must be positive")

    if args.geometry == "double_slit":
        experiment = double_slit_effect_experiment(
            nx=80 if args.nx is None else args.nx,
            ny=80 if args.ny is None else args.ny)
        sigma_t = 0.2 if args.sigma is None else args.sigma
    else:
        experiment = SpatialEffectExperiment(
            nx=64 if args.nx is None else args.nx,
            ny=64 if args.ny is None else args.ny)
        sigma_t = 0.6 if args.sigma is None else args.sigma
    updates = {}
    if args.delay_step is not None:
        updates["delay_step"] = args.delay_step
    if args.horizon_sigmas is not None:
        updates["horizon_sigmas"] = args.horizon_sigmas
    if updates:
        experiment = replace(experiment, **updates)
    figure, animation, evolution = build_animation(
        experiment, sigma_t, args.lambda_strength, args.fps)

    args.output_mp4.parent.mkdir(parents=True, exist_ok=True)
    print(f"[SAVE] animation -> {args.output_mp4}")
    animation.save(
        args.output_mp4, writer="ffmpeg", fps=args.fps, dpi=args.dpi,
        metadata={"title": "Complete spatial effect instrument"})
    if args.snapshot_output:
        args.snapshot_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.snapshot_output, dpi=args.dpi, bbox_inches="tight")
        print(f"[SAVE] final frame -> {args.snapshot_output}")
    print(
        f"[DONE] {evolution.delays.size} delay components; "
        f"complete probability = "
        f"{np.sum(evolution.cumulative_probabilities[-1]):.12g}")
    plt.close(figure)


if __name__ == "__main__":
    main()
