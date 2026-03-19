from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def debug_plot_phase_density_composite_vis(
    state_vis,
    extent,
    title="Phase + density composite",
    density_gamma: float = 0.35,
    density_floor: float = 1e-12,
):
    if state_vis.ndim == 2:
        psi_vis = state_vis
        rho_vis = np.abs(psi_vis) ** 2
        phase_vis = np.angle(psi_vis)

    elif state_vis.ndim == 3:
        psi_vis = state_vis
        rho_vis = np.sum(np.abs(psi_vis) ** 2, axis=0)
        phase_vis = np.angle(psi_vis[0])

    else:
        raise ValueError(f"Unsupported state_vis ndim={state_vis.ndim}")

    rho_norm = rho_vis / (np.max(rho_vis) + 1e-30)
    value = np.clip(rho_norm, 0.0, 1.0) ** density_gamma
    value[rho_norm < density_floor] = 0.0

    hue = (phase_vis + np.pi) / (2.0 * np.pi)
    sat = np.ones_like(hue)

    hsv = np.stack([hue, sat, value], axis=-1)
    rgb = hsv_to_rgb(hsv)

    plt.figure(figsize=(8, 5))
    plt.imshow(rgb, extent=extent, origin="lower", aspect="auto")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def debug_plot_phase_density_composite_with_contours_vis(
    state_vis,
    X_vis,
    Y_vis,
    extent,
    title="Phase + density composite + contours",
    density_gamma: float = 0.35,
    density_floor: float = 1e-12,
):
    if state_vis.ndim == 2:
        psi_vis = state_vis
        rho_vis = np.abs(psi_vis) ** 2
        phase_vis = np.angle(psi_vis)

    elif state_vis.ndim == 3:
        psi_vis = state_vis
        rho_vis = np.sum(np.abs(psi_vis) ** 2, axis=0)
        phase_vis = np.angle(psi_vis[0])

    else:
        raise ValueError(f"Unsupported state_vis ndim={state_vis.ndim}")

    rho_norm = rho_vis / (np.max(rho_vis) + 1e-30)
    value = np.clip(rho_norm, 0.0, 1.0) ** density_gamma
    value[rho_norm < density_floor] = 0.0

    hue = (phase_vis + np.pi) / (2.0 * np.pi)
    sat = np.ones_like(hue)

    hsv = np.stack([hue, sat, value], axis=-1)
    rgb = hsv_to_rgb(hsv)

    plt.figure(figsize=(8, 5))
    plt.imshow(rgb, extent=extent, origin="lower", aspect="auto")

    levels = [0.05, 0.10, 0.20, 0.40, 0.70]
    plt.contour(
        X_vis,
        Y_vis,
        rho_norm,
        levels=levels,
        colors="white",
        linewidths=0.7,
        alpha=0.7,
    )

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def debug_plot_phase_winding_vis(
    state_vis,
    extent,
    dx: float,
    dy: float,
    title="Phase winding map",
):
    if state_vis.ndim == 2:
        psi = state_vis
    elif state_vis.ndim == 3:
        psi = state_vis[0]
    else:
        raise ValueError(f"Unsupported state_vis ndim={state_vis.ndim}")

    phase = np.angle(psi)

    dpx = np.diff(phase, axis=1)
    dpy = np.diff(phase, axis=0)

    dpx = (dpx + np.pi) % (2.0 * np.pi) - np.pi
    dpy = (dpy + np.pi) % (2.0 * np.pi) - np.pi

    winding = (
        dpx[:-1, :]
        + dpy[:, 1:]
        - dpx[1:, :]
        - dpy[:, :-1]
    ) / (2.0 * np.pi)

    x_min, x_max, y_min, y_max = extent

    plt.figure(figsize=(8, 5))
    plt.imshow(
        winding,
        extent=(x_min, x_max - dx, y_min, y_max - dy),
        origin="lower",
        cmap="seismic",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="phase winding")
    plt.show()

def debug_plot_scalar_field_vis(
    field_vis: np.ndarray,
    extent,
    title: str,
    cmap: str = "viridis",
    colorbar_label: str | None = None,
):
    fig, ax = plt.subplots(figsize=(8.6, 6.4))

    im = ax.imshow(
        field_vis,
        extent=extent,
        origin="lower",
        cmap=cmap,
        aspect="equal",
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    cbar = fig.colorbar(im, ax=ax)
    if colorbar_label is not None:
        cbar.set_label(colorbar_label)

    plt.tight_layout()
    plt.show()


def debug_plot_metric_fields_vis(
    alpha_vis: np.ndarray,
    a_vis: np.ndarray,
    V_metric_vis: np.ndarray,
    extent,
    title_prefix: str = "Metric fields",
):
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2))

    panels = [
        (alpha_vis, "alpha_metric", "viridis"),
        (a_vis, "a_metric", "plasma"),
        (V_metric_vis, "V_metric", "magma"),
    ]

    for ax, (field, title, cmap) in zip(axes, panels):
        im = ax.imshow(
            field,
            extent=extent,
            origin="lower",
            cmap=cmap,
            aspect="equal",
        )
        ax.set_title(f"{title_prefix}: {title}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()