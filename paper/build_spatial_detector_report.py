"""Build the English technical report for the calibrated double-slit study.

The report is rendered directly with Matplotlib so it can be rebuilt in the
project environment without a TeX installation.  Numerical results are read
from the locked 500 + 500 + 500 validation JSON file; study figures are reused
without modifying their data.
"""

from __future__ import annotations

import json
import math
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/quantumtoy-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
VALIDATION_JSON = ROOT / "spatial_detector_validation_500.json"
ARRIVAL_JSON = ROOT / "spatial_arrival_time_study.json"
ARRIVAL_FIGURE = ROOT / "spatial_arrival_time_study.png"
OUTPUT_PDF = PAPER / "QuantumToy_Double_Slit_Detector_Report.pdf"
BOOTSTRAP_FIGURE = PAPER / "spatial_detector_bootstrap_validation.png"

PAGE_SIZE = (8.27, 11.69)  # A4 portrait, inches
NAVY = "#132A44"
BLUE = "#2166AC"
CYAN = "#20A4B8"
PALE_BLUE = "#EAF3F8"
PALE_CYAN = "#EAF8F8"
PALE_RED = "#FCEDEC"
RED = "#B33A3A"
AMBER = "#B66A00"
PALE_AMBER = "#FFF4DF"
TEXT = "#24313D"
MUTED = "#607080"
LIGHT = "#D8E1E8"
WHITE = "#FFFFFF"


def wrapped(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


class Page:
    """Small deterministic page-layout helper."""

    def __init__(self, pdf: PdfPages, number: int, title: str, kicker: str = ""):
        self.pdf = pdf
        self.number = number
        self.fig = plt.figure(figsize=PAGE_SIZE, facecolor=WHITE)
        self.fig.add_artist(Rectangle((0, 0.965), 1, 0.035, transform=self.fig.transFigure,
                                      color=NAVY, linewidth=0))
        if kicker:
            self.fig.text(0.075, 0.943, kicker.upper(), fontsize=8.2, color=CYAN,
                          weight="bold", family="DejaVu Sans")
        title_size = 17.5 if len(title) > 47 else (19.0 if len(title) > 38 else 22.0)
        self.fig.text(0.075, 0.905, title, fontsize=title_size, color=NAVY, weight="bold",
                      va="top", family="DejaVu Sans")
        self.fig.add_artist(plt.Line2D([0.075, 0.925], [0.873, 0.873], color=LIGHT,
                                      linewidth=1, transform=self.fig.transFigure))
        self.y = 0.842

    def heading(self, text: str, size: float = 12.5, color: str = BLUE, gap: float = 0.016):
        self.fig.text(0.075, self.y, text, fontsize=size, color=color, weight="bold",
                      va="top", family="DejaVu Sans")
        self.y -= gap + 0.026

    def paragraph(self, text: str, width: int = 96, size: float = 9.2,
                  color: str = TEXT, gap: float = 0.014, line: float = 0.0184):
        content = wrapped(text, width)
        n_lines = content.count("\n") + 1
        self.fig.text(0.075, self.y, content, fontsize=size, color=color, va="top",
                      linespacing=1.35, family="DejaVu Sans")
        self.y -= n_lines * line + gap

    def bullet(self, text: str, width: int = 90, size: float = 9.1, gap: float = 0.012,
               color: str = TEXT):
        content = wrapped(text, width)
        n_lines = content.count("\n") + 1
        self.fig.text(0.084, self.y, "•", fontsize=12, color=CYAN, va="top", weight="bold")
        self.fig.text(0.105, self.y, content, fontsize=size, color=color, va="top",
                      linespacing=1.34, family="DejaVu Sans")
        self.y -= n_lines * 0.0182 + gap

    def equation(self, text: str, height: float = 0.062, size: float = 12.0):
        y0 = self.y - height
        self.fig.add_artist(FancyBboxPatch(
            (0.075, y0), 0.85, height, boxstyle="round,pad=0.008,rounding_size=0.006",
            transform=self.fig.transFigure, facecolor=PALE_BLUE, edgecolor=LIGHT,
            linewidth=0.8))
        self.fig.text(0.5, y0 + height / 2, text, fontsize=size, color=NAVY,
                      ha="center", va="center", family="DejaVu Sans")
        self.y = y0 - 0.022

    def callout(self, title: str, text: str, tone: str = "blue", height: float | None = None):
        face, edge, title_color = {
            "blue": (PALE_BLUE, BLUE, NAVY),
            "cyan": (PALE_CYAN, CYAN, NAVY),
            "amber": (PALE_AMBER, AMBER, AMBER),
            "red": (PALE_RED, RED, RED),
        }[tone]
        content = wrapped(text, 91)
        n_lines = content.count("\n") + 1
        h = height or (0.055 + n_lines * 0.018)
        y0 = self.y - h
        self.fig.add_artist(FancyBboxPatch(
            (0.075, y0), 0.85, h, boxstyle="round,pad=0.009,rounding_size=0.007",
            transform=self.fig.transFigure, facecolor=face, edgecolor=edge, linewidth=0.9))
        self.fig.text(0.095, y0 + h - 0.020, title, fontsize=10.0, color=title_color,
                      weight="bold", va="top", family="DejaVu Sans")
        self.fig.text(0.095, y0 + h - 0.047, content, fontsize=8.8, color=TEXT, va="top",
                      linespacing=1.32, family="DejaVu Sans")
        self.y = y0 - 0.024

    def table(self, columns: list[str], rows: list[list[str]], widths: list[float] | None = None,
              height: float | None = None, font_size: float = 8.2):
        n = len(rows) + 1
        h = height or min(0.38, 0.039 * n + 0.012)
        ax = self.fig.add_axes([0.075, self.y - h, 0.85, h])
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=columns, cellLoc="left", colLoc="left",
                         colWidths=widths, loc="upper left", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor(WHITE)
            cell.set_linewidth(1.0)
            cell.PAD = 0.075
            if r == 0:
                cell.set_facecolor(NAVY)
                cell.get_text().set_color(WHITE)
                cell.get_text().set_weight("bold")
            else:
                cell.set_facecolor(PALE_BLUE if r % 2 else "#F6F8FA")
                cell.get_text().set_color(TEXT)
        self.y -= h + 0.025

    def image(self, path: Path, height: float, caption: str | None = None):
        image = plt.imread(path)
        cap_h = 0.035 if caption else 0
        ax = self.fig.add_axes([0.075, self.y - height, 0.85, height])
        ax.imshow(image)
        ax.axis("off")
        self.y -= height + 0.008
        if caption:
            self.fig.text(0.075, self.y, wrapped(caption, 105), fontsize=7.7, color=MUTED,
                          va="top", style="italic", linespacing=1.25)
            self.y -= cap_h

    def footer(self):
        self.fig.add_artist(plt.Line2D([0.075, 0.925], [0.046, 0.046], color=LIGHT,
                                      linewidth=0.8, transform=self.fig.transFigure))
        self.fig.text(0.075, 0.025, "QuantumToy • synthetic detector benchmark",
                      fontsize=7.3, color=MUTED, va="center")
        self.fig.text(0.925, 0.025, str(self.number), fontsize=7.3, color=MUTED,
                      ha="right", va="center")
        self.pdf.savefig(self.fig, bbox_inches=None)
        plt.close(self.fig)


def add_flow_box(fig, x: float, y: float, w: float, h: float, title: str, body: str,
                 face: str = PALE_BLUE, edge: str = BLUE):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.008",
                                  transform=fig.transFigure, facecolor=face, edgecolor=edge,
                                  linewidth=1.0))
    fig.text(x + 0.014, y + h - 0.020, title, fontsize=9.2, color=NAVY, weight="bold", va="top")
    fig.text(x + 0.014, y + h - 0.050, wrapped(body, max(18, int(w * 112))), fontsize=7.7,
             color=TEXT, va="top", linespacing=1.25)


def add_arrow(fig, x1: float, y1: float, x2: float, y2: float):
    fig.add_artist(FancyArrowPatch((x1, y1), (x2, y2), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=12, color=CYAN,
                                   linewidth=1.5))


def make_bootstrap_figure(data: dict) -> None:
    lr = np.asarray(data["bootstrap_detection_threshold"]["likelihood_ratios"], dtype=float)
    threshold = float(data["bootstrap_detection_threshold"]["threshold"])
    eval_q95 = float(data["monte_carlo_recovery"][0]["likelihood_ratio_quantiles"][2])
    interval = data["validation_intervals"]["bootstrap_threshold_order_statistic_95_interval"]

    fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    fig.patch.set_facecolor(WHITE)
    bins = np.linspace(0, max(7.25, np.quantile(lr, 0.995)), 30)
    ax_hist.hist(lr, bins=bins, color=BLUE, alpha=0.82, edgecolor=WHITE)
    ax_hist.axvspan(interval[0], interval[1], color=CYAN, alpha=0.16,
                    label="95% order-statistic interval")
    ax_hist.axvline(threshold, color=AMBER, linewidth=2.2, label=f"Bootstrap threshold = {threshold:.3f}")
    ax_hist.axvline(eval_q95, color=RED, linewidth=2.2, linestyle="--",
                    label=f"Independent null 95% quantile = {eval_q95:.3f}")
    ax_hist.set(xlabel="Likelihood-ratio statistic", ylabel="Bootstrap count",
                title="Bootstrap null distribution (n = 500)")
    ax_hist.legend(frameon=False, fontsize=8, loc="upper right")

    ordered = np.sort(lr)
    probs = np.arange(1, len(ordered) + 1) / len(ordered)
    ax_cdf.step(ordered, probs, where="post", color=BLUE, linewidth=2)
    ax_cdf.axhline(0.95, color=MUTED, linewidth=1, linestyle=":")
    ax_cdf.axvline(threshold, color=AMBER, linewidth=2.2)
    ax_cdf.axvline(eval_q95, color=RED, linewidth=2.2, linestyle="--")
    ax_cdf.set_xlim(0, max(7.25, np.quantile(lr, 0.995)))
    ax_cdf.set_ylim(0, 1.01)
    ax_cdf.set(xlabel="Likelihood-ratio statistic", ylabel="Empirical CDF",
               title="Tail quantile is uncertain at 500 bootstrap draws")
    for ax in (ax_hist, ax_cdf):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=LIGHT, linewidth=0.7, alpha=0.8)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.title.set_color(NAVY)
    fig.savefig(BOOTSTRAP_FIGURE, dpi=180, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def cover(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=WHITE)
    fig.add_artist(Rectangle((0, 0.70), 1, 0.30, transform=fig.transFigure,
                             color=NAVY, linewidth=0))
    fig.add_artist(Rectangle((0.075, 0.655), 0.18, 0.012, transform=fig.transFigure,
                             color=CYAN, linewidth=0))
    fig.text(0.075, 0.925, "QUANTUMTOY TECHNICAL REPORT", color=CYAN, fontsize=10,
             weight="bold", va="top")
    fig.text(0.075, 0.855, "A Complete and Calibrated\nDouble-Slit Detector Model",
             color=WHITE, fontsize=27, weight="bold", va="top", linespacing=1.12)
    fig.text(0.075, 0.735, "Temporal-width and coupling inference with a complete outcome law",
             color="#D9EAF2", fontsize=12, va="top")

    fig.text(0.075, 0.595, "Purpose", color=BLUE, fontsize=12, weight="bold", va="top")
    fig.text(0.075, 0.558, wrapped(
        "This report consolidates the progression from an exploratory measurement-guided "
        "simulation to a complete spatial POVM, a locked double-slit geometry, a robust "
        "two-setting detector design, a calibrated observation channel, and an independent "
        "Monte Carlo validation.", 91), color=TEXT, fontsize=10.4, va="top", linespacing=1.45)

    add_flow_box(fig, 0.075, 0.355, 0.252, 0.105, "Physics model",
                 "Unitary propagation, complete click/no-click effects, and a declared temporal mixture.")
    add_flow_box(fig, 0.374, 0.355, 0.252, 0.105, "Detector model",
                 "Efficiency, dark counts, spatial blur, timing jitter, and control calibration.",
                 PALE_CYAN, CYAN)
    add_flow_box(fig, 0.673, 0.355, 0.252, 0.105, "Validation",
                 "Bootstrap threshold plus independent null and signal ensembles.",
                 PALE_AMBER, AMBER)

    fig.add_artist(FancyBboxPatch((0.075, 0.160), 0.85, 0.110,
                                  boxstyle="round,pad=0.012,rounding_size=0.008",
                                  transform=fig.transFigure, facecolor=PALE_RED,
                                  edgecolor=RED, linewidth=1.0))
    fig.text(0.095, 0.245, "Scope and status", color=RED, fontsize=10.5,
             weight="bold", va="top")
    fig.text(0.095, 0.212, wrapped(
        "Synthetic benchmark, not empirical evidence. The response coupling and its numerical "
        "values are not derived from TRF Information-Theoretic Formulation v0.2. The final "
        "false-positive calibration criterion is not met and is reported as a validation result.", 88),
        color=TEXT, fontsize=9.2, va="top", linespacing=1.35)

    fig.text(0.075, 0.075, "22 September 2026", fontsize=8.5, color=MUTED)
    fig.text(0.925, 0.075, "QuantumToy", fontsize=8.5, color=MUTED, ha="right")
    pdf.savefig(fig)
    plt.close(fig)


def build_report() -> None:
    if not VALIDATION_JSON.exists():
        raise FileNotFoundError(f"Missing validation input: {VALIDATION_JSON}")
    if not ARRIVAL_JSON.exists() or not ARRIVAL_FIGURE.exists():
        raise FileNotFoundError("Missing joint arrival-time study artifacts")
    data = json.loads(VALIDATION_JSON.read_text())
    arrival = json.loads(ARRIVAL_JSON.read_text())
    make_bootstrap_figure(data)

    PAPER.mkdir(exist_ok=True)
    with PdfPages(OUTPUT_PDF) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "A Complete and Calibrated Double-Slit Detector Model"
        metadata["Author"] = "QuantumToy project"
        metadata["Subject"] = "Synthetic temporal-width and detector-inference benchmark"
        metadata["Keywords"] = "double slit, POVM, detector calibration, likelihood, bootstrap"

        cover(pdf)

        p = Page(pdf, 2, "Executive summary", "Outcome")
        p.callout("Main result", "The implementation now defines a normalized generative law from "
                  "preparation through detector readout and can recover an injected signal. The "
                  "independent null ensemble nevertheless rejects 9.4% of trials at a nominal 5% "
                  "level, so the current detection threshold is not calibrated well enough for a "
                  "controlled false-positive claim.", tone="red")
        p.heading("What was built")
        for text in [
            "A complete spatial measurement with 16 y-resolved click bins and an explicit no-click outcome. Positivity, normalization, and invariance to propagator global phase are testable properties of the construction.",
            "A unitary double-slit propagation model with a fixed barrier, slit geometry, wave packet, temporal quadrature, and convergence checks.",
            "Joint inference of temporal width sigma_T and coupling lambda. The exact null lambda = 0 removes sigma_T from the observable law, exposing the expected non-identifiability.",
            "A second detector selected by a minimax Fisher-design objective over a 3 x 3 parameter region, followed by an independent 80 x 80 grid verification.",
            "A realistic readout channel for efficiency, dark clicks, y-bin blur, and timing jitter, calibrated on five free-propagation controls and profiled jointly with the signal parameters.",
            "A locked validation run: 500 bootstrap null samples set the threshold; independent 500-sample null and signal ensembles assess false detections, power, bias, RMSE, and joint coverage.",
        ]:
            p.bullet(text)
        p.heading("Decision-ready numbers")
        p.table(["Quantity", "Result", "Interpretation"], [
            ["Robust detector gain", "min log-det +1.661", "Improves joint local information"],
            ["Null detection", "0.094 [0.071, 0.123]", "Nominal 0.05 excluded"],
            ["Signal detection", "1.000 [0.992, 1.000]", "High synthetic power"],
            ["Signal joint coverage", "0.958 [0.937, 0.972]", "Near nominal 0.95"],
            ["Signal lambda bias / RMSE", "+0.0486 / 0.1523", "Bias remains visible"],
        ], widths=[0.29, 0.27, 0.44], height=0.225)
        p.footer()

        p = Page(pdf, 3, "Problem statement and model requirements", "1 • Problem")
        p.paragraph("The original measurement-guided simulation was useful for exploring spatial "
                    "state evolution, but its apparent sensitivity to the temporal-width parameter "
                    "depended strongly on normalization choices for effect and overlap fields. That "
                    "made it difficult to distinguish a physical prediction from a response created "
                    "by the analysis convention.")
        p.heading("Operational question")
        p.paragraph("Can a fully specified double-slit measurement identify a temporal-mixture width "
                    "and an unknown response strength while retaining all outcomes, calibrating detector "
                    "imperfections, and controlling the false-positive probability under an exact null?")
        p.equation(r"$H_0:\;\lambda=0\quad\Longrightarrow\quad p(y,\mathrm{no\ click})\ \mathrm{is\ independent\ of}\ \sigma_T$", height=0.066)
        p.equation(r"$H_1:\;\lambda>0\quad\Longrightarrow\quad (\sigma_T,\lambda)\ \mathrm{are\ inferred\ jointly}$", height=0.066)
        p.heading("Requirements imposed on the solution")
        for text in [
            "Completeness: every click bin and no-click must appear in one probability vector; no postselection on detected events.",
            "Physical operator structure: positive effects sum to identity and temporal alternatives are mixed at the operator level.",
            "Identifiability: sigma_T and lambda must be assessed jointly because either can partly compensate for the other.",
            "Prospective design: detector settings are chosen by a declared objective over a parameter region and then checked on a finer grid.",
            "Instrument realism: detector response is estimated from separate controls and treated as shared nuisance parameters in the final likelihood.",
            "Independent validation: threshold construction and performance evaluation use separate simulated ensembles.",
        ]:
            p.bullet(text)
        p.callout("Interpretive boundary", "The implementation tests whether a declared operational "
                  "model is internally measurable. It does not establish that nature follows the "
                  "chosen lambda response or that sigma_T has the numerical scale used in the synthetic "
                  "benchmark.", tone="amber")
        p.footer()

        p = Page(pdf, 4, "Parameters and locked experimental configuration", "2 • Parameters")
        p.heading("Physics and response parameters")
        p.table(["Symbol / name", "Role", "Value in validation"], [
            ["sigma_T", "Half-Gaussian temporal width", "0.20 (signal and indexed null point)"],
            ["lambda", "Phenomenological response strength", "0 under H0; 1.0 under H1"],
            ["q(lambda)", "Bounded mixture fraction", "1 - exp(-lambda)"],
            ["eta", "Click efficiency", "0.88 injected"],
            ["d", "Per-shot dark-click probability", "0.012 injected"],
            ["b", "Gaussian y-bin resolution", "0.45 bins injected"],
            ["j", "Symmetric timing jitter", "0.05 injected"],
        ], widths=[0.23, 0.48, 0.29], height=0.220, font_size=7.7)
        p.heading("Double-slit geometry")
        p.table(["Quantity", "Locked value", "Quantity", "Locked value"], [
            ["Box", "10 x 8", "Validation grid", "48 x 48"],
            ["Verification grid", "80 x 80", "Barrier center", "x = -0.5"],
            ["Barrier width / height", "0.16 / 35", "Slit centers", "y = +/-0.8"],
            ["Slit half-height", "0.30", "Edge smoothing", "0.08"],
            ["Packet center", "(-2.5, 0)", "Packet widths", "(0.45, 0.65)"],
            ["Packet momentum", "kx = 3, ky = 0", "Click outcomes", "16 y bins"],
            ["Delay / propagation step", "0.025 / 0.005", "Temporal horizon", "4 sigma_T"],
        ], widths=[0.23, 0.24, 0.28, 0.25], height=0.205, font_size=7.5)
        p.heading("Data budgets and selected detector pair")
        p.table(["Component", "Configuration"], [
            ["Original detector", "x = 1.5, width = 0.3, reference time = 0.9"],
            ["Selected detector", "x = 0, width = 0.5, reference time = 0.43333"],
            ["Allocation", "50/50 split of 100,000 test shots"],
            ["Calibration", "Five free controls; 50,000 shots each (250,000 total)"],
            ["Validation ensembles", "500 bootstrap + 500 independent null + 500 signal"],
            ["Design region", "sigma_T = {0.12, 0.20, 0.28}; lambda = {0.5, 1.0, 1.5}"],
        ], widths=[0.29, 0.71], height=0.170, font_size=7.4)
        p.footer()

        p = Page(pdf, 5, "Complete operator-level measurement model", "3 • Method")
        p.paragraph("A Gaussian detector gate G(x) and a partition of the transverse y axis define "
                    "terminal click effects. The complementary effect records every event that does not "
                    "produce one of those clicks.")
        p.equation(r"$\Pi_r=\mathrm{diag}\!\left(G(x)\,\mathbf{1}_{y\in r}\right),\qquad \Pi_{\varnothing}=I-\sum_r\Pi_r$", height=0.074)
        p.paragraph("For delay tau_j and reference time t_ref, unitary propagation moves each terminal "
                    "effect to the preparation time. A normalized half-Gaussian quadrature then mixes "
                    "the unresolved delays as operators, retaining complex off-diagonal information.")
        p.equation(r"$E_{r,j}=U(t_{\rm ref}+\tau_j)^\dagger\Pi_r U(t_{\rm ref}+\tau_j),\qquad E_r(\sigma_T)=\sum_j w_j(\sigma_T)E_{r,j}$", height=0.076)
        p.heading("Declared coupling law")
        p.paragraph("The reference effect E_ref,r uses the zero-delay propagation. The candidate "
                    "response interpolates between that reference and the temporal mixture.")
        p.equation(r"$q(\lambda)=1-e^{-\lambda},\qquad F_r=(1-q)E_{{\rm ref},r}+qE_r(\sigma_T)$", height=0.072)
        p.heading("Consequences")
        for text in [
            "Positivity and completeness survive unitary conjugation and convex mixing; the implementation checks them numerically.",
            "A global phase of any propagator cancels from U-dagger Pi U and cannot change an outcome probability.",
            "At lambda = 0, F_r equals E_ref,r exactly, so sigma_T is absent from the law and cannot be estimated physically.",
            "The exponential q(lambda) is a convenient bounded phenomenological mapping. It is an assumption to be replaced or calibrated if a physical TRF coupling is derived.",
        ]:
            p.bullet(text)
        p.callout("Why this solved the normalization problem", "The measured object is now a complete "
                  "probability law generated by explicit effects. No candidate-dependent maximum "
                  "normalization is needed, and no-click probability prevents hidden conditioning on "
                  "successful detections.", tone="cyan")
        p.footer()

        p = Page(pdf, 6, "Locked double-slit geometry and observable signal", "4 • Double slit")
        p.paragraph("The free propagator was replaced by a unitary symmetric split-step evolution in a "
                    "fixed static double-slit potential. Geometry was locked before profiling sigma_T "
                    "and lambda. The synthetic injection is (sigma_T, lambda) = (0.20, 1.0).")
        p.image(ROOT / "spatial_effect_double_slit_study.png", 0.575,
                "Figure 1. Locked geometry, conditional click modulation, joint profile, and numerical checks. "
                "Conditional click probabilities are diagnostic only; inference uses the full click/no-click law.")
        p.table(["Observable comparison", "Result"], [
            ["Free vs double-slit complete-law total variation", "0.11564"],
            ["Free click probability", "0.16318"],
            ["Double-slit click probability", "0.04780"],
            ["Injected pair recovered on the profile grid", "(0.20, 1.0)"],
        ], widths=[0.67, 0.33], height=0.145)
        p.footer()

        p = Page(pdf, 7, "Numerical robustness and the first held-out failure", "5 • Robustness")
        p.heading("Complete-law sensitivity checks")
        p.table(["Variation", "Total-variation distance", "Meaning"], [
            ["Half delay quadrature step", "1.44e-5", "Temporal integration"],
            ["5/4 spatial grid", "4.16e-4", "Spatial discretization"],
            ["One extra sigma_T of horizon", "1.33e-6", "Tail truncation"],
            ["Half split-step", "3.13e-5", "Propagation step"],
            ["5/4 x-box, fixed resolution", "3.48e-6", "Periodic boundary"],
            ["Half slit-edge smoothing", "3.06e-3", "Geometry sensitivity"],
        ], widths=[0.43, 0.25, 0.32], height=0.245)
        p.paragraph("Every numerical refinement is far below the free-versus-slit signal of 0.11564. "
                    "The slit-edge number is larger because it changes the physical geometry; it is a "
                    "model sensitivity rather than a discretization error.")
        p.heading("An independent detector is not automatically informative")
        p.paragraph("The detector pair chosen earlier for free propagation was tested unchanged in the "
                    "double-slit model. It failed: with the same 100,000-shot budget, splitting shots "
                    "between the two settings made both marginal errors worse.")
        p.table(["Diagnostic", "One slit detector", "Old free-geometry pair"], [
            ["Fisher condition number", "98.61", "287.18"],
            ["Parameter correlation", "-0.725", "-0.907"],
            ["Marginal errors", "baseline", "both increase"],
        ], widths=[0.40, 0.30, 0.30], height=0.145)
        p.callout("Design lesson", "Two detector records help only when their score directions are "
                  "complementary in the geometry actually used. The failed held-out transfer was "
                  "retained and motivated a prospective double-slit-specific design scan.", tone="amber")
        p.footer()

        p = Page(pdf, 8, "Prospective two-setting detector design", "6 • Experimental design")
        p.image(ROOT / "spatial_effect_double_slit_detector_design.png", 0.420,
                "Figure 2. Candidate detector scan and the selected complementary setting. All comparisons use "
                "the same total test-shot budget.")
        p.paragraph("The local scan varied x position, gate width, and reference time. The later robust "
                    "scan evaluated all candidates over nine (sigma_T, lambda) points and maximized the "
                    "worst log-determinant gain relative to the original detector.")
        p.table(["Diagnostic", "Original setting", "Selected 50/50 pair"], [
            ["Selected second detector", "—", "x = 0, width = 0.5, t = 0.43333"],
            ["Click probability", "0.04780", "second setting: 0.17907"],
            ["Fisher condition number", "98.61", "33.68"],
            ["Correlation", "-0.72545", "-0.68220"],
            ["SE(sigma_T)", "0.02242", "0.01956"],
            ["SE(lambda)", "0.15151", "0.08074"],
            ["80 x 80 robust verification", "min log-det +1.661", "SE ratios: 0.838 / 0.731"],
            ["Selected-detector box check", "—", "complete-law change 1.12e-7"],
        ], widths=[0.37, 0.27, 0.36], height=0.265, font_size=7.7)
        p.footer()

        p = Page(pdf, 9, "Realistic detector channel and control calibration", "7 • Observation model")
        p.paragraph("The ideal complete law p is passed through a column-stochastic observation channel. "
                    "This preserves normalization while modelling missed clicks, dark clicks, and finite "
                    "transverse resolution. Symmetric timing jitter is applied before the readout channel.")
        p.equation(r"$p_{\rm obs}=C(\eta,d,b)\,p,\qquad \sum_i C_{ij}=1\ \ \mathrm{for\ every}\ j$", height=0.070)
        y = 0.565
        add_flow_box(p.fig, 0.075, y, 0.205, 0.120, "Ideal law",
                     "16 click bins + no-click from complete effects.")
        add_arrow(p.fig, 0.285, y + 0.060, 0.355, y + 0.060)
        add_flow_box(p.fig, 0.365, y, 0.245, 0.120, "Timing response",
                     "Three-node symmetric jitter quadrature; distinct from the half-Gaussian signal delay.",
                     PALE_CYAN, CYAN)
        add_arrow(p.fig, 0.615, y + 0.060, 0.685, y + 0.060)
        add_flow_box(p.fig, 0.695, y, 0.230, 0.120, "Readout channel",
                     "Efficiency eta, dark probability d, blur b; still complete.",
                     PALE_AMBER, AMBER)
        p.y = 0.515
        p.heading("Calibration design")
        p.paragraph("Five free-propagation controls at reference times 0.55, 0.70, 0.85, 1.00, and "
                    "1.15 use the known lambda = 0 law. No double-slit counts enter the pilot fit. In "
                    "the final analysis, control and double-slit log likelihoods are added and the "
                    "shared detector parameters are profiled again.")
        response = data["pilot_calibration_fit"]["response"]
        p.table(["Detector parameter", "Injected", "Pilot estimate", "Pilot standard error"], [
            ["Efficiency eta", "0.8800", f"{response['efficiency']:.4f}", "0.00375"],
            ["Dark probability d", "0.0120", f"{response['dark_probability']:.4f}", "0.00050"],
            ["Blur b (bins)", "0.4500", f"{response['blur_sigma_bins']:.4f}", "0.02032"],
            ["Timing jitter", "0.0500", f"{response['timing_jitter']:.4f}", "fixed in pilot grid"],
        ], widths=[0.34, 0.18, 0.22, 0.26], height=0.165)
        p.callout("No hidden postselection", "A missed ideal click enters the same dark/no-click branch "
                  "as an ideal no-click. All recorded categories remain in the multinomial likelihood.", tone="blue")
        p.footer()

        p = Page(pdf, 10, "Joint likelihood and independent validation protocol", "8 • Inference")
        p.paragraph("For calibration controls c and the two double-slit settings s, the fit minimizes the "
                    "sum of multinomial negative log likelihoods. Detector-response parameters are shared "
                    "between control and test data and profiled for every candidate (sigma_T, lambda).")
        p.equation(r"$-\ell= -\sum_c\sum_r n_{cr}\log p_{cr}(\theta)-\sum_s\sum_r n_{sr}\log p_{sr}(\sigma_T,\lambda,\theta)$", height=0.074)
        p.paragraph("Here theta = (eta, d, b, jitter). Because sigma_T disappears under lambda = 0 and "
                    "lambda lies on a boundary, a fixed chi-square reference for the likelihood ratio is "
                    "not justified. The threshold is therefore estimated by parametric bootstrap.")
        p.heading("Locked Monte Carlo sequence")
        y = 0.420
        boxes = [
            (0.075, "1. Calibrate", "Draw 50,000 shots per control and fit detector response."),
            (0.300, "2. Bootstrap", "Draw 500 null trials and set the empirical 95% LR threshold."),
            (0.525, "3. Null check", "Draw a fresh 500-trial null ensemble; measure false detections."),
            (0.750, "4. Signal check", "Draw 500 signal trials; measure power, bias, RMSE, coverage."),
        ]
        for x, title, body in boxes:
            add_flow_box(p.fig, x, y, 0.175, 0.140, title, body,
                         PALE_BLUE if x < 0.5 else PALE_CYAN, BLUE if x < 0.5 else CYAN)
        for x in [0.255, 0.480, 0.705]:
            add_arrow(p.fig, x, y + 0.070, x + 0.037, y + 0.070)
        p.y = 0.380
        p.heading("Reported criteria")
        for text in [
            "Detection rate with a Wilson 95% interval, evaluated independently of the bootstrap sample.",
            "Joint 95% region coverage for the signal; under the null, sigma_T coverage is bookkeeping because the width is structurally unidentifiable.",
            "Bias, RMSE, optimizer success, likelihood-ratio quantiles, and a distribution-free order-statistic interval for the estimated bootstrap quantile.",
        ]:
            p.bullet(text)
        p.callout("Predeclared interpretation", "The bootstrap threshold is accepted only if the "
                  "independent null rejection rate is compatible with 0.05 within sampling uncertainty. "
                  "The evaluation ensemble is not used to repair the threshold.", tone="amber")
        p.footer()

        p = Page(pdf, 11, "Calibrated inference diagnostics", "9 • Results")
        p.image(ROOT / "spatial_detector_inference_study.png", 0.575,
                "Figure 3. Detector response, control calibration, robust-design scan, and one profiled "
                "double-slit fit. The example signal fit has LR = 574.10 and is detected.")
        p.heading("Key diagnostic readings")
        p.table(["Diagnostic", "Value"], [
            ["Robust design minimum log-det gain (80 x 80)", "1.661"],
            ["Worst SE ratio: sigma_T / lambda", "0.838 / 0.731"],
            ["Example fit: sigma_T / lambda", "0.2533 / 0.95"],
            ["Example likelihood-ratio statistic", "574.10"],
            ["Successful optimizations across validation", "1,500 / 1,500"],
        ], widths=[0.70, 0.30], height=0.185)
        p.paragraph("The profile surface has the expected correlated valley, but the complementary detector "
                    "and shared controls keep the injected signal region finite. Optimizer success alone "
                    "does not establish frequentist calibration; that is assessed on the next pages.")
        p.footer()

        p = Page(pdf, 12, "Monte Carlo recovery: 500 independent trials per case", "10 • Results")
        p.image(ROOT / "spatial_detector_validation_500.png", 0.405,
                "Figure 4. Locked 500 + 500 + 500 validation. Dashed or shaded references show injected "
                "values and intended operating levels; the null rejection bar exceeds its target.")
        intervals = data["validation_intervals"]["per_scenario"]
        p.table(["Metric", "Null: lambda = 0", "Signal: lambda = 1"], [
            ["Detection rate", "0.094", "1.000"],
            ["Wilson 95% interval", "[0.071, 0.123]", "[0.992, 1.000]"],
            ["Joint coverage", "0.988*", "0.958"],
            ["Coverage Wilson interval", "[0.974, 0.994]", "[0.937, 0.972]"],
            ["sigma_T bias / RMSE", "-0.0375* / 0.0770*", "+0.0010 / 0.0348"],
            ["lambda bias / RMSE", "+0.0139 / 0.0234", "+0.0486 / 0.1523"],
        ], widths=[0.38, 0.31, 0.31], height=0.225, font_size=7.9)
        p.callout("Asterisked null-width entries", "At lambda = 0, sigma_T is absent from the law; "
                  "asterisked entries are grid bookkeeping rather than physical width estimates.",
                  tone="blue", height=0.078)
        p.footer()

        p = Page(pdf, 13, "Bootstrap threshold calibration", "11 • Validation finding")
        p.image(BOOTSTRAP_FIGURE, 0.315,
                "Figure 5. The 500-draw bootstrap threshold is 2.422. Its order-statistic uncertainty "
                "contains the independent null 95% quantile 3.049, indicating an unstable tail estimate.")
        p.table(["Quantity", "Value", "Assessment"], [
            ["Bootstrap 95% LR threshold", "2.422", "Used exactly as locked"],
            ["Threshold order-statistic 95% interval", "[1.984, 3.262]", "Wide tail uncertainty"],
            ["Independent null 95% LR quantile", "3.049", "Above point threshold"],
            ["Independent null rejections", "47 / 500", "9.4%"],
            ["False-detection Wilson interval", "[7.1%, 12.3%]", "Excludes intended 5%"],
        ], widths=[0.43, 0.26, 0.31], height=0.185, font_size=7.8)
        p.callout("Validation decision: criterion not met", "The hypothesis test fails its false-positive "
                  "calibration requirement. The threshold must not be raised after inspecting the independent "
                  "evaluation set. The signal-recovery results remain useful diagnostics of the generative "
                  "model, but they do not repair the null calibration.", tone="red", height=0.112)
        p.heading("Most likely next statistical improvement")
        p.paragraph("Increase the independently seeded bootstrap substantially and replace the coarse "
                    "parameter grid near lambda = 0 with continuous or adaptively refined profiling. Lock "
                    "the new threshold and fitting procedure, then evaluate them once on another untouched "
                    "null ensemble. If 5% calibration still fails, revise the test statistic or calibration "
                    "scheme rather than tuning against evaluation data.")
        p.footer()

        p = Page(pdf, 14, "Conclusions, limitations, and reproducibility", "12 • Conclusions")
        p.heading("What the work now supports")
        for text in [
            "A mathematically complete click/no-click measurement can carry temporal-width information without candidate-dependent normalization.",
            "The double-slit model produces a numerically resolved, geometry-dependent complete-law signal, and the selected second setting materially improves joint information about sigma_T and lambda.",
            "Separate control data can calibrate detector efficiency, dark counts, blur, and jitter, with those nuisance parameters carried into the final profile likelihood.",
            "At the injected lambda = 1 benchmark, recovery is strong: 100% detection in 500 trials, near-nominal joint coverage, and small sigma_T bias.",
        ]:
            p.bullet(text)
        p.heading("What remains unresolved")
        for text in [
            "The lambda response law is phenomenological and is not a derivation from TRF-IT v0.2. Empirical interpretation requires a physical coupling law or independent calibration of that mapping.",
            "The current bootstrap point threshold does not control the independent false-positive rate at 5%. A larger locked calibration and finer profiling are required.",
            "Results are synthetic and conditional on the chosen geometry, noise channel, parameter region, and shot budgets. Real apparatus backgrounds and model mismatch are not represented.",
            "sigma_T is structurally unidentifiable under lambda = 0; no null width estimate should be reported as a measurement.",
        ]:
            p.bullet(text)
        p.heading("Reproduce the locked validation")
        p.callout("Command", "PYTHONPATH=src/quantumtoy python "
                  "src/quantumtoy/analysis/debug/run_spatial_detector_inference_study.py "
                  "--bootstrap-repetitions 500 --repetitions 500 --workers 10 "
                  "--plot-output spatial_detector_validation_500.png "
                  "--json-output spatial_detector_validation_500.json", tone="blue", height=0.132)
        p.footer()

        p = Page(pdf, 15, "A measurable joint spatial/arrival-time hypothesis",
                 "13 • Timing extension")
        p.paragraph("A closer laboratory record associates each pulsed or heralded trial with a "
                    "y bin and finite arrival-time bin, while retaining no-click. The timestamp is "
                    "stored after subtracting the standard time of flight for each detector setting.")
        p.equation(r"$\Delta t_{\rm rec}=\kappa_t\tau+\epsilon_{\rm src}+\epsilon_{\rm det}+\Delta t_{\rm clock}$",
                   height=0.062, size=11.5)
        p.image(ARRIVAL_FIGURE, 0.300,
                "Figure 6. Joint click law, sensitivity to the latent-to-clock coupling and timestamp "
                "jitter, and a 40-record calibrated-nuisance recovery check.")
        precision = arrival["precision_comparison"]
        p.table(["Observation", "SE(sigma_T)", "SE(lambda)"], [
            [precision[0]["label"], f"{precision[0]['standard_errors'][0]:.5f}",
             f"{precision[0]['standard_errors'][1]:.5f}"],
            [precision[1]["label"], f"{precision[1]['standard_errors'][0]:.5f}",
             f"{precision[1]['standard_errors'][1]:.5f}"],
            [precision[2]["label"], f"{precision[2]['standard_errors'][0]:.5f}",
             f"{precision[2]['standard_errors'][1]:.5f}"],
            [precision[3]["label"], f"{precision[3]['standard_errors'][0]:.5f}",
             f"{precision[3]['standard_errors'][1]:.5f}"],
        ], widths=[0.58, 0.21, 0.21], height=0.130, font_size=7.2)
        p.callout("Physical limitation", "TRF-IT v0.2 does not derive the latent-delay-to-clock map "
                  "or fix kappa_t. Source width and detector jitter enter through their quadrature "
                  "sum, so one needs independent calibration. The timing gain is conditional on this "
                  "instrument hypothesis.", tone="amber", height=0.092)
        p.footer()

        p = Page(pdf, 16, "Reproducibility map and evidence trail", "Appendix A")
        p.paragraph("The report is generated from repository-local code and locked result artifacts. "
                    "The original TRF-IT v0.2 document provides the research context; none of its prose "
                    "is treated as an execution instruction, and the numerical response law reported "
                    "here remains explicitly phenomenological.")
        p.heading("Primary implementation and result artifacts")
        p.table(["Artifact", "Purpose"], [
            ["src/quantumtoy/theories/thick_front_measurement_guided.py", "Earlier exploratory measurement-guided model"],
            ["src/quantumtoy/analysis/spatial_effect_measurement.py", "Complete effects, propagation, profiles, design"],
            ["src/quantumtoy/analysis/spatial_detector_inference.py", "Observation channel, calibration, robust inference"],
            ["src/quantumtoy/analysis/spatial_arrival_time.py", "Finite-gate joint spatial/arrival-time law"],
            ["src/quantumtoy/analysis/debug/run_spatial_detector_inference_study.py", "Locked study and parallel Monte Carlo runner"],
            ["src/quantumtoy/analysis/debug/run_spatial_arrival_time_study.py", "Timing sensitivity and recovery study"],
            ["tests/test_spatial_detector_inference.py", "Detector, calibration, design, and validation tests"],
            ["tests/test_spatial_arrival_time.py", "Joint-law completeness and timing-identifiability tests"],
            ["spatial_detector_validation_500.json", "Machine-readable 500 + 500 + 500 result"],
            ["spatial_arrival_time_study.json", "Machine-readable p(y,t) comparison"],
            ["spatial_detector_validation_500.png", "Validation overview used in Figure 4"],
            ["paper/build_spatial_detector_report.py", "Rebuilds this report and Figure 5"],
        ], widths=[0.58, 0.42], height=0.340, font_size=6.4)
        p.heading("Rebuild this report")
        p.callout("Command", "MPLCONFIGDIR=/tmp/quantumtoy-matplotlib python "
                  "paper/build_spatial_detector_report.py", tone="blue", height=0.075)
        p.heading("Evidence hierarchy")
        for text in [
            "Operator identities and numerical invariants establish internal consistency of the implemented measurement law.",
            "Fisher information and synthetic recovery quantify identifiability conditional on the declared model and shot budgets.",
            "Independent null validation governs the detection claim and currently reports a failed 5% calibration criterion.",
        ]:
            p.bullet(text)
        p.footer()

    print(OUTPUT_PDF)
    print(BOOTSTRAP_FIGURE)


if __name__ == "__main__":
    build_report()
