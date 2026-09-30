"""Build the self-contained thick-front clock and double-slit browser demo.

Quantum propagation is precomputed as complete instrument branches. Browser
controls mix these branches by the causal clock law and age the resulting
detector records; no amplitude is rescaled to imitate history selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT / "demo" / "trf_clock_lab.html"


def _load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def _payload() -> dict[str, Any]:
    profile = _load("trf_alpha_convention_profile.json")
    scale = _load("trf_physical_scale_prediction.json")
    arrival = _load("spatial_arrival_time_study.json")
    dynamics = _load("history_dynamics_study.json")

    baseline = profile["baseline"]
    baseline_index = next(
        i
        for i, row in enumerate(profile["rows"])
        if all(
            row[key] == baseline[key]
            for key in (
                "information_deficit",
                "required_copies",
                "coherence_tolerance",
                "hold_time",
            )
        )
    )

    example = arrival["example_first_setting"]
    time_bins = int(arrival["arrival_time_response"]["time_bins"])
    flat_joint = example["joint_probabilities"][:-1]
    y_bins = len(flat_joint) // time_bins
    click_probability = float(example["click_probability"])
    conditional_joint = [float(value) / click_probability for value in flat_joint]

    conventions = []
    for index, row in enumerate(profile["rows"]):
        conventions.append(
            {
                "index": index,
                "d": row["information_deficit"],
                "n": row["required_copies"],
                "eps": row["coherence_tolerance"],
                "hold": row["hold_time"],
                "tau": row["latencies"],
                "locked": row["locked_alpha_widths"],
                "profiledAlpha": row["calibration_profiled_alpha"],
                "profiled": row["calibration_profiled_widths"],
            }
        )

    return {
        "alpha": scale["locked_dimensionless_values"][
            "alpha_required_to_match_spatial_sigma"
        ],
        "gValues": profile["g_values"],
        "baselineIndex": baseline_index,
        "conventions": conventions,
        "scenarios": scale["scenarios"],
        "arrival": arrival["arrival_time_response"],
        "historyDynamics": dynamics["demo"],
        "heatmap": {
            "timeBins": time_bins,
            "yBins": y_bins,
            "conditionalJoint": conditional_joint,
            "clickProbability": click_probability,
        },
        "profileSummary": {
            "resolved": profile["resolved_conventions"],
            "strict": profile["strict_locked_alpha_width_summary"],
            "profiled": profile["calibration_profiled_width_summary"],
            "heldoutSpan": profile["heldout_relative_full_spans"],
        },
    }


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>Thick Front Clock Lab — interactive double-slit demo</title>
<style>
:root{--bg:#071018;--panel:#0d1924;--panel2:#111f2c;--line:#233746;--ink:#ecf6fb;--muted:#8fa7b6;--cyan:#52e6d8;--blue:#63a6ff;--amber:#ffcb6b;--pink:#ff6f9f;--front:32px;--radius:18px}
*{box-sizing:border-box} html{scroll-behavior:smooth} body{margin:0;background:radial-gradient(circle at 12% 0,#12283b 0,transparent 31rem),linear-gradient(145deg,#071018,#07131d 55%,#091722);color:var(--ink);font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;min-height:100vh}
body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.18;background-image:linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px);background-size:32px 32px;mask-image:linear-gradient(to bottom,black,transparent 80%)}
.wrap{width:min(1420px,calc(100% - 32px));margin:auto;padding:38px 0 70px}.topline{display:flex;justify-content:space-between;gap:16px;align-items:center;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;font-size:11px}.top-actions{display:flex;align-items:center;gap:10px}.tag{border:1px solid #2c6b68;background:#0a2b2b;color:#86fff4;padding:7px 11px;border-radius:99px;letter-spacing:.06em}.lang-switch{display:flex;border:1px solid #304759;border-radius:99px;padding:3px;background:#08141e}.lang-switch button{border:0;background:transparent;color:#7892a2;border-radius:99px;padding:5px 9px;font:600 10px/1 system-ui;cursor:pointer}.lang-switch button.active{background:#244153;color:#fff}
h1{font-size:clamp(42px,7vw,92px);line-height:.92;letter-spacing:-.06em;margin:48px 0 20px;max-width:1050px;font-weight:720}.accent{color:var(--cyan)}.lede{font-size:clamp(17px,2vw,23px);color:#b9ccd7;max-width:830px;margin:0 0 38px}.formula{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--amber)}
.grid{display:grid;grid-template-columns:minmax(270px,340px) 1fr;gap:18px}.panel{background:linear-gradient(150deg,rgba(17,31,44,.96),rgba(10,23,34,.94));border:1px solid var(--line);border-radius:var(--radius);box-shadow:0 20px 70px rgba(0,0,0,.24);overflow:hidden}.pad{padding:22px}.panel h2{font-size:13px;text-transform:uppercase;letter-spacing:.12em;color:#9cb3c1;margin:0 0 18px}.controls{grid-row:span 2}.control{padding:0 0 20px;margin:0 0 20px;border-bottom:1px solid var(--line)}.control:last-child{border:0;margin:0;padding:0}label,.label{display:flex;justify-content:space-between;gap:10px;margin-bottom:9px;color:#c7d8e1;font-size:13px}.value{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--cyan)}
select,input[type=range]{width:100%}select{background:#09151f;border:1px solid #304759;border-radius:9px;color:var(--ink);padding:10px 11px}input[type=range]{accent-color:var(--cyan)}.seg{display:grid;grid-template-columns:repeat(5,1fr);gap:5px}.seg button,.mode button{border:1px solid #304759;background:#09151f;color:#b9ccd7;padding:9px 5px;border-radius:9px;cursor:pointer}.seg button.active,.mode button.active{border-color:var(--cyan);background:#123b3d;color:#eafffd}.mode{display:grid;grid-template-columns:1fr 1fr;gap:6px}.mode button{text-align:left;min-height:58px}.mode small{display:block;color:#7f99a9;margin-top:3px}.control-note{color:var(--muted);font-size:11px;margin:7px 0 0}
.scene{position:relative;min-height:410px}.scene svg{width:100%;height:auto;display:block}.scene-controls{display:grid;grid-template-columns:auto auto minmax(140px,1fr);align-items:center;gap:9px;padding:11px 22px 34px;border-top:1px solid var(--line);background:#091721}.scene-controls button{border:1px solid #315164;background:#102a39;color:#d9ebf4;border-radius:8px;padding:8px 11px;cursor:pointer}.scene-controls button:hover{border-color:var(--cyan)}.scene-controls input{margin:0}.scene-note{position:absolute;bottom:9px;left:22px;color:var(--muted);font-size:11px}.event-status{position:absolute;right:22px;bottom:9px;color:var(--cyan);font:11px ui-monospace,SFMono-Regular,Menlo,monospace}.beam{stroke-dasharray:6 10;animation:flow 3s linear infinite}.front{filter:drop-shadow(0 0 12px var(--cyan))}.interactive-wave{filter:drop-shadow(0 0 7px var(--cyan));pointer-events:none}.scene.playing .beam{animation-play-state:running}.scene:not(.playing) .beam{animation-play-state:paused}@keyframes flow{to{stroke-dashoffset:-64}}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px}.metric{background:#09151f;border:1px solid #203543;border-radius:12px;padding:13px}.metric strong{display:block;font-size:clamp(18px,2vw,25px);font-weight:600;letter-spacing:-.03em;color:#f3fbff;overflow-wrap:anywhere}.metric span{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em;margin-top:3px}
.history-lab{margin-top:18px}.history-grid{display:grid;grid-template-columns:minmax(250px,330px) 1fr;gap:18px}.history-controls{display:grid;gap:14px}.history-control{background:#09151f;border:1px solid #203543;border-radius:11px;padding:11px}.history-control label{margin:0 0 5px}.history-readout{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:12px}.history-readout .metric{padding:10px}.history-readout .metric strong{font-size:20px}.equations{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}.equation{border:1px solid #294555;background:#0a1721;border-radius:99px;padding:6px 10px;color:#b8ccd8;font:11px ui-monospace,SFMono-Regular,Menlo,monospace}.history-state{padding:9px 12px;border-left:3px solid var(--cyan);background:#0a1721;color:#bad0dc;font-size:12px;margin-top:10px}.history-canvas{height:310px}
.memory-controls{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0}.memory-controls .history-control{min-width:0}.memory-controls select{font-size:12px}
.two{display:grid;grid-template-columns:1.1fr .9fr;gap:18px;margin-top:18px}.chart{width:100%;height:265px;display:block}.chart-head{display:flex;justify-content:space-between;align-items:baseline;gap:10px}.chart-head p{color:var(--muted);font-size:12px;margin:0}.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12px}.dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:5px}.heatmap{display:grid;gap:1px;background:#142736;border:1px solid #28404e;border-radius:8px;overflow:hidden;aspect-ratio:1.52;margin-top:16px}.cell{min-width:0;min-height:0}.axis-note{display:flex;justify-content:space-between;color:#6f8999;font-size:10px;margin-top:5px}
.convention{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}.chip{background:#09151f;border:1px solid #203543;border-radius:9px;padding:8px}.chip b{display:block;color:var(--amber);font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.chip small{color:var(--muted)}
.evidence{margin-top:18px;display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:var(--radius);overflow:hidden}.evidence article{background:#0c1924;padding:23px}.evidence .num{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--cyan)}.evidence h3{font-size:18px;margin:9px 0}.evidence p{color:#a9beca;margin:0;font-size:13px}.footer{display:flex;justify-content:space-between;gap:24px;color:#6f8999;font-size:12px;margin-top:22px}.warn{color:var(--amber)}
@media(max-width:950px){.grid,.two,.history-grid{grid-template-columns:1fr}.controls{grid-row:auto}.metrics{grid-template-columns:repeat(2,1fr)}.evidence{grid-template-columns:1fr}.scene{min-height:0}}@media(max-width:520px){.wrap{width:min(100% - 18px,1420px)}.metrics,.convention,.history-readout{grid-template-columns:1fr 1fr}.topline{align-items:flex-start;flex-direction:column}.top-actions{width:100%;justify-content:space-between}h1{margin-top:32px}.mode{grid-template-columns:1fr}.scene-controls{grid-template-columns:1fr 1fr}.scene-controls input{grid-column:1/-1}.event-status{display:none}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;scroll-behavior:auto!important}}
@media(max-width:950px){.memory-controls{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:520px){.memory-controls{grid-template-columns:1fr}}
</style>
</head>
<body>
<main class="wrap">
  <div class="topline"><span data-i18n="project">QuantumToy / interactive clock model</span><div class="top-actions"><span class="tag" data-i18n="synthetic">Synthetic conditional prediction</span><div class="lang-switch" aria-label="Language"><button class="active" data-lang="en">EN</button><button data-lang="fi">FI</button></div></div></div>
  <h1 data-i18n-html="headline">How thick is a <span class="accent">measurement moment?</span></h1>
  <p class="lede" data-i18n-html="lede">This interactive double-slit laboratory turns a record-stabilization clock into an observable arrival-time distribution. The value <span class="formula">α = 0.2 / 2.76</span> is a locked reference calibration.</p>

  <section class="grid">
    <aside class="panel controls"><div class="pad">
      <h2 data-i18n="setup">Experiment</h2>
      <div class="control">
        <div class="label"><span data-i18n-html="coupling">Coupling <i>g</i></span><span class="value" id="gValue">1.0</span></div>
        <div class="seg" id="gButtons"><button data-g="0.05">.05</button><button data-g="0.1">.1</button><button data-g="0.5">.5</button><button class="active" data-g="1">1</button><button data-g="2">2</button></div>
        <p class="control-note" data-i18n-html="lowGNote">Values below 0.5 use the model's exact <i>gt</i> time-rescaling.</p>
      </div>
      <div class="control">
        <div class="label"><span data-i18n="convention">Clock convention</span><span class="value" id="conventionValue">—</span></div>
        <input id="convention" type="range" min="0" max="80" step="1">
        <div class="convention" id="conventionChips"></div>
      </div>
      <div class="control">
        <div class="label"><span data-i18n="alphaHandling">How α is treated</span></div>
        <div class="mode" id="modeButtons">
          <button class="active" data-mode="locked" data-i18n-html="lockedAlpha">Locked α<small>same number for every clock</small></button>
          <button data-mode="profiled" data-i18n-html="profiledAlpha">Calibrated α<small>σ(1)=0.2 for each clock</small></button>
        </div>
      </div>
      <div class="control">
        <label for="scenario"><span data-i18n="physicalScale">Physical scale</span></label>
        <select id="scenario"></select>
      </div>
      <div class="control">
        <label for="lambda"><span data-i18n="responseStrength">Response strength λ</span><span class="value" id="lambdaValue">1.00</span></label>
        <input id="lambda" type="range" min="0" max="2" value="1" step="0.01">
      </div>
      <div class="control">
        <label for="kappa"><span data-i18n-html="delayCoupling">Delay coupling κ<sub>t</sub></span><span class="value" id="kappaValue">1.00</span></label>
        <input id="kappa" type="range" min="0" max="1" value="1" step="0.01">
        <p style="color:var(--muted);font-size:11px;margin:7px 0 0" data-i18n-html="kappaNote">κ<sub>t</sub>=1 is an explicit closure assumption used by this demo.</p>
      </div>
    </div></aside>

    <section class="panel scene playing" id="scene" aria-label="Interactive double-slit experiment">
      <svg viewBox="0 0 920 430" role="img" aria-label="Source, double slit, and time-stamping detector">
        <defs>
          <linearGradient id="wash" x1="0" x2="1"><stop offset="0" stop-color="#52e6d8" stop-opacity=".03"/><stop offset=".7" stop-color="#63a6ff" stop-opacity=".16"/><stop offset="1" stop-color="#ff6f9f" stop-opacity=".04"/></linearGradient>
          <radialGradient id="source"><stop stop-color="#fff"/><stop offset=".16" stop-color="#52e6d8"/><stop offset="1" stop-color="#52e6d8" stop-opacity="0"/></radialGradient>
          <filter id="glow"><feGaussianBlur stdDeviation="6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
          <clipPath id="beforeBarrier"><rect x="0" y="0" width="359" height="430"/></clipPath>
          <clipPath id="afterBarrier"><rect x="371" y="0" width="428" height="430"/></clipPath>
        </defs>
        <rect width="920" height="430" fill="url(#wash)"/>
        <g stroke="#263d4c" stroke-width="1"><path d="M0 107.5H920M0 215H920M0 322.5H920"/><path d="M230 0V430M460 0V430M690 0V430"/></g>
        <circle cx="92" cy="215" r="45" fill="url(#source)" class="front"/><circle cx="92" cy="215" r="7" fill="#eafffd" filter="url(#glow)"/>
        <circle id="sourceWave" class="interactive-wave" cx="92" cy="215" r="20" fill="none" stroke="#52e6d8" stroke-width="3" clip-path="url(#beforeBarrier)"/>
        <g fill="none" stroke="#52e6d8" opacity=".6" class="beam"><path d="M105 215C190 160 255 145 355 145"/><path d="M105 215C190 270 255 285 355 285"/></g>
        <g stroke="#aac1ce" stroke-width="10"><path d="M365 22V122M365 168V262M365 308V408"/></g>
        <g fill="none" stroke="#63a6ff" class="beam" opacity=".48"><path d="M370 145Q565 70 790 60M370 145Q565 145 790 145M370 145Q565 220 790 230"/><path d="M370 285Q565 210 790 200M370 285Q565 285 790 285M370 285Q565 360 790 370"/></g>
        <g clip-path="url(#afterBarrier)" class="interactive-wave"><circle id="topWave" cx="365" cy="145" r="40" fill="none" stroke="#63a6ff" stroke-width="3"/><circle id="bottomWave" cx="365" cy="285" r="40" fill="none" stroke="#63a6ff" stroke-width="3"/></g>
        <path id="frontBand" d="M392 108Q535 215 392 322" fill="none" stroke="#52e6d8" stroke-width="24" opacity=".13" class="front"/>
        <rect x="800" y="32" width="42" height="366" rx="10" fill="#102737" stroke="#517085"/>
        <circle id="activeHit" cx="821" cy="215" r="7" fill="#ff6f9f" filter="url(#glow)" opacity="0"/>
        <g fill="#8fa7b6" font-size="12" letter-spacing="1.4"><text x="55" y="382" data-i18n="source">SOURCE</text><text x="324" y="420" data-i18n="doubleSlit">DOUBLE SLIT</text><text x="766" y="420" data-i18n="detector">p(y,t) DETECTOR</text></g>
        <g transform="translate(675 25)"><rect width="103" height="40" rx="8" fill="#07131d" stroke="#294151"/><text x="11" y="17" fill="#8fa7b6" font-size="10" data-i18n="timestamp">TIMESTAMP</text><text id="sceneStamp" x="11" y="32" fill="#52e6d8" font-size="12" font-family="monospace">σ = —</text></g>
      </svg>
      <div class="scene-controls"><button id="toggleMotion" data-i18n="pause">Pause</button><button id="newEvent" data-i18n="newEvent">New event</button><input id="phase" type="range" min="0" max="1" value="0.58" step="0.001" aria-label="Wavefront phase"></div>
      <div class="scene-note" data-i18n="sceneNote">Drag the timeline or create a new event. The wavefront is illustrative; the numerical values come from the analysis runs.</div><div class="event-status" id="eventStatus"></div>
    </section>

    <section class="panel"><div class="pad">
      <h2 data-i18n="prediction">Selected model prediction</h2>
      <div class="metrics">
        <div class="metric"><strong id="sigmaDim">—</strong><span>σ<sub>T</sub> / T₀</span></div>
        <div class="metric"><strong id="sigmaPhysical">—</strong><span data-i18n-html="physicalSigma">physical σ<sub>T</sub></span></div>
        <div class="metric"><strong id="meanDelay">—</strong><span data-i18n="meanDelay">mean delay</span></div>
        <div class="metric"><strong id="responseFraction">—</strong><span data-i18n-html="responseFractionLabel">response fraction 1−e<sup>−λ</sup></span></div>
      </div>
    </div></section>
  </section>

  <section class="panel history-lab"><div class="pad">
    <div class="chart-head"><div><h2 style="margin-bottom:5px" data-i18n="historyTitle">Future, front, and fading past</h2><p data-i18n="historySubtitle">Selection remains fixed even when the accessible record disappears</p></div><div class="legend"><span data-i18n-html="futureLegend"><i class="dot" style="background:var(--blue)"></i>future openness</span><span data-i18n-html="frontLegend"><i class="dot" style="background:var(--amber)"></i>present front</span><span data-i18n-html="selectionLegend"><i class="dot" style="background:var(--cyan)"></i>selection</span><span data-i18n-html="recordLegend"><i class="dot" style="background:var(--pink)"></i>accessible record</span></div></div>
    <div class="equations"><span class="equation">S(a)=Φ(a/σ<sub>T</sub>)</span><span class="equation">O(a)=1−S(a)</span><span class="equation">N(a)=4S(a)[1−S(a)]</span><span class="equation">M(a)=S(a)e<sup>−[((a−a<sub>keep</sub>)<sub>+</sub>/τ<sub>fade</sub>)<sup>β</sup>]</sup></span></div>
    <div class="history-grid" style="margin-top:16px">
      <div class="history-controls">
        <div class="history-control"><label for="historyAge"><span data-i18n-html="observerAge">Observer age a / σ<sub>T</sub></span><span class="value" id="historyAgeValue">0.00</span></label><input id="historyAge" type="range" min="-6" max="14" value="0" step="0.05"></div>
        <div class="history-control"><label for="historyKeep"><span data-i18n-html="retentionTime">Retention a<sub>keep</sub> / σ<sub>T</sub></span><span class="value" id="historyKeepValue">1.50</span></label><input id="historyKeep" type="range" min="0" max="8" value="1.5" step="0.1"></div>
        <div class="history-control"><label for="historyFade"><span data-i18n-html="fadeTime">Fade time τ<sub>fade</sub> / σ<sub>T</sub></span><span class="value" id="historyFadeValue">5.00</span></label><input id="historyFade" type="range" min="0.5" max="12" value="5" step="0.1"></div>
        <div class="history-control"><label for="historyBeta"><span data-i18n="fadeShape">Fade shape β</span><span class="value" id="historyBetaValue">1.50</span></label><input id="historyBeta" type="range" min="0.5" max="3" value="1.5" step="0.1"></div>
        <div class="history-state" id="historyState"></div>
      </div>
      <div><canvas id="historyChart" class="chart history-canvas"></canvas><div class="history-readout"><div class="metric"><strong id="historyOpen">—</strong><span data-i18n="openReadout">future open</span></div><div class="metric"><strong id="historyFront">—</strong><span data-i18n="frontReadout">front active</span></div><div class="metric"><strong id="historySelected">—</strong><span data-i18n="selectedReadout">history selected</span></div><div class="metric"><strong id="historyRecord">—</strong><span data-i18n="recordReadout">record accessible</span></div></div></div>
    </div>
  </div></section>

  <section class="panel history-lab" id="dynamicsPanel"><div class="pad">
    <h2 data-i18n="dynamicsTitle">Wavefunction and absorbing detector</h2>
    <p class="control-note" data-i18n="dynamicsIntro">A one-time upper/lower measurement changes the quantum state. The detector absorbs the evolving wave. Every record fades from its own creation time.</p>
    <div class="history-grid" style="margin-top:16px">
      <div class="history-controls">
        <div class="history-control"><label for="recordStart"><span data-i18n="recordStartLabel">Clock start / T₀</span><span class="value" id="recordStartValue">−2.00</span></label><input id="recordStart" type="range" min="-4" max="1" step=".05" value="-2"></div>
        <div class="history-control"><label for="readoutWait"><span data-i18n="readoutWaitLabel">Readout wait / σT</span><span class="value" id="readoutWaitValue">0</span></label><input id="readoutWait" type="range" min="0" max="20" step=".25" value="0"></div>
        <div class="history-state" id="dynamicsClock"></div>
        <p class="control-note" data-i18n="dynamicsScope">Acquisition: 0–2 T₀; absorption rate: 4/T₀. Clock center = start + σT/α. The prepared wave has no earlier selection. λ controls the probability of selection. The keep/fade/β controls above set record lifetimes; observer age only moves the envelope cursor. κt belongs to the separate delay hypothesis below.</p>
      </div>
      <div>
        <canvas id="dynamicsChart" class="chart" aria-label="Unconditional detector distributions"></canvas>
        <div class="legend"><span data-i18n="dynamicsNull">Blue: no path selection</span><span data-i18n="dynamicsPhysical">Amber: physical detection</span><span data-i18n="dynamicsSaved">Pink: readable record</span></div>
        <div class="history-readout"><div class="metric"><strong id="dynamicsAbsorbed">—</strong><span data-i18n="absorbedLabel">physically detected</span></div><div class="metric"><strong id="dynamicsRecorded">—</strong><span data-i18n="savedLabel">readable click</span></div><div class="metric"><strong id="dynamicsErased">—</strong><span data-i18n="erasedLabel">erased record</span></div><div class="metric"><strong id="dynamicsMissing">—</strong><span data-i18n="missingLabel">no record created</span></div></div>
      </div>
    </div>
    <div class="two">
      <div><h2 data-i18n="densityTitle">Surviving quantum probability density</h2><canvas id="dynamicsDensity" class="chart"></canvas><p class="control-note" id="dynamicsTrace"></p></div>
      <div><h2 data-i18n="dynamicJointTitle">Readable p(y,t), per preparation</h2><canvas id="dynamicsJoint" class="chart"></canvas><p class="control-note" data-i18n="jointAxes">Horizontal: detection time 0–2 T₀. Vertical: detector position y.</p></div>
    </div>
    <p class="control-note" data-i18n="dynamicsCaveat">New candidate dynamics. Forgetting a classical record cannot undo the measurement back-action. The older delay heatmap below is a fixed benchmark from a different detector model.</p>
  </div></section>

  <section class="panel history-lab" id="memoryPanel"><div class="pad">
    <h2 data-i18n="memoryTitle">Reference memories and repeated delayed reads</h2>
    <p class="control-note" data-i18n="memoryIntro"></p>
    <div class="memory-controls">
      <div class="history-control"><label for="referenceCopies"><span data-i18n="referenceCopiesLabel"></span><span class="value" id="referenceCopiesValue"></span></label><input id="referenceCopies" type="range" min="0" max="16" step="1" value="1"></div>
      <div class="history-control"><label for="delayedCopies"><span data-i18n="delayedCopiesLabel"></span><span class="value" id="delayedCopiesValue"></span></label><input id="delayedCopies" type="range" min="0" max="16" step="1" value="1"></div>
      <div class="history-control"><label for="memoryReads"><span data-i18n="memoryReadsLabel"></span><span class="value" id="memoryReadsValue"></span></label><input id="memoryReads" type="range" min="1" max="16" step="1" value="1"></div>
      <div class="history-control"><label for="memorySpacing"><span data-i18n="memorySpacingLabel"></span><span class="value" id="memorySpacingValue"></span></label><input id="memorySpacing" type="range" min="0" max="5" step=".25" value="1"></div>
      <div class="history-control"><label for="memoryReadEfficiency"><span data-i18n="memoryReadEfficiencyLabel"></span><span class="value" id="memoryReadEfficiencyValue"></span></label><input id="memoryReadEfficiency" type="range" min="0" max="1" step=".05" value=".9"></div>
      <div class="history-control"><label for="referenceSurvival"><span data-i18n="referenceSurvivalLabel"></span><span class="value" id="referenceSurvivalValue"></span></label><input id="referenceSurvival" type="range" min="0" max="1" step=".005" value=".995"></div>
      <div class="history-control"><label for="memoryLossMode" data-i18n="memoryLossLabel"></label><select id="memoryLossMode"><option value="independent" data-i18n="memoryIndependent"></option><option value="shared" data-i18n="memoryShared"></option></select></div>
      <div class="history-control"><label for="memoryPreparations"><span data-i18n="memoryPreparationsLabel"></span><span class="value" id="memoryPreparationsValue"></span></label><input id="memoryPreparations" type="range" min="100" max="10000" step="100" value="1000"></div>
    </div>
    <div class="history-state" id="memorySchedule"></div>
    <canvas id="memoryChart" class="chart" aria-label="Expected unique record counts by creation time"></canvas>
    <div class="legend">
      <span style="color:#8fa7b6" data-i18n="memoryCreatedLegend"></span>
      <span style="color:#63a6ff" data-i18n="memoryReferenceLegend"></span>
      <span style="color:#ff6f9f" data-i18n="memoryLastLegend"></span>
      <span style="color:#ffcb6b" data-i18n="memoryLogLegend"></span>
      <span style="color:#52e6d8" data-i18n="memoryUnionLegend"></span>
    </div>
    <div class="history-readout">
      <div class="metric"><strong id="memoryCreated"></strong><span data-i18n="memoryCreatedLabel"></span></div>
      <div class="metric"><strong id="memoryLast"></strong><span data-i18n="memoryLastLabel"></span></div>
      <div class="metric"><strong id="memoryLogged"></strong><span data-i18n="memoryLoggedLabel"></span></div>
      <div class="metric"><strong id="memoryRecovered"></strong><span data-i18n="memoryRecoveredLabel"></span></div>
    </div>
    <p class="control-note" id="memoryAccounting"></p>
    <p class="control-note" data-i18n="memoryScope"></p>
  </div></section>

  <section class="two">
    <article class="panel"><div class="pad">
      <div class="chart-head"><div><h2 style="margin-bottom:5px" data-i18n="latentDelay">Latent thick-front delay</h2><p data-i18n="delaySubtitle">Half-normal response branch plus a zero-delay mass</p></div><div class="legend"><span data-i18n-html="responseBranch"><i class="dot" style="background:var(--cyan)"></i>response branch</span><span data-i18n-html="meanLegend"><i class="dot" style="background:var(--pink)"></i>mean</span></div></div>
      <canvas id="delayChart" class="chart"></canvas>
    </div></article>
    <article class="panel"><div class="pad">
      <div class="chart-head"><div><h2 style="margin-bottom:5px" data-i18n="jointDistribution">Joint p(y,t | click)</h2><p data-i18n-html="fixedBaseline">Fixed baseline: σ=0.2, κ<sub>t</sub>=1</p></div><span class="tag" id="clickProbability">—</span></div>
      <div class="heatmap" id="heatmap" aria-label="Position and arrival-time joint distribution"></div>
      <div class="axis-note"><span data-i18n="earlyTime">early time ←</span><span data-i18n="detectorPosition">detector position y ↑</span><span data-i18n="lateTime">→ late time</span></div>
    </div></article>
  </section>

  <section class="panel" style="margin-top:18px"><div class="pad">
    <div class="chart-head"><div><h2 style="margin-bottom:5px" data-i18n="clockSensitivity">Clock-convention sensitivity</h2><p data-i18n="clockSubtitle">All 81 resolved conventions; dot = current selection</p></div><div class="legend"><span data-i18n-html="lockedLegend"><i class="dot" style="background:var(--amber)"></i>locked α</span><span data-i18n-html="calibratedLegend"><i class="dot" style="background:var(--cyan)"></i>calibrated α</span></div></div>
    <canvas id="profileChart" class="chart" style="height:220px"></canvas>
  </div></section>

  <section class="evidence">
    <article><span class="num" data-i18n="coreLabel">01 / MODEL CORE</span><h3 data-i18n="coreTitle">What the clock supplies</h3><p data-i18n-html="coreText">A stabilization latency τ<sub>stab</sub>, a half-normal response shape, and the relation σ<sub>T</sub>=ατ<sub>stab</sub>. The scale α must be calibrated.</p></article>
    <article><span class="num" data-i18n="closureLabel">02 / DEMO CLOSURE</span><h3 data-i18n="closureTitle">What this demo adds</h3><p data-i18n-html="closureText">Locked α=0.2/2.76, the Schrödinger time unit T₀=mL₀²/ℏ, event timestamps, and the κ<sub>t</sub>=1 closure.</p></article>
    <article><span class="num" data-i18n="experimentLabel">03 / DECISIVE EXPERIMENT</span><h3 data-i18n="missingTitle">What is still missing</h3><p data-i18n="missingText">An independent alpha prediction or a new calibration, together with a measured detector response. Until then this is a testable conditional prediction, not an empirical discovery.</p></article>
  </section>
  <div class="footer"><span data-i18n="footerData">Data: QuantumToy analysis runs · 81 clock conventions · joint p(y,t)</span><span class="warn" data-i18n="footerWarning">Research demo — not an empirical result</span></div>
</main>
<script>
const DATA=__DATA__;
const TEXT={
  en:{title:'Thick Front Clock Lab — interactive double-slit demo',project:'QuantumToy / interactive clock model',synthetic:'Synthetic conditional prediction',headline:'How thick is a <span class="accent">measurement moment?</span>',lede:'This interactive double-slit laboratory turns a record-stabilization clock into an observable arrival-time distribution. The value <span class="formula">α = 0.2 / 2.76</span> is a locked reference calibration.',setup:'Experiment',coupling:'Coupling <i>g</i>',convention:'Clock convention',alphaHandling:'How α is treated',lockedAlpha:'Locked α<small>same number for every clock</small>',profiledAlpha:'Calibrated α<small>σ(1)=0.2 for each clock</small>',physicalScale:'Physical scale',responseStrength:'Response strength λ',delayCoupling:'Delay coupling κ<sub>t</sub>',kappaNote:'κ<sub>t</sub>=1 is an explicit closure assumption used by this demo.',source:'SOURCE',doubleSlit:'DOUBLE SLIT',detector:'p(y,t) DETECTOR',timestamp:'TIMESTAMP',pause:'Pause',play:'Play',newEvent:'New event',sceneNote:'Drag the timeline or create a new event. The wavefront is illustrative; the numerical values come from the analysis runs.',prediction:'Selected model prediction',physicalSigma:'physical σ<sub>T</sub>',meanDelay:'mean delay',responseFractionLabel:'response fraction 1−e<sup>−λ</sup>',latentDelay:'Latent thick-front delay',delaySubtitle:'Half-normal response branch plus a zero-delay mass',responseBranch:'<i class="dot" style="background:var(--cyan)"></i>response branch',meanLegend:'<i class="dot" style="background:var(--pink)"></i>mean',jointDistribution:'Joint p(y,t | click)',fixedBaseline:'Fixed baseline: σ=0.2, κ<sub>t</sub>=1',earlyTime:'early time ←',detectorPosition:'detector position y ↑',lateTime:'→ late time',clockSensitivity:'Clock-convention sensitivity',clockSubtitle:'All 81 resolved conventions; dot = current selection',lockedLegend:'<i class="dot" style="background:var(--amber)"></i>locked α',calibratedLegend:'<i class="dot" style="background:var(--cyan)"></i>calibrated α',coreLabel:'01 / MODEL CORE',coreTitle:'What the clock supplies',coreText:'A stabilization latency τ<sub>stab</sub>, a half-normal response shape, and the relation σ<sub>T</sub>=ατ<sub>stab</sub>. The scale α must be calibrated.',closureLabel:'02 / DEMO CLOSURE',closureTitle:'What this demo adds',closureText:'Locked α=0.2/2.76, the Schrödinger time unit T₀=mL₀²/ℏ, event timestamps, and the κ<sub>t</sub>=1 closure.',experimentLabel:'03 / DECISIVE EXPERIMENT',missingTitle:'What is still missing',missingText:'An independent alpha prediction or a new calibration, together with a measured detector response. Until then this is a testable conditional prediction, not an empirical discovery.',footerData:'Data: QuantumToy analysis runs · 81 clock conventions · joint p(y,t)',footerWarning:'Research demo — not an empirical result',deficit:'deficit',copies:'copies',hold:'hold',delayAxis:'delay Δt',densityAxis:'density',conventionAxis:'clock convention',zeroBranch:'zero-delay branch',responseEvent:'response branch',electronSlits:'Electron · 1 µm slit separation',neutronSlits:'Neutron · 10 µm slit separation',c60Slits:'C₆₀ · 100 nm slit separation',electronVelocity:'Electron · 10⁶ m/s'},
  fi:{title:'Paksun rintaman kellolaboratorio — interaktiivinen kaksoisrakodemo',project:'QuantumToy / interaktiivinen kellomalli',synthetic:'Synteettinen ehdollinen ennuste',headline:'Kuinka paksu on <span class="accent">mittaushetki?</span>',lede:'Interaktiivinen kaksoisrakolaboratorio muuntaa tietueen stabiloitumiskellon havaittavaksi saapumisajan jakaumaksi. Arvo <span class="formula">α = 0.2 / 2.76</span> on lukittu referenssikalibrointi.',setup:'Koeasetelma',coupling:'Kytkentä <i>g</i>',convention:'Kellokonventio',alphaHandling:'α:n käsittely',lockedAlpha:'Lukittu α<small>sama numero kaikille kelloille</small>',profiledAlpha:'Kalibroitu α<small>σ(1)=0.2 jokaiselle kellolle</small>',physicalScale:'Fysikaalinen skaala',responseStrength:'Vasteen voimakkuus λ',delayCoupling:'Viivekytkentä κ<sub>t</sub>',kappaNote:'κ<sub>t</sub>=1 on tässä demossa käytetty eksplisiittinen sulkeumaoletus.',source:'LÄHDE',doubleSlit:'KAKSOISRAKO',detector:'p(y,t) DETEKTORI',timestamp:'AIKALEIMA',pause:'Pysäytä',play:'Toista',newEvent:'Uusi tapahtuma',sceneNote:'Vedä aikajanaa tai luo uusi tapahtuma. Aaltorintama on havainnollinen; numeeriset arvot tulevat analyysiajoista.',prediction:'Valitun mallin ennuste',physicalSigma:'fyysinen σ<sub>T</sub>',meanDelay:'keskimääräinen viive',responseFractionLabel:'vasteosuus 1−e<sup>−λ</sup>',latentDelay:'Latentti paksun rintaman viive',delaySubtitle:'Puolinormaali vastehaara ja nollaviiveen massa',responseBranch:'<i class="dot" style="background:var(--cyan)"></i>vastehaara',meanLegend:'<i class="dot" style="background:var(--pink)"></i>keskiarvo',jointDistribution:'Yhteinen p(y,t | click)',fixedBaseline:'Kiinteä vertailuasetus: σ=0.2, κ<sub>t</sub>=1',earlyTime:'varhainen aika ←',detectorPosition:'detektoripaikka y ↑',lateTime:'→ myöhäinen aika',clockSensitivity:'Kellokonvention herkkyys',clockSubtitle:'Kaikki 81 ratkaistua konventiota; piste = nykyinen valinta',lockedLegend:'<i class="dot" style="background:var(--amber)"></i>lukittu α',calibratedLegend:'<i class="dot" style="background:var(--cyan)"></i>kalibroitu α',coreLabel:'01 / MALLIN YDIN',coreTitle:'Mitä kello antaa',coreText:'Stabiloitumisviiveen τ<sub>stab</sub>, puolinormaalin vasteen muodon sekä suhteen σ<sub>T</sub>=ατ<sub>stab</sub>. Skaala α täytyy kalibroida.',closureLabel:'02 / DEMON SULKEUMA',closureTitle:'Mitä tämä demo lisää',closureText:'Lukittu α=0.2/2.76, Schrödingerin aikayksikkö T₀=mL₀²/ℏ, tapahtumakohtaiset aikaleimat ja κ<sub>t</sub>=1-sulkeuma.',experimentLabel:'03 / RATKAISEVA KOE',missingTitle:'Mitä vielä puuttuu',missingText:'Riippumaton alfaennuste tai uusi kalibraatio sekä mitattu detektorin vaste. Siihen asti tämä on testattava ehdollinen ennuste, ei empiirinen löytö.',footerData:'Data: QuantumToy-analyysiajot · 81 kellokonventiota · yhteinen p(y,t)',footerWarning:'Tutkimusdemo — ei empiirinen tulos',deficit:'vajaus',copies:'kopiot',hold:'pitoaika',delayAxis:'viive Δt',densityAxis:'tiheys',conventionAxis:'kellokonventio',zeroBranch:'nollaviivehaara',responseEvent:'vastehaara',electronSlits:'Elektroni · 1 µm rakoväli',neutronSlits:'Neutroni · 10 µm rakoväli',c60Slits:'C₆₀ · 100 nm rakoväli',electronVelocity:'Elektroni · 10⁶ m/s'}
};
TEXT.en.lowGNote='Values below 0.5 use the model\'s exact <i>gt</i> time-rescaling.';
TEXT.fi.lowGNote='Arvot alle 0.5 käyttävät mallin täsmällistä <i>gt</i>-aikavenytystä.';
Object.assign(TEXT.en,{historyTitle:'Future, front, and fading past',historySubtitle:'Selection remains fixed even when the accessible record disappears',futureLegend:'<i class="dot" style="background:var(--blue)"></i>future openness',frontLegend:'<i class="dot" style="background:var(--amber)"></i>present front',selectionLegend:'<i class="dot" style="background:var(--cyan)"></i>selection',recordLegend:'<i class="dot" style="background:var(--pink)"></i>accessible record',observerAge:'Observer age a / σ<sub>T</sub>',retentionTime:'Retention a<sub>keep</sub> / σ<sub>T</sub>',fadeTime:'Fade time τ<sub>fade</sub> / σ<sub>T</sub>',fadeShape:'Fade shape β',openReadout:'future open',frontReadout:'front active',selectedReadout:'history selected',recordReadout:'record accessible',ageAxis:'age a / σT',weightAxis:'weight',stateFuture:'Alternatives are still open on the future side.',stateFront:'The observer is inside the selection front.',stateYoungPast:'The outcome is selected and its record remains accessible.',stateOldPast:'The outcome stays selected while its accessible record fades.'});
Object.assign(TEXT.fi,{historyTitle:'Tulevaisuus, rintama ja häviävä menneisyys',historySubtitle:'Valinta pysyy lukittuna, vaikka saavutettava tietue häviää',futureLegend:'<i class="dot" style="background:var(--blue)"></i>tulevaisuuden avoimuus',frontLegend:'<i class="dot" style="background:var(--amber)"></i>nykyhetken rintama',selectionLegend:'<i class="dot" style="background:var(--cyan)"></i>valinta',recordLegend:'<i class="dot" style="background:var(--pink)"></i>saavutettava tietue',observerAge:'Havaintoikä a / σ<sub>T</sub>',retentionTime:'Säilytysaika a<sub>keep</sub> / σ<sub>T</sub>',fadeTime:'Häviämisaika τ<sub>fade</sub> / σ<sub>T</sub>',fadeShape:'Häviämisen muoto β',openReadout:'tulevaisuus avoin',frontReadout:'rintama aktiivinen',selectedReadout:'historia valittu',recordReadout:'tietue saavutettava',ageAxis:'ikä a / σT',weightAxis:'paino',stateFuture:'Vaihtoehdot ovat vielä avoimia tulevaisuuden puolella.',stateFront:'Havaitsija on valintarintaman sisällä.',stateYoungPast:'Tulos on valittu ja sen tietue on vielä saavutettavissa.',stateOldPast:'Tulos pysyy valittuna saavutettavan tietueen hävitessä.'});
const state={g:1,mode:'locked',convention:DATA.baselineIndex,scenario:0,lambda:1,kappa:1,lang:'en',phase:.58,playing:true,eventY:215,eventDelay:.14,eventResponds:true,lastFrame:0,historyAge:0,historyKeep:1.5,historyFade:5,historyBeta:1.5};
const $=id=>document.getElementById(id);
const scenarioKeys={electron_1um_slits:'electronSlits',neutron_10um_slits:'neutronSlits',c60_100nm_slits:'c60Slits',electron_1e6mps:'electronVelocity'};
const t=key=>TEXT[state.lang][key]||key;
$('convention').value=state.convention;
function fmt(x,d=3){return Number(x).toFixed(d)}
function timeFmt(s){const a=Math.abs(s);if(a>=1)return fmt(s,3)+' s';if(a>=1e-3)return fmt(s*1e3,3)+' ms';if(a>=1e-6)return fmt(s*1e6,3)+' µs';if(a>=1e-9)return fmt(s*1e9,3)+' ns';if(a>=1e-12)return fmt(s*1e12,3)+' ps';if(a>=1e-15)return fmt(s*1e15,3)+' fs';return s.toExponential(2)+' s'}
function erfApprox(x){const sign=x<0?-1:1,a=Math.abs(x),p=.3275911,t0=1/(1+p*a),y=1-(((((1.061405429*t0-1.453152027)*t0+1.421413741)*t0-.284496736)*t0+.254829592)*t0)*Math.exp(-a*a);return sign*y}
function historyWeights(x){const S=.5*(1+erfApprox(x/Math.sqrt(2))),O=1-S,N=4*S*O,fadeAge=Math.max(x-state.historyKeep,0),M=S*Math.exp(-Math.pow(fadeAge/state.historyFade,state.historyBeta));return {S,O,N,M}}
function widthAt(row,key,g){const exact=DATA.gValues.indexOf(g);if(exact>=0)return row[key][exact];const reference=DATA.gValues.indexOf(1);return row[key][reference]/g}
function current(){const r=DATA.conventions[state.convention],sig=widthAt(r,state.mode==='locked'?'locked':'profiled',state.g);return {r,sig,alpha:state.mode==='locked'?DATA.alpha:r.profiledAlpha,sc:DATA.scenarios[state.scenario]}}
function refreshScenarios(){const select=$('scenario');select.innerHTML='';DATA.scenarios.forEach((s,i)=>select.add(new Option(t(scenarioKeys[s.name])||s.name,i)));select.value=state.scenario}
function setLanguage(lang){state.lang=lang;document.documentElement.lang=lang;document.title=t('title');document.querySelectorAll('[data-i18n]').forEach(el=>el.textContent=t(el.dataset.i18n));document.querySelectorAll('[data-i18n-html]').forEach(el=>el.innerHTML=t(el.dataset.i18nHtml));document.querySelectorAll('[data-lang]').forEach(el=>el.classList.toggle('active',el.dataset.lang===lang));refreshScenarios();updateMotionLabel();update()}
function update(){
  const {r,sig,alpha,sc}=current(),g=state.g,T0=sc.time_unit_s,q=1-Math.exp(-state.lambda),eff=state.kappa*sig,mean=q*eff*Math.sqrt(2/Math.PI);
  $('gValue').textContent=String(g); $('conventionValue').textContent=`${state.convention+1} / ${DATA.conventions.length}`;
  $('lambdaValue').textContent=fmt(state.lambda,2); $('kappaValue').textContent=fmt(state.kappa,2);
  $('sigmaDim').textContent=fmt(sig,4); $('sigmaPhysical').textContent=timeFmt(sig*T0); $('meanDelay').textContent=timeFmt(mean*T0); $('responseFraction').textContent=fmt(100*q,1)+' %';
  $('sceneStamp').textContent='σ = '+timeFmt(sig*T0); $('frontBand').style.strokeWidth=(8+Math.min(58,sig*95))+'px';
  $('conventionChips').innerHTML=`<div class="chip"><b>${r.d}</b><small>${t('deficit')}</small></div><div class="chip"><b>${r.n}</b><small>${t('copies')}</small></div><div class="chip"><b>${r.eps}</b><small>ε</small></div><div class="chip"><b>${r.hold}</b><small>${t('hold')}</small></div>`;
  [...document.querySelectorAll('#gButtons button')].forEach(b=>b.classList.toggle('active',Number(b.dataset.g)===state.g));
  [...document.querySelectorAll('#modeButtons button')].forEach(b=>b.classList.toggle('active',b.dataset.mode===state.mode));
  drawDelay(eff,q,T0); drawProfile(); drawHistory(); updateWavefront();
}
function setupCanvas(canvas){const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);return {c,w:rect.width,h:rect.height}}
function drawAxes(c,w,h,xLabel,yLabel){c.strokeStyle='#29404f';c.lineWidth=1;c.beginPath();c.moveTo(48,15);c.lineTo(48,h-34);c.lineTo(w-16,h-34);c.stroke();c.fillStyle='#718b9b';c.font='11px system-ui';c.fillText(yLabel,10,16);c.textAlign='right';c.fillText(xLabel,w-16,h-10);c.textAlign='left'}
function drawHistory(){const {c,w,h}=setupCanvas($('historyChart')),left=48,right=w-16,top=20,bottom=h-34,xmin=-6,xmax=14,plotW=right-left,plotH=bottom-top,xp=x=>left+(x-xmin)*plotW/(xmax-xmin),yp=y=>bottom-y*plotH;c.clearRect(0,0,w,h);drawAxes(c,w,h,t('ageAxis'),t('weightAxis'));const curves=[['O','#63a6ff'],['N','#ffcb6b'],['S','#52e6d8'],['M','#ff6f9f']];curves.forEach(([key,color])=>{c.beginPath();for(let i=0;i<=320;i++){const x=xmin+(xmax-xmin)*i/320,y=historyWeights(x)[key];i?c.lineTo(xp(x),yp(y)):c.moveTo(xp(x),yp(y))}c.strokeStyle=color;c.lineWidth=key==='S'?1.5:2.2;c.setLineDash(key==='S'?[6,5]:[]);c.stroke();c.setLineDash([])});const marker=historyWeights(state.historyAge),mx=xp(state.historyAge);c.strokeStyle='#eef8fc';c.globalAlpha=.55;c.beginPath();c.moveTo(mx,top);c.lineTo(mx,bottom);c.stroke();c.globalAlpha=1;curves.forEach(([key,color])=>{c.fillStyle=color;c.beginPath();c.arc(mx,yp(marker[key]),4.5,0,Math.PI*2);c.fill()});for(let i=-5;i<=10;i+=5){c.fillStyle='#6f8999';c.font='11px system-ui';c.textAlign='center';c.fillText(String(i),xp(i),h-16)}c.textAlign='left';const {sig,sc}=current(),physicalAge=state.historyAge*sig*sc.time_unit_s;c.fillStyle='#dcecf4';c.fillText('a = '+timeFmt(physicalAge),Math.min(mx+7,right-100),top+12);$('historyAgeValue').textContent=fmt(state.historyAge,2);$('historyKeepValue').textContent=fmt(state.historyKeep,2);$('historyFadeValue').textContent=fmt(state.historyFade,2);$('historyBetaValue').textContent=fmt(state.historyBeta,2);$('historyOpen').textContent=fmt(100*marker.O,1)+' %';$('historyFront').textContent=fmt(100*marker.N,1)+' %';$('historySelected').textContent=fmt(100*marker.S,1)+' %';$('historyRecord').textContent=fmt(100*marker.M,1)+' %';const key=state.historyAge<-.8?'stateFuture':marker.N>.45?'stateFront':marker.M>.2?'stateYoungPast':'stateOldPast';$('historyState').textContent=t(key);drawDynamics()}
function drawDelay(scale,q,T0){const {c,w,h}=setupCanvas($('delayChart'));c.clearRect(0,0,w,h);drawAxes(c,w,h,t('delayAxis'),t('densityAxis'));const left=48,right=w-16,top=24,bottom=h-34,plotW=right-left,plotH=bottom-top,maxX=Math.max(.25,scale*4.1),safe=Math.max(scale,.002);let vals=[];for(let i=0;i<=180;i++){const x=maxX*i/180;vals.push(q*Math.sqrt(2/Math.PI)/safe*Math.exp(-x*x/(2*safe*safe)))}const maxY=Math.max(...vals,1);c.beginPath();vals.forEach((v,i)=>{const x=left+plotW*i/180,y=bottom-plotH*v/maxY;i?c.lineTo(x,y):c.moveTo(x,y)});c.lineTo(right,bottom);c.lineTo(left,bottom);c.closePath();const grad=c.createLinearGradient(0,top,0,bottom);grad.addColorStop(0,'rgba(82,230,216,.48)');grad.addColorStop(1,'rgba(82,230,216,.02)');c.fillStyle=grad;c.fill();c.beginPath();vals.forEach((v,i)=>{const x=left+plotW*i/180,y=bottom-plotH*v/maxY;i?c.lineTo(x,y):c.moveTo(x,y)});c.strokeStyle='#52e6d8';c.lineWidth=2;c.stroke();const mean=q*scale*Math.sqrt(2/Math.PI),mx=left+plotW*mean/maxX;c.strokeStyle='#ff6f9f';c.setLineDash([4,5]);c.beginPath();c.moveTo(mx,top);c.lineTo(mx,bottom);c.stroke();c.setLineDash([]);c.fillStyle='#ff6f9f';c.fillText('E[Δt] '+timeFmt(mean*T0),Math.min(mx+6,right-105),top+10);c.fillStyle='#ffcb6b';c.beginPath();c.arc(left,bottom,4+9*(1-q),0,Math.PI*2);c.fill();c.fillStyle='#8fa7b6';c.fillText('P(Δt=0) '+fmt(100*(1-q),1)+' %',left+10,bottom-9);for(let i=0;i<=4;i++){const x=maxX*i/4;c.fillStyle='#6f8999';c.textAlign=i===0?'left':i===4?'right':'center';c.fillText(timeFmt(x*T0),left+plotW*i/4,h-16)}c.textAlign='left'}
function drawProfile(){const {c,w,h}=setupCanvas($('profileChart')),rows=DATA.conventions,left=48,right=w-18,top=20,bottom=h-34,vals=rows.flatMap(r=>[widthAt(r,'locked',state.g),widthAt(r,'profiled',state.g)]),lo=Math.min(...vals)*.92,hi=Math.max(...vals)*1.05;c.clearRect(0,0,w,h);drawAxes(c,w,h,t('conventionAxis'),'σT/T₀');const x=i=>left+(right-left)*i/(rows.length-1),y=v=>bottom-(bottom-top)*(v-lo)/(hi-lo);[["locked",'#ffcb6b'],["profiled",'#52e6d8']].forEach(([key,color])=>{c.beginPath();rows.forEach((r,i)=>{const value=widthAt(r,key,state.g);i?c.lineTo(x(i),y(value)):c.moveTo(x(i),y(value))});c.strokeStyle=color;c.lineWidth=1.6;c.globalAlpha=.82;c.stroke();c.globalAlpha=1});const r=rows[state.convention];['locked','profiled'].forEach((key,j)=>{c.fillStyle=j?'#52e6d8':'#ffcb6b';c.beginPath();c.arc(x(state.convention),y(widthAt(r,key,state.g)),5,0,Math.PI*2);c.fill()});c.fillStyle='#718b9b';c.font='11px system-ui';c.fillText(lo.toFixed(3),9,bottom);c.fillText(hi.toFixed(3),9,top+4);c.textAlign='right';c.fillText(String(rows.length),right,h-14);c.textAlign='left';c.fillText('1',left,h-14)}
function buildHeatmap(){const h=DATA.heatmap,el=$('heatmap'),max=Math.max(...h.conditionalJoint);el.style.gridTemplateColumns=`repeat(${h.timeBins},1fr)`;el.style.gridTemplateRows=`repeat(${h.yBins},1fr)`;for(let yi=h.yBins-1;yi>=0;yi--)for(let ti=0;ti<h.timeBins;ti++){const v=h.conditionalJoint[ti*h.yBins+yi]/max,cell=document.createElement('span');cell.className='cell';const hue=195+125*v,light=8+62*Math.pow(v,.55);cell.style.background=`hsl(${hue} 88% ${light}%)`;cell.title=`p(y,t|click)=${h.conditionalJoint[ti*h.yBins+yi].toExponential(2)}`;el.appendChild(cell)}$('clickProbability').textContent='P(click) '+fmt(100*h.clickProbability,2)+' %'}
function sampleDetectorY(){const h=DATA.heatmap,m=Array(h.yBins).fill(0);for(let ti=0;ti<h.timeBins;ti++)for(let yi=0;yi<h.yBins;yi++)m[yi]+=h.conditionalJoint[ti*h.yBins+yi];let u=Math.random()*m.reduce((a,b)=>a+b,0),yi=0;for(;yi<m.length-1;yi++){u-=m[yi];if(u<=0)break}return 48+(h.yBins-1-yi)*334/(h.yBins-1)}
function gaussianAbs(){let u=0,v=0;while(u===0)u=Math.random();while(v===0)v=Math.random();return Math.abs(Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v))}
function sampleEvent(){sampleDynamicalPreparation()}
function updateMotionLabel(){$('toggleMotion').textContent=t(state.playing?'pause':'play');$('scene').classList.toggle('playing',state.playing)}
function updateWavefront(){const p=state.phase,before=Math.min(p/.38,1),after=Math.max(0,Math.min((p-.28)/.58,1)),fade=Math.max(0,1-Math.max(0,p-.78)/.22);$('phase').value=p;$('sourceWave').setAttribute('r',18+before*285);$('sourceWave').style.opacity=p<.46?String(.9*fade):'0';['topWave','bottomWave'].forEach(id=>{const el=$(id);el.setAttribute('r',18+after*470);el.style.opacity=after>0?String(.75*fade):'0'});$('frontBand').setAttribute('transform',`translate(${after*300} 0)`);$('frontBand').style.opacity=after>0&&after<.94?String(.18*fade):'0';const hit=Math.max(0,Math.min((p-.82)/.08,1));$('activeHit').setAttribute('cy',state.eventY);$('activeHit').setAttribute('r',6+6*Math.sin(hit*Math.PI));$('activeHit').style.opacity=state.eventResponds?String(hit):'0';const T0=current().sc.time_unit_s;$('eventStatus').textContent=(state.eventLabel?t(state.eventLabel):t('sceneNote'))+(state.eventLabel==='preparationSaved'?' · t '+timeFmt(state.eventDelay*T0):'')}
function animate(now){if(state.playing){if(state.lastFrame){const old=state.phase;state.phase=(state.phase+(now-state.lastFrame)/6200)%1;if(state.phase<old){sampleEvent()}}state.lastFrame=now;updateWavefront()}else state.lastFrame=0;requestAnimationFrame(animate)}
__DYNAMICS_JS__
__MEMORY_JS__
[['historyAge','historyAge'],['historyKeep','historyKeep'],['historyFade','historyFade'],['historyBeta','historyBeta']].forEach(([id,key])=>$(id).addEventListener('input',e=>{state[key]=Number(e.target.value);drawHistory()}));
$('gButtons').addEventListener('click',e=>{if(e.target.dataset.g!==undefined){state.g=+e.target.dataset.g;update()}});$('modeButtons').addEventListener('click',e=>{const b=e.target.closest('button');if(b){state.mode=b.dataset.mode;update()}});$('convention').addEventListener('input',e=>{state.convention=+e.target.value;update()});$('scenario').addEventListener('change',e=>{state.scenario=+e.target.value;update()});$('lambda').addEventListener('input',e=>{state.lambda=+e.target.value;update()});$('kappa').addEventListener('input',e=>{state.kappa=+e.target.value;update()});$('phase').addEventListener('input',e=>{state.phase=+e.target.value;state.playing=false;updateMotionLabel();updateWavefront()});$('toggleMotion').addEventListener('click',()=>{state.playing=!state.playing;state.lastFrame=0;updateMotionLabel()});$('newEvent').addEventListener('click',sampleEvent);document.querySelectorAll('[data-lang]').forEach(el=>el.addEventListener('click',()=>setLanguage(el.dataset.lang)));$('scene').querySelector('svg').addEventListener('pointerdown',e=>{const box=e.currentTarget.getBoundingClientRect(),x=(e.clientX-box.left)*920/box.width,y=(e.clientY-box.top)*430/box.height;if(x>770){state.eventY=Math.max(45,Math.min(385,y));state.eventLabel='manualMarker';state.eventResponds=true;state.phase=.86;state.playing=false;updateMotionLabel();updateWavefront()}});window.addEventListener('resize',update);buildHeatmap();const query=new URLSearchParams(location.search),queryG=Number(query.get('g'));if([.05,.1,.5,1,2].includes(queryG))state.g=queryG;setLanguage(query.get('lang')==='fi'?'fi':'en');requestAnimationFrame(animate);
</script>
</body>
</html>'''


def build_html() -> str:
    """Return the complete demo document with compact embedded data."""

    data = json.dumps(_payload(), ensure_ascii=False, separators=(",", ":"))
    dynamics_js = Path(__file__).with_name("history_dynamics_demo.js").read_text(encoding="utf-8")
    memory_js = Path(__file__).with_name("memory_banks_demo.js").read_text(encoding="utf-8")
    return HTML.replace("__DATA__", data).replace("__DYNAMICS_JS__", dynamics_js).replace("__MEMORY_JS__", memory_js)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(), encoding="utf-8")
    print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
