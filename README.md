# QuantumToy

QuantumToy is a modular Python simulation environment for exploring quantum dynamics and alternative quantum interpretations.

The project implements several wave equation models and allows experimentation with different theoretical extensions such as:

* Schrodinger dynamics  
* Continuous measurement models  
* Dirac equation  
* Thick reality front models  
* Hybrid Dirac plus Thick Front theories  

The simulator includes tools for:

* wavefunction propagation  
* double slit experiments  
* Bohmian trajectory overlays  
* ridge tracking of probability flow  
* retrodictive weighting (Emix)  
* animated visualization  

The goal is to provide a playground for quantum dynamics experiments and theoretical exploration.

---

## Features

The simulator supports:

### Wave dynamics

* Schrodinger equation using FFT split operator  
* Dirac equation using a two spinor relativistic model  

### Experimental models

* double slit barriers  
* absorbing boundaries  
* detection screens  

### Analysis tools

* ridge tracking of probability flow  
* velocity field visualization  
* Bohmian trajectory integration  
* divergence diagnostics  
* retrodictive weighting (Emix)  

### Visualization

* animated density maps  
* ridge trajectory overlay  
* Bohmian paths  
* flow arrows  
* mp4 video export  

---

### Legacy experiments

The repository also contains an experimental/ directory.

This folder includes earlier standalone research scripts that were used during the development of the project.
Some of these scripts reproduce the originally demonstrated animations and visual experiments.

These legacy experiments often use a slightly different workflow compared to the newer modular theory classes:

* forward wave evolution is computed first

* a retrodictive weighting field (Emix) is constructed from detector-conditioned backward propagation

* the visible ridge structure is then derived from the overlap of forward density and the backward effect field

The newer theory modules in src/quantumtoy/theories attempt to internalize similar ideas directly into the dynamical models (for example Thick Front and worldline-style extensions).

The experimental scripts are kept in the repository because they reproduce the original research visualizations and may still be useful for comparison or further exploration.

---

## Example simulation

Typical experiment simulated by the code:
wave packet -> double slit -> interference -> detection screen

The simulation computes:

* forward wave evolution  
* backward click conditioned propagation  
* ridge trajectory through the probability landscape  
* optional Bohmian trajectories  

---

## Installation

Clone the repository:
git clone https://github.com/YOURNAME/quantumtoy.git
cd quantumtoy

Create a virtual environment:
python -m venv venv
source venv/bin/activate

On Windows:
venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

---

## Requirements

Main Python dependencies:
numpy
matplotlib
scipy

Video export requires ffmpeg.

Linux:
sudo apt install ffmpeg

Mac:
brew install ffmpeg

---

## Tests

Run the reference and simulation regressions from the repository root, using
the Python environment in which the project dependencies are installed:

```bash
PYTHONPATH=src/quantumtoy MPLBACKEND=Agg python -m unittest discover -s tests -v
```

The TRF-IT v0.2 tests cover operational quantum probabilities, effect mixtures,
and the sampled stabilization clock. See [tests/README.md](tests/README.md)
for their scope and interpretation.

Run the declared two-path weak-probe and physical-record experiment with:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/run_record_environment.py \
  --g 0 0.5 1 2 --duration 8 --dt 0.02
```

Add `--json-output record_environment.json` to save the complete joint record
tables and time series. The model and fixed record conventions are described
in [paper/record_environment_model.md](paper/record_environment_model.md).

Run the shared-threshold sensitivity, time-sampling convergence, and minimal
TRF-candidate identifiability study with:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/run_trf_calibration_study.py \
  --json-output trf_calibration_study.json
```

The candidate law and parameter policy are declared in
[paper/trf_minimal_candidate.md](paper/trf_minimal_candidate.md).

Optimize the measurement allocation and run full multinomial recovery tests
for both the quantum null and a declared TRF signal with:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/run_trf_inference_study.py \
  --repetitions 30 --json-output trf_inference_study.json
```

Measure the temporal response width on a separate quantum-eraser ensemble and
test `alpha=sigma_T/tau_stab` at held-out couplings with:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/run_trf_response_study.py \
  --json-output trf_response_study.json
```

The measurement and held-out prediction protocol is described in
[paper/trf_response_measurement.md](paper/trf_response_measurement.md).

`TRF_SIGMA_T` is the single physical-time width used by both the post hoc TRF
products and `THEORY_NAME=thick_front_measurement_guided`. The latter samples
its backward half-Gaussian at
`TRF_MEASUREMENT_BACK_STRIDE * dt` and truncates it at
`TRF_MEASUREMENT_BACK_HORIZON_SIGMAS * TRF_SIGMA_T`. The old registry name
`thick_front_measured_guided` remains available as a compatibility alias.

Profile the same width from the full pre-click detector distribution of a
compact spatial double-slit run with:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/run_measurement_guided_response_study.py \
  --json-output spatial_response_study.json
```

This also checks the exact zero-strength worldline null, raw and
baseline-corrected grid recovery, and per-run, fixed, and absent normalization
rules. See
[paper/spatial_response_study.md](paper/spatial_response_study.md).

Run the separate operator-level spatial reference with:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/run_spatial_effect_measurement.py \
  --json-output spatial_effect_measurement.json
```

This compact benchmark represents every y-bin and no-click outcome as a full
POVM effect, mixes the backward effects with declared half-Gaussian weights,
and checks positivity, completeness, phase invariance, the exact `lambda=0`
null, and numerical convergence. Its assumptions and comparison with the
exploratory measurement-guided response are in
[paper/spatial_effect_instrument.md](paper/spatial_effect_instrument.md).

Create an MP4 and a final-frame overview of the same complete measurement:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/visualize_spatial_effect_measurement.py \
  --output-mp4 spatial_effect_measurement.mp4 \
  --snapshot-output spatial_effect_measurement.png
```

---

## Running a simulation

Run the main simulation:
python3 main.py

The simulation will generate a video file such as:
schrodinger_modular_api.mp4
and open a visualization window.

---

## Configuration

Simulation parameters are controlled through:
config.py

Example configuration:
THEORY_NAME = "schrodinger_measurement"

Dataclass settings can also be supplied as environment variables. For example,
this adds a finite post-slit obstacle to the upper channel while running the
entanglement theory:

```bash
cd src/quantumtoy && THEORY_NAME=thick_front_entanglement USE_SIMPLE_BARRIER=true SIMPLE_BARRIER_CENTER_X=2 SIMPLE_BARRIER_CENTER_Y=2 SIMPLE_BARRIER_HALF_HEIGHT=1.2 python3 main.py
```

Set `SIMPLE_BARRIER_ABSORPTION` above zero for an absorbing obstacle. TRF runs
save the full forward density separately from `rho_realized`; visualize the
latter with `visualize.py output.npz --render-mode realized_trf`.

For a direct comparison with detector-anchored ridge tracking:

```bash
python3 visualize.py output.npz --split-view --left-mode density --right-mode realized_trf
```

Use `--snapshot-frames 0 180 260 333 --snapshot-dir trf_snapshots` to save
reproducible frame comparisons without joining unrelated ridge segments.

Entanglement runs use a configurable low-rank two-arm approximation
(`TRF_TWO_ARM_APPROX=true`). One joint spin outcome is sampled first; A and B
positions then come from the corresponding conditional sign marginals. New run
bundles store separate A/B backward effects. Display them in green/cyan with:

```bash
python3 visualize.py output.npz --render-mode realized_trf_arms
```

---

| Theory                 | Equation                 | State       | Collapse   | Relativistic |
| ---------------------- | ------------------------ | ----------- | ---------- | ------------ |
| Schrödinger            | (i\hbar \partial_t \psi) | scalar      | optional   | no           |
| SchrödingerMeasurement | stochastic               | scalar      | continuous | no           |
| Dirac                  | relativistic spinor      | 2-component | no         | yes          |
| ThickFront             | modified wave evolution  | scalar      | emergent   | no           |
| DiracThickFront        | hybrid                   | spinor      | emergent   | yes          |

---

## Author

Project by:

Saku Hamalainen

---

## Acknowledgements

The simulator builds on standard numerical methods used in quantum simulation, including:

* split operator FFT methods  
* Dirac spinor propagation  
* Bohmian trajectory integration  
* retrodictive weighting methods
