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

Jointly profile the unknown temporal width and response strength with:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/run_spatial_effect_joint_profile.py \
  --plot-output spatial_effect_joint_profile.png \
  --json-output spatial_effect_joint_profile.json
```

The resulting likelihood surface and Fisher matrix expose the strong local
`sigma_T`–`lambda` tradeoff that a fixed-coupling width profile hides.

Select and combine a second detector location under the same total shot
budget with:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/run_spatial_effect_two_setting_profile.py
```

The default Fisher design retains the first detector at `x=1.5`, selects a
second detector at `x=0`, and writes a one-versus-two-setting comparison plot
and JSON report.

Run the locked unitary double-slit robustness study with:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/run_spatial_effect_double_slit_study.py
```

Render its interference evolution and complete detector law with:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/visualize_spatial_effect_measurement.py \
  --geometry double_slit \
  --output-mp4 spatial_effect_double_slit.mp4 \
  --snapshot-output spatial_effect_double_slit.png
```

The study fixes the barrier, slits, packet, propagation clock, and detector
before profiling. It also tests the earlier free-geometry detector pair as a
held-out design rather than retuning it after seeing the double-slit result.

Design a complementary detector specifically for the locked double-slit
geometry with:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/run_spatial_effect_double_slit_detector_design.py
```

The declared scan varies the downstream detector plane, Gaussian gate width,
and readout time relative to classical arrival. It splits the same 100,000-shot
budget equally between the original and candidate settings. Candidates must
improve both marginal Fisher standard errors; the remaining candidate with the
largest combined Fisher determinant is selected. The fixed scan selects
`x=0`, width `0.5`, and reference time `0.43333`. Relative to the original
detector alone, the pair changes the condition number from `98.61` to `33.68`,
the `sigma_T` standard error from `0.02242` to `0.01956`, and the `lambda`
standard error from `0.15151` to `0.08074`. The x-box check changes the complete
law by only `1.12e-7`. These are synthetic local-design results at the declared
injection `(sigma_T, lambda)=(0.2,1)`, not empirical detector calibration.

Run the calibrated robust-design and repeated-inference layer with:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/run_spatial_detector_inference_study.py
```

This layer keeps the ideal complete POVM and follows it with a normalized
detector channel containing efficiency, dark-click probability, y-bin blur,
and readout-time jitter. Five known free-propagation controls calibrate those
nuisances before the double-slit setting is selected. The detector scan then
maximizes the worst Fisher log-determinant gain over the declared
`sigma_T x lambda` region, rather than at one injected point. The final
likelihood profiles the same nuisance response jointly against independent
calibration and double-slit counts.

The detector design selects the same second setting
`(x,width,time)=(0,0.5,0.43333)`. On the independent 80 by 80 verification
grid, the worst standard-error ratios relative to the original detector are
`0.838` for `sigma_T` and `0.731` for `lambda`. Because `sigma_T` is not
identified at `lambda=0`, a separate parametric bootstrap calibrates the
likelihood-ratio threshold.

The locked validation run used 500 bootstrap nulls followed by independent
ensembles of 500 null and 500 signal samples. Its bootstrap threshold is
`2.422`, but the independent false-detection rate is `0.094` with Wilson 95%
interval `[0.071,0.123]`. The intended `0.05` level is outside this interval,
so the hypothesis test does **not** yet pass calibration. The bootstrap
threshold itself has a wide order-statistic 95% interval `[1.984,3.262]`,
which contains the independent null 95% LR quantile `3.049`. Signal detection
is `1.000` (`[0.992,1.000]`) and signal joint-region coverage is `0.958`
(`[0.937,0.972]`). Thus the model recovers the injected signal, but more
bootstrap samples or a better-calibrated continuous test are required before
claiming a controlled false-positive rate. A null `sigma_T` estimate has no
physical meaning.

Reproduce the long run with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/run_spatial_detector_inference_study.py \
  --bootstrap-repetitions 500 --repetitions 500 --workers 10 \
  --plot-output spatial_detector_validation_500.png \
  --json-output spatial_detector_validation_500.json
```

Audit the fit grid on identical null and signal records with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
PYTHONPATH=src/quantumtoy .venv/bin/python \
  src/quantumtoy/analysis/debug/run_spatial_detector_grid_comparison.py \
  --repetitions 10 --workers 4 \
  --json-output spatial_detector_grid_comparison_10.json \
  --plot-output spatial_detector_grid_comparison_10.png
```

This holds the selected detector geometry, 48 x 48 spatial grid, jitter grid,
and shot budgets fixed. The current 7 x 22 sigma/lambda grid is compared with
a nested 13 x 43 grid formed by adding midpoints. Each pair receives the same
saved calibration and test counts. Calibration uses 50,000 shots **per** free
control (250,000 total), and the two test settings share 100,000 shots.
The `.inputs.json` file records the configuration and counts before fitting;
`.jsonl` checkpoints completed pairs, and the final JSON includes summaries.
Best/null optimizer status, detector estimates, boundary flags, nonwinning
optimization failures, and common-grid likelihood differences are recorded.
Future bootstrap and recovery outputs also include seed-addressable replicate
records. This small comparison does not estimate or recalibrate a detection
threshold; the previous 500-trial validation remains a separate result.

For the locked seed `20260923`, all 20 paired datasets completed. Across the
coarse and refined fits, all 42,780 profile optimizations, all best/null fits,
and all calibration optimizations succeeded with finite objectives. The
common-point profile surfaces and null minima agreed exactly. Under `lambda=0`,
the largest LR change was `0.0483` (median `0`); under `lambda=1`, it was
`0.1405` (median `0.0253`) against signal LR values `512`–`643`. The refined
grid changed the selected parameter pair in 3/10 null and 6/10 signal records,
but only by resolving intermediate points. It produced no nested-minimum
violations. Null width estimates hit a sigma-grid boundary in 7/10 coarse and
8/10 refined fits, consistent with the declared non-identifiability. Therefore
the coarse sigma/lambda grid is not a plausible main explanation for the prior
9.4% false-detection result; the next validation should target bootstrap tail
precision and calibration construction.

Compare the current spatial marginal with stronger temporal observations:

```bash
PYTHONPATH=src/quantumtoy MPLBACKEND=Agg .venv/bin/python \
  src/quantumtoy/analysis/debug/run_spatial_temporal_observation_comparison.py
```

This local 48 x 48 Fisher benchmark uses the selected two-detector geometry,
100,000 total shots, `(sigma_T,lambda)=(0.2,1)`, the calibrated detector
response, and one fixed delay grid. The time-tagged observation merges the
reference and zero-delay components and applies an additional timestamp blur
of `0.05`, or two delay bins. The coherent observation purifies the declared
mixture into a temporal register, scans a phase offset, and reads all Fourier
ports. Summing either observation over its temporal labels reproduces the
current complete y/no-click law to below `9e-16`.

| observation | `SE(sigma_T)` | ratio to current | `SE(lambda)` | ratio to current |
| --- | ---: | ---: | ---: | ---: |
| current y only | `0.03327` | `1.000` | `0.11225` | `1.000` |
| sharp y only | `0.02860` | `0.860` | `0.10037` | `0.894` |
| time tagged, two-bin blur | `0.00206` | `0.0618` | `0.02164` | `0.1928` |
| ideal time tag | `0.00171` | `0.0514` | `0.01386` | `0.1235` |
| coherent phase ports | `0.00297` | `0.0892` | `0.01503` | `0.1339` |

The large gain comes from observing a temporal register that the current
instrument deliberately traces out. It is not evidence that such a register
is experimentally accessible. The coherent purification, its stable relative
phases, and the locally selected phase setting are additional instrument
assumptions, not consequences of TRF-IT. The calculation also treats the
detector response as known and is not a likelihood-ratio calibration.

The next, more realistic timing benchmark stores an arrival residual for each
click and keeps the complete no-click result. It models

```text
t_recorded - t_standard = kappa_t * tau + source-time error
                            + detector timestamp error + clock offset.
```

Here `tau` is the latent nonnegative delay, while `kappa_t` is an explicit
instrument coupling. TRF-IT v0.2 does not derive this map or require
`kappa_t=1`. The implementation uses finite timestamp bins and an acquisition
gate; arrivals outside the gate remain in the dark/no-click channel. Run its
local precision, coupling/jitter sensitivity, and small synthetic recovery
study with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src/quantumtoy MPLBACKEND=Agg .venv/bin/python \
  src/quantumtoy/analysis/debug/run_spatial_arrival_time_study.py
```

At 100,000 total shots in the 48 x 48 double-slit benchmark, the spatial
marginal gives local errors `(0.03183, 0.10585)` for
`(sigma_T, lambda)`. The joint timestamp law with fixed timing calibration
gives `(0.00253, 0.02329)`. Profiling source width and clock offset degrades
this to `(0.00276, 0.04357)`; independent timing controls with standard errors
`(0.005, 0.003)` improve it to `(0.00265, 0.03526)`. In 40 synthetic records,
the calibrated nuisance grid recovered the exact injected grid point in 39
cases; this is a model check, not an empirical detection claim.

The sensitivity scan shows why a real test can still be difficult. Reducing
`kappa_t` from `1` to `0.1` increases the width error from `0.00253` to
`0.01477`, and detector timestamp jitter `0.4` increases it to `0.01557`.
At `kappa_t=0`, the time label adds no direct delay measurement and the width
error is `0.02768`, close to the spatial-only result. Source-pulse width and
detector jitter enter only through their quadrature sum, producing an exact
unidentified direction unless one is calibrated independently. All time
values remain dimensionless until the simulation clock is mapped to a
specific particle, length scale, and laboratory apparatus.

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
