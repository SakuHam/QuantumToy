# Complete spatial effect-instrument reference

The earlier measurement-guided simulation is useful as an exploratory state
evolution, but its apparent `sigma_T` sensitivity depends strongly on how its
effect and overlap fields are normalized. This reference therefore asks a
narrower question: what does a complete spatial measurement look like if the
temporal mixture is implemented directly at the operator level?

It is a synthetic identifiability benchmark. It does not derive `sigma_T`,
`lambda`, or their numerical values from TRF-IT v0.2, and it is not an
empirical measurement.

## Complete detector

The benchmark uses a compact periodic 2D lattice and exact free-particle
spectral propagation. At the terminal time, a Gaussian detector gate
`0 <= G(x) <= 1` and a partition of the y axis define

```text
Pi_r       = diag(G(x) 1[y in bin r]),
Pi_no-click = I - sum_r Pi_r.
```

Every effect is positive and their sum is the identity. Consequently the
reported vector includes every click bin and no-click; no event is discarded
or conditioned away. The readable conditional y distribution is reported
only as a diagnostic.

For a nonnegative delay `tau_j` and a fixed reference propagation time
`t_ref`, the terminal effects are represented at the preparation time by

```text
E_(r,j) = U(t_ref + tau_j)^dagger Pi_r U(t_ref + tau_j).
```

The half-Gaussian temporal prior has declared width `sigma_T`, quadrature
step, and truncation horizon. Unresolved temporal alternatives are mixed as

```text
E_r(sigma_T) = sum_j w_j(sigma_T) E_(r,j).
```

This arithmetic operator mixture preserves off-diagonal complex entries.
Changing an individual propagator by a global phase leaves the effect and all
probabilities invariant. Positivity and completeness follow from unitary
conjugation and convex mixing and are also checked numerically.

## Explicit candidate coupling

The experiment exposes a nonnegative `lambda_strength` through the declared
convex interpolation

```text
q(lambda) = 1 - exp(-lambda),
E_ref,r = U(t_ref)^dagger Pi_r U(t_ref),
F_r = (1 - q) E_ref,r + q E_r(sigma_T).
```

This makes `lambda = 0` an exact, `sigma_T`-independent null. The exponential
mapping is a convenient bounded candidate response, not a TRF prediction.
Any empirical use would have to fix or jointly estimate this nuisance
coupling from a physical model and calibration data.

## Default synthetic result

With `sigma_T = 0.6`, `lambda = 1`, and candidates
`[0.3, 0.45, 0.6, 0.8, 1.0]`, the exact complete law recovers `0.6`. Its total
variation distance from the `lambda = 0` law is about `0.07838`. The
`lambda = 0` distributions agree exactly across widths.

The complete-law total-variation changes are approximately:

| refinement | distance |
| --- | ---: |
| half temporal quadrature step | `1.33e-5` |
| 5/4 spatial grid | `1.23e-3` |
| one extra `sigma_T` of horizon | `1.72e-6` |

Spectral propagation is unitary and evaluated directly, so there is no
forward integrator time step to converge. The temporal-step check concerns
the half-Gaussian quadrature.

## Comparison with the exploratory spatial response

| property | measurement-guided heuristic | complete effect instrument |
| --- | --- | --- |
| Primary object | nonlinear state correction | POVM effects |
| Temporal alternatives | scalar effect/overlap fields | full complex operators |
| Outcome law | terminal bins plus no-click | terminal bins plus no-click |
| Candidate-dependent normalization | present in `per_run_max` | absent |
| Positivity and completeness | output law is normalized | operator properties tested directly |
| Exact response null | parent worldline update at strength zero | unsmeared POVM at `lambda = 0` |
| Status | exploratory TRF implementation | exactly solvable phenomenological reference |

The two signal magnitudes should not be compared as estimates of the same
physical constant: their geometry, width scale, dynamics, and response
couplings differ. The useful comparison is structural. The new reference
shows that width inference can be well-defined without maximum normalization,
while also making clear that the response coupling remains an assumed and
currently unknown quantity.

Run the benchmark with:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/run_spatial_effect_measurement.py \
  --json-output spatial_effect_measurement.json
```

Render the delay components, detector geometry, half-Gaussian weights, and
the accumulating complete outcome law with:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/visualize_spatial_effect_measurement.py \
  --output-mp4 spatial_effect_measurement.mp4 \
  --snapshot-output spatial_effect_measurement.png
```

The animation frames label unresolved alternatives in the operator mixture;
they are not successive hidden positions of one detected particle.

## Joint width and coupling profile

Because `lambda` is not fixed by the document, a one-dimensional width fit can
overstate what the detector identifies. The joint benchmark evaluates the
complete multinomial law on a two-dimensional `(sigma_T, lambda)` grid. At
each width it also profiles over `lambda` and evaluates the local multinomial
Fisher matrix at the injected point.

For the default synthetic case with 100,000 nominal shots, the injected and
recovered grid points are both `(0.6, 1.0)`. The local results are:

```text
Fisher eigenvalues       = [112.20, 50237.47]
Fisher condition number  = 447.74
local standard errors    = [0.0222, 0.0919]
parameter correlation    = -0.9784
```

The strong negative correlation is the main finding: an increased temporal
width can be compensated partly by a reduced response strength. The finite
smaller Fisher eigenvalue shows that the default complete detector law still
separates the parameters locally, but much more weakly along this ridge than
across it. At `lambda=0`, the Fisher information for `sigma_T` is exactly zero,
as required by the sigma-independent null.

These standard errors use the synthetic model, its declared shot count, and
the local asymptotic Fisher approximation. The plotted `2N KL` contours are
likelihood-distance guides. They are not uncertainty estimates from observed
data and do not include model or calibration error.

Generate the profile, figure, and machine-readable result with:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/run_spatial_effect_joint_profile.py \
  --plot-output spatial_effect_joint_profile.png \
  --json-output spatial_effect_joint_profile.json
```

## Second independent detector setting

The strong one-setting ridge can be reduced by measuring the same shared
`(sigma_T, lambda)` with a second complete detector law. A predeclared design
scan changes only the detector x position and maximizes the determinant of the
combined per-shot Fisher matrix. For candidates
`[-0.5, 0.0, 0.5, 1.0, 2.5]`, the selected second position is `x=0`; the first
detector remains at `x=1.5`.

The two multinomial records are independent conditional on the shared
parameters. Their KL divergences and Fisher matrices add with the declared
shot fractions. Using the same total budget of 100,000 shots, split equally
between the settings, gives:

| diagnostic | one setting | two settings |
| --- | ---: | ---: |
| Fisher condition number | `447.74` | `14.28` |
| parameter correlation | `-0.9784` | `-0.6335` |
| local sigma standard error | `0.0222` | `0.00819` |
| local lambda standard error | `0.0919` | `0.0227` |

The injected pair `(0.6, 1.0)` remains the exact joint grid minimum. The
improvement comes from the second setting's differently oriented score, not
from increasing the total shot count or omitting no-click outcomes.

The design scan is local to the synthetic fiducial pair and the declared
candidate detector locations. A real design would have to optimize expected
performance over prior parameter uncertainty, calibration error, and feasible
detector geometry.

Run the two-setting design and comparison with:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/run_spatial_effect_two_setting_profile.py
```

This writes `spatial_effect_two_setting_profile.png` and
`spatial_effect_two_setting_profile.json` by default.

## Locked double-slit robustness study

The operator reference now also supports a real, static double-slit potential
with a unitary symmetric split-step propagator. The geometry is fixed before
the width and coupling profile:

| quantity | value |
| --- | ---: |
| box | `10 x 8` |
| grid | `80 x 80` |
| barrier center and Gaussian width | `x=-0.5`, `0.16` |
| barrier height | `35` |
| slit centers | `y=+-0.8` |
| slit half-height and edge smoothing | `0.3`, `0.08` |
| packet center, widths, and momentum | `(-2.5,0)`, `(0.45,0.65)`, `kx=3` |
| detector center and width | `x=1.5`, `0.3` |
| y outcomes | `16` click bins plus no-click |
| reference time | `0.9` |
| split-step and delay quadrature steps | `0.005`, `0.025` |
| temporal horizon | `4 sigma_T` |

At the synthetic point `(sigma_T, lambda)=(0.2,1)`, the injected pair is the
exact profile minimum. The double slit changes the complete outcome law from
the matched free propagation by total variation `0.11564`; click probability
changes from `0.16318` to `0.04780`. The conditional click law contains the
resolved modulation across the 16 y bins.

The numerical complete-law total-variation checks are:

| variation | distance |
| --- | ---: |
| half delay quadrature step | `1.44e-5` |
| 5/4 spatial grid | `4.16e-4` |
| one additional sigma of horizon | `1.33e-6` |
| half split-step | `3.13e-5` |
| 5/4 x-box at fixed resolution | `3.48e-6` |
| half slit-edge smoothing sensitivity | `3.06e-3` |

The edge-smoothing number is a declared geometry sensitivity rather than a
discretization error. It is still much smaller than the free-versus-slit
signal but larger than the numerical refinements.

The detector pair selected for the earlier free geometry was also tested
without redesign. In the double-slit model, splitting the same 100,000 shots
between `x=1.5` and `x=0` changes the Fisher condition number from `98.61` to
`287.18`, the parameter correlation from `-0.725` to `-0.907`, and both local
standard errors increase. Thus detector independence alone is insufficient:
the settings must have complementary parameter responses in the actual
geometry. The result is retained as a failed held-out design test rather than
being optimized away after inspection.

### Double-slit detector design

We therefore define a prospective scan for the locked double-slit geometry.
It varies the second detector over `x in {0, 0.5, 1, 1.5, 2}`, Gaussian gate
widths `{0.2, 0.35, 0.5}`, and reference-time offsets
`{-0.4, -0.2, 0, 0.2}` from the packet's classical arrival. The original and
candidate settings receive equal shares of the same total shot budget. A
candidate is eligible only if the inverse combined Fisher matrix gives smaller
marginal errors for both `sigma_T` and `lambda`; among eligible candidates the
predeclared objective maximizes the combined per-shot Fisher determinant.

The selected second setting is `x=0`, gate width `0.5`, and reference time
`0.43333`. At 100,000 total shots the results are:

| diagnostic | original detector | selected 50/50 pair |
| --- | ---: | ---: |
| click probability of each setting | `0.04780` | second: `0.17907` |
| Fisher condition number | `98.61` | `33.68` |
| parameter correlation | `-0.72545` | `-0.68220` |
| `SE(sigma_T)` | `0.02242` | `0.01956` |
| `SE(lambda)` | `0.15151` | `0.08074` |

The injected pair `(sigma_T, lambda)=(0.2,1)` remains the exact combined
profile minimum. For the selected detector, increasing the x box by 5/4 at
fixed spatial resolution changes the complete outcome law by `1.12e-7`. This
check matters because a later detector candidate initially appeared favorable
in the smaller periodic box but failed the same boundary test. The detector
choice is a synthetic, local design conditional on the declared parameter
point and geometry; experimental calibration or a robust prior-averaged design
would still be needed for data collection.

Run the design study with:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/run_spatial_effect_double_slit_detector_design.py
```

### Calibrated observation model and robust inference

The ideal effects alone do not describe a laboratory detector. We retain their
complete probability vector `p` and apply a column-stochastic observation
channel `C`, so the recorded law is

```
p_observed = C(eta, d, b) p.
```

Here `eta` is click efficiency, `d` is the per-shot dark-click probability,
and `b` is Gaussian y-bin resolution. A missed ideal click enters the same
dark/no-click branch as an ideal no-click. Consequently every column of `C`
sums to one and the model never conditions on a recorded click. Symmetric
readout-time jitter is applied before `C` with a three-node quadrature having
the declared jitter variance. The TRF half-Gaussian delay and instrumental
symmetric jitter therefore remain distinct model components.

Five free-propagation controls at reference times
`{0.55, 0.7, 0.85, 1.0, 1.15}` calibrate the detector using the known
`lambda=0` law. The seeded pilot injection
`(eta,d,b,jitter)=(0.88,0.012,0.45,0.05)` gives
`(0.8777,0.0120,0.4368,0.05)`. No double-slit counts enter this pilot fit.
During the final fit, calibration and double-slit log likelihoods are added and
the shared detector parameters are profiled again. Thus calibration uncertainty
is carried into the test rather than replacing the nuisances by exact values.

The prospective detector design uses the Cartesian region

```
sigma_T in {0.12, 0.20, 0.28}
lambda  in {0.5, 1.0, 1.5}.
```

For each candidate it evaluates the gain
`log det(F_pair) - log det(F_original)` at all nine points and maximizes the
minimum gain. A 48 by 48 grid is used for the scan and the selected setting is
then checked independently at 80 by 80. Both resolutions select
`(x,width,time)=(0,0.5,0.43333)`. On the verification grid the minimum log
determinant gain is `1.661`; the worst marginal standard-error ratios are
`0.838` for `sigma_T` and `0.731` for `lambda`.

The repeated synthetic procedure redraws both calibration and test counts and
profiles `(eta,d,b,jitter)` together with `(sigma_T,lambda)`. Since `sigma_T`
is absent from the law at `lambda=0`, the null is nonregular and a fixed
chi-square likelihood-ratio threshold is not justified. The locked validation
uses 500 parametric-bootstrap nulls to set the threshold, followed by new and
independent 500-sample null and signal ensembles. The fit grid starts at
positive `lambda=0.005` near the boundary and has spacing `0.05` around the
injected signal. The results are:

| injection | detection rate | joint 95% coverage | lambda bias | lambda RMSE |
| --- | ---: | ---: | ---: | ---: |
| `lambda=0` | `0.094` | `0.988` | `0.0139` | `0.0234` |
| `lambda=1` | `1.000` | `0.958` | `0.0486` | `0.1523` |

All 1,500 profile fits report successful optimization. The bootstrap 95%
threshold is `2.422`, while the independent null ensemble rejects 47 of 500
samples. The false-detection Wilson 95% interval is `[0.071,0.123]`, excluding
the intended `0.05`, so this hypothesis test fails its calibration criterion.
The bootstrap threshold's distribution-free order-statistic 95% interval is
wide, `[1.984,3.262]`, and contains the independent null 95% LR quantile
`3.049`. This points to unstable tail-quantile estimation rather than a failed
optimizer, but the threshold must not be raised after inspecting the evaluation
set. Signal detection is `1.000` with Wilson interval `[0.992,1.000]`; its joint
coverage is `0.958` with interval `[0.937,0.972]`. The positive signal bias
`0.0486` remains visible. A larger independent bootstrap or a continuously
profiled and separately calibrated test is required before making a controlled
false-positive claim. Under `lambda=0`, the width estimate remains structurally
meaningless; its displayed coverage only records inclusion of an arbitrarily
indexed grid point.

Run the complete calibrated study with:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/run_spatial_detector_inference_study.py \
  --bootstrap-repetitions 500 --repetitions 500 --workers 10 \
  --plot-output spatial_detector_validation_500.png \
  --json-output spatial_detector_validation_500.json
```

Run the study with:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/run_spatial_effect_double_slit_study.py
```

Render the complete double-slit mixture with:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/visualize_spatial_effect_measurement.py \
  --geometry double_slit \
  --output-mp4 spatial_effect_double_slit.mp4 \
  --snapshot-output spatial_effect_double_slit.png
```

### Paired fit-grid diagnostic

The fit now reports both winning and null optimizer status, iteration and
function-evaluation counts, failed/nonfinite profile attempts, shared detector
estimates, and boundary flags. A successful winning optimizer no longer hides
an unsuccessful nonwinning profile attempt in the diagnostic record. Boundary
flags distinguish lower/upper edges from fixed or interior parameters; a width
selected at `lambda=0` is explicitly marked unidentifiable. Bootstrap and
recovery results retain compact per-replicate records with their generated seeds.

The small comparison uses ten null and ten signal datasets, generated once at
`sigma_T=0.2`, `lambda in {0,1}` and the declared true detector response. Both
fits receive exactly the same saved control and test counts. The 48 x 48
spatial grid, selected detector pair, timing-jitter candidates, and shot budgets
stay fixed. The current 7 x 22 sigma/lambda grid is embedded in a 13 x 43 grid
by adding midpoint candidates without expanding its bounds. The comparison
checks equality of the common-point likelihood surface and null optimum;
the refined alternative optimum should be no worse than the coarse optimum.

The five controls each receive 50,000 shots (250,000 calibration shots total).
The double-slit settings each receive 50,000 shots (100,000 test shots total).
Every completed pair is checkpointed to JSONL; the input manifest and final
JSON retain actual counts, seeds, software versions, configuration, and source
hashes. The seed for this comparison is `20260923`, distinct from the previous
validation. No detection threshold is fitted or applied in this diagnostic:
`detected=false` in its fit records means detection was disabled, not a failed
test. It cannot determine the false-positive rate from ten null trials, and it
does not replace the previous independent 500-trial validation.

All 20 paired datasets completed. The current and refined fits jointly ran
42,780 profile optimizations; none failed or returned a nonfinite objective.
Every best and null fit and every calibration optimization also succeeded.
At grid points shared by both fits, the complete profile likelihood surfaces
and null minima agreed exactly, and the refined minimum never violated the
nested-grid ordering. The numerical comparison is:

| injection | median LR change | largest absolute LR change | selected pair changed |
| --- | ---: | ---: | ---: |
| `lambda=0` | `0` | `0.0483` | `3/10` |
| `lambda=1` | `0.0253` | `0.1405` | `6/10` |

The signal LR values span `512`–`643`, making their grid changes negligible
for detection. The null fits have LR range `0`–`2.58` on the current grid and
`0`–`2.62` on the refined grid. Five of ten null records select `lambda=0`
itself; their fitted width is explicitly marked unidentifiable. More broadly,
the best null-record width lands on a sigma boundary in 7/10 current-grid and
8/10 refined-grid fits, as expected near the structurally unidentified null.
No continuous detector nuisance reaches a bound; one record per scenario uses
an endpoint of the discrete timing-jitter candidates. These results rule out a
large sigma/lambda grid error or hidden optimizer failure as the main cause of
the earlier `0.094` false-detection rate. They do not rule out small
classification changes for records lying extremely close to a threshold.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
PYTHONPATH=src/quantumtoy .venv/bin/python \
  src/quantumtoy/analysis/debug/run_spatial_detector_grid_comparison.py \
  --repetitions 10 --workers 4 \
  --json-output spatial_detector_grid_comparison_10.json \
  --plot-output spatial_detector_grid_comparison_10.png
```

### Temporally resolved and coherent observation benchmark

A separate local Fisher comparison asks how much information is lost by
tracing out the temporal alternatives. It uses a fixed 0.8-unit delay horizon,
the 48 x 48 double-slit grid, the selected equal-shot detector pair, 100,000
total test shots, `(sigma_T,lambda)=(0.2,1)`, and the declared calibrated
detector response. All laws retain click and no-click outcomes.

The current observation sums over the temporal register. The resolved
observation retains its nonnegative delay label but merges the response-null
branch with the temporal component at exactly zero delay, because an arrival
timestamp cannot distinguish them. A conservative resolved case adds a
Gaussian timestamp blur of `0.05`, equal to two delay bins. The coherent case
instead treats the mixture as a coherent purification with orthogonal temporal
modes, applies a discrete Fourier readout, and selects one phase offset from a
predeclared 12-point local scan. Summing the resolved or coherent click labels
reproduces the current spatial marginal to a maximum error below `9e-16`.

| observation | `SE(sigma_T)` | ratio | `SE(lambda)` | ratio | log-det gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| current y only | `0.03327` | `1.000` | `0.11225` | `1.000` | `0` |
| sharp y only | `0.02860` | `0.860` | `0.10037` | `0.894` | `0.448` |
| time tagged, two-bin blur | `0.00206` | `0.0618` | `0.02164` | `0.1928` | `8.133` |
| ideal time tag | `0.00171` | `0.0514` | `0.01386` | `0.1235` | `9.228` |
| coherent phase ports | `0.00297` | `0.0892` | `0.01503` | `0.1339` | `7.963` |

Thus improved spatial sharpness gives a modest gain, while access to the
temporal label changes identifiability by roughly an order of magnitude or
more. With the conservative tag blur, direct timing gives the smaller width
error and the coherent port gives the smaller coupling error; an ideal time
tag is best for both parameters in this comparison. These are optimistic
measurement-design bounds. The temporal register, coherent purification,
phase stability, and Fourier-port measurement are new physical assumptions,
not predictions derived from TRF-IT. Detector nuisance parameters are fixed
at their injected values, and no repeated-sample or false-positive calibration
is performed.

```bash
PYTHONPATH=src/quantumtoy MPLBACKEND=Agg .venv/bin/python \
  src/quantumtoy/analysis/debug/run_spatial_temporal_observation_comparison.py \
  --json-output spatial_temporal_observation_comparison.json \
  --plot-output spatial_temporal_observation_comparison.png
```

## Finite-resolution joint spatial/arrival-time instrument

The resolved-register calculation above is an information bound: it labels a
latent temporal mode directly. A closer laboratory model instead starts each
trial with a pulsed or heralded source, subtracts the standard time of flight
for that detector setting, and stores the record

```text
(pulse id, y bin, arrival-time bin), or explicit no-click.
```

The candidate response law is

```text
t_recorded - t_standard = kappa_t tau + epsilon_source
                            + epsilon_detector + clock_offset,
```

where both errors are Gaussian. The response retains a finite acquisition
window and equal timestamp bins. A genuine arrival outside the window or an
inefficient physical click enters the residual channel; that channel yields a
uniform dark `(y,t)` record with the calibrated dark probability or the
explicit no-click result. Thus every emitted trial remains represented.

The spatial law for latent delay `tau` is still evaluated at
`t_reference + tau`. The timestamp scale `kappa_t` is separate. This matters:
TRF-IT v0.2 does not derive that a latent temporal width shifts a detector
clock one-for-one. `kappa_t=1` is an optimistic instrument hypothesis, while
`kappa_t=0` means the timestamp carries no direct latent-delay label.

The locked small study uses the selected two-detector double-slit geometry,
48 x 48 spatial points, 24 time bins over residual times `[-0.2,1.0]`, source
width `0.03`, detector jitter `0.05`, clock offset `0.01`, and 100,000 total
shots. All quantities use the simulation's dimensionless time unit.

| observation model | `SE(sigma_T)` | `SE(lambda)` | correlation |
| --- | ---: | ---: | ---: |
| spatial marginal of this instrument | `0.03183` | `0.10585` | `-0.759` |
| joint `p(y,t)`, timing fixed | `0.00253` | `0.02329` | `-0.389` |
| joint `p(y,t)`, source width and offset fitted | `0.00276` | `0.04357` | `+0.084` |
| joint `p(y,t)`, independent timing controls | `0.00265` | `0.03526` | `-0.086` |

The controls constrain source width to standard error `0.005` and clock
offset to `0.003`. Source width and detector jitter cannot both be learned
from these records alone: they occur only as
`sqrt(source_width^2 + detector_jitter^2)`. The five-parameter per-shot Fisher
matrix has a smallest eigenvalue about `4.1e-13`, so an independent source or
detector timing calibration is structurally required.

| response variation | `SE(sigma_T)` | `SE(lambda)` |
| --- | ---: | ---: |
| `kappa_t=0` | `0.02768` | `0.10056` |
| `kappa_t=0.1` | `0.01477` | `0.08664` |
| `kappa_t=0.25` | `0.00691` | `0.06057` |
| `kappa_t=0.5` | `0.00357` | `0.03530` |
| `kappa_t=1` | `0.00253` | `0.02329` |
| detector jitter `0.2` at `kappa_t=1` | `0.00626` | `0.05665` |
| detector jitter `0.4` at `kappa_t=1` | `0.01557` | `0.09046` |

A 40-record multinomial smoke study with a calibrated nuisance grid recovered
the exact injected `(sigma_T,lambda,source width,offset)` grid point in 39
records. This checks implementation and local recoverability under the stated
law. It does not establish the physical map, calibrate a null test, or provide
evidence for TRF. The main experimental bottleneck is therefore upstream of
counting statistics: a concrete system must predict `kappa_t`, set the
dimensionful clock scale, and distinguish the delay shape from ordinary
source and detector response.

Run the study with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src/quantumtoy MPLBACKEND=Agg .venv/bin/python \
  src/quantumtoy/analysis/debug/run_spatial_arrival_time_study.py \
  --json-output spatial_arrival_time_study.json \
  --plot-output spatial_arrival_time_study.png
```
