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
