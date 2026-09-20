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

For a nonnegative delay `tau_j`, the terminal effects are represented at the
preparation time by

```text
E_(r,j) = U(tau_j)^dagger Pi_r U(tau_j).
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
F_r = (1 - q) Pi_r + q E_r(sigma_T).
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
