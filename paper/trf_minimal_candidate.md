# Minimal TRF candidate and parameter policy

This file declares the first falsifiable extension built on the fixed
two-path record experiment. The rule is a phenomenological candidate. It is
not derived by TRF-IT v0.2 and it is not yet connected to the spatial
`thick_front_measurement_guided` dynamics.

## Parameter policy

| Class | Quantities | Treatment |
| --- | --- | --- |
| Calibrated apparatus | preparation, probe strength `kappa`, products `g r_k`, detector efficiencies | determine with quantum-reference calibration data |
| Operational convention | `delta`, `R*`, coherence tolerance, hold time | declare before analysis and report a common sensitivity grid |
| Numerical | time step and observation duration | establish clock-sampling convergence; do not fit |
| TRF candidate | `lambda` | estimate or bound with held-out joint-record data |
| Width convention | `alpha` in `sigma_T=alpha tau_stab` | fix before fitting; the first study uses `alpha=1` |
| Calibrated nuisance | ordinary dephasing rate | include in identifiability and likelihood analyses |

Only the products `g r_k` occur in the memory rotation. Their overall scale is
therefore not separately identifiable. The default convention fixes the listed
relative `r_k` values and treats `g` as the measured common rate scale.

## Candidate channel

The normalized half-Gaussian kernel has cumulative weight

```text
W(t; sigma_T) = erf[t / (sqrt(2) sigma_T)],
sigma_T = alpha tau_stab.
```

The candidate preserves path populations and changes the system coherence by

```text
rho_01^theta(t) = rho_01^Q(t) exp[-lambda W(t; sigma_T)],
lambda >= 0.
```

This is a phase-damping quantum channel because its coherence factor lies in
`[0,1]`. Consequently the resulting joint probabilities are positive and
normalized. The channel is applied after the recorded weak probe and memory
formation and before the complete terminal detector. Summing over every
terminal outcome leaves the earlier `p(a,Y)` marginal unchanged. `lambda=0`
recovers the quantum-reference joint distribution exactly.

This choice is intentionally minimal. It supplies a complete observable joint
law without assigning physical meaning to the gain, competition-relief, blur,
or refresh parameters of the exploratory spatial implementation.

## Declared observable and identifiability

The first scalar diagnostic is the complete-record detector contrast

```text
V(g,t) = p(d=+) - p(d=-),
```

where no-click has score zero and remains part of the normalized distribution.
The full multinomial `p(a,Y,d|g,t)` should be used for inference; `V` is a
readable summary rather than a sufficient statistic.

The synthetic design evaluates several times and several environment
couplings. It includes a common coupling-scale nuisance, probe-strength
nuisance, and ordinary Markovian dephasing. The expected Fisher matrix reports
local identifiability and parameter correlations. A full-rank matrix does not
establish empirical evidence. In particular, a strong correlation between
`lambda` and ordinary dephasing means independent noise calibration or a more
discriminating time/coupling design is necessary. A one-time design cannot
separate the two dephasing sources and is explicitly tested as rank deficient.

Clock thresholds are scanned as shared conventions across every coupling.
The underlying states are analytic, so the time-step study measures only the
sampled onset and hold-time error. Spatial-grid, adjoint-step, blur, and refresh
convergence remain separate work for the main simulator.

## Design optimization and recovery

The design study allocates a fixed shot budget over candidate `(g,t)` settings.
It profiles calibrated uncertainties in the common coupling scale, probe
strength, and ordinary dephasing. It also adds a continuous clock scale as a
fifth local design parameter. The greedy objective minimizes the worst
profiled lambda variance over the minimum, central, and maximum shared clock
conventions found in the threshold sweep. A singular one-time design is seeded
with additional settings until the full design is identifiable.

The Monte Carlo study draws every `(a,Y,d)` count from its multinomial joint
distribution. The fit constrains `lambda>=0`, profiles the three calibrated
nuisance parameters, and profiles the finite clock-convention set. It compares
the fitted candidate with the profiled `lambda=0` model using a one-sided
likelihood-ratio threshold. Null simulations measure the actual false-positive
rate of this procedure; signal simulations measure recovery bias and power.

Continuous clock profiling and discrete convention profiling are deliberately
reported separately. A near-unit lambda–clock correlation means that the
mapping from stabilization latency to kernel width, rather than ordinary noise
calibration, limits the physical interpretation of lambda. In that case a
reported detection can establish a deviation for the candidate family while
the numerical value of lambda remains convention dependent.

## Independent width measurement

The clock degeneracy can be tested by measuring the response width from a
separate quantum-eraser ensemble. The record ensemble still determines
`tau_stab`; the response ensemble determines `sigma_T` from the rise of excess
dephasing. Their ratio `alpha=sigma_T/tau_stab` can then be calibrated at one
coupling and predicted at held-out couplings. The complete protocol and its
universality likelihood-ratio test are specified in
`trf_response_measurement.md`.
