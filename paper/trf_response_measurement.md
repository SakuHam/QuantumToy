# Independent temporal-width measurement

The candidate width can be treated as a measured quantity rather than fixed
by the stabilization convention. The dimensionless quantity to test is

```text
alpha(g) = sigma_T(g) / tau_stab(g).
```

The two times are obtained from separately prepared ensembles. This avoids
using the same record twice to define the clock and estimate the response.

## Two declared ensembles

The **record ensemble** reads every memory fragment in the X basis. Those
physical path records define the redundancy, residual-coherence, and hold-time
criteria, and hence `tau_stab(g)`. This is the protocol in
`record_environment_model.md`.

The **response ensemble** reads the fragments in the Z basis. In the toy
controlled-X memory interaction this is a quantum-eraser readout: it removes
the X-basis which-path record while retaining every memory outcome. It is not
postselection. The terminal detector is scanned at phases 0 and pi/2 and its
complete `+`, `-`, and `no_click` distribution is fitted jointly with the
probe and memory outcomes.

The separately prepared response runs are necessary because reading a stable
X-basis record has already destroyed the detector contrast needed to measure
the later temporal response. The Z readout restores that contrast without
changing the record ensemble used to define `tau_stab`.

## Response model

For the minimal half-Gaussian candidate, the excess dephasing relative to the
ordinary quantum reference is

```text
D(t) = -log(C_theta(t) / C_Q(t))
     = lambda erf(t / (sqrt(2) sigma_T)).
```

The plateau determines `lambda`, while the rise shape determines `sigma_T`.
The half-rise time supplies a direct readable width statistic,

```text
t_50 = sqrt(2) erfinv(1/2) sigma_T
     = 0.6744897502 sigma_T.
```

Ordinary exponential dephasing is included as a calibrated nuisance rate.
Inference uses the full multinomial distribution rather than only a fitted
contrast curve.

## Calibration and held-out test

The executable study calibrates `lambda`, `sigma_T`, and the nuisance rate at
`g=1`. It then predicts

```text
sigma_T(g) = alpha_cal tau_stab(g)
```

at held-out couplings `g=0.5` and `g=2`. Each held-out width is also fitted
freely. A pairwise likelihood-ratio comparison refits the calibration and
held-out data together, so uncertainty in the calibration is propagated into
the universality test. Finally, an all-coupling comparison tests one common
`alpha` against one independent width per coupling.

Run the synthetic protocol with:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/run_trf_response_study.py \
  --json-output trf_response_study.json
```

The generated observations are a regression fixture. Agreement with the
injected `alpha` demonstrates recoverability for this declared experiment; it
is not an empirical measurement or evidence for the TRF candidate.

## Spatial-simulator parameter

The fitted physical width maps to the main simulator through the single
configuration value `TRF_SIGMA_T`. The post hoc products and the
`thick_front_measurement_guided` theory read this same value. Its backward
kernel is sampled using physical delays `j * stride * dt`; its sample count is
derived from a truncation horizon expressed as a multiple of `sigma_T`.
Consequently changing `dt` or the storage stride refines the same temporal
kernel instead of changing its physical width.
