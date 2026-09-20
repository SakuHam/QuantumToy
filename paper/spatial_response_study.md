# Spatial measurement-guided response study

This study checks whether the physical-time width `sigma_T` changes an
operational output of the spatial measurement-guided theory. It is an
end-to-end numerical identifiability test, not an empirical measurement.

## Declared observable

The benchmark evolves a compact double-slit geometry and reads the terminal
screen distribution before any click is sampled. A Gaussian detector gate
`G(x)` defines the readable conditional profile

```text
p(y | sigma_T) = integral dx |psi(x,y)|^2 G(x)
                 / integral dx dy |psi(x,y)|^2 G(x).
```

Inference uses the complete outcome vector consisting of every unnormalized
screen-bin probability and the complementary no-click probability. The
conditional profile above and total detector mass are reported as readable
diagnostics. Thus the fit avoids click-sampling noise without discarding
no-click runs or postselecting a detector bin.

The spatial nuisance settings are fixed. Flow-direction competition and the
frozen worldline bias are disabled in this benchmark, leaving the smooth
measurement-overlap correction as the response under test. The default grid
is `64 x 48`, the detector is centered at `x=2.5`, and the terminal time is
`13/6`. The phase-neighborhood, competition radius, and Gaussian blur widths
are specified in physical lengths and converted to pixels on each grid.

## Width profile and null

A distribution generated at a declared `sigma_T` is compared with candidate
widths using

```text
KL(p_true || p_sigma).
```

For a nominal sample size `N`, the reported expected likelihood deviance is
`2 N KL`. Counts are not drawn in this test; the factor only expresses the
separation on a familiar multinomial scale.

`TRF_MEASUREMENT_RESPONSE_STRENGTH=0` is defined as an exact state-evolution
null. It calls the parent `ThickFrontWorldLineTheory` update, and the regression
test requires both the terminal state and detector distribution to agree
exactly with a separately constructed parent theory.

The response strength scales the spatial model's measurement-specific gain
and competition relief. It is not identified with the `lambda` of the minimal
dephasing candidate; deriving or rejecting such a mapping is a later task.

## Normalization controls

The study compares three declared scale rules:

- `per_run_max` divides each effect and overlap field by its own maximum;
- `fixed` obtains one effect scale and one overlap scale from a parent-theory
  calibration run at a predeclared reference width, then reuses them for every
  candidate width;
- `none` applies no effect or overlap rescaling.

This control is essential. With 100,000 nominal shots, the current default
fixture gives maximum profile deviances of about `1.83`, `0.050`, and `0.0008`
for `per_run_max`, `fixed`, and `none`, respectively. Most apparent width
sensitivity therefore comes from candidate-dependent maximum normalization.
Until a physical normalization rule is derived, the spatial model does not
provide a robust measurement of `sigma_T`.

## Convergence

The same terminal distribution is recomputed with:

- half the forward time step,
- five-fourths as many grid points in both dimensions, and
- one additional `sigma_T` of backward-kernel support.

Each comparison reports both the raw complete-distribution distance and the
distance for the excess response

```text
Delta p = p_measurement_guided - p_worldline.
```

The separately simulated parent-theory baseline removes ordinary diffraction
error from the response comparison. The cross-grid recovery likewise reports
both a raw-distribution fit and a baseline-corrected excess-response fit.

Run the study with:

```bash
PYTHONPATH=src/quantumtoy python \
  src/quantumtoy/analysis/debug/run_measurement_guided_response_study.py \
  --json-output spatial_response_study.json
```

With the default synthetic fixture, the injected width `0.1` is the same-grid
profile minimum and the exact-null errors are zero. The raw grid comparison is
still larger than the signal and its raw fit is biased. After subtracting the
independently computed worldline baseline, the `64 x 48` fit recovers `0.1`
from an `80 x 60` response. The excess-response total-variation distances are
approximately `0.000031` for `dt/2`, `0.000042` for the longer horizon, and
`0.000319` for the refined grid.
