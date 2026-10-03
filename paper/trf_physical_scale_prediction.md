# Physical clock scale and direct arrival-delay prediction

This note closes the unit system of the spatial double-slit benchmark and
states the strongest arrival-time prediction currently supported by an
explicit additional hypothesis. It also identifies what TRF-IT v0.2 still
does not determine.

## Schrödinger unit conversion

The implemented free kinetic phase corresponds to

```text
i hbar d(psi)/dt = -(hbar^2 / 2m) Laplacian(psi) + V psi.
```

Write `x = L0 x_tilde` and `t = T0 t_tilde`. Keeping the code's
dimensionless `hbar=m=1` fixes

```text
T0 = m L0^2 / hbar,
v0 = L0 / T0 = hbar / (m L0),
E0 = hbar / T0.
```

The locked slit centers are at `y=+-0.8`, so a chosen physical slit
center-to-center distance `d` gives `L0=d/1.6`. The locked packet momentum
`kx=3` then fixes the packet-center velocity to `3 hbar/(m L0)`. Equivalently,
choosing mass and velocity fixes `L0=hbar kx/(m v)` and therefore the physical
geometry. Length, velocity, and time cannot be selected independently while
retaining the locked dimensionless experiment.

## Audit against TRF-IT v0.2

The source document now stored at
`paper/TRF_Information_Theoretic_Formulation_v0.2.pdf` fixes the status of each
part of the prediction:

- Section 6, equation (12), defines the normalized half-Gaussian kernel and
  requires `sigma_T>0`. Because the kernel has inverse-time units, `sigma_T`
  is a time, but the equation does not set its magnitude.
- Section 8, equation (17), defines the event-anchored stabilization latency
  `tau_stab`.
- Section 8, equation (18), postulates `sigma_T=alpha tau_stab` and explicitly
  says that dimensionless `alpha` is fixed by an independently declared
  convention or calibration. It is not a derived universal constant.
- Section 10 states that v0.2 does not yet supply a distinct observable TRF
  prediction and asks for a complete measurable joint model.
- Section 11 states directly that the motivating information bounds do not
  derive a TRF timescale or kernel.

The document gives unresolved timing offsets as an example for arithmetic
effect mixing, but it does not map a latent delay to an eventwise detector
arrival timestamp. Consequently the following `kappa_t=1` closure is new
model content rather than an interpretation forced by v0.2.

## Direct-delay closure

The spatial instrument evaluates a temporal component with
`U(t_reference + tau)`. To turn that latent component into an arrival-time
record, the minimal closure is

```text
Delta t_recorded = T0 tau_tilde + source error + detector error + clock offset.
```

Because both sides use the same physical clock conversion, this closure gives

```text
kappa_t = d(Delta t_recorded) / d(T0 tau_tilde) = 1.
```

This value follows from defining `tau` as an actual additive propagation
delay. TRF-IT v0.2 does not require that interpretation. If its temporal
mixture is only an unresolved operator average, an eventwise timestamp and
therefore `kappa_t` do not follow.

For the current spatial fixture, `sigma_T=0.2`. Under the direct-delay closure,

```text
sigma_T(physical) = 0.2 m L0^2 / hbar.
```

At `lambda=1`, the responding fraction is `q=1-exp(-1)=0.63212`. The response
branch has a half-normal delay. Its mean is
`sqrt(2/pi) sigma_T`; the full ensemble also has zero-delay weight `exp(-1)`,
and hence mean delay

```text
E[Delta t] = q sqrt(2/pi) sigma_T(physical)
           = 0.10086 T0.
```

## Worked physical mappings

These are unit conversions of the locked synthetic geometry, not proposed
optimal apparatus designs.

| mapping | `L0` | packet speed | `T0` | spatial `sigma_T=0.2T0` | ensemble mean | current jitter `0.05T0` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| electron, 1 micrometer slit separation | 0.625 micrometers | 556 m/s | 3.374 ns | 674.8 ps | 340.4 ps | 168.7 ps |
| neutron, 10 micrometer slit separation | 6.25 micrometers | 0.0302 m/s | 620.4 microseconds | 124.1 microseconds | 62.58 microseconds | 31.02 microseconds |
| C60, 100 nm slit separation | 62.5 nm | 0.00423 m/s | 44.29 microseconds | 8.857 microseconds | 4.467 microseconds | 2.214 microseconds |
| electron, packet speed 1,000,000 m/s | 0.347 nm | 1,000,000 m/s | 1.042 fs | 0.208 fs | 0.105 fs | 0.052 fs |

The comparison exposes a strong apparatus constraint. A one-micrometer
electron geometry maps the locked packet to a very slow electron, while a
roughly 2.84 eV electron maps it to a 0.556 nm slit separation and a
sub-femtosecond width. A physical proposal must change and revalidate the
dimensionless geometry if neither regime is suitable.

## Conflict between the two existing clocks

The independent record model gives a stabilization latency
`tau_stab=2.76` at dimensionless coupling `g=1`. Its earlier response fixture
used the declared convention

```text
sigma_T = alpha tau_stab, alpha=1.
```

That predicts `sigma_T=2.76T0`, whereas the spatial detector fixture injects
`0.2T0`. The difference is a factor of 13.8. Making the two current fixtures
describe one width requires

```text
alpha = 0.2 / 2.76 = 0.07246377.
```

For example, the neutron mapping gives `sigma_T=124.1 microseconds` in the
spatial fixture but `1.712 ms` under the record fixture's `alpha=1`
convention. This is not a numerical discretization problem. `alpha` is a
missing physical bridge.

The mismatch also changes the existing acquisition design. Ignoring timestamp
blur, the current one-unit upper delay gate contains `0.9999996` of the full
latent mixture when `sigma_T=0.2`, but only `0.54670` when
`sigma_T=2.76`; about 45.3% of the latter mixture lies beyond the gate. Its
four-sigma propagation horizon would also reach `11.04T0`, well outside the
time range for which the present periodic spatial box was validated. The
`alpha=1` branch therefore requires a redesigned time gate, a larger
nonperiodic spatial domain, and a new convergence study before inference.

## Fast profile over clock conventions

The reference value

```text
alpha_ref = 0.2 / 2.76 = 0.0724637681
```

is locked to the default `g=1` convention. A discrete profile evaluates all
81 shared combinations of information deficit, required record copies,
coherence tolerance, and hold time. Every convention resolves at every
coupling.

Keeping the numerical alpha fixed while changing the clock convention gives
the conservative envelope:

| coupling | minimum `sigma_T` | median | maximum |
| ---: | ---: | ---: | ---: |
| `g=0.5` | `0.25507` | `0.40000` | `0.62029` |
| `g=1` | `0.12754` | `0.20000` | `0.31014` |
| `g=2` | `0.06377` | `0.10000` | `0.15507` |

This envelope no longer preserves the calibration condition at `g=1`.
Operationally, changing the definition of `tau_stab` also changes the
calibrated numerical alpha. The appropriate nuisance profile therefore
calibrates alpha separately for each candidate shared convention using only
the `g=1` control, then predicts the held-out couplings without refitting:

| coupling | minimum `sigma_T` | median | maximum |
| ---: | ---: | ---: | ---: |
| `g=0.5` | `0.39813` | `0.39886` | `0.40000` |
| `g=1` | `0.20000` | `0.20000` | `0.20000` |
| `g=2` | `0.10000` | `0.10000` | `0.10099` |

The full held-out spans are only `0.469%` at `g=0.5` and `0.990%` at `g=2`.
The absolute profiled alpha still ranges from `0.04673` to `0.11364`; its
numerical value is convention dependent. The held-out width ratios are stable
because the same threshold convention changes all three latencies almost
proportionally in this rate model.

This is a deterministic discrete convention profile, not yet a likelihood
profile from empirical response counts. It demonstrates that a calibration
control can absorb most of the clock convention while leaving sharp held-out
predictions in the current analytic record model.

```bash
MPLCONFIGDIR=/tmp/quantumtoy-matplotlib PYTHONPATH=src/quantumtoy \
  .venv/bin/python \
  src/quantumtoy/analysis/debug/run_trf_alpha_convention_profile.py
```

If the record interaction has a calibrated physical rate `G` in inverse
seconds, its dimensionless rate in the spatial clock is `g=G T0`. The record
latency is then `2.76/G`. A common model must specify or measure `G` and must
predict `alpha`; setting `G=1/T0` merely aligns the two dimensionless rate
units and does not determine `alpha`.

## Prediction status

The implemented result is a falsifiable conditional prediction:

1. choose a particle and one physical geometry or velocity scale;
2. derive `T0=mL0^2/hbar`;
3. adopt the direct-delay closure, which gives `kappa_t=1`;
4. choose either the spatial width `0.2T0` or a record-clock width
   `alpha tau_stab`, with `alpha` declared before observing the test data;
5. test the complete joint `p(y,t)` law, including no-click.

There is no parameter-free numerical TRF prediction yet. The Schrödinger
clock `T0` is derived from the implemented dynamics. `kappa_t=1` is the new
direct-delay postulate. The dimensionless width still requires a TRF
derivation of `alpha` or an independent calibration followed by held-out
predictions.

Reproduce the scale table and figure with:

```bash
MPLCONFIGDIR=/tmp/quantumtoy-matplotlib PYTHONPATH=src/quantumtoy \
  .venv/bin/python \
  src/quantumtoy/analysis/debug/run_trf_physical_scale_prediction.py
```
