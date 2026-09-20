# TRF-IT v0.2 validation

Run from the repository root with the project dependencies installed:

```bash
PYTHONPATH=src/quantumtoy MPLBACKEND=Agg python -m unittest discover -s tests -v
```

With this checkout's virtual environment, replace `python` with
`.venv/bin/python`. No GUI, video generation, external data, PDF file, or new
test dependency is required. The analytic regression values are recorded in
the tests from *An Information-Theoretic Formulation of the Thick Reality
Front*, version 0.2, September 2026.

| File | Checks |
| --- | --- |
| `test_quantum_reference.py` | Section 4 joint table and ensemble recovery; unread versus absent instruments; no-click and loss completeness; impossible and rare records; scalar rank-one conditioning; weak-probe back-action; dissipative propagation and adjoint effects; later-setting marginals; singlet marginals and correlations. |
| `test_emix_reference.py` | Section 5 arithmetic versus geometric mixture; posterior component weights; full operator phase invariance; existing scalar, spinor, and entangled Emix builders; epsilon error bound and impossible-event rejection. |
| `test_record_environment.py` | Declared two-path preparation; weak-probe back-action; controlled physical memory fragments; complete probe–memory–detector joint records; restricted accessible information and Holevo bounds; residual coherence; redundancy and cross-coupling stabilization clock. |
| `test_record_sensitivity.py` | Shared operational-threshold grid, monotonic threshold behavior, unresolved cases, and event-clock time-sampling convergence. |
| `test_stabilization.py` | Section 8 monotone rate example at several couplings and sampling resolutions; already stable records; transient stability and hold-endpoint failure; insufficient observation; combined criteria; irregular sample times. |
| `test_trf_candidate.py` | Minimal half-Gaussian dephasing candidate; exact quantum limit; positivity, normalization, earlier-record marginals, observable contrast, and synthetic Fisher identifiability with nuisance parameters. |
| `test_trf_inference.py` | Robust shot-allocation optimization, independent calibration priors, complete-record multinomial sampling, nuisance and clock-convention profiling, likelihood-ratio recovery, and null/signal Monte Carlo summaries. |
| `test_trf_response.py` | Independent temporal-response width recovery, X-record versus Z-eraser ensembles, half-rise convention, universal-alpha and free-width fits, and held-out coupling predictions. |
| `test_measurement_guided_kernel.py` | Physical-time sigma wiring into the spatial measurement-guided theory; canonical and compatibility registry names; time-step, stride, and truncation-horizon convergence. |
| `test_spatial_response.py` | Complete pre-click detector/no-click laws, exact worldline null, fixed and absent normalization controls, baseline-corrected cross-grid width recovery, and spatial `dt`, grid, and horizon convergence. |
| `test_spatial_effect_measurement.py` | Full spatial POVM positivity and completeness; complex operator coherences and global-phase invariance; complete no-click laws; exact `lambda=0` null; injected-width recovery; visualization-frame consistency; temporal quadrature, grid, and horizon convergence. |
| `test_trf_conditioning.py` | Existing simulation sampling, trajectory geometry, and propagation regressions. |

## Reference API

`analysis.quantum_reference.joint_record_probabilities` accepts a density
matrix, an instrument, later effects, and an optional intervening channel.
Each instrument outcome is a sequence of Kraus operators; the channel is
another sequence of Kraus operators. The result has axes `[a, R]`. The
instrument, channel, and final outcome set must each be complete. Missing
outcomes are errors, not a reason to renormalize a surviving subensemble.
The implementation supports finite square operators, including the two-state
reference and the two-party singlet tests.

`analysis.spatial_effect_measurement` extends the same operator discipline to
a compact 2D lattice. It constructs a complete terminal detector POVM,
unitarily propagates its full complex effects backward, and mixes unresolved
delays arithmetically. The faster profile path evaluates the identical effects
in the forward picture; a regression test requires both calculations to
agree.

`mix_record_effects` mixes one record's effects using declared prior weights.
All effects must refer to the same earlier time and the weights must be
state independent. `rank_one_posterior` implements the scalar special case
and optional numerical floor; general probes use the operator model.

## Meaning of the Emix regression

The existing `build_Emix_from_phi_tau` intentionally sums amplitudes. Its
phase-sensitive result is tested as a contrasting example: it is not the
arithmetic effect mixture of section 5 for unresolved classical alternatives.
This test passes when that distinction is demonstrated; it does not certify
that builder as a v0.2 effect implementation.

`build_Emix_density_from_phi_tau` is invariant to each component's global
phase and matches the diagonal reference mixture in the tested fixture. This
does not establish a full operator instrument for the simulation's detector
seeds, time lookup, state-dependent guidance, or absorbing boundaries. No
existing guidance dynamics or Emix builder is changed by this test suite.

## Clock and physical scope

`analysis.stabilization.sampled_stabilization` takes an explicit Boolean
stability predicate. The first time sample is the event origin. It returns
the onset, the confirming sample time, and the latency of the first passing
run spanning the hold time. Confirmation requires every sample in that run
to pass, including the confirming sample. It does not interpolate between
samples. `None` means the finite observation window has not established
stabilization, not that the physical latency has been proven infinite.

The isolated rate example assumes the document's analytic information profile
and that the other stability criteria already hold. It checks the clock
against `log(1 / delta) / gamma` with a one-sample onset error bound.

`analysis.record_environment` now supplies the physical reference model for
the next validation stage. It computes the information obtainable through a
fixed X measurement on each declared qubit fragment, the corresponding Holevo
upper bounds, residual path coherence, redundancy, and the complete joint
record probabilities. It applies the same thresholds and hold time across the
coupling sweep. See `paper/record_environment_model.md` for its equations and
conventions. This is a small exactly solvable memory model, not yet the spatial
double-slit environment in the main simulator.

The no-signalling tests verify selected quantum-reference protocols, not a
general proof or the admissibility of the existing TRF dynamics. A shared
width convention and a minimal candidate joint law are now declared in
`paper/trf_minimal_candidate.md`. They remain a phenomenological choice rather
than a derivation. The synthetic Fisher study assesses local identifiability;
it is not empirical evidence or an estimate from observed data. Passing these
tests is reference validation, not evidence of new physics.

`analysis.trf_inference` treats the stabilization-convention scale in two
ways. The design Fisher matrix profiles it continuously, exposing local
lambda–clock degeneracy. The recovery study profiles the finite set of shared
threshold conventions used in the sensitivity report. These uncertainties
answer different questions and their quoted lambda errors need not agree.

`analysis.trf_response` supplies the complementary direct-width protocol. It
uses separate preparations: X-basis fragment outcomes determine the physical
record clock, while complete Z-basis quantum-eraser outcomes preserve the
phase contrast used to fit `sigma_T`. A calibration coupling defines `alpha`,
and joint likelihood-ratio fits test the resulting held-out predictions while
propagating calibration uncertainty. The present counts are synthetic.
