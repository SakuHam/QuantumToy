# Dynamical history selection, detection and memory loss

## Status and scope

This is a specified **candidate open-system instrument**, not a derivation of
unique dynamics from the temporal envelope. It changes the propagated quantum
state and the first-detection distribution. It is implemented in
`analysis/history_dynamics.py` and used by the interactive clock demo.
It does not silently replace the older nonlinear guidance theory or its
terminal-POVM likelihood library. Those are different models and their old
validation results do not validate this candidate.

The construction uses ordinary completely positive quantum instruments and
quantum trajectories. Background: [Preskill, quantum channels](https://www.preskill.caltech.edu/ph219/Ph-CS-219A-Slides-2020/Ph-CS-219A-Lecture-4-Channels.pdf)
and [Castin, Dalibard and Mølmer, wave-function approach](https://arxiv.org/abs/0805.4002).
These establish the mathematical framework, not the new clock, pointer or
memory hypotheses used here. The public demo is self-contained.

## The assumptions that turn an envelope into dynamics

1. The wavepacket follows the existing 2D Schrödinger Hamiltonian with the
   real double-slit potential and periodic spatial box.
2. A state-independent clock schedules at most one upper/lower-half-plane
   projective measurement. The pointer is the sign of y at the selection
   time, not a claim that a trajectory passed a particular slit. Projecting
   the entire half-plane is an idealized, spatially extended instrument.
3. The selected label is irreversible. Subsequent Schrödinger motion can
   cross y=0; the historical label does not change with the later position.
4. A detector continuously removes probability at rate Γ times its x gate.
   The first absorbed event is terminal and has a y-bin and a time.
5. Classical records can subsequently be erased. Erasure does not feed a
   force back into a particle that has already been measured or absorbed.
   The demonstration uses the same lifetime parameters for path records and
   detector records; this is a declared simplification, not a measured fact.

Different pointer operators, spatially local monitoring, feedback Hamiltonians
or nonunitary fundamental dynamics would be different models with additional
parameters. Multiplying a wavefunction by the fading envelope and then
renormalizing would not specify such a model: it hides the missing outcomes.

## Clock and finite preparation boundary

Time t=0 is wavepacket preparation. Let σ>0 be the temporal width, c the center
of the formation clock, and q=1−exp(−λ). We explicitly prepare the particle
**without an earlier selected record**. Consequently the clock CDF after
preparation is

    F(t) = [Φ((t−c)/σ) − Φ(−c/σ)] / [1−Φ(−c/σ)],    t ≥ 0.
    Q(t) = 1 − q F(t).
    h(t) = q F′(t) / Q(t).

Q is the probability the selection clock has not fired. It includes the
1−q branch that never undergoes this extra selection. Hence a plot of Φ alone
is not the actual selected fraction. For finite λ, full eventual selection
does not hold for the entire ensemble; each **selected** history is fixed.
The λ→∞ limit removes the never-selected fraction.

In the demo, c = t_start + σ/α and the default t_start = −2 T0, giving
c=0.76 T0 for σ=0.2 T0, α=0.2/2.76. This relative preparation/clock timing is
an explicit new control, not a derived parameter. Conditioning on no
preparation-time selection is essential when c<0. Small g with the same
t_start can move the front well beyond the 0–2 T0 acquisition window; it
therefore has almost no extra path-measurement effect inside that window.
This is different from extrapolating the older terminal delay mixture.

For time edges t_i the exact nonnegative clock masses are

    w_i = q [F(t_{i+1})−F(t_i)],
    w_none = 1−q F(T),
    Σ_i w_i + w_none = 1.

The unobserved late tail is retained, not normalized into the observed window.
The implementation uses log normal-survival functions to avoid subtracting
nearly equal CDFs. Selection inside each bin is represented at its midpoint;
that quadrature approximation is tested by refinement.

## Coupled state and detector dynamics

Let P_+ and P_- partition the y lattice, with P_+ + P_- = I, and let D_j be
the Gaussian x detector gate restricted to y-bin j. Define D=Σ_j D_j.
Use separate density blocks for the unselected state ρ_u and the selected
histories ρ_+,ρ_-. They are unnormalized, accounting for detector survival.
The continuous-time limit is

    dρ_u/dt = −i[H,ρ_u]/ℏ − (Γ/2){D,ρ_u} − h(t)ρ_u,
    dρ_h/dt = −i[H,ρ_h]/ℏ − (Γ/2){D,ρ_h} + h(t)P_hρ_uP_h,
    dJ_j/dt = Γ Tr[D_j (ρ_u+ρ_++ρ_-)],       h ∈ {+,−}.

Thus Tr(ρ_u+ρ_++ρ_-)+Σ_j J_j = 1. An absorbed particle cannot form a later
path record. The code accounts for this when reporting the actually selected
fraction, which can be smaller than the clock's scheduled fraction.

On the augmented register u,+,−, the selection jump operators are
L_h(t)=sqrt(h(t)) P_h ⊗ |h><u|. There is no transition from a selected register
back to u. Before selection the unnormalized no-jump wavefunction has
H_eff,u=H−iℏ[ΓD+h(t)I]/2; after selection it has H_eff,h=H−iℏΓD/2.
The scalar clock term cancels from the normalized conditional state; the
implementation accounts for it through the schedule weights. Selection gives

    ψ → P_h ψ / sqrt(<ψ|P_h|ψ>).

The outcome is sampled with its Born probability. Detector jumps transfer the
particle to absorbing record states. One normalized wavefunction cannot
represent the unread mixed ensemble. The implementation instead propagates
all unnormalized branch kets and sums their outer products. An exact
finite-grid trajectory sampler returns either a normalized surviving branch
and its fixed label, or the absorbed outcome.

For each finite detector step Δt:

    K_no = exp(−ΓD Δt/2),
    K_j†K_j = 1_{y-bin j} [I−exp(−ΓD Δt)].

These obey K_no†K_no+Σ_j K_j†K_j=I exactly. Each integration interval uses
U(Δt/2), an optional projective selection, U(Δt/2), and this detector channel.
The timestamp is the interval midpoint. The map is positive and complete
on every grid, but the split-step propagation and event/time placement are
numerical approximations, not an exact continuum solution.

## Fading is aged from the birth of each record

For a record created at s and read at T_read≥T, define

    R(T_read−s) = exp(−[(T_read−s−a_keep)_+ / τ_fade]^β).

Its register becomes

    |j,s><j,s| → R |j,s><j,s| + (1−R)|erased><erased|.

This is a trace-preserving classical erasure channel. The past outcome is
fixed even when its locally readable copy is gone. In general accessible
history is a sum/integral over birth times,

    M(T_read) = ∫ R(T_read−s) dP(actual selection at s),

not Φ((T_read−c)/σ) multiplied by one common fading factor. The latter remains
only the old illustrative envelope panel. Detector records have a separate
birth distribution J_j(s), computed from the evolving wavefunction.

Finite efficiency and y blur act on first-detection probabilities. Timestamp
jitter scatters records between finite time bins; outside-gate probability
is kept. A dark record is added at most once per preparation when no genuine
record was produced. Memory loss is then evaluated using actual creation time,
before timestamp jitter. The final law includes

    Σ_{j,k} p(saved j,k) + p(erased record) + p(no accepted record) = 1.

The last category includes rejected out-of-gate timestamps when jitter is
nonzero. In the demo jitter is zero, so it is labeled “no record created.”

Erased and never-created records may be indistinguishable in an experiment
without an independent latch. In that case the likelihood must combine them;
the demo shows them separately as model accounting, not automatically
observable categories. Any ordinary no-click channel is their sum.

Local trace-preserving erasure of an external register leaves the particle's
unconditional reduced density operator invariant. Restoring interference by
decreasing its decoherence factor as a classical record fades would violate
this model. A quantum eraser or actual feedback interaction requires a
different, explicitly specified experiment.

## Parameters and numerical verification

### Forward Theory integration and visual playback

`theories/history_instrument.py` implements `HistoryInstrumentTheory`, registered
as `history_instrument`. It accepts a scalar wavefunction or a batch with shape
`(shots, Ny, Nx)` and uses the simulator's grid/potential/mass/ℏ. A live
trajectory is conditionally normalized by `dx*dy`; an absorbed trajectory is
exactly zero and stays zero. Ensemble density and current average over **all**
preparations, not only survivors. Branch wavefunctions are not coherently added.

At initialization, each trajectory samples the same truncated Gaussian clock
and the never-selected probability exp(−λ) as the analytical branch model.
Within a step, a due selection is placed at the midpoint and its upper/lower
outcome is sampled with the Born probability. The detector channel follows
the two unitary half steps. Its no-click result is normalized conditional on
survival; detection terminates that trajectory. The selected historical label
never changes even if the later wavefunction crosses the pointer boundary.
Independent quantum and readout random streams prevent classical memory
settings from altering quantum trajectories through random-number consumption.

The simulator can also have an existing complex absorbing potential
`V_real−iW`. The new Theory propagates `V_real` unitarily and accounts for W as
a separate terminal boundary/environment loss with rate `2W/ℏ`. In each
instrument step the combined hazard is `a(x,y)=ΓD(x,y)+2W(x,y)/ℏ`. The loss
effect is `1−exp(−a Δt)`; the terminal channel is chosen in proportion to its
local rate. These losses are not detector clicks. With W=0 this is the
reference detector instrument; with nonzero W it is a specified splitting
approximation requiring time refinement. The locked visual preset uses W=0.
The app rejects `USE_SCREEN_CAP=True` for this Theory to prevent an additional
screen absorber from duplicating the instrument's detector channel.

The specialized runner saves `TheoryStepResult.aux` events, bypasses the legacy
detector/terminal-click sampler, and continues classical readout after the
quantum acquisition window. It does not run the old Emix/backward or Bohmian
pipeline. A register-aware adjoint is not implemented: backward calls raise
`NotImplementedError` rather than substituting unitary Schrödinger reversal.

Electronics are the existing **window-level** response: at most one accepted
genuine record, otherwise a possible dark record. Dark records are assigned
after the entire acquisition and are not a continuous causal Poisson detector
model. True creation times are retained separately from jittered timestamps;
memory ages always use creation times. The reference memories and delayed
copy lifetimes are sampled once, then reused at every read, so loss cannot
randomly reverse. Successful reads are kept in the ideal external log.

The supplied HTML viewer plays computed Python frames at λ=0,1,4, with 512
preparations per case. Its scene menu does not solve new parameter settings
in the browser. Individual and ensemble panels have separately declared colour
scales; both compared ensembles share one scale. The display is quantized;
NPZ densities are float32 and the initial complex state/potential are archived.
After acquisition, the spatial panels explicitly show the final acquisition
state while the timeline advances through readout. The simulation event log
includes hidden selection and absorption events, not a claim of experimental
access to that entire history. Sampling error may exceed the change in total
click probability at this small sample count.
Illustrative trajectories are selected after acquisition to include a detected
particle and surviving upper/lower/unselected histories when available. They
are reconstructed deterministically from the original selection/absorption
records, with no fresh random draws, and checked against the original propagated
states. Their display selection does not affect the all-preparation statistics.

`run_history_instrument.py` provides the reference geometry and parameterized
exports. Its optional reference width law σ=0.2/g is the chosen baseline
calibration; the Theory itself accepts σ and c directly. The full app selects
the same Theory with `THEORY_NAME="history_instrument"` but uses the app's
geometry/packet and `HISTORY_*` parameters, so its numerical results need not
match the reference preset.

Verification includes a 12,000-trajectory comparison of unconditional clicks
and surviving density against `build_history_dynamics_basis`, a unitary null,
irreversible absorption, separate boundary loss, analytic versus sampled
memory readouts, digitizer gate rejection, and app routing. These verify the
implemented candidate; they do not establish its empirical correctness.

### Adjustable memory banks and repeated reads

`analysis/memory_banks.py` acts on the classical record law **before erasure**.
Every original event, including a dark record, can be copied at birth into
`n_ref` reference memories and `n_delayed` ageing memories. Zero copies is a
valid setting. Copying does not repeat the quantum measurement and cannot
recover an event that never produced a detector record.

Reference copies retain their records independently with probability `r_ref`,
assumed constant over the studied delay range. Their collective accessibility
is `A_ref = 1 − (1−r_ref)^n_ref`. This is an idealized stable archive, not a
prediction of a fundamental protection mechanism.

Delayed copies are read K times at `T_k = T_first + k δ`, k=0,…,K−1. Each has
one persistent lifetime drawn from the survival law R, measured from record
birth s. Reading is nondestructive, does not refresh that lifetime, and succeeds
with probability η if the record remains. Read failures are independent across
copies and attempts. Successful results are retained in an ideal external log.
The probability of obtaining a particular copy's record at least once is

    b(s) = Σ_k R(T_k−s) η(1−η)^k.

Survival must not be redrawn independently on each read: that would incorrectly
allow a dead memory to revive. Conditional on being alive at T_k it was alive
at all earlier read times, which gives the expression above.

With independent copy lifetimes, the delayed bank's unique-event log and last
read probabilities are respectively

    B_log(s) = 1 − [1−b(s)]^n_delayed,
    B_last(s) = 1 − [1−η R(T_last−s)]^n_delayed.

With a **shared lifetime within the delayed bank**, set
`η_bank = 1 − (1−η)^n_delayed`, then

    B_log(s) = Σ_k R(T_k−s) η_bank(1−η_bank)^k,
    B_last(s) = R(T_last−s) η_bank.

For zero delayed copies both probabilities are zero. In the shared mode,
additional copies can overcome read errors but cannot overcome the shared
erasure. With η=1 the first read already captures everything that this bank
can ever reveal: more reads do not improve its log and later reads may see
less. With η<1, retries can recover previously unread, still-existing records.

The two banks have independent failures. The union of the reference bank and
delayed read log has accessibility

    A_union(s) = A_ref + (1−A_ref) B_log(s).

Multiply these functions by the original unconditional record law, at each
true creation time, before summing. Unique events are never added across
copies or reads. The mutually exclusive outcomes are: in both archives,
reference only, read log only, original record unrecovered, and no original
record. They sum to one per particle preparation. “Unrecovered” includes read
failures and zero allocated copies as well as erasure. The plotted counts are
expectations for N preparations; changing N changes counts, not probabilities.

The demo uses zero timestamp jitter, so the time bins are creation-time bins.
For nonzero jitter, memory ageing requires true creation time as an additional
latent index; passing a timestamp-blurred marginal to this API is not valid.

Shared loss here concerns only the delayed bank. The reference archive and
external read log explicitly remain available. This is not universal erasure
of every physical trace, nor proof of a TRF effect. Memory settings leave the
wavefunction and original absorption law unchanged. Tests enumerate classical
copy outcomes and persistent-lifetime/read sequences independently; the browser
checker compares all joint laws with Python across both loss modes and limits.

### Propagation checks

The default numerical demonstration uses the existing double-slit potential,
48×48 sites, 100 intervals over 2 T0, Γ=4/T0, σ=0.2 T0, λ=1 and c=0.76 T0.
The history sliders specify a_keep=1.5σ, τ_fade=5σ, β=1.5. Readout is at the
end of acquisition plus an adjustable nonnegative wait. At fixed slider
ratios, changing σ also changes these physical memory lifetimes; that choice
is a demo convention, not a universal equality of the two mechanisms.

`run_history_dynamics_study.py` generates all instrument branches, reports
null, selection and memory-loss cases, and checks finer time and space grids.
The standalone demo embeds these branch distributions. Browser controls take
their positive mixture, so they genuinely update the surviving spatial density
and the detector p(y,t), without rerunning a solver in JavaScript. Detector
jitter is set to zero in this interactive view; the Python API and study also
support nonzero jitter. The older snapshot/delay heatmap is labeled separately.

Tests compare the propagated density against an independent dense operator
calculation, verify Kraus completeness, the unitary null, positivity, no
post-absorption selection, no recoherence on erasure, late-clock tail handling,
trajectory frequencies and time refinement. Numerical convergence is reported
separately from mathematical probability conservation. The periodic box,
sharp half-plane pointer and finite grid remain physical/numerical limitations;
this does not establish a realistic slit-local measurement apparatus or a
validated continuum experiment.

Run:

```sh
OPENBLAS_NUM_THREADS=1 python src/quantumtoy/analysis/debug/run_history_dynamics_study.py
python src/quantumtoy/analysis/debug/build_trf_clock_demo.py
PYTHONPATH=src/quantumtoy python -m unittest discover -s tests -p 'test_history_dynamics.py' -v
```
