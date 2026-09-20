# Declared two-path record experiment

This is the finite-dimensional reference experiment used to implement stages
1–2 of the TRF-IT v0.2 validation programme. It is standard quantum mechanics;
it does not introduce a TRF probability law.

The path system uses the fixed pointer basis `|0>, |1>` and starts in

```text
sqrt(p)|0> + exp(i phi) sqrt(1-p)|1>,       0 < p < 1.
```

The weak intermediate probe has recorded outcomes `a=+,-` and Kraus operators

```text
M_a = sqrt((I + a kappa Z) / 2),            0 <= kappa <= 1.
```

This explicitly includes probe back-action. Summing over `a` retains the
unread measurement and is different from omitting it.

Each physical environment fragment is a qubit initially in `|0>`. Conditional
path rotations create

```text
|f_0,k> = cos(theta_k/2)|0> + sin(theta_k/2)|1>
|f_1,k> = cos(theta_k/2)|0> - sin(theta_k/2)|1>
theta_k(t) = (pi/2) [1 - exp(-g r_k t)].
```

The fixed rates `r_k`, fragment identities, X-basis readout, pointer basis,
information threshold, required copy count, coherence tolerance, and hold time
are part of the declared protocol. They are not selected after a sweep.
At large `g t`, the two conditional states become the orthogonal X states and
each fragment can carry one independently readable copy of the same branch
label. Redundancy counts copies; their information values are not added and
called new branch information.

For every fragment, the code reports the mutual information `I(B:Y_k)` obtained
by the declared X readout and the Holevo quantity `chi(B:F_k)`. The former is
the accessible information under this readout restriction; the latter remains
an upper bound. It also reports `I(B:Y_1,...,Y_N)` for the combined record.

The residual system coherence is the normalized l1 coherence in the fixed path
basis,

```text
C_l1(t) = 2 |rho_01(t)|.
```

For the balanced default preparation this has the independent analytic value

```text
C_l1(t) = sqrt(1-kappa^2) product_k cos(theta_k(t)).
```

A sample is stable only if the predeclared redundancy and coherence criteria
both pass. The existing event-anchored clock then requires an uninterrupted
hold interval. Failure to establish one inside the simulated window is
reported as unresolved.

Every run reads all fragments and makes a terminal system measurement with
outcomes `+`, `-`, and `no_click`. The saved array is the complete joint
distribution

```text
p_Q(a, y_1,...,y_N, d | g, t),
```

including the state updates associated with each instrument. In the array,
the fragment record axis is the integer whose fixed-width binary
representation gives `(y_1,...,y_N)`, with `0` and `1` denoting the `+X` and
`-X` outcomes respectively. The final axis is ordered `+`, `-`, `no_click`.

Summing over the terminal outcome is invariant under the terminal setting.
This supplies the quantum reference distribution and the physical record
metrics requested by the document. A separate, admissible `p_theta` is still
needed before the experiment can predict distinct TRF statistics.

The temporal-response study uses a separately prepared ensemble with Z-basis
fragment readout as a quantum eraser. It retains all fragment outcomes and
does not replace this X-basis ensemble when defining the stabilization clock.
See `trf_response_measurement.md` for the two-ensemble protocol.
