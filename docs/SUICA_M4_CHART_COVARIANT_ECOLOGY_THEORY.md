# GRAIN M4-C.2 Chart-Covariant Opportunity Ecology

Date: 2026-07-31  
Status: V2 frozen `M4_C2_NO_GO_CHART_TRANSPORT`; V1 development NO-GO
preserved; psychological interpretation closed

## 1. Development problem

M4-B recovered a finite family of opportunity-selection, opportunity-creation,
response, return/recovery, history-gate, and compensation mechanisms. Its
conditions were planted category coordinates. M4-C discovered anonymous
condition relation geometry without reading the response, but transported
only a static conditional-response surface.

M4-C.2 tests the missing seam:

\[
\text{response-safe condition chart}
\longrightarrow
\text{complete registered M4-B operator family}.
\]

The target is not equality of coefficients across coordinate systems. It is
equality of physical predictions and operator actions on the same held-out
opportunities.

## 2. Frozen condition basis

Let \(r(z)\in\mathbb R^L\) be the selected M4-C landmark representation. On
the reference-calibration panel only, fit

\[
q(z)
=
\widehat\Sigma_r^{\dagger/2}
[r(z)-\widehat\mu_r].
\]

The ecology basis retains a separate mass coordinate:

\[
\phi(z)
=
\begin{bmatrix}
1\\q(z)
\end{bmatrix}.
\]

The constant coordinate prevents centering from deleting menu size. If an
equivalent full-rank chart is written as \(r'(z)=Ar(z)+b\), reference
whitening leaves only an orthogonal gauge on the identified support:

\[
\phi'(z)=T\phi(z),
\qquad
T=\operatorname{diag}(1,O),
\qquad O^\top O=I.
\]

This is the domain on which isotropic Ridge is covariant. Non-equivalent or
noninjective charts are not repaired by whitening and must be refused.

The implementation freezes chart family, source transforms, landmarks,
bandwidths, retained rank, whitening map, and a provenance hash. Reference
calibration and reference selection use disjoint authors.

## 3. Conditions as finite measures

For \(J\) candidate opportunities at event \(t\), define

\[
\nu_{u,t}
=
\frac1J
\sum_{j=1}^J
M_{u,t,j}\delta_{z_j},
\qquad
o_{u,t}
=
\int\phi(z)\,d\nu_{u,t}(z).
\]

External and author-generated sources induce separate measures
\(\nu^{ext}\) and \(\nu^{gen}\). The selected response condition is

\[
c_{u,t}
=
\begin{cases}
0,&Q_{u,t}=0,\\
\phi(z_{Q_t}),&Q_{u,t}>0.
\end{cases}
\]

Thus category names are unnecessary. Candidate identity is retained only as
the physical opportunity on which a query is made.

## 4. Registered ecology operators

The availability-restricted choice law is fitted in the frozen basis:

\[
P(Q_t=j\mid M_t)
\propto
M_{t,j}
\exp\{a_u^\top\phi(z_j)\}.
\]

The response transition is

\[
x_{t+1}
=
F_ux_t+G_uc_{u,t}+R_uh_t+\epsilon_{t+1}.
\]

The generated-opportunity hazard contains a condition surface, direct
persistence/recency, expression feedback, and a history-gated feedback
candidate:

\[
\begin{aligned}
\operatorname{logit}P(M^{gen}_{t+1,j}=1)
=\;&
b_u^\top\phi(z_j)
+\alpha_u M^{gen}_{t,j}
+f_u(d_t(z_j))\\
&+\phi(z_j)^\top V_u x_{t+1}
+g_u(h_t)\phi(z_j)^\top V_u^Gx_{t+1}.
\end{aligned}
\]

The implemented recency state is relation-based rather than a semantic
category label:

\[
r_t(z)
=
\max_{s<t}
\exp[-(t-s)/\tau]\,
k_{\mathcal M}(z,z_{Q_s}),
\qquad
d_t(z)=-\tau\log\max(r_t(z),\epsilon).
\]

Calibration fits the registered `base`, `return`, `feedback`, and `gate`
families. A separate mechanism-selection panel chooses regularization and
model family. The untouched evaluation panel is opened once.

## 5. Covariance and physical action

The local operators transform as

\[
\widetilde B^{C\leftarrow O}
=TB^{C\leftarrow O}T^{-1},
\]

\[
\widetilde B^{X\leftarrow C}
=B^{X\leftarrow C}T^{-1},
\qquad
\widetilde B^{O\leftarrow X}
=TB^{O\leftarrow X}.
\]

The loop

\[
L_u
=
B_u^{O\leftarrow X}
B_u^{X\leftarrow C}
B_u^{C\leftarrow O}
\]

therefore obeys

\[
\widetilde L_u=TL_uT^{-1}.
\]

Raw matrix elements remain coordinate-dependent. The primary estimand uses a
frozen physical query bank of held-out menus and condition replacements:

\[
\mathcal T_u(q)
=
\left(
\Delta P(Q),
\Delta P(M^{gen}_{t+1}),
\Delta x_{t+1},
\Delta o_{t+1},
\Delta o_{t+2}
\right).
\]

Oracle and discovered charts are compared through choice probabilities,
creation hazards, response impulse curves, physical loop kernels, return
curves, recovery curves, and author-relation geometry. Axes, coefficients,
and matrix cells are excluded.

## 6. Return and recovery

Return is defined in relation geometry:

\[
\tau^{return}_{u,r}
=
\inf\{\Delta>0:
d_{\mathcal M}(z_{t+\Delta},z_t)\le r\}.
\]

The implementation reports a curve at the frozen 10%, 20%, and 30%
reference-distance radii. Recovery follows registered condition-replacement
impulses:

\[
\Delta x_{u,k}
=
F_u^kG_u[\phi(z_b)-\phi(z_a)].
\]

Because \(G_uT^{-1}T\Delta\phi=G_u\Delta\phi\), the response impulse curve is
gauge invariant even when the chart coordinates differ.

## 7. Evidence roles

The fitting order is:

1. fit chart candidates on reference-calibration authors;
2. choose the chart from response-free geometry on disjoint
   reference-selection authors;
3. freeze chart, whitening, rank, landmarks, and hash;
4. transform mechanism conditions without reading paths or responses;
5. fit ecology candidates on mechanism calibration;
6. choose ridge and hazard family on mechanism selection;
7. refit on calibration plus selection;
8. evaluate physical actions once on the untouched panel;
9. open synthetic truth and run oracle/discovered comparison.

The oracle and discovered routes receive the same dynamic paths, query bank,
candidate models, and role split.

## 8. Registered worlds and attacks

The V2 battery contains:

- a null ecology on an identifiable chart;
- exogenous selection;
- endogenous source partition under the exact same realized union-menu,
  environment, and condition paths as the selection arm;
- endogenous creation that may expand the total opportunity set;
- source-rotated feedback;
- fast return and slow hysteresis with equal stationary menu rates;
- history-gated creation;
- selection/creation compensation;
- a noninjective condition alias;
- an observationally identical external/generated source alias;
- author leakage;
- response leakage;
- evaluation support shift; and
- source-topology mismatch.

In the matched pair, let \(U_{u,o,t,j}\) be one shared opportunity envelope.
The selection arm sets

\[
M^{ext,S}=U,\qquad M^{gen,S}=0,
\]

whereas the source-partition arm samples a response-dependent source mark
\(G\le U\) and sets

\[
M^{gen,P}=G,\qquad M^{ext,P}=U-G.
\]

Thus both arms have the same union menu element by element. Source partition
does not claim net opportunity creation; the separate expansion world carries
that estimand. The selected-condition distribution is not matched because it
is an outcome carrying the selection effect. It is reported as a diagnostic.
Fast and slow return worlds use a stationary-preserving transition so
persistence does not silently alter the menu marginal.

An additional gauge audit applies a shared random orthogonal transform to the
whitened relation coordinates, refits the ecology, and compares physical
actions. Response permutation must leave the selected chart and chart
features unchanged.

## 9. Identification and refusal

The operational chart route refuses when:

- provenance declares author or response fields;
- chart geometry is underresolved;
- evaluation conditions leave reference support;
- source topology is incompatible and cannot be represented locally;
- external and generated menu sources are observationally identical; or
- the observed mechanism is underresolved.

Condition alias is a different claim. In a truth-open synthetic audit, define

\[
S_\chi
=
\frac12 S_{\chi,\mathrm{choice}}
+\frac12 S_{\chi,\mathrm{hazard}},
\qquad
\Delta_{\mathrm{alias}}=S_O-S_D,
\qquad
R_{\mathrm{retain}}=S_D/S_O.
\]

Information loss is confirmed only if \(S_O\ge .50\),
\(\Delta_{\mathrm{alias}}\ge .20\), \(R_{\mathrm{retain}}\le .70\), and the
paired author-bootstrap lower 95% bound for \(\Delta_{\mathrm{alias}}\) is
positive. This is not an operational condition-alias detector: when the oracle
is unavailable, the cause of underresolution remains unknown.

The oracle route must recover the planted finite ecology before any loss is
attributed to chart discovery. A failed oracle is
`INVALID_WORLD_OR_ESTIMATOR`, not evidence against M4-C.

## 10. Claim boundary

A passing M4-C.2 battery can establish only that the registered finite
opportunity ecology has a response-safe chart-covariant formulation in
synthetic worlds. It can support conditions-as-measures, action-level
covariance, and explicit refusal under alias, leakage, and support failure.

It cannot establish:

- a unique coordinate system or named topic ontology;
- that PCA, Isomap, landmark count, dimension, or thresholds are theory
  primitives;
- that the finite candidate family is a complete ecology law;
- natural-text recovery;
- personality, emotion, cognition, diagnosis, or clinical meaning; or
- cross-language or population validity.

M4-D may start only after the fresh V2 oracle route is valid, discovered
action transport passes absolute and oracle-relative gates, condition-alias
information loss closes `8/8`, at least `6/8` repetitions pass every
substantive gate, and all operational refusal worlds resolve.

## 11. Development history

The V1 eight-repetition development run returned
`M4_C2_NO_GO_IDENTIFICATION`. Its mechanism transport was strong, but
condition alias resolved only `5/8` under an estimand-misaligned absolute
skill rule, and expected-rate calibration did not produce realized matched
menus. Those outputs are preserved under
`results/m4_chart_ecology_v1_development_no_go/`.

V2 is a new estimand with a fresh seed bank. Its protocol, thresholds,
development-only power calibration, and frozen source hashes are recorded in
`docs/SUICA_M4_C2_V2_CONFIRMATION_PROTOCOL.md`. A V2 result cannot
retroactively convert the V1 NO-GO into a pass.

## 12. V2 confirmation result

Across eight fresh repetitions and fifteen worlds:

- oracle mechanism macro-F1: `.9878`;
- discovered mechanism macro-F1: `.9647`;
- choice action geometry: `.7905`;
- creation action geometry: `.7736`;
- return geometry: `.8452`;
- recovery geometry: `.9765`;
- truth-open condition-alias information loss: `8/8`;
- source-alias refusal: `8/8`;
- null false-positive rate: `0`;
- basis and response invariance: `1`;
- matched condition, environment, and union-menu maximum difference: `0`;
- matched menu and entropy MMD: `0`; and
- matched outside-option gap: `.0104`.

The composite loop geometry was `.6519`, below the frozen `.70` gate. Only
`2/8` repetitions passed every substantive gate, below the required `6/8`.
The decision is therefore:

```text
M4_C2_NO_GO_CHART_TRANSPORT
```

This establishes that strong transport of individual physical actions does
not imply transport of their independently estimated plug-in product. It does
not invalidate the response-safe chart, condition-measure representation, or
the atomic action results. M4-C.3 separately audits physical edge composition;
M4-D remains blocked.
