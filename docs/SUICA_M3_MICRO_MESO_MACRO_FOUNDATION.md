# GRAIN M3 Micro-Meso-Macro Foundation

Status: prospective mathematical contract, not sealed and not a psychological
claim.

Implementation note: the first V1 internal 60-seed run is retained only as a
matched-family engineering baseline. It directly shares the declared condition
basis between an additive generator and estimator, uses saturated condition
means for the nonlinear remainder, and does not implement the full event-level
\(K^S K^C K^X\) kernel below. It therefore does not satisfy this contract's P0
closure requirements.

## 1. Purpose

GRAIN currently mixes three kinds of objects:

1. author-process objects such as choice, stable position, and conditional
   response;
2. measurement objects such as reliability shrinkage, uncertainty, and the
   unresolved channel;
3. interpretation objects such as personality, state, emotion, or behavior.

M3 separates them:

\[
\text{microscopic process}
\xrightarrow{\mathcal C_m}
\text{mesoscopic author structure}
\xrightarrow{\mathcal E_m}
\text{estimated structure}
\xrightarrow{\Psi}
\text{macroscopic construct}.
\]

The first experiment tests only the first two arrows. The interpretation map
\(\Psi\) remains closed.

### Mechanism-atlas refinement

The event law is not assumed to connect to one author coordinate through a
single universal projection. The M3 mechanism-atlas discovery refines the
middle arrow into competing coarse-grainings:

\[
\mathcal N_u
\longrightarrow
\left(
\mu_u,\mathcal R_u,\mathcal K_u,\mathcal C_u,\mathcal P_u
\right)
\longrightarrow
\widehat{\mathcal A}_u.
\]

These objects respectively describe state distribution, conditional response,
temporal evolution, interactional coupling, and higher-order path memory.
They may coexist, but a combined feature vector does not by itself prove
mechanism decomposition. The discovery protocol and its remaining attacks are
documented in `docs/SUICA_M3_MECHANISM_ATLAS_THEORY.md`.

## 2. Microscopic object

For author \(u\), occasion \(o\), and event \(t\), let

\[
\mathcal N_u
=
\sum_{o,t}
\delta_{(o,t,O_{uot},C_{uot},S_{uo},\Xi_{uot})},
\]

where \(O\) is an expression opportunity, \(C\) is a selected or assigned
condition, \(S\) is transient occasion state, and
\(\Xi\in\mathcal H\) is a local vector-space response.

The declared factorization is

\[
K_u(ds',dc,d\xi\mid H_t,s,o)
=
K_u^S(ds'\mid H_t,s,o)
K_u^C(dc\mid H_t,s',o)
K_u^X(d\xi\mid H_t,s',c,o).
\]

This ordering is a model declaration, not a causal fact. Alternative orderings
must be treated as non-identification attacks.

Relative to a frozen reference kernel \(K_0\),

\[
\ell_u
=
\log\frac{dK_u}{dK_0}
=
\ell_u^{S}+\ell_u^{C}+\ell_u^{X}.
\]

The corresponding KL chain rule separates state, choice, and response
information under the declared factorization. It does not say that any channel
is personality.

## 3. Two-phase observation design

The synthetic closure uses two distinct phases.

- Free phase: conditions follow \(K_u^C\). This phase identifies selected
  exposure and transition structure.
- Fixed phase: conditions are assigned under a common design \(Q_0\). This
  phase identifies response under shared support.

Free selection does not substitute for fixed response coverage. Fixed response
does not estimate natural choice.

## 4. Mesoscopic projection

Define the author-relative response field

\[
G_u(z)
=
\mathbb E[\Xi\mid u,z]
-
\mathbb E_{U\sim P_0}[\Xi\mid z].
\]

In \(L^2(Q_0;\mathcal H)\), with a centered condition basis
\(\varphi(z)\), define

\[
a_u=\int G_u(z)\,dQ_0(z),
\]

\[
B_u^{\mathrm{proj}}
=
\arg\min_B
\int
\|G_u(z)-a_u-B\varphi(z)\|^2
\,dQ_0(z),
\]

and

\[
h_u(z)=G_u(z)-a_u-B_u^{\mathrm{proj}}\varphi(z).
\]

When the constant, linear, and nonlinear bases are orthogonal under \(Q_0\),

\[
\|G_u\|_{Q_0}^2
=
\|a_u\|^2
+
\|B_u^{\mathrm{proj}}\varphi\|_{Q_0}^2
+
\|h_u\|_{Q_0}^2.
\]

\(B_u^{\mathrm{proj}}\) is a global linear projection. It is not the same
object as the local Jacobian

\[
J_u(z)=DG_u(z)
\]

unless the response field is linear on the declared support.

The unresolved nonlinear field \(h_u\) is not noise. It is structure outside
the registered linear response family.

## 5. Measurement layer

At evidence resolution \(m\), the ideal squared-error estimator is

\[
M_u(m)=\mathbb E[G_u\mid\mathcal F_m].
\]

The existing reliability operator \(W_m\) is a finite-dimensional linear
approximation:

\[
\widehat G_u^{(m)}
=
\widehat\mu_0+
W_m(X_u^{(m)}-\widehat\mu_0),
\]

\[
R_u^{(m)}
=
(I-W_m)(X_u^{(m)}-\widehat\mu_0).
\]

Thus \(W_m\), \(R_u^{(m)}\), uncertainty, and refusal are measurement
properties. They are not author constructs.

## 6. Required invariants

The first battery tests the following operational invariants.

| Transformation | Required invariant | Non-invariant quantity |
| --- | --- | --- |
| Common orthogonal response rotation | pairwise distances, Gram matrices, singular values | coordinate coefficients |
| Common response translation | author-relative \(G_u,a_u,B_u,h_u\) | absolute reference origin |
| Weighted slice coarsening within author/occasion/condition | conditional means and projected mesoscopic objects | individual slice values |
| Factor rotation | subspace and operator spectrum | factor names and signs |
| Reference change | declared cross-reference contrasts | absolute author score |

Changing language, tokenizer, embedding model, or nonlinearly distorting the
space is not assumed invariant. Such transport requires an independently
estimated map.

## 7. Non-identification constructions

The battery must contain at least these attacks.

### 7.1 Same mesoscopic occupancy, different microscopic dynamics

For a fixed stationary distribution \(\pi\), the transition families

\[
P_\kappa
=
(1-\kappa)\mathbf 1\pi^\mathsf T+\kappa I
\]

share \(\pi\) but have different persistence. Therefore \(\pi_u\) alone does
not identify the microscopic choice process.

### 7.2 Stable-position/state alias

With one occasion, or state constant across all observed occasions,

\[
a_u+S_u
\]

is identified but \(a_u\) and \(S_u\) are not. Two planted worlds with the
same sum must produce identical observations and force refusal.

### 7.3 Rank-deficient representation

If an observation map \(P\) has a non-trivial kernel, then distinct response
operators satisfying

\[
P B_u=P B_v
\]

are observationally equivalent. No downstream estimator may claim to recover
the lost direction.

## 8. Two-phase microkernel V2 operationalization

The free phase instantiates an event-level joint state-choice chain:

\[
S_{uo,t+1}\sim K^S(S_{uo,t},\cdot),\qquad
C_{uo,t+1}\sim K^C_u(S_{uo,t},C_{uo,t},\cdot),
\]

The fixed phase independently assigns \(Z\sim Q_0\) under occasion-level
state \(S_{uo}\), then samples

\[
J_{uo,z,r}\sim K^J_u(S_{uo},Z,\cdot),\qquad
\Xi_{uo,z,r}=v_{J_{uo,z,r}}+\varepsilon_{uo,z,r}.
\]

Here \(J\) is a hidden discrete emission code and \(v_J\) is a hidden vector
prototype. The estimator receives neither object. The generator's nonlinear
condition functions are also hidden. Public estimator inputs contain only:

1. ordered condition sequences from independent train/test occasions;
2. fixed-condition vector readouts;
3. independent reference-population readouts;
4. public condition coordinates, \(Q_0\), support masks, and protocol version
   metadata.

Choice occupancy and ordered transition kernels are estimated separately.
Transition information is scored conditionally on the author's own marginal
occupancy, so a gain over shuffled sequences cannot be credited merely to
topic preference. Sparse author transition matrices are hierarchically shrunk
toward the population kernel; the shrinkage strength is selected by held-out
training occasions. This separates shared sequential structure from
author-specific transition information.

The response estimator does not receive the generator's hidden finite basis,
but this is only basis-blinded near-family recovery: Gaussian random-Fourier
functions and Gaussian RBF regression belong to the same kernel family.
Bandwidth and ridge strength are selected only by
leave-one-training-condition-out error. Recovery is scored on unseen
conditions and independent occasions. The fitted field is projected under
\(Q_0\) to obtain
\(a_u,B_u^{\mathrm{proj}},h_u\). These are public-coordinate projections, not
hidden generator coefficients.

Occasion state is estimable only in the within-author relative gauge:

\[
\widetilde s_{uo}
=
\bar \Xi_{uo\cdot}
-
\frac{1}{O_u}\sum_{o'}\bar \Xi_{uo'\cdot}.
\]

Absolute stable position and a one-occasion state remain observationally
aliased. V2 must refuse the state claim when only one occasion exists.

## 9. Required design-derived refusals

| Failure | Required result |
| --- | --- |
| no common condition support | refuse response field |
| rank-deficient public coordinates | refuse public projection |
| unknown missingness mechanism | refuse response comparison |
| representation-version mismatch | refuse cross-phase response comparison |
| reference-version mismatch | refuse absolute relative position |
| fixed phase not randomized | retain association, refuse causal wording |
| one occasion | refuse stable/state separation |
| dependent technical streams | refuse reliability interpretation |
| mixed-condition coarse blocks | refuse coarse-graining invariance |

These decisions derive from observable protocol metadata. They do not detect
hidden world type from generated data.

## 10. Current battery licenses

### 10.1 V1 matched-family baseline

The internal 60-seed V1 result may establish only known-basis additive
projection, basic refusal guards, and algebraic invariants. It is not an
event-level recovery result.

### 10.2 V2 discovery

The current V2 result is `PARTIAL`. It supports only:

1. event-level choice occupancy plus separable shared and author-specific
   transition information in the registered synthetic worlds;
2. public-coordinate response projection and within-author relative occasion
   state under shared support;
3. author response reidentification across independent occasions, while the
   null-author world stays at chance;
4. response geometry survives registered rotations and translations;
5. protocol metadata triggers the declared refusal interface.

Nonlinear remainder geometry correlates with planted truth, but general
nonlinear predictive improvement is not established: held-out incremental
\(R^2\) is positive only in the P3 nonlinear-heavy-tail world. P2 and P4 fail
that conjunction.

Numerical evidence and the independent ruling are recorded in
`reports/SUICA_M3_EVENT_KERNEL_DISCOVERY.md` and
`reports/SUICA_M3_TWO_PHASE_MICROKERNEL_INDEPENDENT_AUDIT.md`.
The unsealed next-stage design is
`docs/SUICA_M3_CROSS_FAMILY_CONFIRMATION_PROTOCOL.md`.

A discovery result cannot establish:

- that human text follows the generator;
- that any recovered object is personality;
- cross-language or cross-embedding invariance;
- clinical validity;
- construct validity;
- that all residual information is recoverable.

## 11. Promotion sequence

1. P0 synthetic recovery and falsification.
2. P1 multi-view real-text persistence without psychological labels.
3. P2 fixed-condition human feasibility using data not used for training.
4. P3 frozen anonymous macro modes across independent views.
5. P4 external behavioral or psychological anchoring.

The macro interpretation map remains closed until P4.
