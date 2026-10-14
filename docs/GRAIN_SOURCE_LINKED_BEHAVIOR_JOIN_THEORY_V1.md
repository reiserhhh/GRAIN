# Source-linked behavioral joins: measurement construction V1

HM270, 2026-10-14 JST. This is a source observation-law construction, not a
new probability theorem or a prediction-accuracy promotion.

## Question That Changes the Mainline

Two comments can jointly change a personality-related behavioral reading more
than the sum of their separate changes. There are at least three explanations:
posterior normalization, a shared source-author relation, and the reader's
serialization/meaning mechanism. General nonadditivity does not distinguish
them. HM270 connects the first two explanations to the exact native evidence
already observed in HM263; it does not recollect answers or tune the instrument.

HM196 already studied replacement/depletion, HM176/209 already implemented
paired source laws, and HM267 supplied event-level continuous emissions. The
new interface joins those existing capabilities to the actual HM263 corners.
It is not a rediscovery of their underlying mathematics.

## Source Observation Contract

Let g be one of the 32 registered five-bit ENACO profiles, not an unobserved
human personality class. For each source author i, define the fixed emission

\[
L_{iv}(x)=\frac{1}{T_v}\sum_{t\in v}
\exp\{-\|x-h_{it}\|^2/(2\sigma^2)\}.
\]

The common Gaussian normalizing factor cancels because all compared emissions
use the same dimension and variance. Source author weights are uniform within
profile, and the profile prior is empirical, pi_g=n_g/386.

For the same two numeric evidence points x,z, compare

\[
J_g(x,z)=\mathbb E_{i\mid g}[L_{i0}(x)L_{i1}(z)],\qquad
I_g(x,z)=\mathbb E_{i\mid g}[L_{i0}(x)]
          \mathbb E_{j\mid g}[L_{j1}(z)].
\]

Both laws have identical one-event conditional marginals and the same profile
prior. J keeps one source-author index; I resets that index within the SAME
full profile. I therefore retains profile association; it is not a world with
all person information removed. The exact unnormalized difference is

\[
J_g-I_g=\operatorname{Cov}_{i\mid g}(L_{i0}(x),L_{i1}(z)).
\]

Conditional independence of emissions given a source author is an instrument
assumption, not a proven law of natural text generation. The fixed variance
is inherited, not selected against native answers.

For A in {J,I}, define the normalized profile reading and target mean

\[
P_A(g\mid x,z)=\frac{\pi_g A_g(x,z)}{\sum_k\pi_k A_k(x,z)},\qquad
\psi_{Aq}(x,z)=\sum_g b_q(g)P_A(g\mid x,z).
\]

Here b_q is the registered profile bit, not the author's continuous Big Five
score or a validated probability of human behavior. When r_g=J_g/I_g,

\[
\psi_{Jq}-\psi_{Iq}=
\frac{\operatorname{Cov}_{P_I}(b_q,r)}{\mathbb E_{P_I}[r]}.
\]

These are standard covariance and Bayes reweighting identities. They explain
why a positive source covariance need not produce a positive target change.

## Same-Evidence Mixed Action

The four actual corners 00,01,10,11 replace own/foreign evidence independently
in the two registered blocks. For either law,

\[
M_{Aq}=\psi_{Aq}(00)-\psi_{Aq}(01)-\psi_{Aq}(10)+\psi_{Aq}(11),
\qquad M_{Dq}=M_{Jq}-M_{Iq}.
\]

M_I can be nonzero even if each profile has only one source author, when J=I
and M_D=0. Multiplying independent emissions and then normalizing is already
nonlinear. The toy unit test exhibits this directly. Thus M_J or native
nonadditivity alone is not an author-linkage test.

Each source component is compared to the frozen native balanced behavioral
mixed action H_r on the same edges and input order:

\[
C^{A}_{qr}=\frac1{24}\sum_u M_{Aq,u}H_{r,u}.
\]

All signed 25 cross-target cells are retained, as is the native common-support
reading. C is an unnormalized contact, not Pearson r, AUC, a variance fraction,
a significance result, or a causal mediation estimate.

## Two Source Views, Not a Temporal Causal Claim

Pooled comments use all 24 source prototypes for both input blocks. Swapping
the blocks leaves both source laws and their mixed actions exactly unchanged
by symmetry. Native order-dependent action can still change their contacts.

Registered windows read the first input block against source prototypes 0-11
and the second against 12-23. This creates input-slot sensitivity. Its singleton
marginals differ from the pooled contract, so comparing the contracts does not
isolate pure temporal dependence or a person's psychological change.

GRAIN15 preserves the covariance of the nonorthogonal directions by using an
orthonormal basis of their span; full E5 is retained alongside it. Neither
space nor source view is selected from the resulting contacts.

## Empirical Decision

In E5/pooled comments, the additional source linkage has positive N contact
in both native orders (+.000240210,+.000023670), while E/A/C/O are negative.
E5/registered windows also has positive N contact in both orders, but C/O
change signs. GRAIN15/pooled N changes sign (-.000052530,+.000029765).
These descriptions share the same panel; they are not independent replications.

The result provides a local linkage-versus-normalization discriminator, not a
universal benefit of author linkage. The inherited profile shuffle also has
substantial linkage effects. Effect existence or norm alone is not target
specificity. The pooled source law's exact order invariance means that it
cannot explain the native order-odd component on its own.

## What Is New and What Is Borrowed

The new construction is a cold, same-event comparison of marginal-matched
source relations against an already frozen behavioral observation. Gaussian
mixtures, conditional covariance, Bayes normalization, and four-corner mixed
differences are mature mathematics. No new RKHS inverse, convergence proof,
tree reconstruction threshold, or psychology theorem is claimed.

The following sources inform the distinction rather than replace the test:

- [Kernel Bayes' Rule (2013)](https://jmlr.org/papers/v14/fukumizu13a.html):
  prior art for kernel-based conditional inference; its RKHS inverse estimator
  is not implemented here. Abstract checked.
- [Causal Abstraction (2025), sections 2-3](https://arxiv.org/html/2301.04709v4):
  correspondence of model mechanisms requires a specified alignment and
  interventions. A signed observational contact is not that correspondence.
- [Pinned OpenAI/math foundations, lines 166-215](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/build/sections/02-foundations.tex):
  product combination is conditional on an observation independence contract.
  This design lesson is used; the spin/tree theorem is not transferred.

## Next Scientific Decision

Before another experiment, inventory existing event/role qualification,
condition-dependent meaning, and actual vector-address interventions in
HM209/244/258/259/263. A useful next test must produce different predictions
for source-author linkage and reader serialization/qualification on the SAME
evidence, or construct a new independently usable measurement operation.
Another Gaussian width, rank, norm, donor, cutoff, or fidelity grid will not
answer that question. Do not fit the native behavioral answers to manufacture
an alignment. Choice, condition selection, state, and negative effects remain.

Canonical artifacts: `results/grain_math_response_v1/linked_join_r1`.
The source module is `grain_research/construction/math_linked_join.py`; the
cold CLI is `scripts/score_grain_linked_join_cold.py`. Upstream vectorization
still depends on the historical neural encoder. Original18/100/44 and the
full project goal remain OPEN.
