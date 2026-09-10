# Derivative-enhanced token potentials and the remaining additive restriction

2026-09-11. NEXT62/pass51, empirical report
[here](../reports/grain_vector_mechanisms/HERMITE_TOKEN_POTENTIAL_REPORT_V1.md).
This note concerns a learned numerical reader of frozen model H16 states, not a
new personality factor model or a universal statement about natural-language tokens.

## 1. One scalar with directional local features

Fix source U, center mu, coordinate SD s, landmarks l_m and sigma. For BODY token
t, z_t=(H_t U-mu)/s; p_t is the frozen BODY-order degree1..3 Legendre coordinate.
Let v_t=[z_t,p_t], c=(H_last U-mu)/s. We use

    k_m(v)=exp(-||v-l_m||^2/(2 sigma^2))
    psi_mj(v)=sigma*d k_m/d l_mj=(z_j-l_mj) k_m/sigma
    f(v)=sum_m b_m k_m(v)+sum_mj d_mj psi_mj(v)
    Lhat=b+a^T mean(z)+beta^T c+mean_t f(v_t).

For BODY coordinates,

    K_t=T*d Lhat/d z_t=a+grad_z f(v_t)
    d psi_mj/d z_a=k_m/sigma*[delta_aj-(z_j-l_mj)(z_a-l_ma)/sigma^2]
    ghat_t=(K_t/s) U^T/T
    tauhat_f=sum_t <ghat_t,D_ft>.

At a center, radial-kernel gradient is zero while the new derivative features
give an independently oriented local response. Every response still comes from
one scalar; cross-partials are symmetric and line integrals reproduce finite
scalar changes. GRAIN D is a fixed evaluation direction, not itself differentiated.
Setting d=0 exactly reproduces the frozen NEXT60 function. This is an explicit
finite Hermite-type expansion, not a claim that the basis is new mathematics.

## 2. Three different capacities

Source-normalized least squares uses equal total weight for each source author:

    J(theta)=||V theta-ell||^2+||D theta-k||^2+.01||theta_nonintercept||^2.

Expanded coefficient count is593 vs81 with4sources,449 vs65 with3. Nestedness
implies min J_new<=J_old at padded old coefficients; it does not imply monotonic
score loss, field loss or new-author performance separately. The observed source
field loss increases slightly while score loss and total penalty decrease.

With fixed relative SVD cutoff1e-10, the field column-space capacity is

    E_span=||P_col(D) k||^2, E_out=||k-P_col(D)k||^2,
    ||k||^2=E_span+E_out=1.

An augmented QR of[D,k] preserves this geometry and avoids a huge left-singular
matrix. All4 E_span=.260542/.217058, old=.039838/.028048. This is neither a
psychological information fraction nor a new-author guarantee. The all4 field
inverse coefficients have norms~3.19e4/3.55e4 and design condition~5.25e5/4.44e5.
Capacity, regularized estimability, and application utility must remain distinct.
Equal coefficient penalty after a basis extension is not the same prior over
functions. No parameter-count-matched superiority or basis-invariant norm is claimed.

## 3. A structural reason to inspect context next

For this entire additive-after-local-reading family, for fixed source reference,
fixed p, and independent H16 coordinates,

    d K_t/d c = 0,
    d K_t/d z_s = 0 for s != t.

These exact zeros hold however many local centers or Hermite terms we add.
The last token can change Lhat through beta, but it cannot change which BODY
directions influence that scalar. A family such as

    Lhat=b+beta^T c+mean_t f_q(z_t,p_t,c)

can represent nonzero mixed derivatives. A nonlinear outer reader of aggregate
features can also do this; the zero property is not a theorem about all pooling.
The current implementation uses positional coordinates, so it is not a plain
permutation-invariant bag-of-words algorithm either.

This is an input-coordinate claim after the neural prefix: H16 already contains
context. It does not show that real texts have independent words, or that query-
last is necessarily the dominant missing input. NEXT63 tests exactly that missing
dependence with a small source-only native state intervention before choosing a
context-dependent extractor. Do not diagnose a specific native pathway from the
mere existence of an architectural zero.

## 4. Link to the user's same-material interface

The same span has both a GRAIN direction D_f and a query-conditioned numerical
response tau_qf. NEXT62's confirmation/F03 allocation response r=.881 is one
observed connection, alongside negative/small cells. This enables source-addressed
descriptions of which expression changes a model reads, not an automatic
psychoanalytic/ego-state interpretation. The
[same-material interpretation contract](GRAIN_SAME_MATERIAL_INTERPRETATION_THEORY_V1.md)
still separates numerical evidence, candidate explanatory coding, and independent
psychological/behavioral evidence. No new clinical classifier is fitted here.

## 5. Prior mathematical tools and our adaptation

- [Chmiela et al., 2017, GDML](https://arxiv.org/abs/1611.04678): derivative-domain
  learning of conservative molecular force fields motivates preserving scalar/
  gradient compatibility. Our finite Gaussian token expansion is not their
  molecular Matérn estimator and does not assert psychological energy conservation.
- [Czarnecki et al., 2017, Sobolev Training](https://arxiv.org/abs/1706.04859): values
  and derivatives can jointly teach a function. That principle motivates our
  native score+BODY-gradient targets; it is not a theorem that this dictionary
  will generalize from4authors.
- [Zaheer et al., 2017, Deep Sets](https://arxiv.org/abs/1703.06114): set-function
  representation motivates distinguishing local feature aggregation from a full
  nonlinear set reader. The specific mixed-partial zeros above are derived from
  our restricted additive formula, not a limitation proved for all Deep Sets.

Sources checked2026-09-11. Standard mathematical principles are the tools;
the project-specific evidence is how compatible readers behave on fixed same-
author GRAIN directions and personality-query-conditioned model states.
