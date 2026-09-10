# GRAIN: a common displacement can change the reading of individual differences

2026-09-10, NEXT55/pass44. Computational mechanism, not a named personality factor.
Evidence: [complete-function report](../reports/grain_vector_mechanisms/WHOLE_PATH_ASSEMBLY_REPORT_V1.md).

## 1. Same-material integration now has a functional test

For the same permitted author text and fixed query q, let X be its native H16
state and D_f its GRAIN direction. A frozen source function Psi supplies both
the substituted state and its derivative. With actual suffix S_q:

    Lhat_q(X)=S_q(Psi(X))
    tauhat_qf(X)=<grad S_q(Psi(X)), D Psi(X)[D_f]>.

This connects the user's two views without adding unrelated scalar scores:
GRAIN supplies a comparative perturbation; the query-conditioned mechanism
supplies the model's response to it, at exact token addresses. The integration
depends on both *where the material is represented* and *how it moves*.
The original0–16 prefix and22–end suffix remain actual neural dependencies.

NEXT53 established useful approximate attention22 key-component responses on
new8. NEXT55's complete S does not preserve that component's success. This is a
change of tested functional object, not a contradiction of the earlier result.
Component-specific readout error can compensate for another approximation;
one such match does not fix all residual, value, query and downstream channels.

## 2. Exact finite state/response decomposition

Write g_i=grad S(X_i) and d_i for two candidate states/directions. Then

    Delta_tau=<g_1,d_1>-<g_0,d_0>
             =<g_1-g_0,(d_1+d_0)/2> + <(g_1+g_0)/2,d_1-d_0>.

This identity requires no small-change assumption. Its first term is a changed
reader at a different state; the second is changed transported direction. Four
crossed contractions compute both and their interaction exactly. Crossed pairs
are not automatically derivatives of one compatible function.

NEXT55 new8 shows the first term dominates identity-to-joint response change:
direction changes~1.82%, but gradient norm drops to~20% and gradient cosine
to~.20–.25. Query-contrast r stays .775 for identity-state/joint-direction but
falls to .057 for joint-state/identity-direction. The tested loss is therefore
mainly state-conditioned downstream reading, not destruction of D_f itself.
These are candidate-state comparisons; they do not measure every intermediate
state on the original native path or identify a human psychological mechanism.

Where S is twice continuously differentiable, the reader difference can also be
written as the integrated Hessian action along the finite state displacement:

    g_1-g_0 = integral_0^1 Hess S(X_0+t Delta_X)[Delta_X] dt.

Thus a small tangent error does not control response error if relocating X
changes the reader strongly. One immediate bound is

    |Delta_tau| <= ||Delta_g|| ||dbar|| + ||gbar|| ||Delta_d||.

The finite decomposition is measured; the Hessian integral is an analytic
description, not a new empirically sampled Hessian map. A simple witness is
S(x)=tanh(x), Psi(x)=x+b: D Psi=1 for every b, yet the response is sech^2(x+b),
which can become nearly zero. No new noise assumption is needed.

## 3. A common term need not be irrelevant to who differs

For the frozen pointwise source map, with fixed layout and source features:

    Psi(X)=X+Z(X)B = X + 1 b_source^T + V_source(X).

The coefficient on the explicit constant feature is b_source. Its derivative
with respect to X is zero, whereas the downstream derivative need not be
translation invariant:

    D[S(X+b_source)] = grad S(X+b_source), not generally grad S(X).

This preserves an important GRAIN principle: a shared condition may govern how
individual departures are expressed/read. Removing something merely because
it is common, or treating a common shift as harmless, can both be mistaken.
The common offset here is a learned hidden-state coefficient, not an identified
topic, situation or psychological condition. Those levels must not be renamed.

Observed token-constant correction energy is99.95%. It measures the norm of each
material's token mean correction, not the contribution of b_source alone.
V_source can have a material-dependent mean; its cross term with b_source is
signed. NEXT56 tests the actual coefficient and remainder separately. Neither
the current results nor this identity establish that all common signal is noise.

## 4. Three quantities must be developed together

- Output origin: joint logit bias is -3.72/-2.71 for N/confirmation.
- Author-dependent scale: joint SD is .056/.057 vs native .904/1.039, despite
  score-order r=.548/.420. The preserved ordering is useful, but strongly compressed.
- Query-conditioned response: full GRAIN action r=.049/.022 after joint mapping;
  identity query contrast instead retains r=.774 with imperfect absolute amplitude.

These are different functional properties, not a single leaderboard number.
NEXT56 targets their mechanism, not a held-author recalibration. Error allocations
retain cancellation; projections above1 do not mean more than100% personality.

## 5. Methodological connection and next discriminating test

Model stitching evaluates whether one representation works when supplied to
another network segment, rather than only matching geometry.
[Bansal, Nakkiran & Barak (2021)](https://arxiv.org/abs/2106.07682) use fixed
networks connected by a trainable bridge;
[Csiszarik et al. (2021)](https://proceedings.neurips.cc/paper/2021/file/2cb274e6ce940f47beb8011d8ecb1462-Paper.pdf)
explicitly distinguish representation and functional similarity. These are
methodological precedents, not studies of our GRAIN/personality pathway.

Our contribution in this packet is the same-material, GRAIN-direction-specific
full-score/full-response comparison and crossed-reader localization. It supports
a targeted experiment on the source offset's role in functional compatibility.
See [NEXT56](GRAIN_SOURCE_OFFSET_MECHANISM_PLAN_V1.md). A successful scalar local
readout remains valuable even when it cannot be used as a full hidden-state substitute.

Psychological interpretation remains downstream of fixed scores, actual evidence
addresses and a declared construct/criterion. The existing
[same-material interpretation interface](GRAIN_SAME_MATERIAL_INTERPRETATION_THEORY_V1.md)
allows candidate dialogue interpretations without making them diagnostic scores.
