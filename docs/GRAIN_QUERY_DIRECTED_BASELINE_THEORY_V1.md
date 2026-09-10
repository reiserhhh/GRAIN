# From proximity to action: a query-directed conditional baseline

2026-09-10, twentieth finite logic pass, NEXT32. Native construction `c70332e`,
analysis first frozen `33966d9`, inherited plan in `e174ae8`. This is a positive
mechanism-development result, not a new human personality accuracy benchmark.

## 1. The missing reading is now partly executable

Four fixed old N reference authors supply source endpoints and Jacobians. The
other20 old N authors are evaluation. The same4-source context kernel has score
r=-.050; equal averaging of source affine response functions gives r=.727
[.443,.885], RMSE=.011660 versus kernel .015616. The paired score MSE reduction
versus kernel is about44.2%, with a pointwise fixed-prediction bootstrap interval
below zero. Compared with source mean, the MSE interval narrowly includes zero.

This comparison changes the information supplied to construction: gradients
contain much more information than four endpoint values. It is a mechanism
extraction experiment, not a claim of equal-information supervised-learning
efficiency. No held author endpoint or human label fits any candidate.

The target remains z0: all twelve comments collapsed to their own means, the
retained states preserved, P/Q fixed to the old anchor. The upstream block16
encoder remains. The final numerical reader does not generate reasoning text.

## 2. Conditional chain rule

Let M_j be the hidden mean repeated at every position in comment j and R_t the
state at retained role t. Source r has its own fixed layout L_r:

    z_r = centered_logits(F_suffix(M_r,R_r;L_r,PQ0)),
    J_M[r,j] = sum_{t in comment_j} partial z / partial H_t,
    J_R[r,t] = partial z / partial H_t.

The sum in J_M is forced by the chain rule, not a weighting preference. Averaging
those token derivatives would change the represented intervention. A known
linear observer and actual source central differences test this implementation.
Eight vector checks (two epsilons across four references) have80/80 matching
candidate signs; maximum relative vector error3.50%, and at eps=.002 at most1.25%.

Define each source-local affine function, then their equal average:

    f_r(c) = z_r + J_Mr:(M-M_r) + J_Rr:(R-R_r),
    zhat(c) = mean_r f_r(c) = b + Jbar_M:M + Jbar_R:R,
    Bhat(c) = S_original(zhat(c)).

No outcome-dependent source weights or intercept correction was added. All
derivatives and coefficients are frozen before evaluation. Source and target
role grids align; layouts still vary. The formula averages functions originating
at different L_r, so cross-layout operation is an observed local transfer, not
the mathematical assertion that L never matters.

## 3. Shared reading and individual positions are complementary objects

In the declared M/R coordinates, the four flattened candidate-Jacobian fields
have74.7% common-mean field energy; R alone80.6%, M46.0%. These are parameter
geometry ratios, not fractions of human personality variance. The scalar-score
gradient field has70.7% common-mean energy and mean source cosine.639.

The flattened candidate-Jacobian Gram matrix is a comparison between operators.
It is NOT the input active-subspace matrix sum J_r^T J_r. The scalar-score
gradient rows do define an empirical input-gradient covariance, with rank at
most4 from four sources, not a proved population dimension.

This fits the useful principle behind
[Constantine & Gleich's active-subspace analysis](https://arxiv.org/abs/1408.0545):
input directions should be defined relative to the response of interest. Their
sampling guarantees are not transferred to our four fixed references. Local
gradient-based response linearization also has precedent in
[Simonyan, Vedaldi & Zisserman](https://arxiv.org/abs/1312.6034); here the inputs
are conditional hidden text states rather than images, and transfer is measured.

## 4. A reader-induced geometry that can connect to GRAIN

For the frozen numerical readout Jbar and a declared positive-semidefinite
output weighting W, define

    d_q(c,c')^2 = ||W^(1/2) Jbar(c-c')||^2
                = (c-c')^T Jbar^T W Jbar (c-c').

This is a pseudometric: it can assign zero distance to distinct text states that
the selected reader cannot distinguish. With centered ten-candidate outputs,
the exact readout rank is at most9, not a claim that personality has nine factors.
It measures similarity of the represented query response, not semantic or human
similarity in general. No large dense input metric needs to be materialized.

This is the pullback construction in a possibly rank-deficient setting. In
contrast, a nonsingular Riemannian metric requires additional rank assumptions;
see [Arvanitidis et al. (2022)](https://proceedings.mlr.press/v151/arvanitidis22b.html).
We do not infer a human manifold or recover geodesics merely by writing J^T W J.

GRAIN supplies group-referenced expression positions/directions on the same
material; this reader supplies a query-specific weighting of their changes.
Earlier GRAIN Cw/position-specific native tangent results stay valid. The next
empirical bridge should compare those declared directions, not rename every
high-derivative token as a personality factor.

## 5. Averaging readers preserves a reader-position covariance

Write c_r=(M_r,R_r). Pure algebra gives

    mean_r [z_r+J_r(c-c_r)]
      = zbar + Jbar(c-cbar) - C_Jc,
    C_Jc = mean_r [(J_r-Jbar)(c_r-cbar)]
         = mean_r [J_r(c_r-cbar)].

C_Jc is a candidate-logit vector, with separate M/R contributions. It is not a
personality dimension or a noise estimate. In the four source references its
RMS is.052493 (M .039713, R .047584); this re-expression reproduces all20 outputs
within2.45e-15. The pairing of source position and source reading matters even
when both have an average. This is an actual connection to earlier GRAIN local
co-variation/averaging logic, now inside a compiled response function.

No source-centered replacement was evaluated or promoted. Dropping C_Jc would
be a different estimator, not a harmless recentering of the same method.

## 6. Variation and calibration are different tasks

The affine reader has worse whole-logit RMSE (.098 versus kernel .083), yet its
author-centered logit error is about half the native variation, versus1.061
for kernel. The exact identity is

    mean_u ||zhat_u-z_u||^2
      = ||mean_u(zhat_u-z_u)||^2
        + mean_u ||(zhat_u-zhatbar)-(z_u-zbar)||^2.

Affine logit offset/variation MSE=.006689/.002923; kernel=.000940/.005916.
The score projection changes which logit deviations matter. This explains why
whole-vector error alone would miss useful recovery of author differences.
The evaluation means here only diagnose error; they do not calibrate predictions.

Next: [source-matched complete pair assembly](GRAIN_QUERY_DIRECTED_PAIR_ASSEMBLY_PLAN_V1.md).
Keep this positive baseline reader; test composition before new calibration grids.
