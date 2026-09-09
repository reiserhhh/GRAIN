# NEXT13: scalar curvature, component compression, and aggregation order

2026-09-10. Fixed before fitting this finite packet; exposed-source discovery.
This is a mechanism comparison, not a new benchmark or a parameter search.

## Decision

The eighth pass found a same-observer native-to-fixed T/F criterion difference
(.348 versus .201) despite output fidelity .911. A non-affine remainder alone
does not imply an omitted input: if `p=x` and `t=x^2`, the scalar already suffices.
Test whether the current gap is accessible from the same scalar, its material
distribution, or the three existing function components before obtaining new
hidden states. Keep the current release and previously retained readers unchanged.

## Prior work to retain, not repeat

| Existing packet | Established result | Different question now |
|---|---|---|
| M11 | Same24 authors/query: conditional anchor and function-transport remainder both have source links | Initially treated as a possible reference change; the cache reconciliation below establishes identical TF/N functions. Cohort-specific percentages still cannot transfer |
| M12-M13 | Query-attention baseline estimate locally useful for TF; changing aggregation improves N target fit without solving criterion linkage | Here test a scalar function of the current f, not another attention-baseline regressor |
| M16-M18 | State/position-conditioned action fields can carry different author links; quadratic functions have a parity restriction | Do not label scalar cubic success a new token interaction or a replacement for those fields |
| M22 | Same-vector trees are not universally better; N contextual prefix + Ridge retains more source association than E5 + Ridge | Here the input is only fixed f or M/C/Q, not a high-dimensional embedding or tree grid |
| M28 / NEXT11 | Nine query functions transport locally; source gaps differ by query and runtime | Use same FP32 observer and fixed source author membership |

## Cache and leakage boundary

- Two queries: T/F63 MBTI-only authors; N213 Big5-only authors. Three exact
  source cue-mask views each. Reuse the reading-level five author folds,
  seed20260910; keep the one previously excluded MBTI author out.
- Predictors: M29's frozen per-view `f`, `mean`, `configuration`, `final`.
  Targets: S9's existing FP32 native per-view score. Match author/query/view;
  replay their three-view means against the frozen source files. D1 BF16 is
  not the target. No text/embedding/model recomputation is needed.
- Construction reads only model outputs and fold metadata. No Y, bridge, MEPS,
  external API, lockbox, or model-weight training. Source self-reports are joined
  only after all predictions and model artifacts are hashed. All source labels
  have previously been exposed; this is not newly blinded confirmation.
- All scaling fitted in the training fold. Source authors never cross folds.
  Models and author data remain in ignored `results/`; public tables aggregate.

## Fixed seven arms

1. `raw_f`: original fixed function, no fitting.
2. `affine_author`: affine f-to-native function fitted to author means.
3. `cubic_author`: degree3 scalar polynomial fitted to author means.
4. `components_author`: linear readout of the three author-mean components.
5. `affine_view`: affine scalar fitted on views, averaged for each author.
6. `cubic_view`: degree3 scalar fitted on views, then author-averaged.
7. `components_view`: linear components fitted on views, then author-averaged.

Use standardized inputs; scalar cubic uses powers1/2/3, with training-only
feature standardization. Ridge objective is weighted mean square error plus
`.01*||beta||^2`, unpenalized intercept. Each author has total weight1; views
each weight1/3. No clipping, sign reversal, nonlinear-degree choice or alpha
search. Record out-of-range scores and training-range extrapolation instead.
These are native-score approximations, not calibrated personality probabilities.

The view contrast is necessary: `mean g(f_v)` generally differs from
`g(mean f_v)`. For one fitted cubic in standardized z,

`mean g(z) - g(zbar) = gamma2 Var(z) + gamma3[3*zbar Var(z)+mean((z-zbar)^3)]`.

Save this exact Jensen-gap account for the same fitted model. Do not confuse
that algebraic gap with the difference between two independently fitted models,
or with a real temporal state change (these are sampling views).

## Evaluation and branch rule

- Primary: matched-author OOF native MSE/MAE/Pearson, per-fold MSE gains,
  and paired1999-author-bootstrap differences. Training-fold mean is a separate
  MSE comparator. Teacher fidelity is the construction criterion, not a claim
  of human accuracy. Keep every arm, including negative ones.
- Freeze first, then report unflipped source Y Pearson (T/F AUC too), paired
  fixed-output intervals and exact signed residual accounts. Human Y cannot
  select an arm, transform, query orientation, or deployment promotion.
- Six fixed paired contrasts: cubic versus affine within each aggregation;
  components versus affine within each aggregation; view versus author for
  cubic and for components. Pointwise uncertainty, no training-uncertainty claim.
- Scalar gain supports an accessible link within the tested function family;
  component gain suggests information lost in scalar compression. View-only
  gain motivates distribution statistics. Failure of these fixed families is
  not a theorem that all scalar functions fail.
- NEXT14 is conditional: only revisit the existing conditional-anchor/transport
  caches if these simpler interfaces leave a decision-relevant native gap.
  No deeper-layer or teacher-response queue is authorized by this plan alone.

## Mathematical precedents and output

[Hastie & Tibshirani (1986)](https://doi.org/10.1214/ss/1177013604) separate linear
predictors from smooth coordinate functions; our degree3 map is a finite
parametric diagnostic, not an implementation of their full GAM algorithm.
[Szabo et al. (2016)](https://www.jmlr.org/papers/v17/14-510.html) analyze regression
from sampled distributions. It motivates retaining a material distribution,
but its consistency guarantees do not automatically apply to three correlated
views or a fixed small polynomial. The Jensen identity here is exact algebra.

Output: `results/grain_vector_mechanisms/logic_scalar_component_v1/` (new run
directory), explicit prediction freeze, private models/views/author predictions,
teacher metrics, criterion metrics, paired comparisons, Jensen account, aggregate
report, tests and local commits. Cache-only CPU, no long compute or new dependencies.

## NEXT14 amendment after NEXT13, before new conditional accounts

The seven-arm packet has now completed without a clear nonlinear/component
gain over its matched affine teacher control. Reconcile rather than repeat M11.
A model-only cache check finds all48 old anchored authors in the current panel,
24/query, with native error5.55e-17, fixed-PQ error1.11e-16 and reference-anchor
error2.22e-16. **For these two queries M28 reused the old function exactly.**
The earlier possible-reference-change assumption was incorrect; only the cohort
and newly fitted approximation differ. No criterion was read for this check.

The next finite account uses those already measured anchors and all seven
frozen NEXT13 author-OOF predictions, without refitting or neural calls:

`F = b(E) + (f-b_ref) + transport`.

For each frozen proxy p, apply the same affine residual operator Q_p to all
three terms, and retain their aligned total as a fourth contribution:

`F = P_p F + Q_p b(E) + Q_p(f-b_ref) + Q_p transport`.

All criterion covariances share `sd(F)*sd(Y)` on the same24 authors/query.
Whole-author1999 bootstrap recomputes the diagnostic projections. This asks
which **measured** conditional term remains outside each current proxy, not
which old raw term merely correlates with Y. The projector uses no Y for its
coefficients and does not yield a deployable recalibration. Public results are
aggregate; small-subset descriptive uncertainty cannot replace the full-source
estimates. This amendment changes no completed NEXT13 fit, parameter, or arm.
