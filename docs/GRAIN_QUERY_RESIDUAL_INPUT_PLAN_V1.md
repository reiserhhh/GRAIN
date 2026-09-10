# NEXT35: query-linked residual input directions, not another alpha sweep

2026-09-10. Frozen next after NEXT34 and its source-span analysis. Decision:
does the compressed-out content contain linearly usable query-reading information?
This tests the coefficient mechanism, not a new human personality benchmark.

## Inputs and immutable roles

- Reuse NEXT34's4 old authors/3 queries, frozen3-source/1-held input preprocessing,
  shared reader rank8, and content-only36-column heads W0. No new model calls.
- Context H/G/order packets are numeric; source gradients produce A=T J V^T.
  Held gradient is evaluator-only. Do not fit the residual subspace on held H,
  J, GRAIN directions, native effects, or questionnaires.
- Leave old source plans/code/assets immutable; add a new module/runner.
  Same scalar natural-state target, not NEXT32's mean-null candidate field.

## Exactly two equal-width extensions

For each source split, retain the exact original standardized Phi0 columns.
With frozen source center and orthonormal PCA rows P, compute unscaled
Rc=(H-hbar)(I-P^T P). Source weights remain1/(3T_u).

1. **Residual variance control:** fit8 leading directions in sqrt(w)Rc by
   randomized SVD (oversamples10,n_iter5,seed4400+fold); reuse for all queries.
2. **Query residual directions:** predict source coefficients with frozen W0;
   E=A-Phi0 W0. Compute C=sum w Rc^T E (1536x8), then its left singular vectors.
   Keep up to8 directions with singular value>1e-10 of the largest; if C=0,
   record rank0. Source rank determination is numerical, not held-based selection.
   Numerically reproject into P's orthogonal complement and orthonormalize,
   recording tolerance/rank and sign convention. No iterative PLS/boosting loop.

Source-center and scale the added coordinates (constant columns use scale1);
append to the unchanged36 base columns, giving44 when rank8. Refit all weights
on the same3 source authors with alpha=.01 and unpenalized intercept, using the
same weighted density loss. Retain old content and GRAIN arms as comparisons;
do not add GRAIN interactions to these two new arms.

This is a fixed cross-covariance-informed subspace comparison, motivated by
[multivariate reduced-rank regression](https://pmc.ncbi.nlm.nih.gov/articles/PMC3444519/).
It is not a claim to implement the full estimator from that paper or the optimal
subspace among all nonlinear functions. Matching the number of added directions
separates generic extra capacity from useful query-linked residual selection.

## Outputs and decisions

- Per query/factor: source fit, held coefficient error, gradient error, native
  action r/SSE-to-zero/RMSE/RMS/signs. Keep factor-centered author r as well as
  pooled r;4 reused authors are not a population significance test.
- Report added coordinate source variance, cross-covariance singular values,
  orthogonality to P and held coordinate ranges/source scales.
- Improved source and held actions specifically with query directions supports
  a usable target-linked residual channel. Do not rename that channel by a
  questionnaire term simply because its teacher query has that name.
- Source gain only identifies available in-source information and transport
  loss; inspect the current numeric support before collecting more people.
- If neither extension helps source fit, move from isolated H_t to declared
  wider-context/nonlinear inputs rather than keep enlarging linear ranks.
- If variance-only helps equally, retain the simpler explanation of discarded
  linear input information; no need to claim query supervision is necessary.

## Resource bound and checks

At most4 residual PCA fits,12 small cross-covariance SVDs and24 Ridge heads;
zero neural calls. First-query resource check with240s projected numerical bound.
One exact replay. Export target-free inputs/models, deny undeclared reads at
inference, and retain source membership hashes privately. Test orthogonality,
zero residual covariance, a synthetic low-variance signal lost by PCA but found
by C, and paired source/held independence. Public report/aggregate tables only.

GRAIN measurement, earlier complete-score results and interpretation protocols
remain in place. This is the next finite piece of the ACTIVE whole-system goal.
