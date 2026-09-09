# NEXT08/09: level-specific reading versus input-information loss

2026-09-10, after `22e5f30`. This is a cache-only mechanism comparison, not a
human-label optimization grid. The prior T/F residual readout error retained
+.147 criterion covariance contribution. That alone did not say whether the
cause was amplitude shrinkage, a shared reading rule, or missing input features.

## Questions and fixed decisions

1. **Affine scale versus a different component.** Given a fixed OOF teacher
   target t and readout p, describe t=a+b*p+e using only t,p, not criterion Y.
   Then t-p=a+(b-1)*p+e. Decompose its criterion covariance into amplitude and
   non-affine terms with the same sd(t)sd(Y). This full-source descriptive
   alignment is NOT a new fitted evaluation score or a held-set calibration.
   Also report within-CV-fold centered accounting to expose intercept mixing.
2. **Different estimands, same input geometry.** On source training authors,
   separate author means and author-centered view deviations. Does the same
   vector program need a different reading operator for each level?
3. **Extend the defined interface.** Fit all nine fixed query configurations,
   not only the three with recently promising criteria. Prior T/F/N pooled
   outputs must numerically reproduce. New queries are exploratory extension,
   not independent confirmation of the observed E/A/O links.

## Exact mathematical distinction

For equal J=3 views, let A=I_authors tensor (11'/J), W=I-A. Then A,W are
orthogonal projectors, AW=0, and for any prediction residual v:

    ||v||^2 = ||Av||^2 + ||Wv||^2.

Thus pooled mean squared loss equals between-author mean squared loss plus
within-author mean squared loss. A common coefficient minimizes their sum;
two distinct slopes solve a different, explicitly multilevel problem.
No part is discarded as noise. These views are sampled comments, NOT repeated
clinical visits or equal-stimulus psychological change.

## Fixed programs

- Source same276 authors, three12-comment views and frozen GRAIN factors;
  E5 feature cache from the previous pass. FiveBig5 queries on213 authors,
  fourMBTI on63. No new text embeddings or neural calls.
- Mean E5 feature only. Use identical source-view centering/trace norm for
  pooled, between and within programs. No new query-pole embeddings needed:
  the mean-only input does not depend on poles; query dependence is in the
  teacher target and fitted coefficients.
- `pooled`: prior average view loss +.01||w||^2, intercept fitted.
- `between`: average author-mean loss +.01||w||^2, intercept fitted, retaining
  pooled feature norm. Ridgealpha=.01*n_authors (not n_views).
- `within`: average centered-view loss +.01||w||^2, no intercept. It has NO
  identified author-level offset, so never report its author-level criterion
  prediction. Evaluate only centered views or material changes.
- All train folds build their own GRAIN between reference using the same
  alpha100 and their own feature norm. Sameauthor views stay in onefold.
- Final programs can score the old12-author full/E1/E2 material, but native
  and configuration response evaluation is only T/F/N where exact caches
  exist. E2 has no P/Q configuration target. Do not fabricate seven new native
  comparisons or train on these held authors.

## Outputs and decisions

Freeze source-OOF and held predictions before reading existing source labels.
No labels change source membership, targets, signs, loss weights or models.
Source descriptive Pearson/binaryAUC links compare author3viewmeans. Report
within fidelity separately, with no view-IID significance. Paired author
bootstrap intervals on primary T/F/N contrasts and E1 observed response.
Affine alignment is an accounting analysis, not a new prediction arm.

Record exact pooled-loss identity, coefficient/action differences, matching
prior outputs, same-input natural changes, and criterion-retention differences.
Use results to decide whether a two-level reading interface is useful and
where a genuine conditional-input experiment is still needed. No forced
promotion if the levels behave similarly; no accuracy-only long run.

## Prior work used

- [Curran & Bauer(2011)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3059070/):
  disaggregate within- and between-person processes instead of interchanging
  coefficients. Current repeated materials are sampling views, not their
  longitudinal psychological setting.
- [Mundlak(1978)](https://people.stern.nyu.edu/wgreene/Econometrics/Mundlak-1978.pdf):
  pooling repeated units requires explicit heterogeneity assumptions. We use
  the distinction, not a claim of implementing his causal/panel estimator.

Private models/author scores remain gitignored; aggregate report, math, tests
and code are committed locally. Existing score releases/lockboxes unchanged.
