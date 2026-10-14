# GRAIN conditional-response construction V1

2026-09-13. Execution addendum to C01/C12/C15/C17 of the active all-plan goal.
Frozen before this tranche's outcomes. Existing development fold is not a new
lockbox. No questionnaire targets, MEPS, bridge fitting, neural calls or uploads.

## Question and inheritance

Can an author's evidence responses in observed conditions support a prediction
in an entirely hidden condition? This fills the model-based branch left by
`GRAIN_CONDITIONAL_REFERENCE_PLAN_V1.md`. It is not another text-prediction head
or repetition of `vector_mechanisms/conditional_models.py`: that older kernel
maps text representations to fixed-model responses. Here the stochastic index
is **condition within author**, and an entire author-condition cell is hidden.

Reuse Rasmussen & Williams, *GPML* ch.2, equations 2.23-2.24, 2.38 and 2.40:
https://gaussianprocess.org/gpml/chapters/RW2.pdf. A Gaussian random intercept plus
a condition kernel is a standard covariance model, not a new theorem. The
construction contribution is an explicit observed/prior/personal decomposition
and a source-only, whole-condition evaluation on the existing evidence interface.

## Fixed model and units

Use all 624 PANDORA Big5-only authors, 36 original comments each, the exact
float64 E5/IPIP-keyed 30 response functions and recorded subreddit conditions
from `conditional_reference_r2`. Frozen GRAIN24 comment coordinates come from
`local_bilinear_r2`; these are not 24 named psychological factors.

For author u, condition c, evidence coordinate j:

    m_ucj = mu_cj + a_uj + f_uj(c) + epsilon_ucj
    cov(a+f) = sigma_j^2 [alpha 11' + beta R]
    R_cc' = exp(-||z_c-z_c'||^2 / (2 ell^2))
    var(epsilon_ucj) = sigma_j^2 / n_uc

mu is the source-only equal-author mean within condition. z is the corresponding
equal-author GRAIN24 mean, standardized over source conditions. ell is the
median nonzero pairwise distance of at most 2,048 source conditions, sampled
with seed 913, independent of outcomes. sigma_j^2 is pooled within-author-cell
unit residual variance on source authors only; it is a working noise model.
Comments need not actually be independent. Model uncertainty is conditional on
frozen estimated moments and does not include source-mean/kernel uncertainty.

Three main arms: source-condition mean (alpha=beta=0), random author intercept
(alpha in {0.1,1}, beta=0), GRAIN condition GP (alpha,beta in {0.1,1}). Select
within each learned family using official fold1 authors, with folds2/3/4 source.
Selection loss: squared error divided by source sigma_j^2, averaged across 30
coordinates, then cells within author, then authors. Refit all source moments
on folds1/2/3/4 after selection; evaluate fold0 once for this fixed family.

One diagnostic control permutes condition descriptors with seed913 and uses
the selected GP hyperparameters without reselection. It tests whether actual
condition-neighborhood geometry adds value beyond generic borrowing. No further
bandwidth, rank, pooling, alpha or kernel search in this tranche.

## Whole-condition test

For each validation/held author, hide all comments in one source-known
condition; predict their empirical mean using other known conditions only.
Unknown source conditions are recorded, not imputed or removed from the stored
actual score. Known targets without another known cell receive prior-only
prediction and are reported separately. No author-cell target or held text
enters source means, descriptors, standardization, noise or model selection.

Report per-author equal-cell MSE/normalized MSE/MAE, trait-coordinate errors,
supported-author/target counts, and 2,000 paired author bootstrap draws of error
differences. Negative delta favors the first named arm. Comparisons are reference
development evidence, not human personality accuracy. No multiple cells/facets
treated as independent subjects and no held-out questionnaire correlations.

## Fixed-reference completion

Retain the previous source nu targets, full and common5. Let S be observed
source-known cells and U the missing target cells. Preserve observed means:

    T_model = sum_S nu_c m_uc + sum_U nu_c mu_c
              + nu_U' K_US [K_SS+diag(1/n_S)]^-1 (m_uS-mu_S)

Export the three terms separately: observed contribution, missing-source prior,
and missing-personal adjustment. Missing mass and actual coverage stay visible.
The old empirical `point=None` remains None if incomplete; this is a separately
named **model-completed functional**, not additional observed evidence.
Conditional latent missing-functional variance is

    sigma_j^2 (nu_U'K_UU nu_U - k_US A^-1 k_SU).

No multi-coordinate interval is produced under a false independence assumption.
Gaussian outputs are not clipped: compare with the previous unit-ball completion
envelope and count any violations. Both empty-support and full-support limits
must hold. Essays has one recorded condition, so it cannot identify a distinct
cross-condition surface; retain that boundary rather than invent extra contexts.

## Delivery / decision

Code, tests, JSON/NPZ source model, private cell predictions, source selection,
author summaries, paired intervals, completion components, numerical audit and
public aggregate report under `reports/grain_measurement_foundation/`.
Results: `results/grain_vector_mechanisms/all_plan_closure_v1/conditional_response_r1/`.
Budget: existing CPU cache only; no scheduled long job. If GP improves beyond
intercept and shuffled geometry, retain it as a local conditional-response
reference. Otherwise retain the simpler model and the negative comparison;
do not rescue it with another hyperparameter sweep. Completion of this tranche
does not close all plans, genuine opportunity selection, or psychological validity.
