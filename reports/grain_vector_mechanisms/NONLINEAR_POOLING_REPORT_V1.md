# NEXT69: content-dependent response exists, but this pooled readout is not better

2026-09-11. **Complete; pooled-quadratic family closed without predictive
promotion.** Primary implementation `102da28` was frozen before two numeric
fits. Same source4, four views each, two queries,1305 coefficients and alpha=.01.
No new neural/API calls, human labels or MEPS use. The disjoint new8 panel was
already exposed in earlier development; it is not a fresh confirmation set.

## What changed

Replaced16 pooled linear/position moments by16 source-frozen Gaussian token
features, evaluated before pooling. Original64 centers, bandwidth, U8, feature
scales, local1169 prefix and all four source objective terms are unchanged.
The new field is the derivative of the same scalar score, not an independent
unconstrained path model. The two old NEXT68 arms and five older controls are
retained without refit.

The intended content dependence is now observable:57.81%/53.26% of the **new
predicted response energy** lies outside the old constant-plus-position span
(N/confirmation). This percentage describes the candidate's output shape, not
the amount of true model response or personality information recovered.

## Source mechanism comparison

All ratios use zero response as denominator unless stated otherwise.

| Source4 / query | Matched local | Linear pooled quadratic | Nonlinear pooled quadratic |
|---|---:|---:|---:|
| N: coordinate B-response SSE | 1.000000 | .973321 | .993427 |
| Confirmation: coordinate B-response SSE | 1.000000 | .974318 | .995628 |
| N: four GRAIN B-actions SSE | 1.000000 | .960824 | .962056 |
| Confirmation: four GRAIN B-actions SSE | 1.000000 | .961599 | 1.058208 |

The new candidate reduces coordinate error by.657%/.437% against zero, less
than NEXT68. N GRAIN action error improves3.794% against zero; confirmation error
increases5.821%. Their pooled16-action correlations are .214768/-.242055.
These16 pairs are four directions from four authors, not16 independent people.
Full-hidden-field relative SSE author means are .998914/.999207; coordinate
fit and whole hidden-field fit remain distinct.

## New8 application

| Query / model | Native whole-score r | Whole-score RMSE | Full GRAIN-action SSE/zero |
|---|---:|---:|---:|
| N: original-material reader | .503312 | .821332 | 1.080006 |
| N: matched local | .143813 | .919341 | 1.085740 |
| N: linear quadratic | .137133 | .921382 | 1.092754 |
| N: nonlinear quadratic | .143683 | .919363 | 1.085216 |
| Confirmation: original-material reader | .538314 | .903460 | 1.092412 |
| Confirmation: matched local | .356246 | .975583 | 1.053263 |
| Confirmation: linear quadratic | .386932 | .967955 | 1.086398 |
| Confirmation: nonlinear quadratic | .356251 | .975585 | 1.053008 |

Whole scores are effectively those of matched local. Tiny action-error decreases
do not justify a full replacement; the original-material reader remains stronger
on whole scores. Confirmation seeking is not Big5 conscientiousness. All these
targets are native-model readings, not newly measured questionnaire accuracy.

## Decision and theory update

NEXT68 proved the precise affine-position restriction. NEXT69 removes that
restriction in the implementation and actual predictions, but a more flexible
response shape does not automatically recover the desired query-conditioned
response. Keep both findings. No more center/rank/alpha/width sweeps in this family.

Next development changes the functional operation: compare unnormalized
kernel-weighted evidence with normalized relative selection, keeping total kernel
mass as a separate feature. This reconnects to the earlier native key-selection
mechanism instead of another arbitrary pooled quadratic. It is a next hypothesis,
not an explanation already proven by this negative comparison.

## Verification and files

- 61 focused tests across12 modules pass; two inherited SWIG warnings. A pre-run
  contract-error ordering mismatch was corrected before the primary freeze.
- 48 input-only prediction packets and12 numerical CSVs replay exactly.
- 192 actual GRAIN directional differences: max scalar/gradient error1.11651e-10.
  Source objective replay error0; prior private control rows retained unchanged.
- 23.445285s stagedCPU, below180s;2fits,0new neural calls. No live model job.
- New module `grain_research/vector_mechanisms/nonlinear_pooling.py`, runner
  `scripts/run_grain_nonlinear_pooling.py`, publisher
  `scripts/report_grain_nonlinear_pooling.py`, focused test module of the same name.
- Public7 aggregate tables/hashes: [nonlinear_pooling_v1](nonlinear_pooling_v1/).
  Private lineage/models/predictions remain ignored under
  `results/grain_vector_mechanisms/logic_nonlinear_pooling_v1/`.

[Theory](../../docs/GRAIN_NONLINEAR_POOLING_THEORY_V1.md) and
[NEXT70 plan](../../docs/GRAIN_NORMALIZED_EVIDENCE_READER_PLAN_V1.md).
H16 still requires the encoder; release/scorer/lockbox/clinical interpretation
boundaries remain unchanged. Overall goal ACTIVE, no global closure claim.
