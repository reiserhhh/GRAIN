# NEXT68: a learned cross-token response and its precise range limitation

2026-09-11. **Finite experiment complete; retain the mechanism, not a predictor
replacement.** Primary implementation frozen at `4df8e00`. Source4/two queries,
four views per author, four numeric fits; zero new neural/API calls or human
criterion use. New8 is source-disjoint but previously exposed, not a fresh test.

## Finding

The compatible quadratic reader learns a nonzero response on unchanged late
tokens when early tokens move. The matched local scalar predicts exactly zero.
Source coordinate response SSE decreases by2.67%/2.57%; the four GRAIN-direction
effects decrease by3.92%/3.84% in squared error. This develops NEXT67's detected
dependence into a single numeric score-and-derivative function.

| Source4 endpoint | Neuroticism | Confirmation seeking |
|---|---:|---:|
| Local cross-response SSE/zero | 1.000000 | 1.000000 |
| Quadratic coordinate cross-response SSE/zero | .973321 | .974318 |
| Full hidden-field relative SSE, author mean | .995399 | .995102 |
| Four GRAIN actions SSE/zero | .960824 | .961599 |
| Four GRAIN actions correlation, pooled16 pairs | .226789 | .195567 |
| Predicted action RMS / actual RMS | .149893 | .130224 |

These are infit source responses, not participant-score correlations. The
pooled16 effects are four directions from four authors, not16 independent people.
Confirmation seeking is a separate query, not Big5 conscientiousness.

## Matched new-author application

All current predictions were frozen before joining current target scores or
gradients. Both arms fit identical source views and objective terms. The five
existing comparison arms are retained unchanged in the public tables.

| Query / model | Native whole-score r | Whole-score RMSE | GRAIN full-action SSE/zero |
|---|---:|---:|---:|
| N: original-material reader | .503312 | .821332 | 1.080006 |
| N: matched local | .143813 | .919341 | 1.085740 |
| N: pooled quadratic | .137133 | .921382 | 1.092754 |
| Confirmation: original-material reader | .538314 | .903460 | 1.092412 |
| Confirmation: matched local | .356246 | .975583 | 1.053263 |
| Confirmation: pooled quadratic | .386932 | .967955 | 1.086398 |

Confirmation's score improves locally while direction error increases; N does
not improve. Neither query beats the retained original-material reader on whole
score. The source response gain does not justify predictive replacement. These
targets are fixed-model readings, not observed questionnaire or behavior labels.

## Why the small gain is informative

Post-outcome source-only arithmetic identifies the family limitation. For
`v=mean concat(z,p*z)-source_center`, the quadratic's response on unchanged B is
`delta_k_j = a + p_j*b`. It cannot give two different local-content-dependent
responses beyond this affine positional form. The old local part cancels there.

Projecting each source B target onto this permitted positional subspace, while
relaxing all cross-author/shared-M restrictions, gives an optimistic bound:

| Coordinate-density account | N | Confirmation |
|---|---:|---:|
| Representable target energy fraction | .029389 | .030344 |
| Unconstrained SSE/zero floor | .970611 | .969656 |
| Actual quadratic SSE/zero | .973321 | .974318 |
| Actual SSE reduction / optimistic possible reduction | .907787 | .846373 |

Thus the fitted model already obtains about91%/85% of this generous permitted
reduction. Increasing the penalty weight or fitting longer cannot remove the
remaining positional-subspace residual within this family. This is a restriction
of this coordinate-response function, not a statement that97% of personality or
text information is missing. The other hidden coordinates and all finite model
responses remain separately reported.

The projection is labeled post-outcome: least-squares coefficients are discarded,
never used as predictions on another author, and no diagnostic fit is promoted.
Actual predicted B responses leave the allowed subspace by at most4.10e-17 RMS;
projection energy identity error is at most2.12e-22.

## Reproduction and artifacts

- Module: `grain_research/vector_mechanisms/coupled_pooling.py`.
- Runner: `scripts/run_grain_coupled_pooling.py`, stages prepare/fit/predict/
  evaluate, then predict/evaluate with `--folder replay`, then audit.
- Publisher: `scripts/report_grain_coupled_pooling.py`.
- Public aggregate CSVs and hashes: [coupled_pooling_v1](coupled_pooling_v1/).
- Private model/input/prediction lineage: ignored
  `results/grain_vector_mechanisms/logic_coupled_pooling_v1/`.
- 55 focused tests pass across11 named test modules; two inherited SWIG warnings.
  Ruff passes. 48 prediction packets and11 numerical CSVs replay exactly.
- 384 actual directional finite differences have max error1.36575e-10;
  source objective reconstruction error0. Primary staged CPU29.464103s,
  below180s. Publication/range accounting is separately timed in publication.json.

No background model job, deployment change, old lockbox opening or participant
text export. Goal ACTIVE. [Theory](../../docs/GRAIN_COUPLED_POOLING_THEORY_V1.md)
derives the interface; [NEXT69](../../docs/GRAIN_NONLINEAR_POOLING_PLAN_V1.md)
tests a same-size nonlinear token summary rather than another weight grid.
