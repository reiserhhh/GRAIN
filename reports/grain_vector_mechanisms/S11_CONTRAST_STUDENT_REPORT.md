# S11: ordinary E5-mean students learn the background, not the useful contrast

2026-09-08. COMPLETE.213 N source authors, existing E5 cue-mask mean vectors,
same S4 outer author folds. Target-only inner selection; no questionnaire fit.
6.4seconds,0 neural forwards/backwards.30 fitted affine models are saved and
replayed. Every OOF score was frozen before the outcome join.

## Result

| Coordinates | Teacher target | Target fidelity r | External N r |
|---|---|---:|---:|
| PCA64 | Full FP32 teacher | .5768 | .0421 |
| PCA64 | Projected suffix endpoint | .6338 | .0503 |
| PCA64 | Configuration contrast | -.0187 | -.0058 |
| PCA64 | Independently fitted components summed | .5768 | .0421 |
| PLS2 | Full FP32 teacher | .5528 | .0345 |
| PLS2 | Projected suffix endpoint | .6110 | .0296 |
| PLS2 | Configuration contrast | -.0183 | .0311 |
| PLS2 | Independently fitted components summed | .5345 | .0352 |

The native contrast itself has N r=.2116 in S9; these two mean-vector student
families do not recover it, even as a teacher target. They predict the projected
background substantially better. This localizes why copying the complete teacher
can look successful while retaining little of its observed psychological relation.

The fixed common-PCA/alpha10 linearity control holds to2.22e-16:
P(M+C)=P(M)+P(C). Renaming components alone does not create a better learner.
PCA's independently selected sum is also identical to its full-target result
in this run. PLS's target-dependent subspaces change the sum, without improving
the outcome connection. No coordinate family or alpha was selected on Y.

This supports leaving the **tested mean-vector class**, not declaring that E5
contains no such information or that all nonlinear learners must fail. The new
target choice follows the exposed S9 result, so these are discovery comparisons,
not a fresh external validation or an exhaustive impossibility proof.

## Next action

Move to a bounded query-conditioned/joint-vector representation comparison on
the same source authors. Distinguish body-state means, the already-contextualized
trait-query state, and the layer17 key/value-pairing vector. Preserve matched
author folds and teacher-only selection. S10 provides a direct measurable
operation for the last representation; it should now be connected to the source
contrast rather than expanding the ordinary E5 hyperparameter grid.

## Artifacts

Plan: `docs/GRAIN_CONTRAST_STUDENT_S11_PLAN.md`.
Runner: `scripts/run_grain_contrast_student.py`.
Ignored outputs: `results/grain_vector_mechanisms/contrast_student_s11_r1/`:
`manifest.json`, `private_models/`, `private_oof_before_outcomes.csv`,
`oof_frozen.json`, `selected_models.csv`, `metrics.csv`, `paired_intervals.csv`,
`summary.json`. Two added unit tests cover linearity and target-only selection.
