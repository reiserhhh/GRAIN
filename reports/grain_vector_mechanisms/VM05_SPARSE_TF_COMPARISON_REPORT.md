# VM05: completed sparse TF three-arm source comparison

2026-09-06. **All TF arms and saved heads complete; JP/full neural comparisons pending.**

The frozen source-exclusive material is6,393 MBTI-only authors,36 comments each,
with official TF outer folds and inner-only hyperparameter selection. Word/char
TF-IDF plus LogisticRegression is a strong sparse reference, not an LLM prompt,
personality generation system, or new GRAIN observer. Bridge outcomes stay closed.
No fitting/feature choices changed between the three arms.

| Arm | Pooled AUC | Mean fold AUC | BA at .5 | Macro-F1 at .5 |
|---|---:|---:|---:|---:|
| Raw source | .767212 | .767390 | .657990 | .665807 |
| Frozen cue mask | .765381 | .765561 | .657083 | .664790 |
| Same-count placebo mask | .767227 | .767381 | .658898 | .666823 |

Positive F prevalence is .352417. Predicted-positive rates are .215705/.215548/
.215861 respectively. Ranking and threshold performance both exceed the all-one-
class TF reference here, but predictions are still skewed and not proven calibrated.
SN remains different: AUC about .670, but BA=.500 at .5 for all three arms.

| Paired contrast | Delta AUC | 95% conditional author bootstrap CI |
|---|---:|---:|
| Cue minus raw | -.001831 | [-.002377,-.001294] |
| Placebo minus raw | +.000014 | [-.000499,+.000514] |
| Cue minus placebo | -.001846 | [-.002428,-.001248] |

The frozen cue set has a small selective association with TF ranking in this
sparse representation. This differs from the earlier weak lexical-projection
head, whose cue-mask comparison did not show the same degradation. Therefore
"TF is unaffected by cue removal" is not representation-independent. However,
the retained AUC .7654 is essentially the same practical scale of performance;
this intervention does not explain away the entire association or establish
which retained features are psychological rather than topical/community-related.

The intervals use1,000 paired-author draws (seed731), conditional on already
fitted OOF heads. They do not repeat feature/model selection or correct a family
of broad exploratory comparisons. Limited regex masking is not an exhaustive
leakage control; the stronger heads still need the appropriate source-exposure,
shared-discussion and full-null checks. Do not inherit those conclusions from
previous weaker-head controls without rerunning the actual strong model.

## Execution and evidence

- Raw/cue/placebo times:1,874.58 /1,634.85 /1,610.72 seconds.
- Sampled peak RSS:6.015 /6.434 /7.788GB; all below the frozen8GiB hard stop.
- Every one of15 TF saved fold heads independently replays its held-out scores;
  maximum absolute error1.11e-16. No refit or new label-value read in the auditor.
- Historical r2 failed state stays preserved. The memory-lifetime fix preceding
  r3 did not alter material, model, grid or folds.

Private batch: `results/grain_vector_mechanisms/vm05_sparse_mbti_r3/TF_cont_*`.
Replay packets: `vm05_sparse_tf_raw_replay_r1`, `vm05_sparse_tf_cue_replay_r1`,
`vm05_sparse_tf_placebo_replay_r1`. Aggregate snapshot:
`vm05_sparse_sn_tf_complete_summary_r1` (six completed SN/TF arms of the nine-arm
r3 batch, not a complete four-axis experiment).

No source winner, new blend, probability threshold or bridge head is selected
from these partial-program results. Finish JP and all frozen observer caches,
then compare the complete identical-evidence source candidates and controls.
