# EI/raw: complete five-seed source sensitivity

2026-09-06. **One control family complete:7/84 planned strong-source conditions.**
This report supersedes the unfinished five-seed interpretation in
VM05_STRONG_SOURCE_CONTROL_CHECKPOINT.md, without replacing that historical
checkpoint or claiming the other axes/arms are complete. No neural source result,
bridge outcome, MEPS questionnaire or new personality mechanism is claimed.

## Result

All conditions use the same6,294 MBTI-only test authors, original EI outer folds,
inner3-fold selection, fixed word/character features and Logistic head. The99
historical reference authors are excluded before each refit. Every outer and
inner random-count selection matches the corresponding purge training count.

| Condition | Pooled OOF AUC | Mean official-fold AUC | Balanced accuracy at .5 |
|---|---:|---:|---:|
| Retain shared discussions | .670973 | .671157 | .503971 |
| Five exact-count random deletions, mean of scores | .648344 | .652613 | Not an ensemble classifier |
| Purge shared discussions | .637082 | .646169 | .500000 |

Random-count seed AUCs731-735 are .644727, .645304, .647596, .660838, .643254.
Their mean is not the score of blended predictions. The frozen bootstrap treats
these five fitted seeds as fixed references, not five independent populations.

Original predeclared pooled purge-minus-count contrast is-.011262, paired
author-bootstrap95% CI[-.021184,-.001484]. The total purge-minus-retain contrast
is-.033891[-.044132,-.023754]. Outer training counts fall from5,012-5,051 to
2,382-2,505. Consequently a substantial part of the total decrement also occurs
when equally many authors are removed without the discussion rule.

## Pooling sensitivity changes the strength of the claim

After seeing these complete results, an explicitly **post-hoc** audit preserves
the same predictions but resamples authors within each original fold. It reports
both pooled AUC and the unweighted mean of five within-fold AUCs. These are
different estimands: pooled AUC compares predictions made by different fitted
models, and can change under fold-specific offsets even if every within-fold
ranking is unchanged. We do not re-calibrate scores on test labels to remove this.

| Contrast | Pooled estimate [95% CI] | Mean-fold estimate [95% CI] |
|---|---:|---:|
| Mean count minus retain | -.022629 [-.027867,-.016609] | -.018544 [-.023613,-.013023] |
| Purge minus mean count | -.011262 [-.021749,-.001395] | -.006444 [-.015413,+.002832] |
| Purge minus retain | -.033891 [-.044446,-.023472] | -.024988 [-.034569,-.015599] |

All intervals in this table use1,000 within-fold author draws, seed731, conditional
on already fitted models. They do not include training-set resampling or
familywise multiplicity control. The original unconditional-author bootstrap
above remains preserved, not replaced by the follow-up.

Purge-minus-count fold effects are-.036082,-.010280,+.015930,+.006900,-.008689.
Thus2/5 folds favor purge, and the average-fold interval crosses0. **There is a
source-composition sensitivity signal, but no stable isolated discussion-leakage
effect established across these estimands.** It would also be incorrect to infer
equivalence/no effect from the mean-fold interval crossing0.

## What this contributes to the mechanism program

For performance functional A, the exact contrast identity is

```
A(retain) - A(purge)
  = [A(retain) - mean_s A(count_s)]
    + [mean_s A(count_s) - A(purge)].
```

This standard addition/subtraction makes two alternatives visible. The first is
training-evidence/count sensitivity under random deletion; the second additionally
changes who/which discussions remain. Neither is a pure causal quantity or a
percentage of personality information. Equal counts do not equalize class mix,
vocabulary, community or difficulty. No element has been identified as disposable
psychological noise by this experiment.

At the same time ranking and classification must be kept separate. True positive
rate E is .207658; retain predicts positive only .003654, while purge predicts
none at .5. Purge AUC .637 is compatible with balanced accuracy .5. This is weak
to moderate label-related ranking information in this source protocol, not an
accepted individual assessment tool or proof of LLM semantic reasoning.

## Evidence and next action

- `vm05_strong_source_audit_ei_raw_complete_r1`:7 cases,168 independently rebuilt
  selection records, five nested count matches, three eligible contrasts.
  Status remainsPARTIAL for the complete84-condition battery.
- Saved outer-head replay now covers35 models across the existing five replay
  packets; latest count734/735 packet max error9.71445e-17. No refits in replay.
- `vm05_ei_source_gap_posthoc_r1`: identity
  `0b74773c9a18e43c1105a17e2e590e69fea5f52043480dd21195a137d3be695e`;
  additive contrast identity error0. Ranked-AUC implementation and bootstrap
  instrumentation have synthetic tests, including the fold-offset counterexample.
- `vm05_ei_source_gap_rank_audit_r1`: independent sklearn reconstruction of36
  observed scalar values from saved CSV rows; max error2.26e-16. It does not
  independently recompute all bootstrap draws, which is explicitly recorded.

Private packets are under `results/grain_vector_mechanisms/`. Keep all author
rows/labels/threads private. Continue the remaining source controls and complete
observer queue; compare source-only neural/readout candidates and then same-ready
increments. Strong permutation nulls and source-frozen bridge evaluation remain
open. Do not adapt frozen thresholds or choose layers to rescue this EI result.
