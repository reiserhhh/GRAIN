# VM05 SN source controls: removal and sample size must remain separate

2026-09-07, round32. **SN raw7 conditions plus cue-retain accepted; whole
strong-control family remains29/84, PARTIAL.** No new model was fitted for
this acceptance. Pending cue/placebo controls and TF/JP are not negative results.

## Independently verified evidence

The cumulative29-case audit reconstructs696 training-selection records and20
count-matched pairs. New SN8 cases have40 saved official-fold model replays,
maximum absolute error1.1102230246251565e-16. Together with the previously
accepted EI21/105 folds this verifies29 cases/145 folds, not the entire84 grid.
All source outcomes are MBTI-only; bridge text membership can define discussion
exclusions, but bridge outcome values are never read.

Each SN outer fold evaluates the same held-out members of6,294 reference-free
authors. Retain uses5,032-5,039 training authors; purge uses2,394-2,443. Its
five random-count controls match these sizes at every nested/full selection,
not only at the outer boundary. Purge removes shared-discussion authors, not
individual words from the test text. Changing training composition also changes
support and generalization difficulty, so it is not an isolated noise deletion.

## Raw results

| Head/control | Pooled AUC | Mean official-fold AUC |
|---|---:|---:|
| Retain |.668861|.671001|
| Purge shared discussions |.621191|.633914|
| Mean of five matched-count seeds |.633852|.649372|

- Purge minus retain: pooled-.047670, pointwise95% CI[-.062480,-.031981];
  mean-fold delta-.037088, positive in0/5 folds.
- Purge minus average matched-count score: pooled-.012661,
  CI[-.026569,.001470]; mean-fold delta-.015459, positive in0/5 folds.
  The pooled interval crosses0. No mean-fold interval was computed in this
  checkpoint, so the pooled interval must not be attached to that estimand.
- Reference exclusion/refit minus the original strong head restricted to the
  same test authors: raw+.000377[-.001482,.002477]. This comparison does not
  indicate reference overlap was the main driver of the prior SN result.
- Cue-retain minus raw-retain:-.000218[-.002129,.001709]. The remaining cue
  controls and all SN placebo controls are uncompleted, not an accepted mask family.

These1,000-draw paired author intervals are conditional on saved fitted OOF
predictions. The count comparator averages five seed-specific scores, not
predictions from a new ensemble. Its raw pooled range is.610964-.650180.
Neither five seeds nor five folds are five independent studies.

## Decision

The larger purge-versus-retain decline combines loss of training volume with
source-composition changes. The count-adjusted point estimate remains negative,
but its pooled uncertainty does not establish an additional source-specific
effect here. Do not call the full-.0477 decline personality information removed,
causal discussion leakage, or a reason to discard context in GRAIN.

Finish the original84-case/count/null family when its CPU lease is available;
do not resume it over the running sensitivity or registered same-ready successor.
No rescue prompt, axis inversion, new learner, bridge tuning or scorer change.

Artifacts:
`results/grain_vector_mechanisms/vm05_strong_source_audit_first29_r1`
and `results/grain_vector_mechanisms/vm05_strong_source_replay_sn_first8_r1`.
Audit entry: `scripts/summarize_grain_vector_source_controls.py --allow-partial`;
replay entry: `scripts/summarize_grain_vector_external_replay.py` with the eight
completed SN folders. The audit exposes all20 paired contrasts, not only SN.
