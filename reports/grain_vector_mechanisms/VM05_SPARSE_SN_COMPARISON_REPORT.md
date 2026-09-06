# VM05: SN strong sparse three-arm comparison

2026-09-06. All three SN source-only arms complete and independently replayed.
This is one axis within an unfinished four-axis/source-observer comparison,
not a completed bridge analysis or a resumed generative MBTI training program.

## Results

Same6393 MBTI-only source authors, same36-comment common material and source
folds, word+char sparse features, frozen inner selection. PCA is not applied.
Do not substitute this denominator for the earlier complete prepared MBTI-only
population8678 or use bridge users to improve the source head.

| Arm | Pooled OOF AUC | Mean fold AUC | BA at .5 | Predicted N | True N |
|---|---:|---:|---:|---:|---:|
| raw | .669877 | .672291 | .500000 | 100% | 89.0192% |
| cue_mask | .669909 | .672293 | .500000 | 100% | 89.0192% |
| placebo_mask | .668310 | .670506 | .500000 | 100% | 89.0192% |

The macro-F1 at .5 is .470953 for all three arms. Ranking performance is
nontrivial but the frozen threshold yields a majority-only decision rule.
The ranking result is not .67 classification accuracy or calibrated probability.
No threshold was chosen on OOF, test, or bridge outcomes. Any future calibration
requires its own source-inner fit and a separately reported comparison.

Paired1000-author bootstrap, seed731, conditional on already-fitted OOF heads:

| Difference | Delta AUC | Descriptive95% interval |
|---|---:|---:|
| cue_mask - raw | +.000033 | [-.001567,+.001536] |
| placebo_mask - raw | -.001567 | [-.003144,-.000073] |
| cue_mask - placebo_mask | +.001599 | [-.000151,+.003256] |

No model refits occur in this bootstrap, no multiple-comparison claim, and no
new source-independent confirmation. The small negative placebo contrast has
a just-negative conditional interval; it is not evidence that the target cues
are causally useful or that they are harmful. The target-versus-placebo contrast
includes0. This dictionary cue ablation does not certify absence of all leakage:
reference exposure, shared threads and representation controls remain separate
conditions. Do not borrow the earlier weaker-head null/control results as if
they were re-run for these stronger word+char heads.

## Execution and acceptance

All15 saved fold heads independently reproduce their saved validation predictions;
each arm's maximum absolute error is1.1102230246251565e-16. The replay reads no
new label values and fits no model. Source labels were used by the original
source training, as permitted; bridge outcome values remain unopened.

The earlier r2 SN job exceeded8GiB at final fitting and remains FAILED_STOP.
The subsequent lifetime fix released completed inner CSR/transform/model objects,
without changing feature settings, folds, regularization or labels. Complete r3
raw/cue/placebo sampled RSS peaks are6.756/7.040/6.597GB, all below unchanged8GiB.
Supervisor durations are1597.9/1774.6/1723.5 seconds. This is an empirical memory
fix for these conditions, not a claim that all later axes are finished or safe.

## Artifacts and next action

Relative to `results/grain_vector_mechanisms/` (private/gitignored):

- `vm05_sparse_mbti_r3/SN_cont_{raw,cue_mask,placebo_mask}`: frozen outputs/models.
- `vm05_sparse_sn_three_arm_summary_r1`: metrics.csv, paired_contrasts.csv,
  hashes and partial-batch summary; statusPARTIAL_VERIFIED_ARMS_ONLY correctly
  refers to3/9 r3 conditions, not failure of the complete SN packet.
- `vm05_sparse_sn_raw_replay_r1`, `vm05_sparse_sn_cue_replay_r1`,
  `vm05_sparse_sn_placebo_replay_r1`: five-fold replay evidence for each arm.

The active r3 supervisor has advanced toTF raw, with six TF/JP conditions still
open. Do not restart it or edit its source modules while running. Finish those
bounded conditions and full three-observer collection; then compare identical
eligible source material and formalGRAIN-ready subsets. Freeze final source
heads before reading bridge outcomes. AUC thresholds and cue choices are not
to be tuned against this three-arm readout.

## Later TF raw snapshot (2026-09-06)

TF raw subsequently completed on the same6393 source authors: pooled AUC=.767212,
mean-fold AUC=.767390, BA@.5=.657990, macro-F1=.665807. Predicted positive rate
.215705 versus true.352417. Unlike SN, it is not majority-only at the fixed
threshold, but calibrated probability quality has not been established.
All5 saved fold predictions replay within1.11e-16
(`vm05_sparse_tf_raw_replay_r1`). Sampled RSS peak6.015GB, supervisor1874.6sec.
The four-arm snapshot is`vm05_sparse_sn_tf_partial_summary_r1`. TF cue/placebo
and JP remain incomplete; this is not a three-arm TF comparison or an accepted
all-axis/head/bridge result. The active supervisor continues without retuning.
