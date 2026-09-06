# VM05 strong-head source controls: first audited checkpoint

2026-09-06. **Partial,3/84 conditions; no shared-thread acceptance yet.**
These controls apply to the completed strong word/character Logistic reference,
not the earlier weak GRAIN projection. They evaluate6,294 MBTI-only authors after
excluding99 historical reference members. All source-side inner choices are refit.
No bridge outcome values are used. This report does not choose a new model.

## What the first completed controls show

| EI raw condition | Pooled OOF AUC | Mean official-fold AUC |
|---|---:|---:|
| Reference-excluded, retain discussions | 0.670973 | 0.671157 |
| Reference-excluded, purge shared discussions | 0.637082 | 0.646169 |
| Exact-count random deletion, seed731 only | 0.644727 | 0.655628 |

On the identical6,294 test authors, the original6,393-trained OOF head restricted
to those authors has AUC0.671436. The reference-excluded **refit** minus that
restricted original is-0.000462, paired author-bootstrap95% CI
[-0.001932,+0.001063],2/5 positive folds. There is no material difference established
for this EI/raw comparison. It does not clear historical exposure for every axis,
observer or ready-only cohort, and restriction is not itself a refit.

Purge minus retain is-0.033891, CI[-0.044132,-0.023754]; mean-fold delta-0.024988,
0/5 positive folds. This is a changed training procedure, not an isolated causal
effect of shared discussion. Outer training counts fall from5,012-5,051 to
2,382-2,505. Thus approximately half the training evidence disappears. Nested
training also follows the frozen purge/count rules. Fewer training authors and
their changed composition can explain some of the loss.

The first equal-count random control also declines. **Do not select that single
seed or finalize the purge-minus-count contrast yet:** seeds731-735 are the frozen
family. The independent summarizer deliberately withholds this aggregate contrast
until all five controls exist. Even a complete contrast remains observational
source sensitivity, not an experimentally manipulated personality mechanism.

All intervals above use1,000 paired author draws conditional on fitted OOF
predictions. They are not full-pipeline refit uncertainty or familywise confirmation.
Pooled AUC mixes scores from different fitted folds; mean-fold AUC is also reported.
Ranking quality must not be substituted for balanced classification at threshold.5.

## Audit and unfinished work

- Frozen batch `vm05_strong_source_controls_r1`, identity
  `de681be2cd42022737c3cf542d0f211876d2442c81b3d85c03b0dae25422ccf0`.
- `vm05_strong_source_audit_first_count_r1` independently verifies3 cases,
  reconstructs72 selection records and checks one nested exact-count control.
  Status remainsPARTIAL; two eligible paired contrasts are reported, not a
  prematurely averaged random-control result.
- Retain, purge and count731 each passed a separate saved5-fold replay;15 heads
  total, maximum error9.71e-17. No refits or new outcome values in those replays.
- Runtime was1,575.71sec retain,757.69sec purge,762.62sec first count control.
  Single Mac CPU child, bounded8GiB RSS; no simultaneous second heavy real fit.

Finish all five EI/raw count controls, then cue/placebo and the remaining axes.
The real199-permutation stronger-head nulls, complete neural source comparisons,
same-ready increments and source-frozen bridge evaluation remain open. These first
controls neither prove nor refute LLM personality reasoning. They refine which
training-evidence mechanisms need to be separated before such an interpretation.

Private directories above are under `results/grain_vector_mechanisms/` and must
not be published with author-level predictions, labels, IDs or thread memberships.
