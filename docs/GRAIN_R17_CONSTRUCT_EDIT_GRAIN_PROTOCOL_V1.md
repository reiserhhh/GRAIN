# GRAIN R17 construct-targeted edit vs nuisance edits (GRAIN lexical reader only), protocol V1

Frozen 2026-09-26 JST before computation. It reuses the 100 frozen R17 natural units
(`R17/natural_sensitivity_r1/manifest.csv`). There are no neural calls: Packet 1's
four-edit limit per unit is reached, so no E5 or BGE read of this fifth version is made.
GRAIN lexical runtime only. No labels.

## Construct contract (declared)

- Construct: first-person singular self-reference, a lexical self-focus marker. Its
  association with Neuroticism is established only as the lockbox H2 association, not as
  a construct.
- Construct edit (self_reference_remove): map the first-person singular forms
  I/me/my/mine/myself, including contractions I'm/I've/I'll/I'd, to third-person
  "the author" forms. The replacements are "the author", "the author's",
  "the author is" and similar.
  - Intended change: self-reference goes to zero.
  - Intended invariance: all other tokens are unchanged, verified by an identical
    non-pronoun token sequence.
  - Grammaticality is not preserved (for example "the author am"). This is declared.
- Applicable when the unit contains at least one first-person singular form. Units
  without one are refused and recorded.

## Comparisons (GRAIN-24 and the historical GRAIN head)

- The construct delta is compared with the Packet 1 nuisance deltas in GRAIN-24.
  - contraction_expand and sentence_reverse are non-zero there.
  - case_fold and punctuation_strip are exactly zero, by preprocessing.
- Ratios, with author bootstrap: construct distance over within-author natural distance,
  and construct distance over the nuisance distance.
- Selectivity via the generalized eigenproblem Sigma_C v = lambda Sigma_N v on
  GRAIN-24. Sigma_C is the construct-edit second moment; Sigma_N is the pooled non-zero
  nuisance second moment plus a 1e-3 trace ridge. Report the top eigenvalues and the
  share of construct energy on the top selective direction.
- Historical GRAIN head, by single-unit substitution in the author block: author score
  shift per trait under the construct edit against the nuisance edits, in held-out
  prediction SD units, with bootstrap CIs. Neuroticism is the pre-named trait of
  interest.
