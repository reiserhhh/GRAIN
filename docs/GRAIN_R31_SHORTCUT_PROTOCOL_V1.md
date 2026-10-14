# GRAIN R31 shortcut exclusion check, protocol V1

Frozen 2026-09-26 JST before computation. Cached outputs and metadata only. No labels, no
neural calls.

- Part A, topic shortcut in historical readouts:
  - Held PANDORA authors (120, local_bilinear_r2).
  - Author topic composition = share of the author's comments in each of the 30 most
    frequent subreddits among held authors, plus "other".
  - The first 5 principal components of the share matrix predict each trait's held-out
    readout prediction (semantic and GRAIN heads). OLS, leave-one-author-out.
  - Metric: cross-validated R^2 per trait, with a 500-permutation null (seed 20260926).
  - "Topic-explainable share" is a shortcut risk to be controlled before trait
    interpretation, not proof of invalidity.
- Part B, task and wording shortcut in the vector-path action (conditional_operator_v1/r2,
  72 application pairs): the one-way ANOVA share of the measured donor action (E and N)
  explained by situation (task) and by persona wording family, each with a
  500-permutation null.
- Rule: a share is flagged when it exceeds the permutation 95th percentile.
