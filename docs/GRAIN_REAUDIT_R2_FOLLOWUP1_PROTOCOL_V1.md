# GRAIN re-audit r2 follow-up packet 1 (FU1), protocol V1

Frozen 2026-09-26 JST before computation. Zero neural calls, no labels, no generation.
Each part answers an item that the r2 auditors named as agent-executable. Seeds 20260926.

- **FU1-a (R28): within- vs between-author structure of GRAIN-24 on real text.**
  - Source: local_bilinear_r2 PANDORA and Essays units.
  - Sigma_W = pooled within-author covariance of unit GRAIN coordinates. Sigma_B =
    covariance of author means, minus Sigma_W / n_units.
  - Report the top-5 eigen subspace similarity S (as in R35), the trace ratio, and the
    first-eigenvector congruence.
  - Question: is the between-person structure the within-person structure?
- **FU1-b (R16): reader variance ratio, qualification target, F16 items.**
  - Both readers' qualification scores are z-scored within reader. Crossed G-study,
    items x readers, one observation per cell.
  - Report the item variance, the reader x item interaction plus residual, and the
    reader-variance ratio = (reader + reader x item) / total. Separately for source and
    application roles.
- **FU1-c (R31): topic residualization of the historical semantic head on 120 held
  PANDORA authors.**
  - Leave-one-author-out OLS of each trait prediction on the 5 topic PCs (the same
    construction as F10/shortcut_r1).
  - Report the residual variance share and the correlation between residualized and
    original predictions per trait.
  - Access is logged before the run. No labels.
- **FU1-d (R01): matched source-fit comparison, 4 vs 24 coordinates.**
  - The same no-intercept OLS is fitted on the 36 source pairs to the full-text contrast,
    for each input.
  - It is evaluated on the 72 application pairs against the measured donor action and
    the full contrast.
  - The estimator is matched and there is no J. The rule reports the r difference with a
    cell-bootstrap CI (2,000 draws).
- **FU1-e (R27): real-text non-identifiability benchmark, opportunity vs choice.**
  - Two planting mechanisms produce pseudo-author subreddit compositions from the
    PANDORA pool.
  - Choice: z_u tilts selection among offered venues; all venues are offered.
  - Opportunity: z_u tilts which venues are OFFERED (exposure); selection among offered
    venues is uniform by frequency.
  - Both use the same w_s and beta 1.0, with 400 pseudo-authors x 5 replicates.
  - Report the recovery of z from composition under each mechanism, and a classifier
    distinguishing the mechanisms from composition alone (cross-validated AUC), with and
    without the exposure log (offered-venue indicators).
  - Prediction: without the exposure log, the mechanisms are indistinguishable (AUC about
    .5); with it, they are distinguishable.
- **FU1-f (R08): G-invariant bisimulation.**
  - The c2_xor world is augmented with a nuisance bit n: a fair coin, re-drawn each step
    and emitted, with no effect on the target.
  - Compute the metric on the full state (window x nuisance), then on the G-quotient
    (G = flipping n; the reward uses the target outcome only).
  - Report the max d between states that differ only in n, before and after the quotient.
- **FU1-g (R36): codified stage gates.** A function over the declared pipeline stages
  (P03 p.32, P04 pp.27-31) returns PASS, FAIL, BLOCKED or NOT_RUN per stage, plus the
  first failing gate. It is applied to the two mapped phenomena, with tests.
