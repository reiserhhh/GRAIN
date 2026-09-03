# V8.3.7F Prospective Confirmation Seal

Status: `SEALED_BEFORE_CONFIRMATION`

## Frozen estimand

V3.7F tests whether GRAIN can use an author-disjoint external score origin,
calibrate nested technical uncertainty, detect stable information omitted by
hard projection, and separate event-sampling decay from positive capacity or
state floors.

## Frozen design

- 30 fresh worlds under canonical seed `20410921`;
- 256 router-reference, 256 zero-reference, 128 calibration, and 96
  evaluation authors, all disjoint;
- exact rank-12, dense-tail-48, dense-state-alias, and author-permutation
  worlds;
- event budgets 32/64/128/256/512;
- 12 tracked evaluation authors;
- 6 reference resamples, 6 calibration/rank resamples, and 64 event draws;
- event coverage uses scorer-known synthetic repeated sampling;
- hard full-space is retained when adaptive rank is 48; otherwise the
  all-positive-energy soft operator is the information-conserving score.

## Frozen primary gates

1. Numeric integrity, disjoint authors, unique RNG streams, row counts,
   parent seal, own seal, and artifact inventory must pass.
2. Evaluation-cohort composition may change a fixed author's score by at
   most \(10^{-12}\).
3. Population-shift direction cosine must be at least .97 and amplitude must
   lie in [.85, 1.15].
4. The author-permutation world must select rank at most 2 in at least 90% of
   worlds and residual AUC must lie in [.47, .53].
5. Event-only and full-pipeline held-out Monte Carlo coverage must each lie in
   [.90, .98]. The 256/64 full-region radius ratio must be at most .70.
6. Functional-ANOVA reconstruction error must be at most .02. Increasing
   each external reference panel from 64 to 256 authors must reduce external
   zero variance by at least 50%.
7. The exact-rank scorer-only residual AUC upper 95% bound must be at most
   .55. Dense-tail residual AUC lower 95% must exceed .65, and its conditional
   excess over chance must exceed .15.
8. Exact-rank capacity-floor upper 95% must be at most .025; dense hard-floor
   lower 95% must exceed .025. The information-conserving rule must reduce
   dense hard floor by at least 15% on average.
9. Event-only asymptotic floor upper 95% must be at most .02, while
   dense-state-alias infinite-floor lower 95% must exceed .10.
10. Null-change false positive must be at most .07 and two-MDC planted-change
    power at least .80.

## Boundary

This seal licenses only a planted synthetic mechanism result. Author
information is not personality information. Nonzero information in every
synthetic direction does not prove that every token is psychologically
informative or that total Bayes error vanishes.
