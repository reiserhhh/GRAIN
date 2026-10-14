# F13 recovery 3 protocol v1: binary next-p uncertainty for B1/B2/M under missingness

Frozen 2026-09-26 before running. Worker: Claude Opus 5.5
(TASK-CLAUDE-GRAIN-FULL-LOOP-20260926). The census gaps this closes:
- F13-02: next-p uncertainty existed for B1 only; B2/M parameter uncertainty
  was open.
- F13-04: missingness was run for linear and dynamic targets only; binary
  (F08) targets were open.

## Target and truth

- Worlds C2 xor and C2 iid, seeds 20260915-17, 200 application authors each,
  48 events.
- The transition law is shared across persons (F02), so the true next
  probability at time t is c2_next_probability(y_{t-2}, y_{t-1}), a function
  of the true history. It is used only by the terminal scorer.
- Predictions are made at every t >= 12 before y_t is seen, exactly as in the
  F08 application scoring.

## Observation masks (application histories only; numeric observations, no reader)

The binary adaptation targets a missing rate of about .4 under each
mechanism:

| mechanism | missing probability |
|---|---|
| none | 0 |
| MCAR | .4 |
| MAR | .2 if the most recent observed value is 0 or there is none; .6 if it is 1 |
| MNAR | .2 if y_t = 0; .6 if y_t = 1 |

- Mask stream: SeedSequence([seed, 13, 30, world_index, author_index,
  mechanism_index]).
- A missing event enters the frozen filter as qualification 0: an
  uninformative likelihood, and the time position is kept.

## Estimator and components

- Source fits:
  - the canonical `fit_observed_source` fit on the 300 source authors gives
    the point prediction;
  - the 200 source bootstrap refits in `F13/binary_source_full_r2` (hashes
    recorded) give the source variance.
- Filter: the frozen QualifiedHistory filtered-mode semantics, vectorised
  across fits. It is verified to equal QualifiedHistory exactly for the
  canonical fit.
- Per prediction and candidate:
  - **state:** variance across lag states of the per-state prediction under
    the canonical fit's filter posterior;
  - **personal:**
    - B1: 0, since it has no personal parameters;
    - B2: sum_s w_s F_s S F_s', where S = .25 (gram + I)^-1 is the ridge
      working posterior covariance and .25 is the declared maximal Bernoulli
      variance;
    - M: sum_s w_s sum_j P_js^2 r_j (1 - r_j)/(n_j + 3), the Beta(c1+1, c0+1)
      variance of the raw personal table projected by P = R'R;
  - **source:** variance of the prediction across the 200 bootstrap refits;
  - **event:** sum_s w_s v_s (1 - v_s), reported and not part of the
    probability interval.
- Probability interval: canonical prediction ± 1.96 sqrt(state + personal +
  source), clipped to [0,1]. This is a working moment interval, not a
  posterior.

## Metrics and checks

- Per author, averaged over t >= 12:
  - squared error versus the true next-p, and bias;
  - Brier versus y;
  - interval coverage of the true next-p;
  - width;
  - component means.
- Pooled summaries by world x mechanism x candidate. Seed-stratified author
  bootstrap: 2000 draws, seed 20260915. Coverage is summarised at author
  level with author-cluster intervals, because events within an author are
  dependent.
- Paired contrasts:
  - masked minus none, for squared error versus the true next-p (cost of
    missingness);
  - MNAR minus MCAR, for bias (a selection effect with the missing rate
    matched).
- Built-in checks:
  - (a) with no mask, per-author Brier equals `F08/end_to_end_r1` complete
    panel `oracle_channel_oracle_source` exactly (tolerance 1e-12);
  - (b) the vectorised canonical-fit predictions equal QualifiedHistory for
    the first 3 authors of every world/seed/mechanism (1e-12);
  - (c) exactly 200 bootstrap fits are loaded per world/seed.
- COMMON decision rule. Coverage has no nominal target claim; the interval is
  a working interval.

## Non-claims

- Synthetic shared-law worlds; person-specific laws are not present, so the
  B2/M personal variance measures adaptation noise.
- No reader or material component; that is F16 text transport.
- No human interval validity.
