# Seminar-05 queued item 4: decision-relevant design checks, protocol V1

Frozen 2026-09-26 JST before computation. Declared simulations only, no data, no neural
calls. Both parts change a design decision: the stopping rule for H04 adaptive probing,
and person N for H01.

## Part A: paired common-random-number comparison of adaptive designs (R26/A07; H04 design)

- The world is as in FU4-a: 5 rivals, 32 coded cells, known effect 1.5 and intercept -1.
- 1,000 persons, 200 per rival, seed 20260926.
- Common random numbers: each person has one uniform sequence u_t, t = 1..60. The
  response at trial t is 1[u_t < P(cell_t, truth)], whichever cell the policy chose. All
  three policies see the same u.
- Policies: FIXED, SCALAR_CAT, RIVAL_INFO, as in FU4-a.
- Stopping thresholds, declared: .90, .95, .97. Budget 60.
- Metrics: trials to threshold, share never reaching it, misclassification at stop,
  misclassification at budget.
- Paired person-bootstrap CIs (2,000 draws) for RIVAL_INFO minus SCALAR_CAT and
  RIVAL_INFO minus FIXED.
- Decision use: the smallest threshold whose misclassification at stop is at most 5% for
  the chosen policy becomes the H04 stopping rule.

## Part B: H01 final-design person-N recovery

- Design from contracts r4: 30 days, 4 contexts per day (z in {-1.5,-.5,.5,1.5}),
  paraphrase facet with 3 paraphrases rotating per day, arms A/B/C assigned to thirds of
  persons.
- Model:
  y = a_u + b_u z + s_{u,d} + pi_{u,phi} + m M_{u,d} + arm noise + e.
  - AR(1) state, rho .6, innovation SD .5.
  - pi ~ N(0, .05) per person and paraphrase.
  - The randomized manipulation M_{u,d} ~ Bernoulli(.2) has effect m = .5.
  - Arm noise variance: A .61 (topic selection plus interpretation), B .25, C 0.
  - e ~ N(0, .7^2). Person slope b ~ N(.5, .5^2).
- Missing days: 15% missing completely at random, plus state-dependent missingness with
  P = logistic(-2.5 + .8 s_{u,d}).
- Estimation: per-person OLS slope over observed readings with the manipulation as a
  covariate. Slopes are standardized within arm.
- Criterion: C = .2 z(b) + noise, observed with reliability .8.
- N in {200, 250, 300, 400}, 200 simulations each.
- Outputs: power of the slope-criterion test (two-sided alpha .05), slope split-half
  reliability, missing share. Recommended N is the smallest grid N with power >= .80.
