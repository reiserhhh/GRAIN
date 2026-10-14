# GRAIN Packet 3 protocol V1: R18 rival mechanisms with person parameters; R21 held-out map

Frozen 2026-09-26 JST under SEMINAR_FABLE_03_DECISION.md Packet 3, before any
computation. No neural calls. Simulation calibrates designs, not human truth.

## Part A (R18): declared factorial simulation

- Persons N = 400; two sessions; five binary factors crossed (32 cells), 3 trials per
  cell per session (96 trials per person-session). Seed 20260926.
- Factors: B private belief certainty (1 = certain), e evaluation stakes, r relationship
  or face threat, k commitment cost, g formal register. Private belief B and choice C
  (acts on the position; C = B with probability .9) are recorded per trial.
- Outcome hedge y ~ Bernoulli(logistic(eta)),
  eta = alpha_u + th1_u (1-B) + th2_u e + th3_u r + th4_u B k + gamma_u g.
  - M1 uncertainty th1, M2 evaluation anxiety th2, M3 relationship preservation th3,
    M4 strategic ambiguity th4: hedges while certain when commitment is costly.
  - M5 convention/proficiency rival: gamma_u = .3 + .8 p_u, with proficiency p_u ~ N(0,1).
- Worlds:
  - W1 all mechanisms. Person SDs (1.0, .8, .8, .8), means (1.0, .5, .5, .5), mechanism
    correlations 0, alpha_u ~ N(-1, .5), plus M5.
  - W2 convention only: every th fixed at its W1 mean (no person variance), plus M5.
  - Existing trait T_u = .5 z(th2) + .3 z(th1) + sqrt(1-.25-.09) noise in W1, pure noise in W2.
- Designs (person-level linear-probability OLS per session; coefficients are the
  estimated person parameters):
  - D1 factorial, B recorded: (1-B), e, r, B k, g.
  - D2 factorial, B not recorded: e, r, k, g. M1 and M4 have no own regressor, and k
    carries M4.
  - D3 factorial, B not recorded, choice C as proxy: (1-C), e, r, C k, g.
  - D4 naturalistic: g not coded and tied to e (g = e on 90% of trials); (1-B), e, r, B k.
- Metrics per world x design x mechanism coordinate:
  - recovery r(est S1, true), W1 only;
  - test-retest r(S1, S2);
  - estimated person SD = sqrt(max(0, var(est) - mean sampling var));
  - W1 redundancy R^2 of est on T;
  - partial retest r(S1, S2 | T).
  - Person bootstrap CIs, 1,000 draws.
- Rules:
  - identified: recovery >= .5 and retest >= .5;
  - non-redundant: R^2 on T <= .5 and partial retest >= .3;
  - false mechanism in W2: retest >= .30 for a coordinate whose true person variance is 0.

## Part B (R21): source-fitted map with held-out evaluation

- Items: the 380 cached F16 synthetic materials, with 24-d historical GRAIN features and
  9 NLI probabilities (`F16/text_transport_full_r1`).
- Split by template, not item: a seeded permutation (20260926) of templates into two
  halves. The map is fitted on one half and evaluated on the other, then the roles swap.
  Every item is held out exactly once and never fitted and evaluated together.
- Maps: OLS with intercept, GRAIN to NLI and NLI to GRAIN. No alpha or rank choice.
- Held-out metrics:
  - R^2 against the source-half mean;
  - RDM Spearman of mapped vs target, against the unmapped RDM Spearman on the same held
    items;
  - sup distance d_C = max ||f(x) - y|| and its 95th percentile, in units of the median
    held-out between-item target distance;
  - design-label RSA of the mapped representation (pole, subject status) against the
    target's own.
- Random-matched baseline: 200 maps fitted with permuted item correspondence on the
  source half.
- Rules:
  - functional agreement at the finite level requires held-out mapped RDM Spearman above
    the random-matched 95th percentile and d_C below the natural median between-item
    distance;
  - otherwise identity is not supported;
  - the sup is over these held-out items only and is not a global supremum.

## Amendment A1 (after Part B, before any Part C computation)

Part C (R21 multi-encoder gauge, same real texts, zero neural calls).
- Items: the R17 natural_sensitivity_r1 cached vectors of the 100 base units and 100
  natural partner units, with E5 (partners from the compatible historical cache) and BGE
  for every item. Edited versions are excluded as near-duplicates.
- Across settings: fit on one corpus (100 items) and evaluate on the other (100 items),
  both ways, in both directions (E5 to BGE, BGE to E5).
- Maps:
  - (i) orthogonal Procrustes after centring on the fit-corpus means, a pure gauge;
  - (ii) minimum-norm least squares with intercept.
  - No alpha or rank choice.
- Metrics:
  - held-out mean cosine of mapped vs target;
  - mapped RDM Spearman vs the gauge-free unmapped RDM Spearman;
  - random-matched q95 over 200 permuted correspondences;
  - sup and 95th-percentile distance over the median held-out between-item target
    distance.
- Rule: as Part B.
