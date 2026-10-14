# R29 correlated measurement error and duplicate information: protocol V1 (frozen before generating results)

Date 2026-09-26 JST. Authority: `SEMINAR_FABLE_10_DECISION.md` Packet 1. Original demand: P02:29–31, P03:5–11, P05:S3.
Keep covariance and interactions, distinguish repeated reports from new events, and never subtract error that is
correlated with authors or conditions mechanically.
- **Runner and output.** `scripts/run_grain_r29_covariance_identification.py`, writing to
  `results/grain_gptpro_foundation_v3/F13/covariance_identification_r1/`.
- **Scope.** Declared simulation plus exact equations. CPU, 2 workers × 1 BLAS thread, cap 30 min. No data, no neural
  calls.

## Model and identification (derived before simulation)

Y[u,m,t] = a_u + b_{u,m} + e_{u,m,t}, with readers m ∈ {1, 2} and occasions t ∈ {1, 2}.
- **Parameters:** σa² = Var(a), σbm² = Var(b_m), κ_m = Cov(a, b_m), κ12 = Cov(b_1, b_2), σem² = Var(e_m).
- **Blind moments:**
  - V_m = Cov(Y_um1, Y_um2) = σa² + 2κ_m + σbm²;
  - S12 = Cov(Ȳ_u1, Ȳ_u2) = σa² + κ_1 + κ_2 + κ12;
  - σem² = Var(Y_umt) − V_m;
  - reader contrast D = Var(b_1 − b_2) = V_1 + V_2 − 2·S12.
- **Count.** 5 identified moments against 8 parameters, so the null space has 3 dimensions.
- **Invariance.** The map a′ = a + c, b′_m = b_m − c leaves every Y unchanged. Var(a), Cov(a, b_m) and the common
  stable method component are not identified by repetition. No optimizer can supply the missing assumption.
- **Independent-effects comparator (assumption A0: κ_1 = κ_2 = κ12 = 0).** σ̂a² = S12 and σ̂bm² = V_m − S12. It is
  assumption-dependent and never labelled "recovered author variance".
- **Identified design (a separate, labelled arm).** Add an observed anchor channel Y[u,0,t] = a_u + e_{u,0,t},
  t ∈ {1, 2}, whose errors are independent across repetitions and of a, b and the reader errors. It is a declared
  assumption with a cost; no latent parameter is given to the estimator. Then:
  - V0 = Cov(Y_u01, Y_u02) = σa²;
  - C0m = Cov(Ȳ_u0, Ȳ_um) = σa² + κ_m, so κ_m = C0m − V0;
  - σbm² = V_m − V0 − 2κ_m;
  - κ12 = S12 − V0 − κ_1 − κ_2.

  This works because the anchor has no method component by design, which splits a from a + b. It is one sufficient
  design, not the only possible identification strategy.
- **Joint error decomposition (anchor arm).** Var(Y_m) = σa² + Var(b_m + e_m) + 2·Cov(a, b_m + e_m), checked
  numerically. The anchor does not license dropping the covariance term.
- **Condition sub-model.** Y[u,m,c,t] = a_u + β_c + g_{u,c} + b_{u,m} + d_{m,c} + e, with fixed condition effect β,
  fixed reader×condition bias d, and random person×condition response g.
  - Blind: only the reader-average condition effect β_1 − β_0 + d̄_{·1} − d̄_{·0} and the reader difference
    (d_{11} − d_{10}) − (d_{21} − d_{20}) are estimable. (β + t, d_m − t) is a 1-dimensional alias.
  - With the anchor (d_{0c} = 0 by design), β is identified.
  - Var(g) is identified from the variance of person condition contrasts after subtracting the error term. Constant
    offsets add no variance.
  - A constant condition offset is never called a personal response.

## Worlds (frozen; chosen by scientific distinction)

| World | Setup |
|---|---|
| W1 independent | σa² 1, σbm² .3, κ = κ12 = 0, σem² .5, anchor σe0² .5 |
| W2 author-correlated reader error (halo) | b_um = λ_m a_u + ν_um, λ = (.4, .2), Var(ν) .3, so κ_m = λ_m and κ12 = λ1λ2 |
| W3 shared stable method component, independent of a | b_um = h_u + ν_um, Var(h) .3, Var(ν) .2, so κ = 0 and κ12 = .3 |
| W4 conditions | W1 core plus c ∈ {0, 1}, β = (0, .3), d_1 = (0, .4), d_2 = (0, 0), σg² .15 |

N = 300 persons, R = 200 draws per world, seed 20260926, person bootstrap B = 200 per draw (on person-level
statistics, fixed centering; declared).

## Duplicate arm (same underlying draws as W1 and W2)

- **Insertion.** Each existing reader event row (u, m, t) gets one exact copy, with the same event_id and value,
  independently with probability r = .057. The observed PANDORA rate 4,775/83,355 motivates r; it is not a
  population parameter.
- **Two treatments:**
  - naive: every row is an independent occasion;
  - event-aware: rows are collapsed by event_id.
- **Reported.** Bias of V_m, σem² and the person-reliability ratio V_m/(V_m + σem²) for both treatments, bootstrap
  CI coverage, and effective evidence (rows vs events). The direction is reported whatever it turns out to be.

## Outputs and checks

- **Tables.** The identified-quantity table; a numeric same-observation counterexample (identical Y, different
  Var(a)); a population counterexample (identical moments, different parameters); per world, estimator bias with
  Monte Carlo SE and coverage with binomial MC SE.
- **Oracle diagnostic.** Mechanical subtraction σ̂a² = V̂_m − σbm²(true), labelled **ORACLE**. It is never a deployed
  estimator.
- **Tests.**
  - A blind fit asked for Var(a) without a declared assumption or an anchor must raise.
  - An anchor fit without an anchor channel must raise.
  - Exact population moments must return the exact parameters.
  - An event-aware fit must count a duplicated event once.
- **Construction rule.** Derived from the results. There is no requirement to force a positive recovery.
