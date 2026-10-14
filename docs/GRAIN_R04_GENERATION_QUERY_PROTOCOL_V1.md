# R04 generating causes versus query-specific sensors: protocol V1 (packet 32 C)

Frozen 2026-09-27 10:07 JST, before any experiment seed is run. Authority: `NEXT_PACKET_32_GENERATION_QUERY.md`;
`SUPERVISOR_CHECK_20260927_0955.md`. Original clauses:
- P04 pp.1–2 and p.8: X ~ P(X | Θ, Z, C, G, A, L, K, H), with G goal, A audience/social-evaluation model, L
  language/culture and K proficiency. Ignoring G/A/L/K makes P(Θ | X) an under-identified inverse problem. Structural
  versus practical non-identifiability.
- P03 pp.12–15: response law, encoder sensor h = φ(X) and query reader y = ρ_q(h, C) are separate.
- Audit r3, R04.u1/u2: G/A/L absent; the "query-specific readers" were one sensor at two noise levels (FU3-e); K scaled
  reader noise.

Declared-world numbers only. Output root `results/grain_gptpro_foundation_v3/F01/generation_query_r1/` (x-mode).

## Scope and inheritance

- **Restricted static subcase.** FU3-e's opportunity, history, state feedback and ξ estimation are HELD FIXED (absent
  here) because they are answered there; this packet isolates G/A/L/K and sensors.
- **Not claimed.** This is not the complete natural-language generating system, and G/A/L/K are declared
  mathematical mechanisms, not validated psychological effects.

## Observation law (module docstring)

- **Expression.** e = θ w_θ + g w_G + a w_A + d_L, with:
  - w_θ = (1, 0, .4), w_G = (0, 2, .5), w_A = (.3, .2, 1);
  - d_0 = 0, d_1 = (.3, −.2, .1).
- **Realization.** K on the SPEAKER: x = K M_L e + .3ε, with M_0 = I, M_1 = diag(1.3, .8, 1) and log K ~ N(0, .25²).
- **Sensors.** Read the same x: y_q = r_qᵀx + λ_q a + .3η.
  - DISTINCT pair: q1, r = (1, .5, 0), λ = 0; and q2, r = (0, .6, 1), λ = .4, a reader-side context response.
  - REPLICA pair: q1 and a matched-noise copy of q1.
- **Target and reference.** θ_u, on the declared reference ν = the L0 source population, standardized. True
  θ ~ N(0, 1) in L0 and N(.4, 1) in L1: a real group difference that belongs to the target.
- **Pre-freeze note.** A scratch-world sanity check (not these seeds) showed that the first declared goal contrast
  (w_G = (0, 1, .3)) made the per-author α/β_G ratio heavy-tailed (practical non-identification). The designed goal
  contrast was set to (0, 2, .5) before freezing. Low-K authors still give weak contrasts, and these are reported.

## Observation availability

- **Observed:** query readouts per event, the recorded designed g and a (in crossed arms) and language L (metadata).
- **Simulator / auditor only:** θ, K, loadings and the generating λ.

## Arms (predeclared; paired on the same authors and noise draws)

- **CROSS_DISTINCT.** Crossed recorded G×A (8 events per cell, 32 events), distinct sensors, bilingual source anchors.
- **CROSS_REPLICA.** The same with replica sensors.
- **CROSS_DISTINCT_NOANCHOR.** The same as CROSS_DISTINCT, but L1 calibrated by within-language standardization
  (no anchors).
- **CONF_DISTINCT.** G / A unrecorded and self-selected per author (g = 1{θ + N(0,1) > 0}, a = 1{−θ + N(0,1) > 0}),
  one condition for all 32 events. θ is REFUSED; a naive standardized readout is reported as a non-estimator
  reference.
- **ORACLE.** Known loadings, labelled reference only.

## Estimator (observations and recorded design only)

- **Per author.** OLS y_q ~ 1 + g + a gives α_q, β_Gq, β_Aq. Then ρ_q = α_q/β_Gq = θ c_q^L + o_q^L, with K cancelling.
- **Source calibration** (independent source authors, frozen before application scoring):
  - L0: c0 = √(Var ρ − mean δ-noise), e0 = mean ρ.
  - L1 with anchors: 100 bilingual source authors observed in both languages (the same θ and K). Errors-in-variables
    slope of ρ¹ on ρ⁰ and intercept.
  - L1 without anchors: within-L1 standardization, which implicitly sets the L1 mean to the reference mean.
- **Scores.** θ̂ = mean_q (ρ_q − off_q^L)/scale_q^L, with delta-method SE conditional on the calibration.
- **Reader-side λ_q.** The errors-in-variables regression of β_A on β_G across source L0 authors has intercept λ̂_q.

## Exact witnesses (analytic, checked numerically)

- **W1, CONF.** For a true author (θ = .5, K = 1, g = 0, a = 1) there is a distinct (θ′, K′) with (g′ = 1, a′ = 1) giving
  identical expected readouts in BOTH distinct sensors, hence identical distributions: θ is not identified.
- **W2, reader-side versus generation-side.** When K is constant across authors, (b_A, λ) and (b_A + δ/K, λ − δ) give
  identical readouts in all cells, even with distinct sensors.
- **W3, no anchors.** (μ_L1, o¹) and (0, o¹ + μ_L1·c¹) give identical L1 observations.
- **W4, rank.** The composite map has rank 2 (distinct) against rank 1 (replica) per event. The per-author Jacobian in
  (θ, log K) has rank 2 in the crossed design; with g and a unknown, the unknowns outnumber the equations.

## Scale, metrics and contrasts

- **Scale.** 3 seeds.
  - Application: 400 authors (200 L0, 200 L1), `SeedSequence([20260927, 32, seed, 1])`.
  - Source: 200 monolingual authors (100 per language), `[.., 2]`.
  - Anchors: 100 bilingual authors, `[.., 3]`. Anchor θ ~ N(0, 1) and log K ~ N(0, .25²); only within-author
    comparison is used.
- **Metrics.** Bias (overall and by L), MSE, 95% coverage, squared correlation corr(θ̂, θ)² (labelled), and λ̂ against
  generating λ (audit). Composite singular values; naive CONF score correlations with θ, K, g and a.
- **Paired application-author bootstrap.** 2,000 draws, `[.., 9]`, on squared error:
  - distinct − replica;
  - anchor − noanchor (all authors and L1 only);
  - distinct − oracle.
- **No hyperparameter grid. No r leaderboard.**

## Bindings

| file | sha256 |
|---|---|
| `grain_research/r04_generation_query.py` | `e1a3694f14cc0cda56d80397ff7f07aa2c095fbfa9893089f91f5f1c955910e6` |
| `tests/test_grain_r04_generation_query.py` | `78d3a807a47bbb4a718d859ac0b0f2d8c29cf8914b12134bae49408c79605db9` |
| `scripts/run_grain_reaudit_r2_followup3.py` | `40685df8047ff048c7f6749b8d3bc05f2db35e7ca3a75a3a6b25b1489d3fb91d` |
