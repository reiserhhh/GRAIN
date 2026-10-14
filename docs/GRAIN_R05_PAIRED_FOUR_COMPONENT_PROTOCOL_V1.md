# R05 paired four-component identification: protocol V1 (packet21 B)

Frozen 2026-09-26 JST before any calculation.
- **Authority.** `NEXT_PACKET_21_FOUR_COMPONENT_IDENTIFICATION.md` Part B; `SEMINAR_FABLE_18_DECISION.md`, including its
  mathematical correction; `SUPERVISOR_CHECK_20260926_2245.md`.
- **Original clause.** P03 pp.21–22. Condition A: free choice of topic. B: the same forced topic for everyone. C:
  identical information material, with interpretation or reaction requested. The question is how much signal comes
  from topic selection, response style, semantic interpretation and linguistic realization.
  - Falsified if selection shows no stable individual differences after controlling opportunity, and no incremental
    external validity. That falsification part is HUMAN.
- **Code.**
  - Module `grain_research/r05_four_component.py` (bound below).
  - Runner `scripts/run_grain_r05_four_component.py`, which records its own code sha at run time.
- **Output root.** `results/grain_gptpro_foundation_v3/F03/paired_four_component_identification_r1/`, created in
  x-mode.
- **Budget.** ≤ 10 min CPU, one process, single-thread BLAS, 0 neural calls. No text, reader or private data. No
  parameter, sample-size, loading or accuracy grid.
- **Not reused as a substitute.** The F03 opportunity/selection/response/condition split is not this clause's
  selection/interpretation/style/realization distinction.

## Model (declared assumptions, not a human law)

    y[u, a, r] = μ_a + L_a θ_u + e_{u,a,r},   θ = (choice Q, interpretation I, style S, realization R) ~ N(0, Σ)

- **Pairing.** Every arm observes the SAME person's θ.
- **Replicates.** r = 1…3, independent given θ.
- **Errors.** e ~ N(0, .8²), independent across arms and replicates. Per-arm noise variances are estimated
  separately.
- **Centring.** Coordinates are group-relative; latent means and arm intercepts are not separately identified. The
  declared arm means are μ = (.2, 0, −.1, .1, .05).

## Declared nominal loadings (used by every estimator)

| arm | row | role |
|---|---|---|
| A | (1, 1, 1, 1) | idealized free choice |
| B | (0, 1, 1, 1) | fixed topic |
| C | (0, 1, 1, 1) | shared external material. **Interpretation is RETAINED.** B = C is a declared illustrative case, not an empirical law |
| D | (0, 1, 1, 0) | ideal realization clamp (idealized channel, not a validated human task) |
| E | (0, 1, 0, 0) | ideal interpretation anchor (idealized channel) |

## Worlds (true loadings and Σ; fixed before outcomes)

1. **W1 independent.** Σ = diag(.36, .25, .49, .16); true L = nominal.
2. **W2 correlated.** Same diagonal, with cov(Q, I) = −.18 and cov(Q, S) = −.168; other off-diagonals 0.
   - Eigenvalues are checked positive.
   - Population Var(A) − Var(B) = .36 + 2(−.18 − .168) = −.336 < 0, although choice variance is .36.
3. **W3 unseen leaky clamp.** True D = (0, 1, 1, .3); the estimator keeps nominal D.
   - Explicit reparameterization: I′ = I, S′ = S + .3R, R′ = .7R, Q′ = Q.
   - L_nominal T = L_true on every arm, so the observable fit is exact while named S and R are wrong.
4. **W4 observable restriction violation.** True C = (0, 1.75, 1, 1); the estimator keeps nominal C = B.

## Seminar-18 correction (analytic witness, required)

- **The claim being corrected.** A changed independent D row need NOT trigger misfit.
  - A, B, D and E form an invertible basis. For any true basis rows M_true with C = B preserved,
    T = M_nominal⁻¹ M_true gives L_nominal T = L_true on all 5 arms.
  - So L_nominal (T Σ Tᵀ) L_nominalᵀ = L_true Σ L_trueᵀ exactly.
- **Checks.**
  - W3's T.
  - A declared changed D style loading, true D = (0, 1, 1.5, 0).
  - An arbitrary invertible gauge G, showing that unknown L admits coordinate changes.
- **What the restriction can detect.** Only the C = B restriction has observable content: 5 moment equalities. Its
  violation (W4) can be detected. Detection does not locate the cause.

## Estimation (source persons only; observed arrays only)

- **Between-person arm covariance.** Ĉ = Cov(person means) − diag(σ̂²_a / R). σ̂²_a is the two-way residual mean square
  (replicate-column centred, ddof (N − 1)(R − 1)).
  - An alternative formula must reproduce Ĉ exactly: the mean cross-replicate covariance, with the diagonal from r ≠ r′
    pairs. Tolerance 1e-12.
  - Negative eigenvalues of Ĉ and Σ̂ are reported, never clipped.
- **Components.** Σ̂ = L⁺ Ĉ L⁺ᵀ with the declared nominal L. The fitted residual Ĉ − L Σ̂ Lᵀ is reported.
- **Restriction moments.** r = (C_CC − C_BB, C_BC − C_BB, C_AC − C_AB, C_CD − C_BD, C_CE − C_BE), reported separately from
  named-component recovery. The Wald statistic uses the bootstrap covariance (approximate χ²₅, descriptive).
- **Contrasts.** Naive Var(A) − Var(B) and Var(B) − Var(C); paired noise-corrected Var(A − B) and Var_between(B − C).
- **A/B/C-only design.**
  - Ranks: A/B/C 2, A/B/C/D 3, A…E 4.
  - Null witness v = (0, 1, −1, 0): Σ and Σ + c·vvᵀ have identical implied covariance under A/B/C and A/B/C/D.
    Replication does not change the rank.
  - Estimable functions reported: Var(Q), Var(I + S + R), Cov.
- **Application scores.**
  - Fixed contrasts on application person means after SOURCE-mean centring: A − B → Q, E → I, D − E → S, B − D → R.
    There is no fitted model and no application truth in inference.
  - Truth is used only for bias and MSE, for the empirical mixing matrix, and against the theoretical mixing
    K = CONTRASTS · L_true.
- **Uncertainty.**
  - 2,000 deterministic whole-person source bootstrap draws per world and seed, seed [21006, world, seed]. All arms
    and replicates of a person move together.
  - Paired whole-application bootstrap for score MSE, seed [21007, world, seed], source fixed.
  - Three-seed coverage is descriptive only.

## Seeds and size

- **Data.** Seeds [21005, s] for s = 1, 2, 3, with common random numbers across worlds (same standard normals).
- **Size.** 300 source and 150 application persons, 5 arms, 3 replicates. 4 worlds × 3 seeds = 81,000 scalar
  observations. Actual counts are reported.

## Acceptance and stop rule

- **PASS means:**
  - ranks, null witnesses and coordinate transformations replay;
  - estimators consume observed data only;
  - the paired covariance and noise identity holds;
  - correct and misspecified worlds are reported without forced improvement.
- **Not required.** CI coverage in every seed.
- **Stop.** Stop once the identification question is answered. No natural-text separator, semantic naming, causal
  variance fractions or precision search.
- **Scope.** Identification is conditional on declared loadings. Real task exclusion, and selection's incremental
  external validity, stay HUMAN.

## Checksum bindings

| file | sha256 |
|---|---|
| `grain_research/r05_four_component.py` | `e4b25b88971d30911dda592daa2388a88ac85bb3ff839a21492f6ab8fa7d5158` |
| `scripts/run_grain_reaudit_r2_followup3.py` | `40685df8047ff048c7f6749b8d3bc05f2db35e7ca3a75a3a6b25b1489d3fb91d` |
