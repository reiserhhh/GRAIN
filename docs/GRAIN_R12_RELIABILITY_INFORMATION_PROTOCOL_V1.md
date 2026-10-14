# R12 reliability, context diversity and informative coverage: protocol V1 (packet 28)

Frozen 2026-09-27 06:05 JST, before any simulation or scoring.
- **Authority.** `NEXT_PACKET_28_RELIABILITY_INFORMATION.md` Parts B–F; `SUPERVISOR_CHECK_20260927_0553.md`.
- **Original clauses.** P01 p.21 (response-slope reliability); P02 p.13 (individual precision vs group law); P02 pp.29–30
  (reliability against tokens AND against context diversity; "number of informative contexts"; FIM rank); P03 p.7 and
  p.31 (reliability paradox: mean effect and individual-difference reliability reported separately). Audit
  `F17/source_audit_r3/AUDIT_A_REPORT.md` R12: diversity-curve curvature EXECUTABLE; FU2-c bootstrap CI omitted.
- **Code.** New module `grain_research/r12_reliability_information.py` and tests (bound). New runner
  `scripts/run_grain_r12_reliability_information.py` holds this protocol's sha. Frozen old code is imported, never
  edited.
- **Output root.** `results/grain_gptpro_foundation_v3/F04/reliability_information_r1/` (x-mode). The FU2-c replay is
  saved in its subfolder `fu2c_replay/`: a new root relative to F09, which stays untouched.
- **Budget.** Compute ≤ 10 min, one process, single-thread BLAS. 0 neural/API calls; synthetic numbers only.

## World (declared, frozen F04 excitation packet; not a real dataset)

y_ut = a_u + b_u z_t + e_ut; a ~ N(0, 1), b ~ N(.5, .5²), e ~ N(0, .5²) iid, so Var(b) = σ² = .25. "Readings" are
simulated readings, NOT measured language tokens; no empirical token claim.

## Analytic explanation (C)

- Per-half OLS slope with intercept: b̂_h = b + Σ zc_t e_t / Szz_h, so v_h = Var(b̂_h | b) = σ²/Szz_h exactly.
- Half reliability (independent half errors, random b): ρ_AB = Var(b)/√((Var(b)+v_A)(Var(b)+v_B)).
- Full-slope reliability: ρ_full = Var(b)/(Var(b) + σ²/Szz_full).
- **Spearman–Brown assumptions.** ρ_full = 2ρ/(1+ρ) exactly when v_A = v_B AND the halves share the same z mean, so
  Szz_full = Szz_A + Szz_B, with independent homoscedastic errors, a common σ², no slope-by-half interaction and b the
  only person-varying slope component. Unequal halves break the identity (tested).
- **K = 1** has Szz = 0 and a rank-1 information matrix. The slope is UNIDENTIFIED and refused. It is not reliability
  0 and not a pseudoinverse solution.
- Bayes posterior variance of b under the source prior (a ~ N(0,1), b ~ N(·, .25)) is reported separately. It is never
  mixed with OLS repeat reliability.

## Designs (old grid exactly; no search)

- Old design: z = K conditions evenly over [−1, 1] (linspace), cycled to n readings; block halves [0, n/2) and
  [n/2, n), each containing every condition equally often.
- **Tokens family** (K = 4): n ∈ {8, 16, 32, 64}.
- **Diversity family** (n = 32): K ∈ {1 refused, 2, 4, 8}.
- **Equal-Szz diagnostic family** (n = 32): K ∈ {2, 4, 8}, z rescaled about its mean so EACH half has centered SS equal
  to the K = 4 half value 80/9 ≈ 8.889. K2 shrinks to ±.745; K8 expands to ±1.139. **Matching spread changes the
  coordinate / support; this is an algebraic diagnostic, not extra empirical contexts.**
- Per design and half: Szz, information rank, smallest positive eigenvalue of X'X/σ², analytic and empirical
  reliability and MSE.

## Paired simulation

- 3 frozen seeds s = 0, 1, 2: `default_rng(SeedSequence([20260927, 28, s]))`. Draw a (400), then b (400), then
  E (400 × 64).
- **Common random numbers.** The same authors and the same noise matrix serve every design. A design with n readings
  uses E[:, :n].
- **Saved.** Individual half estimates b̂_A, b̂_B, full b̂, truths a and b, design identities.

## Estimands per design (per seed and pooled)

- half_r_empirical = corr(b̂_A, b̂_B).
- whole_SB_empirical = SB(half_r_empirical).
- whole_truth_empirical = corr(b̂_full, b)², available only in simulation.
- half_r_theory and whole_theory: analytic, population Var(b).
- mse_full_empirical = mean (b̂_full − b)², against theory σ²/Szz_full.
- mse_half_empirical and theory.
- posterior_var_b_theory: separate label.

## Uncertainty and curve shape (D)

- **Paired cluster bootstrap.** 2,000 draws, `default_rng(SeedSequence([20260927, 28, 900]))`. Authors are resampled
  WITHIN each seed stratum, with the same indices for every design, so whole author trajectories are preserved.
  Per-seed CIs use that seed's block; pooled CIs are stratified by seed. These are conditional simulated-population
  intervals.
- **Contrasts.**
  - Tokens: 16−8, 32−16, 64−32.
  - Diversity and equal-Szz: K4−K2, K8−K4, K8−K2.
  - Equal-Szz minus raw at K2 and K8.
  - Stats: half_r, whole_SB, whole_truth, mse_full.
- **Curvature, defined before computation.** On grid x (tokens: x = log2 n = 3, 4, 5, 6; diversity and equal-Szz:
  x = log2 K = 1, 2, 3; spacing 1; K1 excluded):
  - first divided differences D_i = (f_{i+1} − f_i)/(x_{i+1} − x_i);
  - second divided differences C_i = (D_{i+1} − D_i)/(x_{i+2} − x_i).
  - Applied to whole_SB_empirical (primary) and half_r_empirical, with CIs from the same paired draws, and compared to
    the analytic Szz prediction. Report even if signs disagree.
- **No success threshold.** Negative or inconclusive outcomes satisfy execution. There is no r threshold.

## FU2-c omitted uncertainty (E)

- **Replay only F09 W1 / D1.** Frozen `scripts/run_grain_r18_rival_simulation.py` functions (persons,
  simulate_session, estimate), stream `SeedSequence([20260926, 0, 0])`, 400 persons × 2 sessions.
- **Exact reproduction checks** against the frozen `rival_person_metrics.csv` W1 / D1 rows (≤ 1e-9): every numeric
  column, including retest_ci replayed with `SeedSequence([20260926, 0, 0, 99])` and 1,000 draws in coordinate order.
  If reproduction fails, the clause stays OPEN and no replacement CI is reported.
- **New CIs.** A paired person bootstrap, 2,000 draws, `SeedSequence([20260927, 28, 901])`, for the session-1 LPM mean
  effect (the FU2-c estimand) and retest r, per coordinate M1–M5.
- **Units.** Estimates are linear-probability (LPM) units. True means are logit units, reported separately and never
  compared numerically. Template-level dual reports (F04/dual_report_r1) are not person-level evidence.

## Interpretation rules (declared)

- If the equal-Szz family's K contrasts are inconclusive and analytic values coincide, then **condition count alone is
  not the mechanism in this scalar linear world**. This does not rule out context-specific nonlinear responses or
  real-person diversity. No nonlinear or multidimensional grid is run.
- Computational success, meaning curves, CIs and curvature delivered, is separate from support for monotonically rising
  K curves (the P02 p.30 prediction).

## Checksum bindings

| file | sha256 |
|---|---|
| `grain_research/r12_reliability_information.py` | `916a7e90863caf2d3dea7ca39fb0fe3a18e739585e053bf29562790860fab04d` |
| `tests/test_grain_r12_reliability_information.py` | `884131f38568909d129e4da0c114b1868a4f7ef3b7c2be70b690b6e97e30ceb1` |
| `scripts/run_grain_gptpro_f04_excitation_reliability.py` | `0f7e9a0ec0c2a7cbc7ca7c0b3782be624a6b5f0661fce849251ae04b8d624ef8` |
| `results/grain_gptpro_foundation_v3/F04/excitation_reliability_r2/excitation_reliability.json` | `b5c94ccd669379f38446488f1fa12e2b0fd70d91c8cbde82519bbd58fefb2457` |
| `scripts/run_grain_r18_rival_simulation.py` | `4a17aa4c56376c283bdb2397e0614a37f74974a48e5aa266353d78dabfe667b5` |
| `results/grain_gptpro_foundation_v3/F09/rival_person_simulation_r1/rival_person_metrics.csv` | `ca3bfef28b06ceb4589bce37c8b54ce73e2adc5e5682d2cba5a25c56e0b35924` |
| `results/grain_gptpro_foundation_v3/F09/rival_person_simulation_r1/manifest.json` | `bf86bc59d6f89bdab3531d5851c4c69de046330513f862791a81fd1bca3a2e1b` |
| `scripts/run_grain_reaudit_r2_followup2.py` | `76d6cdd901d073cf6e4f3b4e9a94e70035cf5c9ed435f38188ec58dd717c87c2` |
| `results/grain_gptpro_foundation_v3/F17/reaudit_r2_followup2/followup2.json` | `bd46f325a5264ead38f503b0b612a4fa02db1b7ad0103558bd73c2566615e6d7` |
