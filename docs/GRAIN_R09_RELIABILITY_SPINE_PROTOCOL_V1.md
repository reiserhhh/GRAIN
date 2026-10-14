# R09 level / response / state / crossed-wording reliability spine: protocol V1

Frozen 2026-09-26 JST before any run.
- **Authority.** `NEXT_PACKET_18_PRODUCT_SPACE_AND_RELIABILITY.md` Part B; `SEMINAR_FABLE_16_DECISION.md`. Correction
  3 (the reliability formula) and correction 4 (estimated vs declared variance) are binding.
- **Original clauses.** P01 pp.10–11, P02 pp.22–24, P05 §2–3.
- **Code.** Module `grain_research/reliability_spine.py`; runner `scripts/run_grain_r09_reliability_spine.py`.
- **Output.** `results/grain_gptpro_foundation_v3/F04/reliability_spine_r1/`.
- **Execution.** CPU ≤ 20 min, one process, single-thread BLAS.
- **Not a rerun.** The completed F04 L/T/Q/V battery is not repeated. The only reuse is reading the existing T-world
  artifacts for item 6. All other data are **new draws**, labelled as such.

## Design (fixed)

- **Coordinates and conditions.** Observed coordinates k ∈ {1, 2}. Condition c = (c1, c2).
  - **Observed conditions:** (−1,−1), (−1,1), (1,−1), (1,1).
  - **Held (unseen) conditions:** (−.5,0), (.5,0), (0,−.5), (0,.5). These are disjoint from the observed set and never
    used in any fit.
- **Replication.** Wording facets f ∈ {1,2,3}; occasions o ∈ {1,2}; repetitions r ∈ {1,2} per (c, f, o). That gives 24
  events per occasion and 48 events per person. Each event is one row carrying both coordinates.
- **State clock.** t = 0…23 within each occasion. The event order is one of four fixed permutations P0…P3 of the 24
  (c, f, r) cells, generated once with `default_rng(18090)` and saved. Person u uses P_{u mod 4} in occasion 1 and
  P_{(u+1) mod 4} in occasion 2. There is no hidden reassignment.
- **Persons.** 300 source and 120 disjoint application persons per world and seed.
- **Worlds and seeds.** Worlds LOW, HIGH and CURVED; seeds 1, 2 and 3.
- **Scale.** 9 × 420 × 48 = 181,440 multichannel rows, under the 500,000 cap.

## Generative model (documented units)

- **Response.** y_{u,o,t,k} = a_{u,k} + (B_u c)_k + κ(u,k,c) + g_{f,k} + d_{u,c,f,k} + s_{u,o,t,k} + e.
- **Person parameters.** θ_u = (a1, a2, B11, B12, B21, B22) ~ N(μ_θ, Σ_θ).
  - μ_θ = (0, 0, .5, .2, −.1, .4).
  - Σ_θ: Var(a1) = Var(a2) = 1, Cov(a1, a2) = .3; Var(B) = (.16, .04, .04, .16), Cov(B11, B22) = .048. a and B are
    independent.
- **Wording main effects.** Fixed: g_{·,1} = (.2, −.1, −.1) and g_{·,2} = (−.15, .05, .10). The identification
  constraint is Σ_f g_{f,k} = 0.
- **Crossed nuisance.** d_{u,c,f,k} ~ N(0, σ_d²), iid over person, observed condition, facet and coordinate. It is
  persistent across occasions and repetitions.
  - σ_d² = .05 in LOW and CURVED, and .15 in HIGH. These are generating parameters, not claimed empirical shares.
- **State.** An AR(1) path over t within each occasion: s_0 ~ N(0, σ_s²), s_t = ρ s_{t−1} + w_t.
  - ρ = .6, σ_w² = .16, σ_s² = .25.
  - States restart independently for each occasion and are independent across coordinates.
- **Measurement error.** e ~ N(0, .25), iid.
- **CURVED.** κ = γ_{u,k}(1 − (c1² + c2²)/2), with γ ~ N(.3, .2²) iid. κ = 0 at every observed condition and .875γ at
  every held condition. The observed grid therefore cannot see it. Elsewhere κ = 0.
- **Target at a held condition.** m_{u,k}(c*) = a_{u,k} + (B_u c*)_k + κ(u,k,c*). This is the same target in every
  world; the curved world is not compared against a pseudo-true linear B.

## Estimation (fitted arm; noisy source data only)

- **Model.** A linear mixed model per person, stacked over 96 observations: y_u = Xβ + Z_θ(θ_u − μ_θ) + Z_d d_u + s_u + e.
  - β = (μ_θ, 4 facet contrasts), with sum-to-zero coding (f1 − f3, f2 − f3 per coordinate).
  - V_u = Z_θΣ_θZ_θᵀ + σ_d² Z_dZ_dᵀ + S(ρ, σ_s²; order) + σ_e² I.
- **REML.** −2ℓ_R = Σ_u [log|V_u| + r_uᵀV_u⁻¹r_u] + log|Σ_u XᵀV_u⁻¹X|, with β̂ by GLS.
  - Parameters: the Cholesky factor of Σ_θ (21 values), log σ_d², atanh ρ, log σ_w², log σ_e².
  - Optimizer: L-BFGS-B with numerical gradient, maxiter 1,000.
  - Initial values (declared): Σ_θ = the covariance of per-person source OLS θ; σ_d² = .1; ρ = .5; σ_w² = .2;
    σ_e² = .2.
- **Fallback.** If the optimizer does not succeed, one Powell attempt is made (maxiter 4,000). Both are recorded. If
  both fail, part B stops, and no estimator is fabricated.
- **Recorded.** Boundary flags (any variance < 1e-6, or |ρ| > .99) and Hessian-based SEs (numerical, on the
  transformed scale, delta method).
- **Plug-in scoring.** Application scoring uses the frozen source ψ̂ and β̂ as plug-ins. Source-fit uncertainty is
  **not** propagated into application intervals. This is stated, and the per-seed spread of ψ̂ is reported.

## Scoring (application persons; observed prefix only)

- **Posterior.** Gaussian conditioning of b = (θ − μ_θ, d, the 96 row states) on the chosen rows.
- **Fitted arm:** uses ψ̂ and β̂. **Known-parameter oracle arm:** uses true ψ and β. The oracle is never fed to the
  fitted arm.

## Outputs

1. **Known-model scalar target reliability** (LOW, HIGH, and the θ targets in CURVED):
   R = 1 − E[Var(θ_j | Y)] / Var(θ_j), using total variance. This is **not** prior / (prior + posterior).
   - Var(θ_j | Y) is deterministic given the order pair, and is averaged over the persons' order pairs.
   - Targets: a1, a2, B11, B12, B21, B22; m_k(c*) at the 4 held conditions; state s at occasion 2, t ∈ {0, 11, 23},
     k = 1, smoothed.
   - **Unit-tested limits:** no data gives 0; noiseless and identified gives 1; unidentified directions (a design
     observing only (1,1)) stay below 1.
   - **Monte Carlo check:** corr²(oracle posterior mean, truth), which equals Var(E[θ|Y])/Var(θ) under correct
     specification.
2. **Fitted repeat and wording stability** (EB θ̂ under the frozen fit, from half designs).
   - **Shared wording:** (o1, f) vs (o2, f), for f = 1…3.
   - **Disjoint wording:** (o1, f) vs (o2, f′), f ≠ f′ (6 pairs).
   - Each half has 8 events (4 conditions × 2 repetitions).
   - **Statistics:** ICC(A,1), two-way absolute agreement, single measure, and ICC(C,1), consistency (McGraw & Wong),
     plus Pearson r, each averaged over the pairs.
   - **Compared with target recovery:** r(θ̂_half, θ)² and r(θ̂_full, θ). Stable crossed nuisance inflates the
     shared-wording retest; the difference between shared and disjoint is reported. The definitions are not forced to
     agree.
3. **Same-time state recovery.** At occasion 2, t ∈ {0, 11, 23}, k ∈ {1, 2}:
   - smoothed (all data) and filtered (all of occasion 1 plus occasion-2 events ≤ t);
   - fitted and oracle arms;
   - reported: corr with the true state, RMSE, mean posterior SD, and 95% coverage.
   - Stable-level recovery r(â, a) and r(B̂, B) is reported alongside. This is not an across-time "state retest".
4. **Person × condition × wording variance.** σ̂_d² per seed (REML, with Hessian SE) against the generating value.
   Facet main effects ĝ are compared with the true g as fixed effects. With 3 facets, the facet population variance is
   not estimated.
5. **Held-context vector mean prediction** at the 4 held conditions, per coordinate.
   - **Predictors:** EB fitted, EB oracle, pooled/static (μ̂_a + μ̂_B c*), and personal OLS of y on (1, c1, c2) over
     all 48 events.
   - **Reported:** MSE per coordinate; MSE of the response contrasts [m(.5,0) − m(−.5,0) = B_{k1} and m(0,.5) −
     m(0,−.5) = B_{k2}]; and 95% coverage.
   - **Interval types:** EB plug-in posterior; pooled population predictive; OLS naive iid.
   - CURVED is a misspecification test. No tuning.
6. **Threshold (existing F04 T-world artifacts, no new run).**
   - Per application author, the step location among the 3 gaps of z ∈ {−1, −1/3, 1/3, 1} is estimated by least
     squares, with a + b·1[z > gap] and b free.
   - Estimated separately from each half: first 16 events vs last 16.
   - Reported: half-to-half bracket agreement (proportion and Cohen's κ), and recovery of the true bracket. The true τ
     is used only for scoring, stratified by |b|.
   - Continuous τ within a bracket is **not identified**, and no point reliability is reported.

- **Uncertainty.** Whole-person resampling with 1,000 fixed draws (seed 18091) on the saved per-person metrics,
  pooled over the 3 seeds within a world (360 persons). Rows and replicates are never treated as people.
- **Acceptance.** Numerical identities (unit tests), role separation (source-only fit; oracle never in the fitted
  arm), and real fitted estimates.
  - There is no .8 hurdle and no requirement that fits fall within 2 MCSE.
  - Outcomes may be negative.
  - This does not close R09. Real person-level reliability stays H01.
