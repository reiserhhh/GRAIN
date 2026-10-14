# R10/R29 common-shock episodes: protocol V1 (frozen before any new episode is generated)

Date 2026-09-26 JST. Authority: `NEXT_PACKET_11_PERSONAL_LAW_AND_SHARED_SHOCK.md`, Packet B. Original cards: F05-05
(R10, R11, R29; common shock vs ignoring model, "禁止用旧source猜新冲击", state and prediction reported separately)
and F13 (R24, R25, R29). Existing evidence: `F05/joint_full_r1` and `F05/joint_audit_r1`, which have three
independent episodes per world and no population interval.
- **Runner and output.** `scripts/run_grain_r10_common_shock_episodes.py`, writing to
  `results/grain_gptpro_foundation_v3/F05/common_shock_episodes_r1/`.
- **Scope.** CPU only, 2 workers × 1 BLAS thread, 0 neural calls, cap 30 minutes. `F05/joint_full_r1` and every
  archived file stay unchanged.

## Worlds and law (unchanged from `f05_recovery.generate_dynamics`, T = 64 events)

State and observation for author i in one role episode:

x_{i,t} = ρ_i x_{i,t−1} + c_t + w_{i,t},  y_{i,t} = a_i + b_i z_t + x_{i,t} + e_{i,t},  with z_t = −1 if ⌊t/4⌋ is even,
else +1.

| World | Common shock c_t | Own shock w_{i,t} | Role |
|---|---|---|---|
| **core** | 0 | N(0, .25) | Negative control: no common shock |
| **shared_shock** | N(0, .125), shared by every author of the role episode | N(0, .125) | Declared shared-shock world |

- **Author law.** a ~ N(0, 1), b ~ N(.5, .5²), ρ ~ U(.2, .8), e ~ N(0, .25). The stationary initial state includes
  the common process.
- **Matched marginals.** Both worlds have the same marginal innovation variance .25; only the cross-author
  covariance differs.
- **Scope of the worlds.** There is no common-variance grid; near_root is not part of this packet.

## Episodes, roles and estimand

- **Independent episodes.** E = 32 per world, with episode seeds 20261001–20261032. The original seeds
  20260915–17 are not reused; their archived results remain separate old evidence.
- **Roles within an episode seed.** The source role (300 authors) is an independent source episode, and the
  application role (200 authors) is an independent application episode. Each has its own common-shock stream,
  from SeedSequence([seed, 5, world, role, 1]). The validation role is not generated (count 0), and that leaves
  the source and application streams unchanged.
- **Authors.** Author keys include the seed, world and role, so authors are disjoint across every episode and
  role. No stable author is reused, and there is no crossed author/episode dependence.
- **Estimand.** For each world, the mean over independent episodes of the author-averaged paired difference
  (joint − independent) in each metric, at each declared cutoff.
  - The sampling unit is the episode: an application episode with its paired, independently drawn source episode.
  - Authors within an episode share the common shock and the frozen source law, so they are not independent
    replications.

## Source side (source observations only; one source episode per application episode)

- **Source law.** `fit_source_dynamics(source rows)`, the original two-round ML then MAP fit on the first 48 events,
  gives the measurement and process noise and the population prior for (a, b, η).
- **Common variance.** `learn_common_variance(source rows, source model)` gives the shared/own innovation split: the
  mean off-diagonal centred innovation covariance, raw and clipped values both stored.
- **What never reaches the application.** No application row, truth field or other episode enters the source law.
  Source common-shock trajectories are never passed to the application filter.

## Application side and the forecast cutoff

- **Personal parameters.** For each application author and prefix budget n ∈ {16, 32, 48},
  ψ̂_n = `fit_person(rows[:n], source noise, source prior)`, identical to `score_prefix` personal_MAP.
- **Filter.** The original `panel_predictions`, reused unchanged: `joint_filter` over the application panel, with
  both arms using the same ψ̂, marginal q and r.
  - Independent arm: off-diagonal process covariance 0.
  - Joint arm: off-diagonal covariance equal to the source common variance.
  - The current application common shock is inferred only from the same-episode panel rows with time ≤ t.
- **Cutoffs.**
  - Budgets 16 and 32: the state at t = n − 1.
  - Budget 48: rolling t = 47–62 with ψ̂_48 frozen. The state at t uses rows ≤ t only.
  - Forecasts of y_{t+k} for k ∈ {1, 4}, using the design clock z_{t+k} only.
- **Oracle.** The original labelled comparator, not a working method: true (a, b, ρ), q = r = .25, common .125 in
  shared_shock and 0 in core. It is evaluated on the same panels.

## Metrics (evaluation side; truth enters only here)

- **State level.** Error² (m_t − x_t)²; 95% coverage |m_t − x_t| ≤ 1.96√P_t; width 3.92√P_t.
- **Prediction level.** Mean ŷ = â + b̂ z + ρ̂^k m_t, variance ρ̂^{2k} P_t + q̂ Σ_{j<k} ρ̂^{2j} + r̂.
  - Error² against the observed y_{t+k}, coverage, width.
  - Conditional error² against a + b z + ρ^k x_t.
- **Interval scope.** Every interval is conditional on the frozen episode source law and the plug-in ψ̂.
  - Source-refit uncertainty is not inside it. It is reported separately as the across-episode spread of the source
    estimates (common, total and measurement variance) against the true .125/0, .25 and .25.
  - The across-episode spread of paired differences includes source-law variability, because every episode has its
    own source fit.

## Population uncertainty

- **Episode-aware.** Episode means d_e; mean over the 32 episodes; SE = sd(d_e)/√32; t interval with df 31; plus an
  episode-cluster bootstrap (2,000 draws, seed 20260926).
- **Naive diagnostic.** SE = sd of all author-level paired differences / √(32 × 200). Report the ratio, the
  intra-episode correlation and the design effect.
- **Role of these SEs.** Diagnostics, not a selected winning method. More authors per episode cannot replace
  episodes.
- **Primary contrast (declared).** shared_shock, budget 48 rolling block, personal_MAP, state error²
  (joint − independent). Every other contrast is secondary. There is no minimum-gain threshold.

## Checks (must pass before full-run results are read)

1. **Archived replay.** For `shared_shock` seed 20260915:
   - `generate_dynamics` reproduces the archived `source.jsonl` and `application.jsonl`;
   - `fit_source_dynamics` reproduces `source_model.json` (log_sigma and prior within 1e-10);
   - `panel_predictions` with archived ψ̂ reproduces `joint_full_r1/joint_state_metrics.csv` state and forecast
     values for personal_MAP within 1e-10.
2. **Zero-shared-variance marginal equivalence.** `joint_filter(common = 0)` equals the per-author univariate filter
   (`f05_recovery.filter_arrays`) on one full application panel.
3. **No future access.** Perturbing panel rows after t leaves the filtered state at t unchanged, and ψ̂_n uses rows
   with t < n only.
4. **Independent episodes.** SeedSequence keys are unique, author keys are disjoint across every episode × role,
   and the common-shock arrays differ. The descriptive mean |correlation| of common shocks across episodes is
   reported.
5. **Dense vs efficient.** Exact Gaussian conditioning with the dense joint covariance of (x, y) on a 3-author ×
   5-time panel equals the `joint_filter` means and variances.

## Budget and failure rule

- **Smoke.** 1 episode per world, timing only. An amendment then fixes nothing but the worker count; E = 32 is
  already frozen.
- **Checkpoints.** Per-episode outputs are written as each finishes.
- **Failures.** A source or personal fit failure is recorded and kept, never replaced, and the fallbacks are
  counted.
- **Cap.** If the 30-minute cap is reached, the completed episodes and the exact stage are checkpointed.
  - No true parameters are injected.
  - The 32 episodes are never silently replaced by 3.
  - The number of completed episodes is reported, and the population estimand is not claimed below 30 episodes.
