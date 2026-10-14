# R26 / A07 committed two-observation recovery design: protocol V1 (packet26)

Frozen 2026-09-27 JST before any application outcome.
- **Authority.** `NEXT_PACKET_26_COMMITTED_DESIGN.md`; `SEMINAR_FABLE_20_DECISION.md` (Codex-corrected odd-index
  epochs); `SUPERVISOR_CHECK_20260927_0352.md`.
- **Original clause.** P01 p.24, R26 / A07.
- **Code.**
  - Module `grain_research/f06_committed_design.py` (bound). It reuses the frozen packet-25 module
    `grain_research/f06_recovery_design.py` (bound, unchanged).
  - Runner `scripts/run_grain_r26_committed_design.py`.
- **Output root.** `results/grain_gptpro_foundation_v3/F06/recovery_committed_design_r1/`.
- **Budget.** Compute ≤ 25 min, one process, single-thread BLAS. 0 neural calls.

## What changes, and what does not

- **Unchanged.** The Gaussian world, source priors (F05 core `source_model.json`, β ~ N(.4, .2²)), fitting
  (`fit_prefix`), the expected-Fisher criterion (mean + η covariance terms), ρ = .99 tanh(η) units, the application
  streams (600 authors, seeds 20260915/16/17, 200 each), the worlds (β = 0, distributed), the arms (fixed, scalar_a,
  recovery), 24 observations, budgets 8/16/24, and the quota (exactly 2 actions in slots 8k..8k+6; slot 8k+7 is 0).
- **Changed.** Only the adaptive DESIGN RULE: scored design = executed design.

## Clock and timing (derived)

- **Environment.** s_t = ρ s_{t−1} + β h_{t−1} + w_t; y_t = a + b z_t + s_t + e_t; h_{−1} = 0.
- **Decision information after y_t.** H_t = {(z_u, y_u): u ≤ t} ∪ {h_u: u < t} plus the frozen source model. No truth,
  future innovations or measurements, and no endpoint errors.
- **Commitments.**
  1. z_0 = −1 in all arms.
  2. After y_0: commit (h_0, z_1) with a ONE-observation criterion (scored window y_1). No z_2 is promised.
  3. After y_t for t = 1, 3, …, 21: commit the JOINT pair ((h_t, z_{t+1}), (h_{t+1}, z_{t+2})) and execute both.
     There is no replanning after y_{t+1}; inference updates, but the commitment stands. The scored windows are
     (2, 3), (4, 5), …, (22, 23). No window crosses the endpoints y_7 / y_15 / y_23, because each epoch t ≤ 21 scores
     only t + 1 and t + 2 ≤ 23, and endpoint posteriors use prefixes only.
  4. After y_23: stop, h_23 = 0. h_7 and h_15 are 0 by the block rule. At epochs 7 and 15 both future conditions
     remain design variables.
- **Mean response to a new h_t.** β at y_{t+1} and ρβ at y_{t+2}. Its NEW η sensitivity starts at y_{t+2}, where
  forcing_η(t+2) = forcing(t+1) + ρ·forcing_η(t+1). Past actions and the state covariance can already carry η
  information, so η information at one step is NOT asserted to be zero.
- **At β = 0 truth.** Actions do not change y at fixed z, but the fitted model still estimates an unknown β.
- **Candidates.** z_{t+1}, z_{t+2} ∈ {−1, +1} independently, with no forced opposite continuation. Actions come from
  sequential quota feasibility: h_t is checked against the actual history, then h_{t+1} against history + the
  hypothetical h_t (`allowed_actions`: forced 0 when the quota is spent or the slot is 7; forced 1 when the remaining
  slots equal the remaining quota). No borrowing and no infeasible neutral continuation.
- **Scoring.** The observed prefix is extended by exactly the committed conditions and actions. The final h entry is a
  placeholder that affects no scored observation. No refit on imaginary y. The score is the predicted target variance
  at the current MAP with the frozen prior: scalar_a → Var(a); recovery → (dρ/dη)² Var(η).
- **Ties.** Lexical over (z_{t+1}, h_t, z_{t+2}, h_{t+1}), with z = −1 before +1 and h = 0 before 1, atol 1e-14.
- **fixed.** Alternating z; h at offsets 2 and 5. It must replay packet 25's fixed-arm metrics exactly.

## Metrics, gates, add-on

- **As packet 25:** ρ bias/RMSE/coverage/width; paired 2,000-draw author bootstrap ΔMSE/ΔRMSE against both references
  (seed 20260928); a/b/state/β errors; counts, placements, perturbation; failures, non-finite values; predicted
  Var(ρ) against realized MSE.
- **New:**
  - condition shares, [1, z] design rank, information condition number;
  - repeated-design frequency (share of authors on the arm's modal (z, h) sequence);
  - a direct comparison of scalar degeneracy between packet 25 and this run.
- **Gates.**
  - The streams equal the archive for all 600 authors.
  - The fixed arm equals the packet-25 fixed-arm metrics, for all authors and budgets, to 1e-12.
- **Endpoints.** Budget 24 is primary; 8 and 16 are trajectory diagnostics. Worlds are reported separately.
- **Mechanism add-on (β = 0 only).**
  - Replay each author's two realized adaptive ACTION schedules on the fixed alternating z, with the same streams and
    no optimization.
  - Verify that y is identical to the fixed arm before fitting.
  - Refit with the same UNKNOWN-β inference and compare ρ errors at 8/16/24 against fixed. This is estimator
    attribution under identical observations.
  - Adaptive history against its same-schedule fixed-z counterpart is a conditional z-design contrast, not an additive
    causal decomposition.
- **What is not required.** No balance threshold and no winner requirement. Continuing degeneracy is reported as a
  limitation of the reference.

## Checksum bindings

| file | sha256 |
|---|---|
| `grain_research/f06_committed_design.py` | `20992e6c0cae9d21ce8f17519b0b0d2f3f80f6f8dd3e81c0e5d25f605e4bb15d` |
| `grain_research/f06_recovery_design.py` | `e9492d396f79e7c6196ba0088a290adcc56ad1dca0173585eb353494097d5fb9` |
| `grain_research/gptpro_foundation_v3/f06_gaussian_design.py` | `e83c3db456c254c41d273e7656b932f0d39e34dedb4fc518f1cce764faca96a0` |
| `results/grain_gptpro_foundation_v3/F06/recovery_target_design_r1/private_budget_metrics.csv` | `cc891fb6e8b6bbfa00a68214841e94b1d02f4104a42bbe337e5c8002a4e7190a` |
| `results/grain_gptpro_foundation_v3/F06/recovery_target_design_r1/results.json` | `4b4374d6ca3204f333a20a4bdb8bfb250357de13e21517ac61088bdc8ddc7878` |
| `results/grain_gptpro_foundation_v3/F05/source_full_r2/core_20260915/source_model.json` | `929eaf0d6438ee206be88122afd30b1349fbd2e2b677c6fcca1e78f33e799fca` |
| `results/grain_gptpro_foundation_v3/F05/source_full_r2/core_20260916/source_model.json` | `586a06a737ef7e905500b26c476f2b9538d9a814bd3db4ec2b9a1f97dc9bb7eb` |
| `results/grain_gptpro_foundation_v3/F05/source_full_r2/core_20260917/source_model.json` | `dd92f0d91243a46d24ed88f7ee527c51059806912825ad32c0089368db265b92` |
| `results/grain_gptpro_foundation_v3/F06/gaussian_full_r1/checkpoint.json` | `5f7d52ad278aa0c4d4426b04396467a484e036dc376f2e09067bb87752fbb7c6` |
