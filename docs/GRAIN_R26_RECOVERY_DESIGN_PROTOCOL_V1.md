# R26 / A07 recovery-parameter observation design: protocol V1 (packet25 B)

Frozen 2026-09-27 JST before any application outcome is examined.
- **Authority.** `NEXT_PACKET_25_RECOVERY_DESIGN.md` Part B; `SUPERVISOR_CHECK_20260927_0251.md`.
- **Original clause.** P01 PDF p.24. At the same participant burden, does a response-operator adaptive design
  identify the recovery parameter better than a scalar adaptive design? A loss or tie is an acceptable answer.
- **Code.** Module `grain_research/f06_recovery_design.py` (bound). Runner `scripts/run_grain_r26_recovery_design.py`.
- **Output root.** `results/grain_gptpro_foundation_v3/F06/recovery_target_design_r1/`.
- **Budget.** B compute ≤ 25 min; 0 neural calls; one process, single-thread BLAS.

## World, streams and inference (reused exactly)

- **Model.** y_t = a + b z_t + s_t + e_t, with s_t = ρ s_{t−1} + β h_{t−1} + w_t and ρ = .99 tanh(η).
  - The frozen F05 core source prior and noise come from `F05/source_full_r2/core_{seed}/source_model.json`, plus the
    declared β prior N(.4, .2²) (`source_prior`).
- **Streams.** Truth (a, b, ρ, β), the initial state, innovations and measurement come from the identical SeedSequences
  of `run_grain_gptpro_f06_gaussian_design.run_author`, for role application (index 2).
  - Seeds 20260915/16/17, 200 application persons per seed, worlds β = 0 ("zero") and β distributed. They are paired
    across arms.
- **Inference.** `f06_gaussian_design.fit_prefix`, unchanged in every arm: MAP with 2 starts plus inverse expected
  information, and the original explicit fallback. Optimization failures are retained.
- **Gates.**
  - The streams must equal the archived `gaussian_full_r1` truth for every author (0 tolerance).
  - The first 4 authors of seed 20260915, run with the ORIGINAL fixed policy (no actions), must reproduce the archived
    fixed-arm ρ estimates and a estimates at budgets 8/16/24 (1e-10).

## Arms: matched 24 observations and a matched intervention quota

- **Quota.** EXACTLY 2 actions per 8-observation block. Actions are allowed only at within-block transition slots
  t mod 8 ∈ 0..6. The slot t mod 8 = 7, which feeds the next block, and the final row are always h = 0. Future quota
  is never borrowed.
  - Adaptive arms are forced to act when the remaining slots equal the remaining quota, and forbidden once the quota
    is used.
  - Equal h counts control design burden; the actual latent perturbation is reported, not declared matched.
- **fixed.** z alternates −1, +1, …; actions at within-block offsets 2 and 5.
- **scalar_a.** Scalar-level adaptive REFERENCE: minimizes the predicted Var(a). It is not an implementation of
  published ALBA/IRT.
- **recovery.** Recovery-target adaptive: minimizes the predicted Var(ρ) = (dρ/dη)² Var(η).
- **Shared criterion.**
  - At the current MAP ψ = (a, b, η, β), take the expected Gaussian Fisher information of the observed prefix extended
    by the candidate next observation (z′, h_t), plus ONE declared neutral continuation (−z′, no further action).
  - It includes BOTH the mean term DᵀΣ⁻¹D and the η covariance term ½ tr(Σ⁻¹∂_ηΣ Σ⁻¹∂_ηΣ). The posterior covariance is
    (prior precision + F)⁻¹.
  - The continuation is needed because h_t first enters the η mean derivative at t + 2.
  - The covariance term is candidate-invariant here, because candidates change only the mean path. It is included, not
    omitted.
- **Candidates.** z′ ∈ {−1, 1} × allowed h. Ties go to lexical candidate order, atol 1e-14.
- **First condition.** Chosen by the same criterion at the prior.
- **Information visible to policies.** Observed prefix plus frozen prior and model. No truth, realized future
  response or oracle. No longer-horizon search.

## Metrics (budgets 8 / 16 / 24; world × seed and pooled; paired by person)

- **ρ.** Estimate, bias, MSE/RMSE, coverage and width of the η-scale Wald interval mapped monotonically to ρ.
- **Other targets.** Errors in a, b, current state and β.
- **Burden and placement.** Effective action count, placements, condition share, source-model predicted Var(ρ)
  against realized squared error, accumulated squared perturbation against the same-person no-action path.
- **Integrity.** Fit failures, non-finite values, quota violations, and the identical-policy rate between scalar_a
  and recovery.
- **Paired whole-author bootstrap** (2,000 draws, seed 20260928), within world × budget: ΔMSE and ΔRMSE for recovery −
  fixed, recovery − scalar_a and scalar_a − fixed.
- **Run order.** Fixed application author order: seed-major, ordinal ascending. Per-author gz checkpoints with
  protocol, code and input hashes. Runtime is estimated from the first 12 authors (4 per seed). The run stops at
  complete author boundaries if the cap would be exceeded.

## Checksum bindings

| file | sha256 |
|---|---|
| `grain_research/f06_recovery_design.py` | `e9492d396f79e7c6196ba0088a290adcc56ad1dca0173585eb353494097d5fb9` |
| `grain_research/gptpro_foundation_v3/f06_gaussian_design.py` | `e83c3db456c254c41d273e7656b932f0d39e34dedb4fc518f1cce764faca96a0` |
| `grain_research/gptpro_foundation_v3/f06_policy_state.py` | `e60ef89d82eceb9a808bf4c89414f3f51ae0187f633f16019dbb1656665857bc` |
| `scripts/run_grain_gptpro_f06_gaussian_design.py` | `1f82cc348ab49c8509e89b070fbfbe765946bf25744a7a92a789678e20a2c501` |
| `results/grain_gptpro_foundation_v3/F05/source_full_r2/core_20260915/source_model.json` | `929eaf0d6438ee206be88122afd30b1349fbd2e2b677c6fcca1e78f33e799fca` |
| `results/grain_gptpro_foundation_v3/F05/source_full_r2/core_20260916/source_model.json` | `586a06a737ef7e905500b26c476f2b9538d9a814bd3db4ec2b9a1f97dc9bb7eb` |
| `results/grain_gptpro_foundation_v3/F05/source_full_r2/core_20260917/source_model.json` | `dd92f0d91243a46d24ed88f7ee527c51059806912825ad32c0089368db265b92` |
| `results/grain_gptpro_foundation_v3/F06/gaussian_full_r1/checkpoint.json` | `5f7d52ad278aa0c4d4426b04396467a484e036dc376f2e09067bb87752fbb7c6` |
| `results/grain_gptpro_foundation_v3/F06/gaussian_full_r1/protocol.json` | `6f8661ddf6be3579ef23b1598bc869915e25a82145ad13069d1ab20c22fd2c47` |
