# R09 design-specific generalizability coefficients from saved REML moments: protocol V1 (packet20 D)

Frozen 2026-09-26 JST before computation.
- **Authority.** `NEXT_PACKET_20_ERROR_BUDGET_AND_HISTORY.md` Part D (R09 bullet).
- **Original clause.** P02 p.23: the evaluation standard includes a "generalizability coefficient", alongside
  test–retest, cross-paraphrase invariance, held-out-context prediction and incremental validity.
- **Runner.** `scripts/run_grain_r09_g_coefficients.py`.
- **Output root.** `results/grain_gptpro_foundation_v3/F04/generalizability_coefficients_r1/`, created in x-mode.
- **Scope.** Closed-form arithmetic only. There are no new fits, no new data and no threshold grid. 0 neural calls,
  and CPU seconds.

## Why this is not redundant

- **Earlier G values are for a different declared world.** FU2-b reported per-openness-level G as a closed-form
  evaluation of declared forms, and FU2-a reported ICC(B) .991 algebraically. Neither is computed from the reliability
  spine's own estimated components.
- **The spine** (`F04/reliability_spine_r1`) emitted known-model posterior R, ICC retest and recovery, but no G
  coefficient of its design.
- **So the missing object is:** the G coefficient of declared observed composites under the spine design, computed
  from its saved REML moments.

## Model and inputs (as saved by the spine; not refitted)

- **Covariance.** Per person, V = Z_θ Σ_θ Z_θᵀ + σ_d² Z_d Z_dᵀ + S(ρ, σ_w²/(1 − ρ²); order pair) + σ_e² I over 96 rows
  (2 occasions × 4 conditions × 3 wordings × 2 repetitions × 2 coordinates). `grain_research/reliability_spine.py`
  supplies `ZT`, `ZD`, `state_cov`, `pair_of` and `X`.
- **Plug-ins.** Per world and seed: `Sig_hat`, `sd2_hat`, `rho_hat`, `sw2_hat`, `se2_hat` from
  `F04/reliability_spine_r1/results.json`.
- **Oracle reference.** Generating values (Σ_θ, σ_d² per world, ρ .6, σ_w² .16, σ_e² .25) are reported beside them,
  never mixed with the plug-ins.

## Composites (explicit weight vectors over the 96 rows)

| target | rows | weight | universe-score part |
|---|---|---|---|
| level a_k (k = 1, 2) | all rows of coordinate k | 1/48 | a_k (balanced c and facet contrasts cancel) |
| slope B_k1 | coordinate k | c1/48 | B_k1 |
| slope B_k2 | coordinate k | c2/48 | B_k2 |
| cell mean m_k(c) | coordinate k, condition c | 1/12 | a_k + (B c)_k |

- **Full design** uses all 48 events. **Half** uses the 8 events of one occasion and one wording
  (`rows_half(o, f)`, o = 1, f = 1), with weights renormalized to the same targets.
- **Balance check.** w' X β must not depend on the fixed wording effects (checked).

## Variance parts per composite

- **Universe score:** var_θ = wᵀ Z_θ Σ_θ Z_θᵀ w.
- **Crossed nuisance:** var_d = σ_d² ‖Z_dᵀ w‖².
- **State:** var_s = wᵀ S w, averaged over the 4 order pairs, because persons use P_{u mod 4}.
- **Residual:** var_e = σ_e² ‖w‖².

## Coefficients

- **Random-wording universe** (generalize to new paraphrases; d is error):
  Eρ²_R = var_θ / (var_θ + var_d + var_s + var_e).
- **Fixed-wording universe** (these 3 wordings; d is part of the universe score):
  Eρ²_F = (var_θ + var_d) / (var_θ + var_d + var_s + var_e).
- **Absolute Φ.**
  - Fixed wordings: the design has no random facet main effects (the fixed g cancel by balance; there is no common
    occasion or time effect in the model), so Φ_F = Eρ²_F. This is stated, not assumed silently.
  - Random wordings: Φ_R needs the population variance of wording main effects. Only 3 fixed effects were estimated,
    so Φ_R is NOT_ESTIMATED.
- **Relation to the spine.** The half composite's Eρ²_R and Eρ²_F correspond conceptually to disjoint-wording and
  shared-wording retest. The spine's ICCs are for EB estimates, not unweighted composites, so the comparison is
  descriptive. G coefficients of composites are not EB posterior reliabilities R = 1 − E Var/Var.

## Labels

- Source-estimation uncertainty is NOT propagated: plug-in REML moments, with the 3-seed spread reported. F16
  bootstraps are not a substitute.
- Simulation world only. This is not human reliability, and incremental validity stays H01.

## Checksum bindings

| file | sha256 |
|---|---|
| `results/grain_gptpro_foundation_v3/F04/reliability_spine_r1/results.json` | `a8bca9206bcac299eb260181c820d1471db529312c2b23fb2bf1f93c690b699c` |
| `grain_research/reliability_spine.py` | `700118d84ec4a98d0b2ad96eaff2d437f4710e78afcdee3d4c3659f82ed8fd28` |
