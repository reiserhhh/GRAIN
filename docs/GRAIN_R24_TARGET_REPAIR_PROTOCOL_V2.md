# R24 target repair: protocol V2 (correction revision `personal_law_r2`, frozen before execution)

Date 2026-09-26 JST. Authority: `NEXT_PACKET_12_TARGET_REPAIR_AND_INTEGRATION.md` Part A and
`SEMINAR_FABLE_11_DECISION.md`. The supervisor's covariance and equality corrections control.
- **What is preserved.** `F13/personal_law_r1` (runner sha 92e4e91a…, protocol V1 sha 5df0cdb0…, amendment A1)
  stays unchanged: code, outputs, hashes and decision. This revision adds columns and a repaired evaluation target.
  It does not rewrite r1.
- **Runner and output.** `scripts/run_grain_r24_personal_law_r2.py`, writing to
  `results/grain_gptpro_foundation_v3/F13/personal_law_r2/`.
- **Scope.** CPU only, 2 workers × 1 BLAS thread, 0 neural calls, cap 15 minutes.

## What r1 got wrong

1. **Cross-target interval.** r1's `cover_int` evaluated the latent-state-mixture interval against the integrated
   target. Its decision sentence, "intervals whose state component represents lag uncertainty are aimed at
   T_integrated", is withdrawn. The interval is a working interval for T_lat, and `cover_int` is a cross-target
   diagnostic.
2. **Selection-blind MNAR target.** r1's evaluation filter used a flat likelihood (.5, .5) at every missing step.
   Under MNAR the missingness indicator carries information about y, so r1's MNAR `se_int`, `bias_int` and
   `cover_int` refer to a selection-ignored target, not the conditional expectation given the observed data.

## Targets (evaluation side only; the author law p_u never reaches a working estimator)

Notation: lag state s_t = 2y_{t−2} + y_{t−1}; R_i = 1 if y_i is observed; D_{<t} = {R_i, and y_i where R_i = 1, for
i < t}.
- **T_lat** = p_u(s_t). Legacy.
- **T_ign.** The legacy `integ`: a forward filter with the true law and likelihood (.5, .5) at missing steps. Kept as
  the named comparator "selection-ignored integrated".
- **T_sel** = E[p_u(s_t) | D_{<t}, p_u, declared mask law], with R_t excluded because it is revealed after the
  prediction.
  - **Filter.** The same forward filter as T_ign, with the missing-step likelihood ℓ_i(y) = P(R_i = 0 | y_i = y,
    history) under the declared mechanism:
    - MCAR: (.4, .4).
    - MAR: (π, π), with π = .6 if the last *observed* value is 1, else .2 (and .2 before any observation).
    - MNAR: (.2, .6).
  - **Observed steps.** An observed step contributes the indicator of the observed y times P(R_i = 1 | y_i =
    observed, history). The second factor is constant across the pair states consistent with the data and cancels
    in normalisation.
  - **Consequence.** Under none, MCAR and MAR, T_sel = T_ign up to rounding; under MNAR they differ.
- **T_fw** = w_tᵀ p_u, where w_t is the canonical working filter's lag-state weight vector, the one every working
  estimator uses. It is the evaluation-side target of the fixed-working-weight companion.
  - **Limitation.** It equals T_sel only when w_t equals the true-law posterior weights w*_t(p_u, D_{<t}). Those
    weights depend on the author's law; the working weights come from the shared source law. No oracle weight is
    substituted.

## Intervals and their contracts

Every interval is point ± 1.959963984540054·√V, clipped to [0, 1], with clipping recorded. A variance identity is not
a coverage guarantee: normality and clipping are working assumptions, and nothing is posterior-calibrated.

| Interval (column family) | V | Own target | Conditioning | Status |
|---|---|---|---|---|
| **L**: legacy, alias `legacy_latent_mixture_working` (`cover_lat`, `width`, `clipped`) | state + Σ_s w_s C_ss + source | T_lat | canonical working weights; candidate working personal law; paired source refits | Working approximation. Identity I1 under the working model. Values unchanged. |
| L vs T_ign: legacy `cover_int`, alias `legacy_cross_target_L_vs_int_ign` | same | none | — | Cross-target diagnostic |
| L vs T_sel: new `cover_int_sel_L` | same | none | — | Cross-target diagnostic |
| **FW**: new fixed-weight companion (`width_fw`, `clipped_fw`, `cover_fw_own`, `cover_fw_ign`, `cover_fw_sel`) | wᵀCw + source | T_fw | canonical working weights held fixed; C = candidate's working personal covariance at the canonical fit; source = variance of the integrated point over 200 paired refits | Working approximation, not an executed law-of-total-variance calculation (see below). Coverage against T_ign/T_sel is a cross-target diagnostic. |

**Why FW is a working approximation.** V adds the canonical-fit conditional personal variance to the between-refit
variance of the integrated point.
- The conditional variance is not averaged over refits.
- The bootstrap mixture of refits is a sampling distribution, not a posterior.
- The pairing of weights and predictions within each refit is preserved, exactly as the legacy source term does.

**Working covariance C per candidate, oriented from the actual code** (`EnsembleHistory.per_state`, canonical fit):
- **B1:** C = 0.
- **B2:** value_s = Σ_i coef_i F[s, i], with F = `FEATURES` (4 states × 4 features). Coef covariance is
  Σ = .25(gram + I)⁻¹, the archived working prior and not between-author variance. So C = F Σ Fᵀ, which is
  non-diagonal because the coefficients are shared across states.
- **M:** value_s = m_mean_s + Σ_j (raw_j − m_mean_j) P[j, s], with P = `projector`. The working model treats raw_j
  as independent with rv_j = raw_j(1 − raw_j)/(n_j + 3). So C = Pᵀ diag(rv) P, which is non-diagonal.
- **EB:** C = diag(v), by the declared independent per-state Beta reference; then wᵀCw = Σ w_s² v_s.

**Consistency gate.** diag(C) must equal the legacy per-state personal variance within 1e-12. If it does not, that
candidate's companion is written as NaN and marked unsupported, with the reason. A diagonal-only substitute is never
used for B2/M.

## Mathematical statements (written and tested)

Let S ~ w, independent of a random vector p with mean m and covariance C, and let p̄ = wᵀm.
- **I1 (latent-mixture total variance).** Var(p_S) = Σ_s w_s C_ss + Σ_s w_s (m_s − p̄)².
- **I2 (fixed weights).** Var(wᵀp) = wᵀCw.
- **Gap.** D = I1 − I2 = ½ Σ_{i,j} w_i w_j [(m_i − m_j)² + C_ii + C_jj − 2C_ij] ≥ 0, since C_ii + C_jj − 2C_ij =
  Var(p_i − p_j).
  - For w on the simplex and C positive semidefinite, D = 0 if and only if every pair with w_i w_j > 0 has m_i = m_j
    AND Var(p_i − p_j) = 0.
  - Degenerate weights, and identical random quantities across all supported states, are sufficient special cases.
  - Zero variances alone, or constant means alone, are NOT sufficient. Counterexamples:
    - C = 0, m = (.1, .9), w = (.5, .5) gives D = .16.
    - m = (.5, .5), C = diag(.01, .01) gives D = .005.
  - The numeric example w = (.5, .5), m = (.1, .9), v = (.01, .01) gives I1 = .17 and I2 = .005.
- **Checks.**
  - A non-diagonal example from an actual B2 and an actual M state: wᵀCw vs Σ w_s² C_ss.
  - Independent verification of wᵀCw by Monte Carlo sampling through the explicit linear maps: coef ~ N(coef, Σ)
    for B2, and independent raw_j for M.
  - Exhaustive enumeration of y ∈ {0, 1}^t, with the full selection likelihood, for T_sel.

## Replay design and gate

- **Unchanged design.** Worlds W0/H1/H2, seeds 20260915–17, 200 application authors, masks none/MCAR/MAR/MNAR,
  200 refits, EB only for none and MCAR.
- **Source bundles reused.** r1's `source_side` is called unchanged: archived W0 bootstrap files and the H recipe
  seeds. Points, updates, counts and estimators are unchanged, and no working estimator receives p_u or hidden
  outcomes.
- **Exact gate.** Every legacy per-author column for every candidate must equal r1 `author_metrics.csv` exactly:
  round-trip parse, identical keys, max |diff| = 0.0. The columns are se_lat, bias_lat, se_int, bias_int, brier,
  cover_lat, cover_int, width, state, personal, source, event, clipped and missing_rate.
  `source_validation_calibration.csv` and `source_side.json` must also be equal.
  - On any mismatch: stop, keep the outputs under `failed_gate/`, and report. No relaxation.
- **New columns** (stored separately, never gated against r1): T_sel metrics (`se_int_sel`, `bias_int_sel`,
  `cover_int_sel_L`), FW metrics (`pers_fw`, `width_fw`, `clipped_fw`, `cover_fw_own`, `cover_fw_ign`,
  `cover_fw_sel`, `se_fw`, `bias_fw`) and `diag_consistency_err`.
- **Event trace.** A parquet file, float64: keys, y, R, w_t, per-candidate per-state values, B2/M C (upper triangles),
  EB v, points, legacy state/personal/source, and targets T_lat/T_ign/T_sel/T_fw. A separate evaluation-only truth
  table holds p_u per author.
  - A cold verifier recomputes every new author column from the trace and the truth table (means within 1e-12).
- **Provenance.** r1 file hashes are recorded before and after the run and must be unchanged. Code snapshots are
  kept.

## Report and budget

- **Report.** Complete-history equality: targets exact, L = FW within 1e-12. Unchanged point predictions. Corrected
  MNAR target error. L vs FW width and coverage by target. Which r1 findings survive.
- **Forbidden.** Multiplier tuning, posterior-family choice, hyperparameter search, and relabelling a mixed interval
  as exact.
- **Budget.** 15-minute cap. If the projected runtime exceeds 10 minutes, checkpoint and stop rather than expand.
