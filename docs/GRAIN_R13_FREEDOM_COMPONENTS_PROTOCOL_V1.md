# R13 freedom-dependent components from crossed observations: protocol V1 (packet 30 B)

Frozen 2026-09-27 08:08 JST, before any outcome. Authority: `NEXT_PACKET_30_FREEDOM_COMPONENTS.md` Part B;
`SUPERVISOR_CHECK_20260927_0754.md`. Original clause: P03 pp.16–17 (five response-freedom levels; compare person,
context, person×context and reader variance, information and G). Declared-world simulation only: no text, forwards, API,
generation or new people. Output root `results/grain_gptpro_foundation_v3/F07/freedom_components_r1/` (x-mode).

## What the old formulas cannot identify

- **FU2-b** computes "G" = target / (target + noise) by summing declared variances, with reader variance fixed at
  0.1·o/3. It estimates nothing from observations.
- **F07/openness_mechanism_r1** computes exact information on declared curves.
- **Neither can:**
  - separate author×condition (P×C) variance from occasion residual (that needs repeated texts per author×condition);
  - separate reader×text noise from text residual (that needs more than one reader per text);
  - distinguish repeatability (G) from target fidelity. Adding the variances of every component that appears in a
    target is not the target-aligned variance Cov(x,t)²/Var(t).

## Design and declared world (functional forms stated honestly; not tuned)

- **Facets.** nA = 200 authors, nC = 6 conditions with known centered intensities z = linspace(−1, 1, 6), nO = 2
  occasions (separate texts) per author×condition, nR = 3 readers reading every text.
- **Freedom levels.** f ∈ {0, .25, .5, .75, 1}, paired: the same standard-normal base draws serve every f.
- **Observation model.** y = .5 z_c + λ(f) θ_a + κ(f) s_a + γ(f) β_a z_c + σ_W(f) w_ac + σ_R ρ_r + σ_AR u_ar +
  σ_ACR v_acr + σ_O(f) o_aco + σ_e(f) e_acor, with:
  - λ = .6 + .4f (level expression grows);
  - κ = .8f (a stable selection/style propensity s appears with freedom);
  - γ = 1 − .6f (the response to the manipulated condition dilutes);
  - Var(β) = .25; σ_W² = .10 + .20f; σ_O² = .20 + .30f; σ_R² = .05; σ_AR² = .02; σ_ACR² = .02.
- **Reader regimes.** σ_e²: LOW = .10(1 + f); HIGH = .40(1 + 3f).
- **Mechanisms (small bounded family).** M1: corr(θ, s) = .3. M2: corr(θ, s) = 0.
- **Worlds.** 3 independent seeds (`SeedSequence([20260927, 30, seed])`). 2 mechanisms × 2 regimes × 5 levels ×
  3 seeds = 60 fitted cells.
- **Negative control (aliased design).** The same data restricted to one occasion (nO = 1). Components e, O, ACR and W
  must be flagged UNIDENTIFIED, never 0.

## Estimation (no truth access)

- Balanced ANOVA expected mean squares (module docstring), with the author×condition term split into the author×linear-z
  slope (SLOPE) and the remainder (W).
- A component or combination is estimated only if it lies in the row space of the available EMS rows (rank check).
  Negative estimates are kept and flagged.
- **Estimable repeatability.**
  - G_level = A / (A + W/nC + AR/nR + ACR/(nC nR) + O/(nC nO) + e/(nC nO nR)), for the author aggregate x.
  - G_slope = SLOPE / (SLOPE + (W + ACR/nR + O/nO + e/(nO nR))/Szz), for the observed z-slope b.

## Targets (no psychological names; same draws)

- **T1 = θ** (author level).
- **T2 = β** (conditional response to z).
- **T3 = θ + s** (declared level/selection combination: selection is signal for T3 and nuisance for T1).

Three distinct quantities are kept apart:
1. **Repeatability.** G_level / G_slope, estimated from components.
2. **Target fidelity (reference, analytic, from the declared model).** Cov(x,t)²/(Var(t)Var(x)), with target-aligned
   linear variance Cov(x,t)²/Var(t). Cov(x,θ) = λ + κρ; Cov(x, θ+s) = (λ + κρ) + (λρ + κ); Var(θ+s) = 2 + 2ρ; T2 uses
   b with Cov(b, β) = γVar(β).
3. **Target recovery (audit only; truth used here and nowhere else).** Empirical corr(score, t)².

## Uncertainty and contrasts

- **Bootstrap.** Author-cluster bootstrap, 300 draws per world. Authors are resampled with conditions, readers and
  occasions held fixed (conditioning stated). The estimator is refit per draw. The same author indices serve all five
  levels, so contrasts are paired. Draws: `SeedSequence([20260927, 30, seed, 7])`.
- **Contrasts.** f = 1 − 0, .5 − 0 and 1 − .5 for G_level, G_slope, recovery T1/T2/T3 and components A, W, SLOPE, e.
  Reported per world and pooled (mean over seeds; the across-seed range is reported).
- **Construction decision (declared before running).** A freedom comparison "changes under the target definition"
  when the signs of the recovery (or fidelity) contrasts differ across T1 / T2 / T3 on the same draws. The explanation
  must cite the covariance equations above.
- **No winner selection, no peak required.** Null or negative answers are valid.
- **Independent checks on actual output.** The SS identity; a few components recomputed from explicit closed-form MS
  differences (not the EMS solve); G recomputed from saved components.

## Bindings

| file | sha256 |
|---|---|
| `grain_research/r13_freedom_components.py` | `7b9c474f42c7bddb9ee9f6113a519353a0bfc959eb60d1edaf5e29f86bcbfd22` |
| `scripts/run_grain_reaudit_r2_followup2.py` | `76d6cdd901d073cf6e4f3b4e9a94e70035cf5c9ed435f38188ec58dd717c87c2` |
| `results/grain_gptpro_foundation_v3/F07/openness_mechanism_r1/decision.md` | `4b42138422fa3854d04566bf2c773906f531dff0b3a4290dfaa14bf8119a10d0` |
