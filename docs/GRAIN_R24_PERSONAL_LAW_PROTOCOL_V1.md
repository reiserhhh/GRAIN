# R24 personal laws and honest uncertainty: protocol V1 (frozen before any output inspection)

Date 2026-09-26 JST. Authority: `NEXT_PACKET_11_PERSONAL_LAW_AND_SHARED_SHOCK.md` Packet A. Sources: R24 (P02:29–31,
P03:31, P04:19–21, P05:S3); R25 (source/application separation).
- **Runner and output.** `scripts/run_grain_r24_personal_law.py`, writing to
  `results/grain_gptpro_foundation_v3/F13/personal_law_r1/`.
- **Scope.** CPU only, at most 2 workers × 1 BLAS thread, 0 neural calls, cap 45 minutes.

## Existing evidence (read, not re-labelled)

- **`F13/recovery3_r1`.** The binary C2 worlds, where the transition law is **shared** (F02). Its B2/M personal
  variance measured adaptation to a shared law, and the B2 working covariance .25(gram + I)⁻¹ over-covered (width
  .58–.86).
- **`F13/dynamic_personal_*`.** Laplace personal-parameter (a, b, η) posteriors in the F05 continuous "core" dynamic
  world, with source aggregation pending. That is a different target, and it does not cover binary personal
  transition laws.
- Neither is personal-law validation for the binary targets.

## Worlds (C2 binary, 48 events, prefix 12; lag state s_t = 2y_{t−2} + y_{t−1}; initial y0, y1 iid Bernoulli(.5))

| World | Personal law p_u(s) = P(y_t = 1 given s_t = s, author u) | Role |
|---|---|---|
| **W0** | Shared xor law, p(s) = .75 if 1 = a⊕b else .25 (`f02.generate_world('c2_xor')`, unchanged) | Negative control: no heterogeneity |
| **H1** | Level heterogeneity: p_u(s) = σ(logit p_xor(s) + δ_u), with δ_u ~ N(0, 1) | Author level |
| **H2** | Structure heterogeneity: p_u(s) = w_u·p_xor(s) + (1 − w_u)·p_copy(s), with p_copy(s) = .75 if b = 1 else .25 and w_u ~ Beta(2, 2) | Author transition/interaction |

- **Scope of the worlds.** There is no strength grid, and H1/H2 are not selected for a result.
- **Generator.** Rows use the same schema as F02 (source 300, source-validation 100, application 200 authors). The
  streams are SeedSequence([seed, 24, world_code, role]) with world codes H1 = 1 and H2 = 2. Latent δ_u, w_u and
  p_u(s) sit behind the evaluation interface only.

## Roles

- **Source 300.** The canonical B1/B2/M fits (`fit_observed_source`), 200 source-bootstrap refits and the EB
  hyperparameters come from source observations only.
  - W0 reuses the archived bootstrap files (`F13/binary_source_full_r2`).
  - H1/H2 use `refit_statistics` with the recipe SeedSequence([seed, 13, 8, 10 + world_code, b]).
- **Source-validation 100.** An observed-outcome calibration report only: Brier and calibration-in-the-large. It uses
  no truth and is never used for fitting.
- **Application 200.** Scored at every t ≥ 12, with the prediction made before y_t is seen.

## Estimators and intervals (all predictions before observation)

- **B1, B2, M.** Canonical `QualifiedHistory` filtered semantics, via the vectorised `EnsembleHistory`, with
  recovery3 interval components:
  - state: the variance across the lag-state posterior;
  - personal: B1 0; B2 the **archived working covariance .25(gram + I)⁻¹**, kept as a comparator; M the
    Beta-projection variance;
  - source: variance across the 200 refits;
  - event: v(1 − v), reported but never inside the probability interval.
- **EB, the single new uncertainty alternative.** A source-calibrated empirical-Bayes Beta-binomial personal-law
  reference (Efron & Morris 1975; Kass & Steffey 1989 for conditional EB intervals).
  - For each state s, a method-of-moments between-author variance τ_s² is computed from the source authors'
    per-state counts (x_is, n_is):
    - p̄ = Σx/Σn;
    - S = Σ n_i(p_i − p̄)²;
    - τ² = max(0, (S − p̄(1 − p̄)(k − 1)) / (Σn − Σn²/Σn)).
  - Then α_s + β_s = p̄(1 − p̄)/τ² − 1, with α_s = p̄·(α_s + β_s). If τ² ≤ 0, the prior is capped at α + β = 1e6
    (homogeneous reduction).
  - The application author's posterior is Beta(α_s + x_us, β_s + n_us − x_us), using counts only from complete
    observed triples.
  - Prediction: the posterior mean mixed over the lag-state posterior weights.
  - Interval variance = state + EB posterior variance + source variance of the EB prediction across the 200 source
    refits of (α, β) (hyperparameter uncertainty).
  - **Assumptions:** per-state independent Beta laws, binomial counts given observed lag states, and hyperparameters
    estimated on source (hyperparameter uncertainty carried by the source bootstrap).
- **Interval form.** 95% intervals are point ± 1.96·√variance, clipped to [0, 1]. Clipping counts are reported.

## Targets (stated for every interval)

- **Complete histories:** T_latent = p_u(s_t), the author's next-outcome probability given the full history and the
  author-law parameters. It is not the observed Bernoulli event; event variance is excluded.
- **Masked histories:** two distinct targets, each reported:
  - T_latent = p_u(s_t), which requires the hidden lag state;
  - T_integrated = E[p_u(s_t) | observed prefix, true author law], by forward filtering with the author's true law
    (evaluation side only).
  - Intervals whose state component represents lag uncertainty are aimed at T_integrated. Coverage of both is
    reported and never silently switched.

## Missingness extension (after complete histories)

- **Masks.** none, MCAR .4, MAR (.2 / .6 by the last observed value) and MNAR (.2 / .6 by y_t), exactly as recovery3,
  with mask stream SeedSequence([seed, 13, 31, world_code, author_index, mechanism]). Time positions are retained,
  and missing events enter as q = 0.
- **B1/B2/M.** Run under every mask. The archived filter adds q-weighted model-imputed counts only for *observed*
  events (q = 1), so missing events add no information.
- **EB.** Run for none and MCAR only (complete-triple counts are valid under MCAR). For MAR and MNAR it is a
  **bounded case, not run**: there is no honest selection model, and complete-case counts are biased under
  response-dependent missingness. Hidden lag states are never substituted.

## Metrics, checks, decision

- **Metrics, per author (t ≥ 12).**
  - Bias, squared error vs the target, Brier vs y, coverage, width, mean components, clipping counts.
  - Paired per-author contrasts: EB − B2, EB − M, EB − B1, B2 − B1. Seed-stratified author-cluster bootstrap, 2,000
    draws, seed 20260926. Timesteps are never counted as subjects.
- **Checks.**
  1. Canonical replay: W0, seed 20260915, mechanism none, 200 application authors, archived bootstrap. The
     per-author B1/B2/M se, bias, Brier, coverage and width must equal `F13/recovery3_r1/author_metrics.csv` within
     1e-12.
  2. Homogeneous reduction: EB with τ² = 0 equals the pooled table, tested on constructed counts. W0 reports its
     τ² and max |EB − B1|.
  3. Source/application isolation: hyperparameters are fit from source rows only (asserted).
  4. Prediction before observation, asserted in the loop.
  5. Independent verification: the EB posterior mean and variance by closed form match scipy.stats.beta, and τ² by
     an independent formula on one state.
- **Budget.**
  - Smoke: 1 seed, 20 application authors, 20 refits, mechanism none, **timing only** (no scientific output
    inspected).
  - Then an amendment freezes seeds, masks and refit count from the timing alone; the default target is 3 seeds,
    all masks, 200 refits.
  - Partial results are preserved at the cap. There is no enlargement.
- **Construction rule.** Derived: *when* personal adaptation is warranted, and *what* its uncertainty represents. A
  failed calibration is a legitimate result.
