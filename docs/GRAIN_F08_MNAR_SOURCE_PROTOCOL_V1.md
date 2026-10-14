# F08 MNAR-aware source fit protocol v1 (frozen 2026-09-26 before running)

Worker: Claude Opus 5.5 (TASK-CLAUDE-GRAIN-FULL-LOOP-20260926). Follows
`reports/grain_measurement_foundation/GPTPRO_V3_F08_END_TO_END_20260926.md`,
where gap-aware fitting on reader-read source text failed for the NLI reader
(B1 +.12 Brier).

## Question

When the reader's measured emission law enters the source likelihood, does the
fit recover the oracle-fitted source's application loss? The emission law is
estimated on non-source materials. How much of any residual cost comes from
estimating that law on other materials?

## Data and estimator

- Source readings: exactly those of `F08/end_to_end_r1`, the same render
  function and streams, mixed panel only. The complete panel is already
  identical.
- Emission law P(r | y), r in {missing, read 0, read 1}, under the declared
  render law: status uniform over 4, variant uniform over 4, ancillary pole
  Bernoulli(.5) for unqualified statuses. Hard thresholds are .5.
  - `validation_law`: computed from the validation-role cache rows only. This
    is the deployable arm, because the materials differ from the source
    materials.
  - `source_law`: computed from the source-role cache rows. It is the true law
    of these source readings, so it is an ORACLE_EMISSION reference and not a
    deployable estimator.
- Model: a pooled second-order chain. The pair state (y_{t-1}, y_t) has 4
  values; parameters are p[4] and the initial pair law. Emission is
  conditionally independent given y_t, which is exact under the render law.
- Fit: EM by forward-backward, with p initialised at .5 and a uniform initial
  pair. The M-step is p = (E c1 + 1)/(E n + 2), the fit_b1 form; the initial
  pair law is (1 + E pair counts) normalised. It stops at a max parameter
  change < 1e-10 or after 1000 iterations.
- Derived from the same posterior expected counts:
  - B1 table;
  - B2 as the ridge normal equations (alpha 1, no intercept, fit_b2 design)
    over expected counts;
  - M as the per-author expected tables, their mean and the top-2 SVD right
    vectors;
  - the source mean as the posterior mean of y;
  - the initial pair as the posterior (y0, y1) law, pooled.
- Declared approximation: the E-step uses the pooled chain, so the posterior
  over y ignores author heterogeneity. M's author tables are therefore shrunk
  toward the pooled chain.

## Application and comparison

- The application readings, streams, hard channel and frozen QualifiedHistory
  filter are identical to the downstream and end-to-end arms. The application
  side is not changed; it keeps its MAR filter approximation.
- Arms:
  - `oracle_source`: replays end_to_end_r1 exactly;
  - `gap_aware_read_source`: replays end_to_end_r1 exactly;
  - `em_validation_law`;
  - `em_source_law`.
- Metric: application Brier (t >= 12) for B1, B2 and M. Contrast: arm minus
  oracle_source. Paired author bootstrap: 2000 draws, seed 20260915, stratified
  by seed. The COMMON decision rule applies.
- Diagnostics: EM iterations, final log-likelihood (monotone check) and max
  |B1 table - oracle table|.

## Acceptance for this packet

- Both replays must be exactly 0.
- The EM log-likelihood must be non-decreasing, to 1e-9.
- No new reader forwards, no API and no truth fields in fitting: only the
  source readings and the emission law are used, with oracle observations only
  in the `oracle_source` reference arm.
