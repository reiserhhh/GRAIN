# F08 author-level E-step protocol v1 (frozen 2026-09-26 before running)

Worker: Claude Opus 5.5. Follows `F08/mnar_source_r1`, where a pooled E-step
recovered B1/B2 under the measured NLI emission law but left M worse than the
oracle-fitted source (+.004 to +.006 Brier). The declared cause was that
author tables were imputed under the pooled chain.

## Question

If each source author's missing or misread events are imputed under that
author's own chain, does the M source model recover? The emission law is
unchanged, and B1/B2 are unchanged.

## Estimator `F08_author_estep_v1`

- Inputs:
  - the same mixed-panel source readings as `end_to_end_r1` and
    `mnar_source_r1` (same render function and streams);
  - the same emission laws: `validation_law` (deployable) and `source_law`
    (ORACLE_EMISSION reference);
  - the pooled EM solution from the frozen `em_fit` code path, recomputed and
    required to equal the `mnar_source_r1` diagnostics exactly.
- Per-author EM:
  - author a has its own second-order table p_a[4], initialised at the pooled
    table;
  - the initial pair law is fixed at the pooled solution;
  - the E-step is forward-backward with the author's own transition;
  - the M-step is p_a = (E c1_a + 1)/(E n_a + 2), the fit_m/Laplace form;
  - it stops at a max change < 1e-10 or after 1000 iterations.
- M model:
  - author tables are the converged (E c1_a + 1)/(E n_a + 2);
  - M's mean and top-2 right singular vectors come from the SVD, as in
    fit_m.
- B1, B2, the initial pair and the source mean are copied from the pooled EM
  arm, so B1/B2 application losses must equal `mnar_source_r1` exactly. That
  equality is a check.

## Comparison

- Application scoring: hard channel, mixed panel, frozen QualifiedHistory,
  t >= 12; identical to `mnar_source_r1`.
- Arms:
  - `oracle_source`: replay;
  - `em_pooled_validation_law`: replay of `mnar_source_r1`;
  - `em_author_validation_law`;
  - `em_author_source_law`.
- Contrast: M Brier, arm minus `oracle_source`, and `em_author` minus
  `em_pooled` under the same law. Paired author bootstrap: 2000 draws, seed
  20260915, stratified by seed. The COMMON decision rule applies.

## Acceptance (declared now, including the corrected EM criterion)

1. **Replays:** `oracle_source` and `em_pooled` M losses equal
   `mnar_source_r1` exactly; B1/B2 losses of the `em_author` arms equal the
   pooled arm exactly.
2. **Monotonicity:** the per-author penalized objective (log-likelihood +
   sum log p_a + log(1 - p_a)) is non-decreasing to 1e-9 for every author.
   This is the MAP-EM Lyapunov function; the raw likelihood is not required
   to be monotone.
3. **Boundaries:** no reader forwards, no API. Oracle observations enter only
   the reference arm.

## Scope

- This tests one declared repair. If M still loses, that is retained as a
  negative result.
- No claim about natural text or human validity.
