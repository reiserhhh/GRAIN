# F11 source-refit uncertainty protocol v1 (frozen 2026-09-26 before running)

Worker: Claude Opus 5.5. Census gap F11-04: the plug-in trait intervals omit
source-fit uncertainty, and the conditional-loading world under-covers
(.9289). The oracle arm also under-covers there (.9356), so no single cause is
assigned in advance.

## Data (reused, not regenerated)

- `F11/mtmm_full_r1`: source, validation and application data, the stored
  REML fits and predictions, and truth.
- 3 seeds x worlds base / shared_event / conditional_loading x panels
  all_two_occasions / same_event_three_readers / one_reader_two_occasions.

## Computation

- **Built-in replay:** the stored REML fit applied with `predict_traits` must
  reproduce the stored application predictions and variances to 1e-10, and
  the plug-in coverage must equal `metrics.csv` exactly.
- **Source bootstrap:**
  - B = 200 resamples of the 300 source authors per seed x world, stream
    SeedSequence([seed, 11, 40, world_index, b]);
  - each resample is refit with the same `fit_mtmm` (same conditional and
    event flags);
  - failed fits (success false) are counted and excluded, with their count
    reported.
- **Augmented interval (primary):**
  - centre: the canonical REML prediction;
  - variance: canonical posterior variance + variance across the bootstrap
    refits of the application prediction (per author and trait).
- **Mixture interval (secondary):** centre and variance from the law of total
  variance over refits.
- **Coverage of the true traits:** plug-in, augmented, mixture and oracle,
  pooled over seeds per world x panel. Author-cluster (seed-stratified)
  bootstrap 95% CI (2000 draws, seed 20260915), with Wilson intervals reported
  alongside.
- **Paired contrasts of coverage** (author-level mean over 3 traits):
  augmented minus plug-in, and oracle minus nominal (.95).

## Decision rules (declared)

- Source uncertainty explains the conditional-loading under-coverage only if
  the augmented interval's author-cluster CI contains .95 while the plug-in
  CI does not.
- If the oracle author-cluster CI contains .95, oracle under-coverage is
  attributed to sampling noise; otherwise it is recorded as unexplained.
- Wide intervals are not success; mean width is reported for every arm.

## Scope

Simulated MTMM worlds only. No reader, human or external validity claims.

## Amendment A1 (2026-09-26, recorded after the 3-refit smoke and before the full run)

- **Defect found in smoke:** `fit_mtmm` raised `ValueError("negative
  variance")` on a bootstrap resample. The optimizer moved to extreme Cholesky
  entries, where the PSD check `eigvalsh(sigma).min() < -1e-10` fails
  numerically.
- **Handling:** a raising refit counts as a failed fit, exactly like
  success=false. It is excluded from the bootstrap, its exception is logged,
  and failures are reported by world and seed.
- **Estimator:** unchanged. A robust re-parameterisation would be a new
  estimator revision and is not part of this packet.
- **Tolerance:** if more than 10% of refits fail in any world/seed, the
  augmented interval for that cell is reported as UNRELIABLE_BOOTSTRAP rather
  than used for the decision.
