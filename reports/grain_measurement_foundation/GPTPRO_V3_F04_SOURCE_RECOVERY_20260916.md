# F04 Source Model Recovery

2026-09-16 JST. Active V3 goal; this is not F04 final acceptance.

## Executed

All seven worlds (L/T/Q/V/S/0/H), three seeds and full source budgets were
generated and fitted. Each world/seed has 300 source authors with 32 events,
plus separate 100 validation and 200 application authors for the next stage.
Twenty-one source REML fits completed without falling back to moments.
The output manifest's 288 file hashes were verified. Eight new tests pass.

The source bundle contains pooled OLS, actual REML random-intercept/random-slope
models, noise-corrected four-condition covariance, and the historical
response_sharing double-half method. No application responses or latent truths
are passed to source estimators. Application scoring has **not** been executed.

Canonical source bundle:
`results/grain_gptpro_foundation_v3/F04/source_full_r1/`.
Runner: `scripts/run_grain_gptpro_f04_source_recovery.py`.
Implementation: `grain_research/gptpro_foundation_v3/f04_recovery.py`.

## Repaired Mechanics

- REML uses one L-BFGS attempt and one Powell retry only; a failed fit is labelled
  and falls back to the fixed noise-corrected source moment estimator. This
  failure branch was explicitly tested rather than claimed in JSON.
- Four-cell covariance subtracts `sigma2 * mean((X'X)^-1)` from the covariance of
  source coefficient estimates before the frozen eigenvalue floor. In the
  balanced four-cell case this is exactly `diag(sigma2/n_cell)`.
- Posterior coefficients match an independent precision-form computation.
  Zero random-effect covariance returns the source mean, and rank-one data do
  not acquire a fictitious data-identified slope.
- Duplicate event IDs are rejected rather than increasing apparent information.
- Old double-half noise is scaled from its actual source half-event budget to
  the application's cell-mean budget. Source4/application6 yields sqrt(4/6)
  scaling on each covariance side. This relies on independent event replication;
  it is not a general temporal-noise law.
- S authors are truly paired with identical intercepts and opposite slopes.
  World0's stored slope is actually .5. H/L share latent parameters and uniform
  noise ranks but use variance-matched t5/Gaussian errors. Budgets are nested and
  role streams independent. These properties have executable tests.

## What Remains Before F04 Closure

Run the frozen first24/last8 application prediction comparison and n8/16/32,
K1/2/4 information designs. Report actual four-condition response and contrast
losses, not just linear coefficient errors in nonlinear worlds. Add V/H variance
and tail experiments with RSS degrees of freedom, Gaussian-only chi-square
intervals and the prescribed heavy-tail bootstrap. Preserve unobserved or
rank-deficient outputs as such; source borrowing is not data identification.

Then independently replay losses and source hashes, package F04's original
card outputs, and integrate the shared execution dependency system. Source-fit
completion is not an application-performance result or a personality-validity claim.
