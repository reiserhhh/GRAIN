# F13 dynamic source/person uncertainty: full finite-grid closeout

## Scope and decision

The F05-derived three-world, three-seed dynamic grid has finished. This closes the
execution and numerical replay of this subexperiment, not F13 or the full V3 plan.
Nominal 95% interval calibration fails in retained cells, most clearly for current
latent state in the near-root world. No thresholds or fitted parameters were
changed to obtain acceptance.

All data here are synthetic numeric observations. No new human data, API calls,
neural training, lockbox opening or publication occurred.

## Completed calculation

- 1,800 actual source-author bootstrap fits: 200 in each world/seed cell.
- 200 application authors per cell and three observed-prefix budgets (16/32/48).
- 200 personal-parameter draws per prefix per source fit: 216,000,000 draws,
  representing 1,080,000 source-conditioned prefixes, not independent people.
- Final source aggregation: 5,400 prefixes and 16,200 target rows; no unsupported
  prefixes or missing source cells. Nine complete cells, 81 metric rows.
- Parameter integration uses a local Gaussian Laplace approximation, not an exact
  parameter posterior. Shared-shock source bootstrap is conditional on its episode.

The retained total covariance is

\[
V=E_b E_\theta[V(Y\mid\theta,b)]
 +E_b\operatorname{Cov}_\theta[E(Y\mid\theta,b)]
 +\operatorname{Cov}_b[E_\theta E(Y\mid\theta,b)].
\]

All three targets retain their joint covariance. Coverage uses moment-based
Gaussian intervals, not exact quantiles of the possibly non-Gaussian mixture.

## Final results

The table averages coverage across seeds and prefix budgets for description;
formal frozen decisions remain at each seed/budget/target cell.

| World | Target | Coverage | Mean width | MSE |
|---|---|---:|---:|---:|
| core | current state | .910556 | 1.524032 | .196404 |
| core | future four-observation average | .923889 | 2.001196 | .311189 |
| core | fourth future observation | .938333 | 3.074919 | .679830 |
| near_root | current state | .795556 | 2.537315 | 1.061002 |
| near_root | future four-observation average | .928333 | 3.027294 | .702553 |
| near_root | fourth future observation | .928333 | 4.156975 | 1.312811 |
| shared_shock | current state | .926111 | 1.569255 | .193262 |
| shared_shock | future four-observation average | .947222 | 2.080297 | .289546 |
| shared_shock | fourth future observation | .944444 | 3.178654 | .694485 |

Core: 19/27 frozen Wilson checks pass, 8 fail. Near-root: 10/27 pass, 17 fail.
The 27 shared-shock cells remain descriptive because authors share a disturbance;
independent-author Wilson intervals are not acceptance evidence for those cells.
Prefix budgets also reuse the same authors and are not independent replications.

Against the earlier source plug-in calculation, average-four coverage improves
from .881667 to .923889 (core), and .895556 to .928333 (near-root). This supports
including parameter uncertainty. It does not establish universally calibrated
state recovery. Intercept/state ambiguity, local approximation and mixture shape
are candidate explanations, not separately identified causes in this experiment.

## Numerical verification

1. `dynamic_source_audit_r2`: all 1,800 source identities, resample indices, prior
   moments and noise likelihoods checked. Maximum discrepancies: 4.44e-16 for
   prior moments, 3.64e-12 for noise objective. Zero recorded optimizer failures;
   this is not proof of global optimization.
2. `dynamic_personal_audit_r1`: all 216 million saved-recipe draws replayed through
   an alternative information-form filter and explicit cross-target covariance
   formulas. Maximum discrepancy 1.42e-13; zero unsupported/missing entries.
   This verifies saved draw calculations, not MAP refitting or Laplace validity.
   Extra expected-RNG-identity and resume-status guards were added to the script
   after this process started; do not attribute those added guards to this run.
3. `dynamic_final_table_audit_r1`: 16,200 target rows linked to joint covariance
   records, 81 metrics/decisions recalculated. Statsmodels Wilson calculation
   agrees within rounding tolerance with the production 1.96 convention.
4. 28 related tests passed; Ruff checks for the new replay code passed.

All result roots above are under `results/grain_gptpro_foundation_v3/F13/`.
Canonical producer/consumer roots are `dynamic_source_full_r1` and
`dynamic_personal_grid_r1`; final aggregate is `dynamic_aggregate_r2`.
Partial aggregate/audit r1 outputs are retained as historical snapshots.

## Next executable work

- F08 currently has fixed-source probability/outcome variance decomposition;
  source bootstrap and personal uncertainty remain executable work.
- F13 missingness, correlated errors, evidence-history correction/withdrawal,
  shifted-prior cases and the shared uncertainty interface remain open.
- Do not spend another source-grid run attempting to tune away these failures.
  Any targeted diagnosis must distinguish latent-state recovery from observable
  forecast quality and test a mechanism rather than optimize application coverage.
- F14-F17 and cross-family task contracts remain on the original full-plan queue.

No parent card is promoted by this report. The research goal remains active.
