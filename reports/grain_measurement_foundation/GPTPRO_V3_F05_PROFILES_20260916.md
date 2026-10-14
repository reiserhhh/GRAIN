# F05 Source-Fixed Likelihood Profiles

2026-09-16 JST. Original F05-04 profiling and Hessian diagnostics executed.
The full F05 package and original V3 goal remain open.

## Protocol and Computation

Before evaluating profiles, select the first20 opaque source IDs in lexical
order in each of nine world/seed bundles. All180 selected authors are source
authors, not chosen for application performance. At prefixes16/32/48 evaluate
rho=0,.02,...,.98. Shared noise and population prior remain frozen source r2
estimates. No truth file or application score is read.

Each fixed-rho point eliminates intercept/slope analytically by GLS. A separate
posterior profile adds the source Gaussian prior, including its cross-parameter
precision terms. GLS and a numerical nuisance optimizer agree in tests.
This produces54,000 profile rows:180 authors x3 prefixes x50 rho values x2 modes.

At each personal MAP solution compute data and posterior Hessians in coordinates
(a,b,eta), rho=.99*tanh(eta), by the prescribed central step1e-4. The prior
curvature is reported separately, not counted as text information.

## Findings

Median width of the grid region with twice-objective increase <=2:

| World | Prefix | Data likelihood | With source prior |
|---|---:|---:|---:|
| core |16|.60|.50|
| core |48|.44|.38|
| near_root |16|.67|.20|
| near_root |48|.18|.12|
| shared_shock |16|.68|.42|
| shared_shock |48|.46|.34|

These widths are descriptive diagnostics, not confidence intervals. The grid
is restricted to nonnegative rho by the original protocol; it is not a search
over every allowable continuous eta. No selected MAP fit was outside this
positive grid in this sample, but the implementation records that possibility.

The near-root short-prefix posterior is much narrower than its data likelihood.
This demonstrates population-prior information supplying concentration, rather
than the short text sequence alone identifying personal dynamics. At48, data
likelihood itself becomes more concentrated. The finding helps explain why good
state forecasts and narrow plug-in intervals can coexist with imperfect rho
recovery; it is a diagnostic of the source model, not a new external replication.

At540 MAP points,49 data Hessians have a negative eigenvalue, while every posterior
Hessian is positive definite numerically. MAP points need not minimize the data
likelihood, so this is not an optimizer failure by itself. The maximum error in
H_posterior-H_data-prior_precision is2.15e-6, consistent with finite differences
amplifying floating cancellation. Do not apply exact-covariance 1e-10 tolerance
to a second-derivative approximation or label these Hessians exact identities.

## Evidence

- `results/grain_gptpro_foundation_v3/F05/profiles_full_r1/`: precomputed ID list,
  54,000 full profile rows,1,080 summaries and540 Hessians with fit attempts.
- `results/grain_gptpro_foundation_v3/F05/profiles_audit_r1/summary.json`: all
  54,000 GLS likelihood/penalty values replayed through scalar Kalman likelihood;
  maximum difference1.71e-13. All selected source IDs and50-point grids verified,
  with78 artifact hashes. Same-agent alternative arithmetic, not peer review.
- `grain_research/gptpro_foundation_v3/f05_profiles.py` implements profiles and
  Hessians. Runner/audit are `scripts/run_grain_gptpro_f05_profiles.py` and
  `scripts/audit_grain_gptpro_f05_profiles.py`.
- Twenty-one F05 tests pass, including numerical nuisance optimization and
  quadratic/prior Hessian checks.

## Next Required Work

The principal remaining F05 mechanism is the actual joint-author common-shock
filter using simultaneous visible histories, compared against ignoring that
covariance. Existing shared-shock generation and marginal forecasts do not
complete this requirement. Remaining application/paired interval audits and
original task/DAG packaging must also be completed. Carry plug-in uncertainty
calibration to F13 without tuning these profiles to application outcomes.

No source refit, API, neural training, private data, lockbox, commit or push.
All processes from this packet have completed.
