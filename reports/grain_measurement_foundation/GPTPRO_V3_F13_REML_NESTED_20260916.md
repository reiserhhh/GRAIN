# F13 REML source uncertainty: completed linear branch

## Execution and scope

F04 L saved numeric data, three seeds, 300 source and 200 application authors per
seed. First 24 application events update the random intercept/slope. Target is
Delta=2b. Source-only MixedLM uses the frozen lbfgs200 / powell200 retry contract;
no application labels enter fitting. Each bootstrap copy receives a distinct
group and event address even when the original author is sampled repeatedly.

Completed three plug-in fits plus 600 source-author bootstrap fits, each followed
by 200 joint parameter draws per application author: 24 million conditional
person draws. No bootstrap fallback or boundary floor. Checkpoints every ten
resamples retain indices, parameters, warnings and draw moments. Seeds and random
draws match the moment arm, permitting paired comparisons without new stimuli.

Artifacts: `results/grain_gptpro_foundation_v3/F13/reml_smoke_r1`,
`reml_full_r1`, `reml_audit_r2`. Earlier partial-scope audit r1 retained.

## Results

| Seed | Plug-in coverage | Analytic mixture coverage | Nested MC coverage | Mixture MSE |
|---|---:|---:|---:|---:|
| 20260915 | .925 | .925 | .915 | .0864623 |
| 20260916 | .950 | .950 | .950 | .0677641 |
| 20260917 | .955 | .955 | .955 | .0664597 |

Eleven of twelve seed/method Wilson intervals contain nominal .95. The first
seed's MC coverage is 183/200, interval [.868104,.946254], and fails the frozen
criterion. This reproduces the moment-arm failure; no retuning or replacement
sampling was used. Normal intervals use mixture moments, not exact mixture
quantiles. These are model-conditioned uncertainty calculations, not clinical
minimum differences or empirical human-personality validity.

## Why the REML and moment results agree

For the common full-rank design X, let B_i be each author's OLS coefficient,
G=(X'X)^-1, C=Sigma+sigma2 G and S the centered coefficient scatter. Up to constants,
twice the negative restricted log likelihood factors as

    (n-1) log|C| + tr(C^-1 S)
    + n(T-p) log(sigma2) + RSS/sigma2.

When Sigma is positive definite, its unrestricted optimum is exactly the
noise-corrected moment solution. This is a balanced-design identity, not a
general equivalence for missing, unbalanced or boundary cases. The actual runner
still executes MixedLM; the factorization independently checks its objective.
Maximum gap per source author is 2.823e-7; parameter differences up to .000627
are optimizer tolerance, not floating-point equality. All 600 fits are interior.

## Numerical checks

All 600 bootstrap source fits compared with the balanced exact optimum; all
24 million conditional draw moments replayed using observation-space conditioning
and explicit two-dimensional sampling. Maximum Gaussian discrepancy 1.58e-14,
sampled-moment discrepancy 8.53e-14. All 2,400 final author/method rows, mixture
files, truths, coverage flags and aggregate metrics linked back to the underlying
calculations, maximum final discrepancy 6.44e-15. Plug-in predictions are replayed
from their saved fit; this audit does not claim a separate optimizer refit for
the three plug-in fits. Same-agent alternate-formula audit, not third-party work.

The expanded final-table audit was also applied to the previous moment arm:
2,400 rows, maximum discrepancy 5.50e-15 (`delta_audit_r2`). Nine related tests
and Ruff pass. Source bootstrap uncertainty is not captured by the conditional
Wilson binomial intervals themselves.

## Next executable work

F05 state/four-step forecast and F08 soft-history next-p remain; then specified
missingness, AR correlation, corrections/withdrawals, source shift, variance
component interfaces and card contracts. This completes one F13 branch, not
F13 as a whole, the 100-card plan or the research goal. No neural/API generation,
private text, MEPS fitting, remote training, lockbox, commit or push occurred.
Both full runs and audits are terminal; no job from this branch remains active.
