# F04 Application and Residual Recovery

2026-09-16 JST. Original V3 goal remains ACTIVE. This supersedes the source-only
checkpoint's statement that application scoring has not run. F04 is not yet
finally accepted; no original atomic card is promoted by this report alone.

## Executed

The 21 frozen source bundles were reused, with no source refit or application
selection. Five predictors, seven worlds, three seeds, K=1/2/4 and prefixes
8/16/24/32 produced 378,000 validation/application prediction rows and 75,600
information rows. The primary forecast uses the first 24 observations and
predicts the last eight. Prefix32 has no future score. Rank-one personal OLS
does not report an identified full response profile.

Variance/tail analysis adds 21,600 application rows. T/Q compare linear and
four-cell means; the other worlds use linear means. H uses 2,000 fixed-design
residual-bootstrap refits per prefix, never a Gaussian chi-square interval.
Latent variance/noise enter terminal scoring only. Operational 2*sigma_hat,
true-target 2*sigma, prefix residuals and future prediction errors are separate.

## Findings

Application errors, K4/prefix24, averaged over 600 author instances per world:

| World | Method | Future MSE | Four-cell profile MSE |
|---|---|---:|---:|
| L | Personal OLS | .282927 | .021825 |
| L | REML | .281777 | .021197 |
| T | Personal OLS | .301591 | .047715 |
| T | Four-cell | .282901 | .027453 |
| T | Historical double-half | .282809 | .027561 |
| Q | Personal OLS | .289508 | .052495 |
| Q | Four-cell | .263818 | .028441 |
| Q | Historical double-half | .263538 | .028451 |
| 0 | Personal OLS | .270783 | .021770 |
| 0 | REML | .260811 | .010927 |

The conditional response model matters: nonlinear T/Q profiles are not fully
represented by a straight line. The older double-half method remains useful,
not invalidated by introducing REML. In world0 the source learns that slopes
vary little and shrinks spurious personal slopes. These are simulated response
parameters and forecast losses, not human personality prediction results.

At prefix24, linear T/Q residual variance estimates are .289777/.287478 for
true noise variance .25. Four-cell means reduce these to .257100/.250892.
Omitted conditional structure was being counted as residual noise.

H and L have matched true variance .25 and similar estimated variance
(.248919/.249018), but variance estimation MSE differs (.017986/.006037).
H bootstrap interval coverage is only .763333; L chi-square coverage is .935.
The H interval implementation is algebraically checked but its coverage is
not accepted as 95%. Retain this finite-sample failure for a distinct follow-up,
not a silent replacement or tuning on this application set.

The true two-sigma tail probabilities are .049313 for scaled t5 and .045500
for Gaussian. Heavy tails do not guarantee a huge difference at every chosen
threshold. Prefix fitting reduces residual exceedances; future errors include
mean estimation error, so neither should be called the latent noise tail.

## Verification and Artifacts

- `results/grain_gptpro_foundation_v3/F04/application_full_r1/`: individual and
  aggregate losses, 63 paired-bootstrap contrasts, condition information.
- `results/grain_gptpro_foundation_v3/F04/tail_full_r1/`: individual variance,
  intervals, distinct tail targets and threshold summaries.
- `results/grain_gptpro_foundation_v3/F04/application_audit_r1/summary.json`:
  317 hashes, all 378,000 prediction rows, 21,600 tail rows and 2,713,320 scalar
  checks; maximum discrepancy 3.55e-15.
- New residual module: `grain_research/gptpro_foundation_v3/f04_tail_recovery.py`.
- Runners: `scripts/run_grain_gptpro_f04_application_recovery.py`,
  `scripts/run_grain_gptpro_f04_tail_recovery.py`.
- Audit: `scripts/audit_grain_gptpro_f04_application_recovery.py`.
- Twelve F04 recovery tests pass; Ruff passes. This is a same-agent alternative
  arithmetic audit, not a separate researcher's independent reproduction.

## Remaining

1. Independently replay the 63 paired bootstrap contrasts and tail aggregation.
2. Finish analytical information versus Monte Carlo SE comparison and S-world
   finite-sample mean-versus-profile comparison required by the original cards.
3. Investigate the heavy-tail interval failure with an explicitly separate
   source-only method decision; do not upgrade coverage by assertion.
4. Map evidence to each original F04 card and integrate dependency manifests.
5. Continue F05-F17 and remaining F00/F01 work. No card count or test count
   represents completion of all 44 computational research questions.

All runs are local synthetic CPU computations. No API, private raw data,
MEPS fitting, neural training, lockbox change, commit or push was used.
