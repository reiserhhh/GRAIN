# F06 B full dynamic policy comparison

Date: 2026-09-16. Full computation: `results/grain_gptpro_foundation_v3/F06/gaussian_full_r1`.
Aggregation/interval audit: `gaussian_full_summary_audit_r1`.
Physical-path audit: `gaussian_full_audit_r1` (completion to be checked before final numerical acceptance).

## Decision

The frozen local information-gain heuristic is not a universal measurement policy. Its state-targeted version is a clear adverse counterexample in the prescribed simulation. Preserve that result rather than change the criterion or population to obtain a win.

The complete run used 1,800 simulated individuals (300 source/100 validation/200 application for each of three seeds), two paired reactivity worlds, seven paths per person, 24 observations and three prefix budgets. Four target readouts are reported separately. The population nuisance prior/noise is the frozen F05 core fit, extended by the declared beta prior; no application truth entered the estimator or action-selection functions. The run finished in 963.50 seconds without time-limit restart.

## Application results at 24 observations

The following table uses the distributed-beta world and 600 application individuals. Each row's adaptive arm is optimized for that row's target. MSE units differ across targets and must not be averaged into one score.

| Target | Fixed | Random | Coverage | Targeted adaptive |
|---|---:|---:|---:|---:|
| a, baseline level | .061434 | .098779 | .106904 | .061405 |
| b, condition slope | .013331 | .020420 | .018829 | .013047 |
| beta, intervention response | .039419 | .027970 | .027858 | .025671 |
| current state | .186687 | .242225 | .255478 | .333064 |

Paired adaptive-minus-fixed MSE differences:

- a: -.000029, 95% CI [-.000174,.000113].
- b: -.000285, CI [-.001098,.000533].
- beta: -.013748, CI [-.017251,-.010440].
- state: +.146377, CI [.106849,.190290].

The a/b policies add no clearly resolved gain over the simple alternating-z/no-intervention design. This is not a formal equivalence claim. They do improve relative to random/coverage policies that also change the state through h.

Beta-targeting improves over a design with no beta information at all, but the gain against random is only -.002299, CI [-.005218,.000814], and against coverage -.002187, CI [-.005256,.000740]. Its additional intervention cost is clearly positive in both comparisons. Therefore the current calculation does not justify paying that extra cost instead of using coverage/random when the sole goal is beta estimation.

## Intervention cost and the state-target failure

Mean cumulative squared physical displacement from the same-noise no-action state:

| Policy | Squared displacement | Effective actions |
|---|---:|---:|
| Fixed, scalar-a, scalar-b | 0 | 0 |
| Coverage | 6.8231 | 12.0000 |
| Random | 7.5418 | 11.5067 |
| Scalar-beta | 9.5981 | 12.9767 |
| Scalar-state | 24.1305 | 23.0000 |

The state-targeted policy intervenes at every effective step. Relative to random, its state MSE increases .090839, CI [.065087,.118669], and its squared displacement increases 16.5887, CI [14.7297,18.6728]. It is worse on both axes in this finite comparison.

In the nonreactive world, physical displacement is exactly zero for every policy, yet scalar-state MSE is .395169 versus fixed .186687. Consequently the failure cannot be attributed solely to physically disturbing the simulated state. Selection under an approximate posterior, opportunity variation, and nuisance/prior effects also matter. This packet establishes the failure, not a unique causal attribution among those mechanisms.

There is a mathematical reason not to equate the criterion with low final error: when an intervention changes the predictive covariance, maximizing V_before-V_after need not minimize V_after. A candidate can have a large apparent reduction because it begins with larger uncertainty. The implementation follows the frozen reduction criterion; no posterior-risk or cost-penalized replacement has been selected after viewing application results.

## Calibration and information versus prior borrowing

All 604,800 prefix fits found at least one converged solution. Numerical convergence did not guarantee calibration: fixed-design a and state coverage are about .8983 and .9083 respectively; b coverage is .955. These are empirical checks of the working MAP/expected-information covariance, not exact-posterior guarantees.

With h=0, beta retains its prior: .4 mean and .04 variance. In the distributed-beta world its MSE is .039419 and coverage .9583, despite containing no personal beta evidence. In the zero-beta world the same prior-only estimate has MSE .16 and zero nominal 95% coverage. This is a concrete warning that good population-matched error/coverage can coexist with no individual identification. Beta-targeting reduces zero-world MSE to .074754 but still has strong shrinkage bias; it does not completely solve prior mismatch.

## Evidence and remaining closure

- 302,400 target metric rows rebuilt from all completed author packets; all role counts verified.
- 604,800 common records exported with explicit after-observation action timing; no simulated text is invented.
- All 288 original paired intervals independently replay, maximum difference 1.7764e-15. An additional 144 target-specific loss/cost contrasts and descriptive Pareto rows are saved.
- 1,809 output hashes verify in the aggregate audit. Zero failed prefix fits are retained as a checked result, not assumed.
- The completed physical-path audit reconstructs all 604,800 steps and 302,400 metrics for all 1,800 individuals, including initial policy actions. Its status is `PHYSICAL_PATH_DENSE_STATE_GAIN_METRIC_REPLAY_PASS`, maximum numeric difference 1.7764e-15. This is same-agent alternate-formula verification, not third-party replication. The audit summary's interval-pending wording is superseded by the separately completed aggregate audit above.
- Analytic gradients/Fisher and state-parameter covariance formulas were checked by focused tests. A Gaussian working covariance is still an approximation; calibration limitations above remain scientific outcomes.

Artifacts include `summary_by_policy.csv`, `summary_by_seed.csv`, `paired_comparisons.csv` in the full run; `target_specific_comparisons.csv`, `descriptive_pareto.csv`, `coverage_intervals.csv`, `fit_rates.csv` and `common_records.jsonl.gz` in the aggregate audit.

Original F06 card-level execution-request/DAG integration and the final cross-panel production-rule mapping remain agent work. This run does not complete every F00-F17 requirement or any human validation. No parent cards are automatically promoted. F07 corrected information laws have proceeded independently while this run executed.

No API, neural training, private raw text, MEPS fitting, remote execution, lockbox, commit or push occurred. Overall goal remains ACTIVE.
