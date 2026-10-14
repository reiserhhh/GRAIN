# F13 learned-source masked uncertainty: full nested execution

## Execution and evidence

Completed all 72 world/mask/seed/assumed-kernel cells: IID, AR=.6 and prior shift;
complete/MCAR/MAR/MNAR; three seeds; assumed rho=0/.6. Each cell has 200 application
authors, 200 actual source-bootstrap fits and 200 joint personal-parameter draws
per source fit. Total 14,400 author-condition objects, 576 million parameter draws,
86,400 target/method rows and 432 metric cells. Runtime 48.76s, no unsupported
source fit was omitted. Source prior shift uses the old IID source distribution.

Results under `results/grain_gptpro_foundation_v3/F13/`:

- `missing_nested_full_r1`: predictions, metrics, all per-source conditional
  means/covariances, MC moments, RNG specification and frozen input hashes.
- `missing_nested_audit_r1`: all 576 million draws reconstructed using explicit
  two-dimensional Gaussian formulas; mixture covariance independently recomputed
  from raw second moments. Maximum difference 6.14e-13.
- `missing_nested_smoke_r1`: six-author smoke across 24 cells.

All 432 coverage/width/MSE/decision rows were independently recalculated from
86,400 saved predictions. Fourteen related unit tests and Ruff passed. The full
draw replay checks source-conditioned stored moments; the separate source fit
normal-equation replay and batch-vs-observation conditioning tests establish
other parts of the chain. This is not independent third-party reconstruction.

## Main result

Complete observations, descriptive mean coverage across seeds:

| Actual world | Assumed rho | Target | Plugin | Source mixture |
|---|---:|---|---:|---:|
| AR=.6 | 0 | level | .666667 | .666667 |
| AR=.6 | .6 | level | .946667 | .946667 |
| AR=.6 | 0 | contrast | .985000 | .985000 |
| AR=.6 | .6 | contrast | .941667 | .940000 |
| IID | 0 | contrast | .945000 | .943333 |
| IID | .6 | contrast | .598333 | .596667 |

Source bootstrap accounts for uncertainty within the specified family; it does
not integrate uncertainty over the family of residual kernels. A wrong kernel
can therefore remain confidently wrong after actual source fitting and nested
sampling. This is target-specific: the same error can overcover level and
undercover contrast, or the reverse. There is no universal interval multiplier.

Across 144 cells per method, plugin passes 87 frozen Wilson checks and fails 57;
source mixture and MC each pass 88 and fail 56. All failures remain. These counts
are not independent experiments or a scientific completion percentage.

Mixture variance is E[V|fit]+Var[E|fit], retaining level/contrast covariance.
It need not exceed a separately fitted plugin variance: the average conditional
covariance can be smaller than the plugin conditional covariance. No artificial
floor enforces wider intervals. The MC and analytic summaries remain separate.

## Scope and remaining integration

- Kernel shape rho is declared, not learned; covariance magnitudes come from
  complete independent source authors. Source-side missingness is not covered.
- Selection-ignoring Gaussian inference under MNAR is a working approximation,
  not an oracle conditional posterior despite known noise parameters.
- Source covariance fitting is the explicitly recorded moment arm, not REML.
- This execution closes the learned-source linear level/contrast missingness
  matrix. Dynamic and binary missingness interfaces, cross-component variance
  contracts, paired changes and canonical card manifests remain to reconcile.
- F13-04's former label-only biased-normal PASS remains invalid. Do not claim
  universal calibration or promote F16 as reliable under rejected conditions.

Next prioritize a cold uncertainty contract that carries target, dependence,
support and calibration status into downstream construction; reconcile actual
protocol coverage before adding experiments. No extra parameter sweep to rescue
failed coverage. No human data/API/neural training/commit/push/lockbox action.
