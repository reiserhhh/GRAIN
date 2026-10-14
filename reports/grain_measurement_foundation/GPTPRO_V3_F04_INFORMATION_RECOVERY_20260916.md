# F04 Information, Response and Uncertainty Interpretation

2026-09-16 JST. This completes additional numerical obligations of F04-01/04/06;
it does not promote all F04 cards or the entire V3 goal.

## Exact Design and Simulation

For correctly specified independent linear responses, OLS covariance is
sigma^2 (X'X)^(-1). At n=24 and sigma^2=.25:

- One condition has rank one at every n. More repeats estimate that condition's
  mean more precisely but cannot identify intercept and slope separately.
- Two balanced endpoints give slope variance .0104167.
- Four balanced conditions give slope variance .01875.
- The ratio is 5/9. Endpoint allocation is optimal for this slope target on the
  bounded interval, not a universal best design for unknown nonlinear curves.

In L, actual slope MSE is .010368 (two endpoints) versus .019219 (four cells),
close to the exact second-moment prediction. H has .010429/.019296: matched
finite variance preserves this OLS covariance result even though Gaussian
likelihood/Fisher and chi-square interval assumptions do not carry over.
The output explicitly calls the H quantity second-moment precision, not t5
likelihood Fisher information. V uses the actual per-author variance only in
terminal analytic scoring; estimators never receive it.

All five legal coefficient worlds, three seeds, three condition designs and
four budgets give 180 analytical/MC rows. T/Q are excluded from true linear
coefficient recovery and retain their four-condition response targets.

## Same Mean, Different Response

In S, paired authors share intercepts but have slopes +/- .5. At prefix24,
mean-score AUC is .5093/.5027/.4999 across the three seeds, while slope AUC is
1.0 in all three. At prefix8, slope AUC is .9988/.9997/.9985.

This is the prescribed constructive example: a mean can erase a response
distinction that is recoverable at the same event budget. It does not show
near-perfect real personality prediction. The two slopes and noise magnitude
are intentionally well separated in this synthetic world. Finite sample mean
AUC need not equal .5 exactly.

## Production Rules from the Original Comparisons

1. Define the target first: average response, conditional contrast, whole curve,
   dispersion and tail are different estimands. More data at one condition do
   not identify responses elsewhere without model assumptions.
2. Source-only hierarchical shrinkage is useful for linear response parameters.
   In world0, REML reduces profile MSE by .010843, CI [-.012066,-.009610], by not
   treating all sample slope variation as personal variation.
3. Do not make a straight line stand in for every curve. In T/Q, four-cell
   profile MSE improvements over personal OLS are -.020261
   [-.022658,-.017952] and -.024054 [-.027709,-.020641].
4. Historical response-sharing remains competitive in these worlds. These
   results support a standard hierarchical/conditional mechanism, not a novel
   GRAIN-only advantage. Small non-significant increments are not equivalence.
5. Dispersion must retain omitted-mean versus stochastic-error provenance.
   Better conditional means reduce T/Q residual inflation without deleting
   all residual information.
6. H bootstrap undercoverage is a scientific negative of this finite-prefix
   interval, not a failed algebra test and not permission to silently retune.
   Original F04 asks for the comparison, not guaranteed nominal coverage.
   Report it and carry calibration into F13; do not hold the entire research
   queue in a search for a prettier interval.
7. Original S author-bootstrap intervals are reproducible, but S authors share
   intercepts in pairs. Preserve those protocol outputs and label this
   dependence; pair-cluster sensitivity belongs before population inference.

## Evidence

`results/grain_gptpro_foundation_v3/F04/information_full_r1/` contains exact
information/MC comparisons, twelve S comparisons and 189 seed-wise loss
contrasts. Three new analytical-design tests pass.

`results/grain_gptpro_foundation_v3/F04/uncertainty_audit_r1/summary.json`
independently reconstructs all 63 stored bootstrap intervals without the
production helper and all 396 tail aggregate cells. Maximum discrepancy is
1.11e-16. This is same-agent alternative arithmetic, not external peer review.

Remaining F04 engineering: common-record adapter, original task-level evidence
packaging, dependency and review manifests. Preserve original r1 and all
recovery outputs. F05 unknown source dynamics and shared-shock recovery are
the next major unimplemented mechanisms; they must not be relabelled human work.
