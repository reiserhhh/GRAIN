# F12 pilot-calibrated transport: four channel worlds

## Design executed

Three seeds, each 300 old-source authors,100 independent new-domain pilot authors
(first60 fit,last40 validate),200 new application authors. Source parameter mean
(0,.5), covariance diag(1,.25); new mean(1,.2), covariance diag(1.4,.7). Each has
32 balanced observations at z=-1,-1/3,1/3,1, Gaussian residual variance .25.
Source residual variance and population reference are estimated from observations,
not hidden parameters. First24 application readings estimate personal OLS response;
targets are four expected condition scores and Delta=2b.

Four channels share authors and original observation noise: identity,1+1.5r,
z+(1+.25z)r,and r+.1r^3. Pilot pairs are the same observed material transformed
deterministically, not independently noisy readers. Calibration modes are fixed,
not selected using validation/application performance. New population differences
are not normalized away. Nonlinear adds the prescribed cubic inverse basis
[1,r',r'^3]; nonpositive derivative anywhere on pilot range rejects the inverse.

## Application results

Three-seed mean Delta MSE / nominal95% coverage:

| Channel | Uncalibrated | Global inverse | Conditional inverse |
|---|---|---|---|
| Identity | .07455 / .9583 | .07455 / .9583 | .07455 / .9583 |
| Global | .88650 / .4300 | .07455 / .9583 | .07455 / .9583 |
| Conditional | 6.74529 / .0017 | 5.24618 / .0067 | .07455 / .9583 |
| Nonlinear | 6.55794 / .4417 | .79270 / .6533 | 1.14693 / .6283 |

Uncalibrated/global conditions retain the old error unit in intervals deliberately
to expose invalid transport; they are not calibrated confidence intervals in the
new channel. Correct matched affine inverses reproduce old noisy observations to
floating precision; they do not recover noiseless author truth exactly.

Global transformation preserves ranking (Delta Spearman .984824) while changing
absolute error and coverage substantially. Conditional bias cannot be repaired
by global calibration. Nonlinear global calibration reduces error but leaves
miscalibration; conditional linear calibration does not fix its curvature.

The prescribed cubic inverse is NONMONOTONE_INVERSE in all three full seeds.
It produces zero supported application scores, not a fabricated successful inverse.
We did not replace it or tune its domain after inspecting results. Rank-deficient
single-condition pilots return MISSING_ANCHOR in all four worlds/three seeds.

Fixed-fit paired application-author bootstrap,2000 samples stratified by seed:
global calibration minus uncalibrated Delta MSE=-.81195[-.90721,-.71457];
conditional minus global on conditional channel=-5.17163[-5.41300,-4.93147];
global minus uncalibrated on nonlinear=-5.76525[-8.27772,-3.97432]. These intervals
omit source/pilot fitting uncertainty and cannot be labeled total uncertainty.

## Verification and state

Smoke completed before full run. Full output has 9360 author-method rows and78
metric rows. Alternate QR reconstruction checks27 pilot fits; per-author slopes,
four-cell loss and target values replay with maximum discrepancy8.53e-14.
Five transport tests and Ruff pass. Full score/support and aggregate rank/coverage
tables still require alternate replay; this report distinguishes that from the
completed parameter/author checks. No full F12 card promotion yet.

Artifacts: `results/grain_gptpro_foundation_v3/F12/transport_full_r1`,
`transport_smoke_r1`, `transport_audit_r1`. Source/pilot/application observations,
separate truth, calibration coefficients/status, supported scores and author metrics
are retained. Known-support interpolation and population reference are separate.

### Follow-up: full table replay

`scripts/audit_grain_gptpro_f12_tables.py` independently reconstructs the four
forward channels, each calibrated score, support masks, all missing-value positions,
OLS slopes and four-condition losses, rank-based correlations and interval coverage.
All 299,520 scalar score slots (including 23,040 intentionally missing cubic slots)
and 78 aggregate rows match, maximum difference 4.00e-15. Artifact:
`results/grain_gptpro_foundation_v3/F12/table_audit_r1`. Thus the earlier full-table
replay gap is now closed; source-refit uncertainty is not closed by this audit.

The protocol/card text requests four fixed interventions but does not operationally
specify their values in the F12 section. The next signature packet must explicitly
freeze an operational definition before execution, disclose that elaboration and
not claim it was specified in the original run. Coordinate rotation alone is not
evidence that psychological constructs transport.

Next: finite intervention-signature identity and coordinate rotation certificate,
source uncertainty linkage to F13 and task contracts.
No new human data, translation, API, neural training, remote job, lockbox, commit
or push. Same-agent alternate arithmetic is not independent third-party replication.
