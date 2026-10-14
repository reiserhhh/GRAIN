# F05 Missing-Time Comparison

2026-09-16 JST. Original F05-05 time-unit subexperiment completed; the joint
shared-shock channel and F05 final review remain open.

## Design

Reuse nine frozen source r2 bundles. At nominal budgets16/32/48, retain every
third observation. The correct arm retains times0,3,6,...; the incorrect arm
relabels the identical records0,1,2,... with otherwise identical source models.
Run frozen source parameters, personal MAP and a separately computed oracle.
Application includes 200 authors per world/seed, three seeds and three worlds.

The comparison changes the clock, not the observations or author identity.
At budget48 there are16 observations, last t=45. State scoring targets s45, and
forecasts target t46/t49, not the nominal boundary t47. This prevents accidental
mixing of filtering and unobserved two-step prediction.

Correct transitions are rho^3 and Q*(1+rho^2+rho^4). The wrong-clock arm applies
rho and Q per retained observation and then misuses that unit for original-time
forecasts. It is a deliberate incorrect-model comparator, not another equally
valid parameterization. Correct coarse-time reparameterization must transport
both rho and Q together.

## Results at Budget48

Current-state MSE (correct clock versus wrong clock):

| World | Oracle | Frozen source | Personal MAP |
|---|---:|---:|---:|
| core | .139112 vs .148142 | .786049 vs .854688 | .178820 vs .179976 |
| near_root | .185777 vs .227135 | 1.161050 vs 1.173432 | 1.019084 vs .884331 |
| shared_shock | .143500 vs .153848 | .698955 vs .766426 | .180201 vs .176831 |

Oracle correct-clock state improvement in core is -.009030
[-.013785,-.004071]; near_root is -.041359 [-.055672,-.027555]. These support
the time-transition correction with known parameters. Shared-shock intervals
use the original author bootstrap and do not account for only three independent
episode streams; keep that uncertainty limitation visible.

For personal MAP, core rho MSE falls .061950 -> .036602, paired delta -.025348
[-.029205,-.021820], yet state improvement is inconclusive. Near-root state MSE
is worse under the correct clock (+.134754 [.080557,.197548]), while rho MSE
changes little. Four-step outcome prediction there improves slightly
(1.479091 -> 1.450322). There is no blanket claim that a correct structural
model must win every finite-sample metric when parameters are estimated.

This result motivates the already-required likelihood profile/Hessian work:
separate weak data information, prior shrinkage and parameter-state allocation.
It does not justify adopting the wrong clock or tuning it on the application
set. No source model was refitted and no threshold was changed.

## Evidence

- `results/grain_gptpro_foundation_v3/F05/missing_time_full_r1/`: 32,400 scored
  rows, 162 paired contrasts, observed-only prediction snapshots and manifests.
- `results/grain_gptpro_foundation_v3/F05/missing_time_audit_r1/summary.json`:
  all32,400 rows replayed using dense Gaussian conditioning instead of the
  original Kalman filter, 356,400 scalar checks and95 artifact hashes; maximum
  discrepancy2.58e-14. It is same-agent alternative arithmetic, not an independent
  team's reproduction. Paired bootstrap replay is still a separate obligation.
- Nineteen F05 tests pass, including exact time anchor, future-poisoning and
  dense likelihood/posterior equivalence.

Next: source-fixed profiles/Hessians, actual joint-author common-shock channel,
remaining application/interval audits and original task/DAG evidence packaging.
No API, new neural training, private text, lockbox, commit or push. No running
process remains from this packet. The full V3 goal is still active.
