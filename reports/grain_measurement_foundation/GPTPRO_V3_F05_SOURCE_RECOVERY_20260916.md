# F05 Dynamics Source Recovery

2026-09-16 JST. Active original V3 goal. Source computation completed; application
and scientific acceptance are still open.

## Changes

The old shared-shock generator drew a separate block shock inside each author's
loop. The corrected generator supplies the same innovation to all authors in a
role episode, with common/individual innovation variances .125/.125. Source,
validation and application have independent episodes. Stationary initial
cross-author covariance is .125/(1-rho_i*rho_j), plus individual diagonal
.125/(1-rho_i^2). This is not a fixed common source effect.

Parameter, common shock, own innovation and measurement-noise random streams
are separate. Nested time budgets preserve observations; changing source count
does not change application people. Truth stores latent trajectories for later
terminal evaluation, not model fitting.

The new learner actually optimizes source dynamics rather than assigning rho
from residual autocorrelation. Two alternating source rounds use personal ML
then MAP, fixed rho starts .2/.8, L-BFGS-B at most 100 iterations, shared log
noise scales bounded log(.05)..log(5), and source Gaussian parameter moments
with covariance regularizer 1e-4 I. All personal attempts and shared-noise
optimizer outcomes are retained. Failed personal fits return a declared source
baseline and are explicitly counted; no silent optimizer sweep is used.

The first source trial used all 64 source observations. It remains
`F05/source_full_r1` but is not canonical. To apply the protocol's first48/last16
boundary consistently, r2 limits every source fit to the first48. An executable
test changes all source outcomes after48 by +10000 and verifies the entire
source fit is identical. No application outcome has been inspected or scored.

Source workers are separate processes reading only source.jsonl. Their input
hash, fitted-model hash and optimization records are saved. Each world still
has 300 source, 100 validation and 200 application authors with 64 observations.

## Executed Results

Canonical: `results/grain_gptpro_foundation_v3/F05/source_full_r2/`.
All nine world/seed source fits finished; all 300 authors per world converged
in both rounds, and all 18 shared-noise optimizations reported convergence.
All 69 output-manifest hashes were verified. Source fits take about 3.2-3.5 s
per bundle; fast execution here reflects scalar CPU likelihoods, not missing
optimization. Iteration/parameter records are saved for inspection.

| World | Learned process SD range | Learned observation SD range | rho at prior mean eta |
|---|---:|---:|---:|
| core | .539-.549 | .428-.443 | .380-.398 |
| near_root | .517-.522 | .467-.475 | .856-.868 |
| shared_shock | .535-.602 | .428-.452 | .353-.434 |

True marginal process and observation SDs are both .5. Optimization convergence
does not establish parameter recovery: core and near-root estimates exhibit
finite-prefix bias, and shared shocks are not yet modelled jointly by this
marginal source likelihood. Application comparison will measure the practical
effect; do not tune these results to force rho agreement.

## Mathematical Checks

Nine tests pass. Scalar Kalman likelihood and terminal state posterior match
independent dense Gaussian conditioning for contiguous and every-third-step
observations. Gaps use rho^gap and Q*(1-rho^(2gap))/(1-rho^2).
The cold scoring API uses rho^k*s_T for the k-step forecast, with
rho^(2k)*P_T + Q*sum_{j=0}^{k-1}rho^(2j) + R variance. A four-step dense-state
test checks this independently. These are plug-in intervals; parameter
uncertainty remains F13 work.

## Remaining Original Requirements

- Frozen-source application at prefixes16/32/48, then sequential last16 scoring.
- Source-only, static OLS, source LMM and oracle Kalman comparisons; failed fits
  and successful-only versus all-author losses both reported.
- Actual joint-author shared-shock covariance channel using visible history.
- Missing-time correct versus wrong-one-step experiment, beyond algebra tests.
- Twenty source-fixed likelihood profiles, Hessians and near-root diagnostics.
- Two-state similarity invariance, restricted-coordinate rank and rho=1 shift
  counterexample.
- Independent numerical audit, common-record adapter, original task evidence
  and DAG review packaging. F04 task-level packaging also remains open.

Code: `grain_research/gptpro_foundation_v3/f05_recovery.py`.
Runner: `scripts/run_grain_gptpro_f05_source_recovery.py`.
Tests: `tests/test_grain_gptpro_f05_recovery.py`.
No API, neural training, private data, lockbox change, commit or push. No active
process remains from this source packet. Do not restart source fitting without
an implementation reason; continue from the frozen r2 bundles.
