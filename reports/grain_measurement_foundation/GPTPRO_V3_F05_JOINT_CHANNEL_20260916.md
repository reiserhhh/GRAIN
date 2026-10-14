# F05 Joint Common-Shock Channel

2026-09-16 JST. The remaining joint-input mechanism of F05-05 has now been
implemented and run. Original task/DAG review and other V3 packages remain open.

## Mechanism and Isolation

Source first48 observations estimate the mean off-diagonal covariance of
innovations y[t]-a-bz[t]-rho*(y[t-1]-a-bz[t-1]). Each source-author series is
centered before computing covariance. Raw estimates are retained; common
variance is constrained to [0,.99*learned total process variance] for positive
definiteness. This moment estimator can be biased by estimated parameters and
short source episodes; it is not a claim of exact noise recovery.

Application joint and independent arms reuse identical stored personal MAP
parameters, marginal process variance and observation variance. Only the
off-diagonal process covariance differs. Initial stationary covariance is
P_ij=Q_ij/(1-rho_i*rho_j). At each time, the vector Kalman update uses current
and past observed author responses. Future shocks are never read or copied
from the source episode. Parameters freeze at48 for sequential last16 scoring.

Source, application and previous prediction manifests are linked and verified.
Observed workers run before the terminal scorer reads latent states or computes
the oracle. Core/near-root worlds provide no-common-shock comparisons; the true
.125 common variance is used only by the explicitly labelled oracle arm.

## Results

Source common-variance estimates for the three shared-shock seeds are
.096529/.153626/.177828, versus true .125. Core estimates are .000197-.000547;
two near-root raw estimates are negative and clipped to0, while the third is
.000804. All raw and constrained values are preserved.

At prefix48, current-state MSE in the shared-shock world:

| Parameters | Independent | Joint | Relative reduction |
|---|---:|---:|---:|
| Learned personal MAP | .178882 | .123640 | 30.9% |
| Oracle | .159201 | .088239 | 44.6% |

The learned-model state reduction has the same sign in all three episodes:
.193201 -> .117920; .213188 -> .130678; .130256 -> .122320. The oracle direction
also agrees in all three. Learned plug-in coverage changes .881667 -> .908333;
this does not include uncertainty in source calibration or personal parameters.

Learned four-step observed-outcome MSE changes only .662821 -> .659515, with
mixed episode directions; conditional-mean MSE .047446 -> .047343. Same-time
peer observations help identify a common current state, but cannot reveal future
unobserved common innovations. State recovery and forecast gain must remain
separate outcomes. Core/near-root joint changes are negligible, consistent with
their tiny learned off-diagonal covariance; oracle arms are identical there.

There are only three independent episode streams per world. No author-bootstrap
population confidence interval is claimed for this joint experiment. The result
is a finite simulated mechanism finding, not human emotion/personality validity.

## Evidence

- `results/grain_gptpro_foundation_v3/F05/joint_full_r1/`: 129,600 scored rows,
  source moments, per-episode and aggregate summaries, observed predictions.
- `results/grain_gptpro_foundation_v3/F05/joint_audit_r1/summary.json`: 64,800
  independent-arm states/variances match the prior scalar-filter outputs to
  1.78e-15; all 129,600 state errors/coverage indicators and 237,600 forecast
  errors checked;75 artifact hashes verified.
- Small-panel full Gaussian conditioning independently verifies the multivariate
  filter, the zero-common limit matches scalar filters, and future poisoning
  leaves earlier states identical. Source common moments match direct covariance.
- Twenty-four F05 tests pass. Same-agent numeric audits do not substitute for
  an independent team's implementation or the original task review.

Code: `grain_research/gptpro_foundation_v3/f05_joint.py`.
Runner/audit: `scripts/run_grain_gptpro_f05_joint.py` and
`scripts/audit_grain_gptpro_f05_joint.py`.

## Next

Reconcile F05 original cards against the now-available source/application,
missing-time, profile, invariance and joint-channel evidence; finish remaining
paired-interval/full-application replay and shared-episode qualifications.
Package common-record/dependency manifests without declaring all V3 complete.
Then continue F06's still-unimplemented distinct policy experiments. No API,
neural training, private data, lockbox change, commit or push; no active job
remains from this packet.
