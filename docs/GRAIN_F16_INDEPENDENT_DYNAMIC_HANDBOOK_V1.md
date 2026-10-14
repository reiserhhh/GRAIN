# F16 independent construction handbook: T3

This is an independent-agent reconstruction packet, not a human reconstruction experiment. It specifies the mathematics and data contract without supplying T1/T2 implementation. Preserve all ambiguities, questions and added rules in your construction log. Do not consult other repository modules, tests, outputs or existing T1/T2 source code. No application truth is provided.

## Brief

Frozen brief: `configs/grain_f16_dynamic_briefs_v1.json`. T3 estimates the mean of four future observations under future conditions (+1,-1,+1,-1), given the author's observed prefix and a frozen source ensemble. Do not substitute a C2 binary model or an F09 family classifier. This is a synthetic numeric construction, not a clinical instrument.

## Data and source model

For each seed 20260915, 20260916, 20260917, permitted source observations are
`results/grain_gptpro_foundation_v3/F05/source_full_r2/core_<seed>/source.jsonl`.
Rows have author_id, role=source, world=core, event_id, episode_id, integer t, z and y. Use the first 48 observations per source author. IDs are grouping only, never predictors. There are 300 source authors.

Two hundred already-fitted source bootstrap models are provided at
`results/grain_gptpro_foundation_v3/F13/dynamic_source_full_r1/core_<seed>/fit_<000..199>.json`.
Read status, seed, world, bootstrap, source_ids, fit.prior.{mu,Sigma}, fit.log_sigma. Verify source identities and status=FITTED; record file hashes. No refitting of the source distribution is requested. Retain failed models as failures, never discard and renormalize.

Do not inspect application/validation/truth files or other output roots. The coordinator supplies new input prefixes after construction. Implementation may use standard NumPy/SciPy, but must not import repository mathematical/scoring functions.

## Observation law and personal update

Parameters psi=(a,b,eta), rho=.99*tanh(eta), Q=exp(2*log_sigma[0]), R=exp(2*log_sigma[1]).

`y_t = a+b*z_t+s_t+e_t`, `s_t=rho*s_(t-1)+u_t`, independent u~N(0,Q), e~N(0,R).
The initial observation state uses the stationary zero-mean variance Q/(1-rho²).
For gap d, transition coefficient rho^d and innovation variance Q/(1-rho²)*(1-rho^(2d)).
At each observation, innovation y-a-b*z-predicted_state; innovation variance P+R; gain P/(P+R); state mean += gain*innovation, variance *= (1-gain). Accumulate Gaussian log likelihood. Reject duplicate events, mixed authors/worlds/roles, noninteger or unordered times and nonfinite inputs. A time gap does not imply zero missing observations.

For each source fit, find personal MAP minimizing negative loglik + .5*(psi-mu)'Sigma^-1*(psi-mu). Two fixed L-BFGS-B starts: a=mean(y), b=least-squares slope on intercept and z, eta=atanh(rho/.99) for rho=.2 and .8. maxiter=100, ftol=1e-9, gtol=1e-6. Retain converged finite minimum; if neither converges, record unsupported instead of a successful fallback.

Personal Gaussian Laplace approximation uses the central Hessian with step 1e-4: diagonal second differences; off-diagonal four corners divided by 4*h². A nonpositive Hessian is unsupported, not repaired. Draw 200 parameter vectors MAP+L*xi with L the lower Cholesky of inverse Hessian. For each source bootstrap b, xi is a frozen standard-normal array from `SeedSequence([seed,16,3,b])`; use the same array for every author. This numerical integration rule keeps scores invariant to batch ordering and author renaming; it is not empirical evidence about people. This new construction rule is not a claim to reproduce older F13 draws exactly.

## Target and uncertainty

For each fixed parameter vector, run the filter to terminal mean m and variance P. For k=1..4, mean future observation is a+b*z_k+rho^k*m. Average target mean is a + m*(rho+rho²+rho³+rho⁴)/4, since planned z sums to zero.

Define l=(rho+rho²+rho³+rho⁴)/4. Conditional variance of the average is

`P*l² + Q/16 * sum_(j=1..4) [sum_(l=0..4-j) rho^l]² + R/4`.

Independently construct the full 4x4 predictive covariance: common terminal-state covariance P*rho^(k+h); future innovations contribute Q*sum_(j=1..min(k,h)) rho^(k-j)*rho^(h-j); observation noise adds R on the diagonal. Verify the scalar variance equals ones'V*ones/16. Do not sum only marginal variances.

Across personal draws, average the conditional covariance and add the variance of conditional means (ddof=0). Across the 200 source fits, average within-source totals and add the variance of source means (ddof=0). Preserve separate event/state, personal-parameter and source components. The result is a working bootstrap/Laplace moment distribution, not an exact posterior. The displayed interval is score±1.96*sqrt(total variance), calibration NOT_EVALUATED until an independent evaluation pass.

Empty prefix, if supported, must be explicitly prior-only: draw parameters from each source prior, use stationary state mean0/varianceQ/(1-rho²), and propagate the same planned target. It does not count as observed personal information. Unsupported numerical fits must not emit fake zero scores.

## Bundle and cold API

Implement `build_instrument(brief,source_manifest)` and `cold_score(input,bundle)` in your independent module. Required bundle keys: target_id, object_class=predictive_state, invalidating_result, formula_version, observed_role, unit, reference_mu, reference_sigma, source_hash, reader_hash, world_contract, scorer_code_hash, support_rule, error_model, explanation_rule. Embed the source ensemble and algorithm/RNG settings. Serialize as strict JSON; verify an externally supplied expected SHA256 before CLI loading. Hash the implementation. No pickle.

Source reference mean and sample SD (ddof=1) come from the same T3 scorer on the 300 source prefixes, not from application data. Unsupported source scores must stop reference construction, not be excluded. Freeze the reference before opening application inputs. Cold inputs contain only world/reader identifiers and one observed prefix. Application scoring must not access source/truth files or look up a previous author's score. Return raw score, variance components, source-standardized score and variance, support/prior-only information, approximation and calibration status.

## Acceptance and write scope

Write only `grain_research/gptpro_foundation_v3/f16_independent_t3.py`, `scripts/run_grain_gptpro_f16_independent_t3.py`, `tests/test_grain_gptpro_f16_independent_t3.py`, `reports/grain_measurement_foundation/GPTPRO_V3_F16_INDEPENDENT_T3_BUILD_20260923.md`, and `results/grain_gptpro_foundation_v3/F16/independent_t3_v1/`.

Verify formula/covariance equality, no future leakage, wrong-world rejection, batch/author-ID invariance, source hash rejection, actual warm/cold CLI operation and unsupported-fit retention. Run a tiny synthetic test and permitted source-only construction, checkpointing every10 authors. At most one active numerical CPU worker while the coordinator runs T2. Do not read or execute T1/T2 code. Log exact read paths and added assumptions; evidence from your independent agent is not independent human H06.

No new model training/API, private data, remote execution, commit/push or lockbox. Stop after implementation/tests/source construction, leaving application scoring to the coordinator. If source-only full construction will exceed the current work budget, leave a verifiable resumable checkpoint and exact command, not a completion claim.
