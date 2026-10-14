# F16 T2 actual construction and T3 independent build dispatch

## Current scope

T2 is implemented as an actual source-frozen current-state scorer, not random surrogate scores or constant test verdicts. The finite core-world construction and separate numerical audit are complete on three seeds, each using the existing 200 source bootstrap fits, 300 source reference authors and 200 application authors. Source parameters were not refitted. All 600 application score objects agree exactly between warm and cold processes. Interval calibration is mixed: two seed cells meet the original finite Wilson check and one fails. Implementation completion is not full F16 acceptance.

Run root: `results/grain_gptpro_foundation_v3/F16/t2_cold_full_r1`.
Smoke: `t2_cold_smoke_r1`; alternate smoke audit: `t2_cold_smoke_audit_r1`.
Full post-score audit: `results/grain_gptpro_foundation_v3/F16/t2_cold_audit_r1/summary.json`.

The new brief was frozen before this construction opened application records. These are historical simulation datasets, not a prospective unseen-human experiment. Only core is included in this F16 construction packet; earlier near-root/shared-shock findings remain, not silently promoted to calibrated instruments.

## Actual algorithm

For each source bootstrap b, the observed author prefix updates (a,b,eta) with the existing two-start MAP estimator and central-Hessian Gaussian Laplace approximation. The state recursion uses rho=.99*tanh(eta), stationary initial variance Q/(1-rho²), and actual elapsed time gaps. Source observation/model identities are checked before bundle construction.

For 200 parameter draws per source fit, terminal filtering produces state mean m_bj and conditional variance V_bj. The returned variance is

`mean_b mean_j V_bj + mean_b Var_j(m_bj) + Var_b(mean_j m_bj)`.

All variances over finite draws use ddof0. The components are event/state, personal parameter, and source variation. They are not arbitrary independent psychological factors: this is a nested conditional variance identity under the declared working model.

For numerical integration, the standard-normal array is fixed per seed/target/source replicate and shared across authors. Its key is `[seed,16,2,bootstrap]`. It contains no author ID or current batch index; renaming/reordering authors does not change a score. It differs from the historical F13 integration recipe and therefore does not automatically inherit its interval calibration. Shared integration draws are not independent population replications.

Source reference mean and sample SD are computed by this same scorer on source prefixes of length48 and frozen before application scoring. They are empirical source-score references, not claimed out-of-sample population constants. The application receives only a JSON bundle and its own observed prefix.

Empty prefixes are explicitly prior-only/model-borrowed, not observed zeros. A failed MAP or nonpositive Hessian returns unsupported, without silently dropping that source replicate. Covariance assumptions and Laplace approximation remain visible. State outputs are model-inferred and are not named emotions/personality traits.

## Implementation and evidence

- `grain_research/gptpro_foundation_v3/f16_state_instrument.py`: source template, score, variance components, norms, strict input/world/code/reader contracts and trusted-bundle CLI.
- `grain_research/gptpro_foundation_v3/f16_factory.py`: public `build_instrument(brief,source_manifest)` and `cold_score(raw,bundle)` for implemented T1/T2; T3 not silently redirected.
- `scripts/run_grain_gptpro_f16_t2.py`: resumable source-score checkpoints, application/warm outputs and actual separate-interpreter cold replay. At most three CPU workers, separate seed cells.
- `scripts/audit_grain_gptpro_f16_t2.py`: all stored mixture arithmetic/reference scores; predetermined dense covariance checks on two authors and three source replicates per seed; truth is opened only in the separate post-score evaluator. This does not claim a new full MAP/Hessian refit audit.
- 40 related tests passed. Ruff passed for the new implementation, runner, facade, audit and test file.
- Smoke has three source reference authors/two application authors, three source fits and eight personal draws. Warm/cold scores match exactly; 48 dense conditional checks have maximum discrepancy 2.26e-16. Smoke interval statistics at n=2 are not scientific acceptance evidence.

## Full-run numerical and calibration results

| Seed | Source / application authors | Warm/cold error | State MSE | 95% working interval coverage | Mean width | Wilson check |
|---|---|---|---|---|---|---|
| 20260915 | 300 / 200 | 0 | 0.176146 | 0.925 | 1.449132 | PASS |
| 20260916 | 300 / 200 | 0 | 0.191886 | 0.900 | 1.478628 | FAIL |
| 20260917 | 300 / 200 | 0 | 0.159792 | 0.925 | 1.470340 | PASS |

The check asks whether the Wilson 95% interval for observed coverage contains .95, not whether the point coverage equals .95. For the failed seed, the interval is [.850594,.934330]. No threshold, interval width, source ensemble or random integration rule was changed to rescue this outcome. Nominal coverage and this finite acceptance check should both remain visible.

All saved application mixture moments and all 900 source reference scores were reconstructed. Maximum moment discrepancy is 8.72e-16; reference reconstruction discrepancy is zero. The prespecified two application authors and three source replicates per seed contribute 3,600 parameter-draw checks against full observation-covariance conditioning; maximum discrepancy is 2.69e-16. This checks the conditional filtering and stored mixture arithmetic, not every personal MAP optimum/Hessian with a second optimizer.

The original score bundles retain calibration `NOT_EVALUATED`; the evaluator attaches the result only to these finite core-world target/seed cells. It does not change an arbitrary future score into a validated measure. Three cell runtimes were 719.870, 726.590 and 710.866 seconds (concurrent seed workers). The original run process is terminal, not a live background job.

Production implication: the same source-frozen function can construct references and score new inputs independently of their current batch, while retaining state, personal-parameter and source uncertainty. The remaining undercoverage is a property to explain within this declared model and Laplace integration, not a reason to duplicate the completed precision grid.

## Independent T3 construction

A separate agent, Hume (`01a0cdec-a67e-7e73-9e6e-742b8feecc7c`), was dispatched with no inherited conversation context and only the brief, mathematical handbook and source-data allowlist. It was explicitly forbidden to read T1/T2 code, prior application outputs or target truth. It writes disjoint T3 implementation/tests/results and logs read paths and added rules. Source-only construction is requested; application evaluation is reserved for the coordinator.

Handbook: `docs/GRAIN_F16_INDEPENDENT_DYNAMIC_HANDBOOK_V1.md`.
Brief: `configs/grain_f16_dynamic_briefs_v1.json`.

Dispatch is not completion. This can establish independent-agent construction only, not independent-human H06. Agent status/read log/results must be checked on return.

## Open acceptance items

1. Eight concrete evidence cards, F12 transport and the same-target F08 text qualification interface. Numeric input validation is not semantic subject attribution.
2. T1 four-cell source uncertainty, T3 independent implementation and application evaluation.
3. Target-specific uncertainty and original protocol reconciliation; the failed T2 coverage cell remains a failure, not covered by numeric replay PASS.
4. Original card/DAG/44-question reconciliation, F01/F14/F15/F17 gaps and six human-stage computational preparations.

No parent task is promoted. No API, neural training, private raw data, remote execution, lockbox, commit or push. The full original goal remains active.
