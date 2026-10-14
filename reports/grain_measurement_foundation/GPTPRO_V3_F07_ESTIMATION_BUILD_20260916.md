# F07 categorical parameter recovery

## Purpose and fixed design

The preceding information-law packet computes how opportunity, reader error and cost affect information. This companion tests actual recovery of the individual parameters a and b using the exact categorical likelihood, rather than treating information curves as estimation results.

The original V3 F07 protocol supplies the design: four worlds U/D/P/flat, three channels (base, off-target, MNAR), five levels, and two budgets (400 tokens or 32 opportunities). An additional complete five-category Gaussian ordinal reference uses 32 opportunities. No ordinal token cost was specified, so none is invented. There are 121 designs per author.

Three seeds each generate 300 source, 100 validation and 200 application authors: 1,800 total and 217,800 MAP fits. Source/validation/application RNGs are separate. The declared source prior is fixed N(0,1) x N(.5,.25), with the second arguments interpreted as variances; this packet does not learn or select that prior from application labels. Each author has common underlying uniform opportunity draws across the comparison arms. No row is replaced by an identical copied outcome: each channel separately transforms those draws through its probability law.

The estimator receives observed counts at z=-1,+1 and known channel parameters, not the author's hidden parameter. MNAR missing outcomes enter the likelihood as an explicit third category. The ordinal reference is a separate five-category law. Estimates use BFGS, gtol 1e-8 and maxiter 200. Covariances use expected information at MAP plus prior precision; they are working covariances, not exact posterior variances.

## Tests and smoke evidence

Seven new tests cover four likelihood derivatives at two finite-difference scales, entirely absent channels, informative MNAR absence and malformed counts. Together with channel and Gaussian-design tests, 20 tests pass; Ruff passes.

The smoke run contains 15 authors and 1,815 fits. Independent channel algebra reconstructs every outcome count and objective, maximum objective difference 7.11e-15. There are 123 optimizer non-success flags; their gradients range from 1.00e-8 to 2.66e-7. A predeclared first-three-author Powell reference performs 363 refits, maximum absolute objective gap 2.84e-14. This identifies numerical stopping limitations in that sample, without erasing flags or changing the protocol.

## Artifacts and acceptance boundary

- Runner: `scripts/run_grain_gptpro_f07_estimation.py`.
- Estimator: `grain_research/gptpro_foundation_v3/f07_estimation.py`.
- Audit: `scripts/audit_grain_gptpro_f07_estimation.py`.
- Results: `results/grain_gptpro_foundation_v3/F07/estimation_full_r1`.
- Smoke verification: `results/grain_gptpro_foundation_v3/F07/estimation_smoke_audit_r1`.

The full run is resumable per author, with source-code hashes and a frozen manifest. Do not resume a live writer or change its code. Completion is determined by its checkpoint, not this build note. Full outcome/objective replay, aggregation and paired uncertainty must follow. F08's actual text-reader confusion, task-contract integration and remaining global families are separate unfinished requirements. No full F07 scientific acceptance is asserted by this build report.

No API, neural training, private raw data, MEPS fitting, remote operation, lockbox, commit or push.
