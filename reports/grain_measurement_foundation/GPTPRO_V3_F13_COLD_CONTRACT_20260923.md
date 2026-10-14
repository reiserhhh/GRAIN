# F13 scoped uncertainty interface and actual export

## Implemented

New `f13_contract.py` provides three cold functions:

- `uncertainty_components(names, covariance)` retains the full covariance of
  error components. Off-diagonal contributions enter total variance twice;
  negative cross covariance is retained, invalid joint covariance is rejected.
- `calibrate_intervals(truth, lower, upper, unit_ids, scope)` assesses frozen
  95% intervals with Wilson uncertainty. Duplicate unit IDs are rejected rather
  than counted as independent people. It does not fit an inflation parameter.
- `measurement_result(score, variance, scope, support, calibration)` preserves
  observed/model-borrowed/unobserved distinctions and null values. Calibration
  for a different target/world/method is rejected; numerical scores can be
  retained while explicitly carrying failed or unevaluated calibration.

Existing `change_interval` remains the paired-covariance change/MDC implementation.
These are explicit working-moment contracts, not human validity certification.

## Actual-data exercise

`export_grain_gptpro_f13_contracts.py` reads the completed nested missingness
experiment. It recalculates all 432 calibration cells and cold-exports 86,400
saved scores through the contract. Results match the original coverage, Wilson
limits, widths and decisions. Output `cold_contract_full_r1` under
`results/grain_gptpro_foundation_v3/F13/` contains compressed score JSONL,
calibration records, input hashes and summary.

52,600 score annotations retain PASS; 33,800 retain FAIL. Counts are repeated
score annotations, not independent discoveries or a completion percentage.
Calibration is retrospective evaluation annotation in the exact stated scope;
it is not a license to select models using application results or transfer a
passing label to a new population. Missing-score/null and model-only branches
were tested separately; this full export does not itself exercise empty histories.

Thirteen contract/history/covariance tests and Ruff passed. Negative covariance,
wrong calibration scope, repeated authors, null-vs-zero and retained FAIL have
executable tests. No model fit or neural call was repeated.

## Remaining integration

The F16 legacy runner still writes seven cold-replay checks as literal True and
draws random values in place of instrument scores. These are not usable acceptance
evidence. This export does not silently fix F16: replace those paths with actual
source bundles, observations, score functions and separate-process cold replay.

F13 scientific failures remain visible. The full dynamic/binary missingness
coverage, canonical card packaging and per-requirement reconciliation are still
open; the new API alone does not close them. No automatic parent-card promotion,
human data access, API, training, commit, push or lockbox action.
