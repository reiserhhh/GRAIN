# NEXT11: locate the criterion link before improving imitation

2026-09-10. Cache-only, exposed-source analysis, fixed before the new accounts.
This extends the seven-pass logic ledger, not a new preregistered confirmation.

## Decision and finite scope

Decide whether a query's useful author association is absent from the fixed
target, lost in lightweight extraction, or merely changed in amplitude. Do not
train a new predictor on questionnaire outcomes or select a winner here.

- Reuse all nine queries, the `pooled` arm of `logic_reading_levels_v1`, and its
  exact source membership: 213 Big5-only authors, 63 MBTI-only authors, three
  views each. The older M29 criterion file includes one additional MBTI author;
  use the current reference-disjoint prediction panel, not an outcome filter.
- Native `F_D1`, fixed `f`, and `c` come from M29; `R_cv`, `c_hat`, and `R_hat`
  come from the already frozen five-fold source OOF predictions. GRAIN's
  fold-specific reference is `G_cv = c - R_cv`, not the full-source reference.
- **Observer conditions matter:** D1 native scores used CUDA/BF16, while fixed
  functions use MPS/FP32 states. Thus the all-nine `F_D1-f` difference combines
  observer conditions and approximation. For T/F and N, reuse the S9 FP32
  native scores to split `F_D1-F_FP32` from `F_FP32-f`. No new native collection.
- Match exact author/query keys, validate three complete views and unique folds,
  check prediction freeze digests, and replay the existing configuration values.
  Keep all individual rows and identifiers in ignored `results/` only.

## Accounts to add, not repeat

Previous reports already allocate `f` into mean/configuration/final and `c` into
GRAIN/residual. Add the missing upstream comparison and a common-root chain:

`F_D1 = (F_D1-f) + (f-c) + G_cv + R_hat + (R_cv-R_hat)`.

For the two observer-matched queries split the first term into two terms above.
Use `sd(F_D1)*sd(Y)` as one denominator within each raw chain. Signed covariance
contributions may cancel; they are not variance shares or causal percentages.

For each fixed edge `F_D1 -> f`, `f -> c`, `c -> c_hat`, `R_cv -> R_hat`, project
the target onto the proxy with an intercept, using no Y to estimate the slope:

`t = a + b*p + e`, `Cov(e,p)=0`.

Then `r(t,Y) = r(t,p)*r(p,Y) + Cov(e,Y)/(sd(t)*sd(Y))`.
Also split the raw error's criterion contribution into amplitude and non-affine
terms using the existing `affine_error_account`. Add both observer-matched
edges for T/F and N. Whole-sample descriptive alignment is not deployable
calibration or a new OOF score. A positive residual association does not by
itself locate a missing attention head, nonlinear route or human mechanism.

## Uncertainty and execution

- Pearson for all queries; unflipped AUC for binary queries (E/N/F/J positive).
  No outcome-based sign reversal, clipping, revised thresholds or model fits.
- 1,999 whole-author paired bootstrap draws per query, shared across channels
  and edges. Recompute descriptive alignments inside draws. Report pointwise
  95% intervals and valid draws; no refitting uncertainty claim. Keep SN's
  small minority count visible. Do not sum per-query contributions across
  different populations or different denominators.
- Tests: exact chain reconstruction; pure affine shrinkage; omitted orthogonal
  criterion component; signed cancellation; perfect target imitation with weak
  criterion association; duplicate/missing/changed cache rejection.
- New ignored output directory; no overwrite of previous analyses. Save input,
  code and plan digests, aggregate CSVs, private joins and a report. Numerical
  CSV replay must match. Bound compute to local cached arrays, no neural calls.

## NEXT12 theoretical integration

After accounting, define separately the author reference population, natural
event clock, slice clock and display clock. A display transformation must retain
event metadata if a natural-order score is intended to be invariant. An actual
event-order change is not the same operation. Combine this with the existing
choice/expression likelihood and C2 findings; do not average away author choice
or motion merely to make scores invariant. No new release scorer or lockbox.
