# F13: full B1 replay and history-sampling boundary

## Completed full replay

`binary_mixture_audit_r1` independently recomputes all 259,200 analytic
application predictions under 200 source fits (51,840,000 source-conditioned
predictions). It uses a four-by-four transition matrix and raw second moments,
not the production tensor-filter or centered variance code. Maximum discrepancy
is 4.27e-16. Frozen code/input hashes match. This verifies arithmetic and row
alignment, not observation-model correctness or calibration. The earlier
mixed-reader undercoverage remains unchanged.

## New mechanism made explicit

The frozen B2/M scorers accumulate filtered transition counts online:

\[
C_T^{online}=\sum_{t=2}^{T}q_t
 E[1\{Y_{t-2:t}=(a,b,y)\}\mid O_{0:t}].
\]

Forward-filter/backward-sampling of full paths instead gives

\[
E[C_T^{path}\mid O_{0:T}]=\sum_{t=2}^{T}q_t
 E[1\{Y_{t-2:t}=(a,b,y)\}\mid O_{0:T}].
\]

These are different estimators: later observations revise earlier state
assignments in the latter. Adding a full-history sampler and calling its spread
the original B2/M posterior would silently change the estimator. In addition,
the nonlinear personal refit generally does not commute with expectation.

The new `f13_history_sampling.py` implements exact finite-state FFBS under the
declared fixed-source observation law. It samples full paths with temporal
dependence, and returns their qualification-weighted transition statistics.
It is not a conjugate posterior over personal regression coefficients. Reader
qualifications remain declared weights; no new eligibility calibration is implied.

## Tests and use

- Enumerated all 64 six-event binary histories, including source transition and
  observation likelihoods. 100,000 sampled paths reproduce marginal states and
  weighted transition expectations within .01 absolute tolerance.
- The same finite example shows online and full-history expected counts differ
  by more than .01. The test protects against silently treating them as identical.
- Hard complete histories yield exactly the observed path for all 200 draws.
- Nine related tests pass. Existing B1 full calculations are unchanged.

The next integration should keep frozen online B2/M point estimates as their
own arm and label full-history-refit distributions as a distinct working
multiple-imputation arm. Compare both to the same held-out next-probability
target; report mean shifts and uncertainty separately. Fully observed paths
contain zero missing-history uncertainty, but still have finite-event estimation
uncertainty, which requires an explicitly separate sampling model. Do not declare
zero total personal uncertainty merely because history reconstruction is exact.

All outputs are synthetic/local. No new human access, training, API, commit,
push or lockbox action. F13 and the full goal remain open.
