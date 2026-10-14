# F13 cold history/support uncertainty contract

## Implemented, not a coverage claim

`grain_research/gptpro_foundation_v3/f13_history_contract.py` now exposes
`score_history(history, model, target)` backed by the existing versioned evidence
ledger. Five substantive tests replace old placeholder support flags. Together
with existing covariance tests, ten tests pass; Ruff passes.

- A literal duplicate record or an undeclared second new record of the same
  event is rejected. Explicit aliases add provenance but no information.
- Revision replaces its parent; summaries of replaced evidence become stale.
- Retraction removes active evidence. If nothing remains, a specified prior
  returns a model-borrowed score and prior variance, not an observed score.
- Missing model returns null, not zero or an invented finite bound.
- Target estimability is checked against the observed design row space. A
  one-condition record can identify that condition's linear target, not a free
  condition contrast. Borrowing from the prior is explicitly labeled.
- One author per ledger is enforced; negative/nonfinite working variances fail.

## Actual mathematical bug found

The legacy `evidence_history.score_arrays` support check projects raw y onto the
range of covariance K. With a nonzero mean, Gaussian support instead requires

\[
(I-KK^+)(y-X\mu)=0.
\]

Reproduced counterexample: mu=(2,1), X=(1,1), all variances zero, y=3.
The old function rejects it as a duplicate-direction conflict even though it is
exactly the deterministic model expectation. The new `conditional_moments`
accepts it with mean=(2,1), covariance=0; y=3.1 is correctly rejected.
Frozen historical implementation/artifacts were not silently overwritten. The
new F13 interface uses the corrected conditioning routine. Migration of older
callers remains explicit work; this report does not assert every historical
result was affected. Positive-definite observation covariance hides this bug.

## Uncertainty components

For target c, observation weights w=c'VX'(XVX'+R)^+, the working estimation-error
variance is

\[
(c-X'w')'V(c-X'w') + wRw'.
\]

The adapter reports remaining prior level/response variance and block/event/report
contributions. Their sum is checked against conditional target covariance. This
is not raw-score variance or a universal independent partition: it is restricted
to the existing five-component working model. Cross-component covariance models
and source-bootstrap integration must remain separately declared.

## Remaining original tasks

These tests close concrete history/support interface checks, not the full F13
RUN card. MCAR/MAR/MNAR, AR=.6, source-prior shift, learned-source uncertainty,
paired changes, component-covariance output and canonical card manifests still
require their prescribed integration/experiments. The legacy `run_grain_gptpro_f13`
placeholder flags and clamping helpers are not valid replacement evidence.

No new human data, model training, API, lockbox action, commit or push. Goal active.
