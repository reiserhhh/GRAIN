# NEXT46: why MLP24 counteracts the query contrast

2026-09-10. Planned, not executed. One localized mechanistic comparison, selected
after NEXT45. Do not turn this into another all-layer source-coefficient grid.

## Question / fixed mathematical objects

MLP24 gives a negative signed projection on the final incoming contrast in both
cohorts and all four new8 GRAIN factors. Does this arise mainly by applying a
similar local operator to already different inputs, or by query-dependent local
gate/norm coefficients changing the operator itself?

Keep the same12 authors, N/confirmation queries, four frozen GRAIN directions,
and both own/pooled shared w* readers constructed in NEXT44. No human labels,
source fitting, source expansion, blend selection or layer search.

At the final position immediately before MLP24, define exact J_q=B_q N_q with
the same RMS/SwiGLU formulas already tested in NEXT43:

    delta_update = J_N d_N - J_C d_C
    carried = (J_N+J_C)/2 (d_N-d_C)
    operator_contrast = (J_N-J_C) (d_N+d_C)/2
    delta_update = carried + operator_contrast.

Project all three with the same w*. This is a local update decomposition, not
the residual I+J recurrence or a full independent personality predictor.
Keep input/gate product-rule terms where needed for exact parity; don't rename
them as psychological channels. Operator contrasts are based on actual current
forward contexts, not source-only deployable parameters.

## Smallest data collection

1. Reuse H16 inputs and the saved source shared w*. Current d before MLP24 is
   initial plus NEXT42 updates[:, :15], ending at attention24. The native MLP24
   target is updates[:,15]. Verify the actual stage labels before indexing.
2. Collect only current primal x before MLP24:24 truncated prefix forwards
   using `pre_mlp(...,block=24)`. Stop after attention24, without MLP24/later blocks,
   output head or trait scoring. Export actual block24 weights once; never reuse
   block25 weights just because shapes match.
3. Four fixed boundary JVP checks: source4_a00 and new8_a00, both queries,
   factor F01. Compare to the cached prefix sum under original vector tolerance
   relative<=1e-4 OR absolute<=1e-5; no adaptive extra checks/cap widening.
4. At most28 native prefix executions,120s native time; report loading separately.
   Cache/check hashes, assess marginal usefulness, stop with partial evidence if
   hardware fails. No rescoring, backwards, API calls or raw-text publication.

## Numerical experiment / evidence

- Apply each current-query J to both incoming d arrays: a full2x2 local operator
  exchange, holding recipient d fixed. Use the actual general `gated_update.py`
  formulas with block24's exported matrices.
- Diagonal vectors must reproduce the saved native MLP24 increments; retain
  full-vector parity as well as w*-projected scalars. Do not evaluate only a
  direction on which the error happens to cancel.
- Freeze predicted numerical updates before reading the native increment targets.
  Numerical cap60s. The four boundary JVP checks are input verification, not
  access to the MLP24 output target.
- Report all factors/cohorts/modes: carried versus operator-contrast RMS, signed
  projected shares, covariance/cross-energy, and sum parity. Also express each
  component relative to NEXT45 final T to preserve the counteraction question.
- This exact local decomposition uses current forward contexts. No claim of
  source-disjoint prediction accuracy or unique causal mediation from the split.

## Decision

If carried dominates, retain a common local operator interpretation: MLP24
mainly reshapes differences already routed upstream. If operator contrast is
substantial, preserve that concrete context-coefficient dependency. Either way
identify one next simplification/localization step; do not delete the negative
update or assume it is measurement noise.

Query paraphrase/target-meaning controls remain later and separate. Existing
release/scorer/lockbox, MEPS boundary and interpretation rubric are unchanged.

## Deliverables

Proposed `scripts/run_grain_query_counteraction.py`, algebra/actual-boundary tests,
private `logic_query_counteraction_v1/`, aggregate
`QUERY_COUNTERACTION_REPORT_V1.md`, frozen source lineage and replay, updated
logical dependencies and local commit.
