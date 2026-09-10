# NEXT47: source-fixed common MLP24 counteraction reader

2026-09-10. Planned after NEXT46; not executed. One finite dependency-reduction
test, not a new personality accuracy benchmark or all-layer search.

## Question

NEXT46 carried term dominates local query contrast but uses each current author's
two-query average Jacobian. Can a source-fixed query-common reader retain the
negative local contribution on new authors without their current x/gate state?

## Fixed construction

- Reuse exact12 authors/two queries/four GRAIN directions and own/pooled w*.
  Source4 evaluations exclude that author from the source bank; new8 use all4.
- Use only actual block24 matrices and preMLP24 source x from NEXT46. Compile
  v*_u,m = mean_(s in S_u, q in {N,C}) J24(x_s,q)^T w*_u,m.
  Reuse `compiled_gated_readout.source_readout`, averaging paired full B_s N_s
  adjoints, not product-of-means. No fits/labels/weights chosen by target results.
- Current prediction reads only v*, d_N, d_C:
  predicted_update_q = v*^T d_q; predicted_contrast = v*^T(d_N-d_C).
  Construction must not open new8 x, target MLP24 increments, current gradients,
  or per-author current compiled Jacobians. Prepare separate source bank and
  current pure d/w inputs; file-guard numerical stage before target evaluation.
- Freeze exact12*2queries*2modes*4factors=192 actions and96 contrast values overall.
  This packet's new8 authors are source-disjoint but repeatedly observed, not blind.

## Analysis and acceptance

Report both cohorts/modes and every factor, not only a favorable aggregate:

1. Compilation identity vs directly applying the same source-average J: <=1e-10
   scalar absolute error. This checks implementation, not scientific transfer.
2. Source-fixed predictions vs actual native MLP24 actions and query contrasts:
   descriptive r, RMS, RMSE, SSE/zero, sign agreement. Keep current pair-average
   carried term from NEXT46 as explanatory comparator, not deployable baseline.
3. Signed projection onto unchanged NEXT45 final T; retain covariance and all
   native source records. Do not delete negative update values.
4. Separate the error exactly into source-common minus current-carried, and
   missing operator-contrast term. This locates whether current context matters
   in source transfer or only in the smaller query-specific residual.

If the frozen source reader retains negative aggregate final-T projection and
contrast SSE/zero<1 in both new8 modes, retain it as a useful local candidate;
otherwise retain current conditional operator dependency. Report magnitude,
not binary gate alone. No post-result alpha/source/axis selection or r polishing.

## Boundaries / outputs

0 neural calls/scoring/backwards/APIs/human-label reads/fits. Numeric cap60s,
one exact array replay. No source expansion or new layer. Existing private x/d
and weights make this immediately executable; do not launch a native capture.

Proposed script `scripts/run_grain_source_common_counteraction.py`, private
`logic_source_common_counteraction_v1/`, aggregate report and tables, source
membership/target-denial audit, tests and local commit. Preserve scorer, release,
lockbox and MEPS boundaries. Decide one next mainline step after this packet.
