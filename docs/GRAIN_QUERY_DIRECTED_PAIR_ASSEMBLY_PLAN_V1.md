# NEXT33: put query-directed baseline reading into one source-matched pair function

2026-09-10. Planned next finite pass after NEXT32; not yet executed. Goal: carry
the useful baseline mechanism into a complete conditional response without
mixing incompatible source protocols or borrowing evaluation endpoints.

## Frozen population, inputs and target

- Same4 old N `role=reference` authors, same20 `role=evaluation` authors as NEXT32.
- Both old pairs(0,1) and(0,11), both Fourier bag/position families; no new arm
  selection, query selection, human labels, lockbox, MEPS or neural calls.
- Candidate-logit target is the same conditional pair as NEXT26, not full F.
- M/R/layout from current source states; mean-null affine baseline from the four
  frozen NEXT32 Jacobians. Upstream block16 encoding remains explicit.
- Existing32-dimensional projections and eta are data-independent random draws
  (`run_grain_kernel_action.py`, seed20260910+query_index), not PCA fitted on held
  authors. Verify their frozen provenance. Recompute center/scale using only
  the four source authors' original three views, equal author/view weighting.
  Do NOT reuse18-source fold-specific geometry or coefficients as4-source assets.

## One structured source-only construction

1. Rebuild context kernel on four source authors. Reuse .01 regularization.
2. Shared singleton increment head per Fourier family:12 source rows
   (4 authors x comments0,1,11), kernel(1+k_context)*<phi,phi'>, zero intercept.
3. Symmetric mixed head per family:8 source rows(4 x two pairs), same old tensor
   kernel and .01 regularization. Zero/swap properties remain exact.
4. Compose centered logits and original S:

       zhat_pair = zhat0_affine + dhat_i + dhat_j + mhat_ij.

Five tiny fits maximum: one source-matched base kernel, two singleton and two
mixed heads. The affine source gradients remain untouched. No hyperparameter
sweep, source-weight selection or post-hoc held calibration.

## Fixed comparison and error accounts

For each pair/family compare:

- Query-affine base + predicted increments/mixed: primary complete candidate.
- Context-kernel base + the SAME predicted increments/mixed: matched control.
- Query-affine base alone: diagnostic of whether configuration adds information.
- True current base + predicted increments/mixed: labeled oracle diagnostic.
- Query-affine base + true increments/mixed: labeled oracle diagnostic.

Report per-component and whole-score RMSE/r, author-centered variation versus
common offset, and signed squared/cross error terms. Bootstrap whole authors,
not40 independent pairs. A good whole score must not silently certify that each
increment is learned. Keep all arms, including negative local actions.

## Practical delivery and next decision

Freeze serializable inputs/coefficients/predictions before endpoint analysis.
Run source-file-denied inference on an independent process: only declared
numerical inputs and fixed source assets may be opened. Return original-alphabet
scores as well as logits. Source identity is metadata, never a numerical feature.
Publish aggregate tables, exact replay and tests; retain private arrays locally.

If whole conditional response improves, next develop its input extraction into
the existing early encoder and connect fixed scores back to old external
criterion packets under a separate no-selection plan. If baseline succeeds but
configuration does not, preserve the baseline and investigate the missing
within-comment positional reading rather than tune a whole-score leaderboard.

## Secondary, not automatic detours

NEXT32's reader-position covariance C_Jc=.052493 RMS is a real source-origin
term. Dropping it or recentering the output defines a different estimator. Do
not adjust it here. A later source-only origin experiment is justified only if
composition identifies calibration as the actionable bottleneck.

NEXT31 low-rank token reading is relevant only when it supplies a missing
configuration input/reading. Do not restart old energy or scalar-link grids.
