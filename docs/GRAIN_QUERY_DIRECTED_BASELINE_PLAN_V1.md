# NEXT32: replace global context similarity with query-directed baseline action

2026-09-10. Planned next finite mainline pass, after NEXT26. No new experiment
has yet been executed under this document. Source references and targets are
inherited, not selected by the latest held-out results.

## Decision and target

NEXT26 achieved endpoint-independent numerical execution, but its base head
failed to recover between-author N mean-null scores. Keep mixed-action successes
and earlier E5/four-layer readers. Do not restart a full-model/accuracy grid.
Develop the missing baseline function by measuring what it actually reads.

Use the four old N authors explicitly marked `role=reference` in
`logic_comment_carrier_v1_r2/private_design.json`; the other20 old N authors are
evaluation, not source. This4/20 design differs from NEXT26's18/6 CV. Compare
against a newly source-matched four-author context Ridge and source mean; do not
call old4-fold metrics a matched control.

Target only z0(c;fixed_PQ,layout), centered ten-candidate logits and original S.
Do not move to another trait because it scores better. No human labels, MEPS or
lockbox. Any shape/layout mismatch must be reported rather than interpolated away.

## Step 1: direction-bearing numerical construction

At the four source authors' all-comment-mean-null states, extract candidate-logit
Jacobians from the existing local observer suffix. Keep source endpoints exactly
matched to the old float32 intervention implementation; check replay first.

Aggregate gradients by the actual differentiation rule:

    J_M[r,j] = sum_{token in comment_j} J_hidden[r,token],
    J_R[r,t] = J_hidden[r,retained_token_t].

The sum, not mean, is the derivative when a comment mean is repeated at every
token. Save projected gradients, not text. Fixed source P/Q and original alphabet
are part of the reader address. No weights are trained in the neural observer.

For each source r, construct the fixed local affine candidate

    zhat_r(M,R) = z0_r + J_Mr:(M-Mr) + J_Rr:(R-Rr).

Use the equal source average as the one candidate; no validation-weight selection.
Report source disagreement as geometry, not automatically calibrated uncertainty.
The role grids align, but target layouts vary. These affine functions initially
live at source layouts; layout transfer is a declared approximation, not a hidden
claim that token-length effects disappeared.

## Step 2: bounded measured tests

- On source states, finite differences on fixed mean/retained directions verify
  the gradient aggregation. Use two fixed small step sizes, include centering.
- On20 old held authors, freeze affine and source-matched kernel outputs before
  endpoint scoring. Report whole centered-logit error, original N score error,
  author-centered variation and retained-versus-mean contributions.
- Compare equal affine reading, source-matched Ridge and source mean. Report
  performance plus squared/cross error terms, not just a correlation winner.
- If source derivatives are correct but transport fails, the useful result is
  local versus portable reading; identify whether source-layout drift or field
  variation warrants the next finite contrast before running it.
- If it helps, substitute this baseline in the exact same conditional assembly
  in a later matched-source experiment. Do not silently mix4-source base and
  18-source other heads into a headline control comparison.

## Resource and completion rules

Maximum4 source Jacobians (40 scalar backward targets),24 suffix forwards and
180 seconds after model load. No remote/API queue. Persist each complete source
packet and stop cleanly at the resource bound; resume only for decision value.
Evaluate old held endpoint caches without new neural calls. Keep all coefficients
and author-level data private; publish aggregates, input-address hashes and tests.

The stage completes when source derivatives, frozen candidate, matched controls
and a next-mechanism decision exist. It does not require improvement to count
as a completed discriminating test. No endless choice of kernels/regularization.

NEXT31 low-rank token reading remains secondary. Request user/Claude input only
if the next mechanism cannot be distinguished with an explicit bounded design.
