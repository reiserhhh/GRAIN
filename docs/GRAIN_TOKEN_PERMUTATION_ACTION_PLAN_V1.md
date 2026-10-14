# HM13: Token Placement Versus Unordered Configuration

2026-10-08. Planned, not executed. Result-guided after completed HM12;
same controlled materials, not a new independent text corpus.

## Mechanism Decision

HM12 shows mean displacement carries most action but centered displacement
is not inert. It leaves two possibilities: unordered dispersion/relations,
and the placement of token states. Test this distinction before adding a
larger mean-only frame or higher-order regression on the old twelve numbers.

Permutation preserves the entire token multiset, not just its mean and
covariance. If a fixed downstream reader changes, that reader cannot be
reconstructed exactly from an unordered summary on these endpoints.
This is a test of intervention-preserving observation, not a natural
word-order edit or a human trait measurement.

## Fixed Design Before Calls

- Reuse original block output16, three unchanged queries, supplied property
  spans, lexical source/validation/application8/4/4 roles, receiver inputs
  and original question states. No layer, rank, prompt or alpha selection.
- Include every one of the36 multi-token owner pairs in the frozen HM12
  input design. Select by input token count only, never by output action.
- Add one single-token no-op control per role and axis: the smallest
  original pair_id with that role/axis and token count1. Six controls.
- Two arms for each pair/query: receiver_reverse and centered_reverse.
  Total (36+6)*3*2=252 new local forwards, at most300calls/600sec per stage.
- receiver_reverse reverses the rows of the original float32 property
  matrix. Compare with the saved unchanged receiver score.
- centered_reverse first reconstructs the exact original HM12 float32
  centered-only assignment, then reverses its rows. Compare with the
  saved actual centered-only score, not an unrounded reconstruction.
- Do not reverse the raw input tokens, other event spans, query or all
  decoder states. Recompute later blocks under a fixed original receiver.
- Freeze helper/runner/config/tests/plan and all addresses before calls;
  verify all HM10B3/HM11/HM12 consumed artifacts. No replay of old forwards.

## Checks and Outputs

Check actual hook state against the original captures. Row permutations
must preserve byte-exact multisets: verify sorted row byte records or
the exact permutation identity, not approximate nearest-neighbor matching.
Record full-dimensional mean and covariance replay errors separately.
No new fitting or neural training.

Report every axis/query/role/order and input token count, both signed and
absolute effects. Family intervals are conditional lexical-material
intervals, not human-population or model-estimation uncertainty. Single
token controls must be actual forwards with exact unchanged assignments.

Maintain receiver_reverse-minus-baseline and centered_reverse-minus-
centered as separate estimands. Their difference is a conditional
organization sensitivity, not a fraction of personality information.
Preserve all raw candidate logits/compliance; no output filtering.

Deliver intervention units, permutation-invariance checks, all-cell
summaries, independent arithmetic verification, frozen hashes and a
mechanism report. Old HM12 primary/helper/config/plan/tests are frozen.

## Next Branch

- Order-sensitive outputs: retain token positions/typed edges in the
  observation, then build source-only ordered probes instead of averaging
  the kernel away. Compare target-conditioned action, not only level fit.
- Permutation-insensitive but centered-sensitive: develop unordered
  nonlinear token summaries or relational moments; do not assert two
  moments suffice for high-dimensional sets.
- Very small placement and centered actions on this scope: pursue
  source-supported full-dimensional means with an explicit residual
  action term, rather than declaring all residuals psychological noise.

Raw-state interventions do not by themselves identify a natural-language
mechanism or psychological author parameter B_u. Original18/100/44,
natural author evidence, transport and psychological interpretation remain
in the inherited scope. No API/private MEPS/new neural training/Claude/
automation/commit/push/tag/lockbox or old-queue recovery.
