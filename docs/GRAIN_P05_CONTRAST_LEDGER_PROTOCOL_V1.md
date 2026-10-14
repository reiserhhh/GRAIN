# GRAIN P05 task A / R27 success-failure contrast ledger, protocol V1

Frozen 2026-09-26 JST under SEMINAR_FABLE_03_DECISION.md Packet 2, before any new
computation. Zero neural forwards, no generation, no accuracy/rank/alpha sweep.
Reuses frozen results; new numbers only come from cached arrays.

## Ledger rows (source P05 section 5, P01/P02 references in the requirement ledger)

C1 kept overall structure, lost target action: conditional_operator_v1/r2 compiled
four-coordinate action (E r -.260) versus actual donor action (.464).
C2 relative structure informative, identity unclear: MCD increment (NP+MCD vs NP)
and the failure of condition de-centring.
C3 model or group success is not individual measurement success.
C4 F08 source-reader selection: a reader-fitted source loses B1 accuracy.
Each row records the stable contrast, rival explanations, cached evidence per rival,
the minimal discriminating test, and whether that test was run here or what blocks it.

## Proposed mechanism (Task A candidate) and rivals

- M* objective mismatch: the fitting criterion (hidden-state reconstruction, condition
  de-centring, selection-ignorant source fit) preserves population or reconstruction
  structure and discards the component that carries the target measurement action. The
  information exists in the input, and a different criterion keeps it.
- Rival I information absence: the retained representation never contained the
  target-relevant information.
- Rival T transfer shift: the information exists but a shared map does not transfer
  across conditions (W(c) not W).
- Rival N non-additivity: the action is not additive in the retained and discarded parts.

## Zero-forward tests on C1 (application pairs n=72, E primary, N reported)

- T1 information: leave-one-cell-out ordinary least squares without intercept from the
  four GRAIN coordinate differences to (a) the measured donor action and (b) the
  full-text score contrast. Cells are axis x family x situation, 3 seeds each. Null: 500
  permutations of coordinate rows across pairs (seed 20260915), same LOCO procedure.
  Rule: LOCO r at or below the null 95th percentile supports Rival I at the linear level.
- T2 metric: pooled Euclidean capture 1 - sum||dh-a||^2 / sum||dh||^2 of the compiled
  state change a against the donor change dh, per query segment, versus pooled action
  capture 1 - sum(D-A)^2 / sum D^2 with measured donor action D and compiled action A.
  Rule: Euclidean capture exceeding action capture by at least .10 supports M* at the
  metric level (energy is kept where action is not).
- T3 transfer: the same no-intercept least squares fitted on the 36 source pairs to the
  full-text score contrast, evaluated on application pairs, versus T1(b). Rule: T1(b) r
  exceeding source-fit application r by at least .20 supports Rival T.
- Rival N: cited from the frozen remainder analysis (interaction RMS).
- The proposed J^T J estimator itself needs source-material derivatives (backward
  passes). None are authorized in this packet. It is recorded as the precise missing
  dependency, not replaced by a surrogate.

## Other rows

C2-C4 use frozen numbers with their paths. A new fit is not run. Blocked minimal tests
name the dependency (for example exposure denominators, which no corpus has).

## Amendment A1 (after T1-T3, before any 24-coordinate computation)

- Because T1 supports Rival I for the four coordinates, one richer text-computable
  representation is tested once: T1b repeats T1 (same cells, LOCO, no intercept, 500
  permutations, seed 20260915) with the 24-coordinate deterministic GRAIN lexical runtime
  (`common_source_comment_representation.joblib`) applied to the full synthetic texts.
  The estimator is minimum-norm least squares (numpy lstsq). No alpha, rank or feature
  selection. Zero neural calls. Same rule. This is one declared alternative, not a sweep.
- The four-coordinate result stays the primary C1 outcome.
