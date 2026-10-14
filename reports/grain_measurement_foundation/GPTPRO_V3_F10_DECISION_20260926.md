# F10 decision packet: action path vs zero, cross-fitted shrinkage, derivative triage

Date: 2026-09-26 JST. Worker: Claude Opus 5.5. Builds on
`GPTPRO_V3_F10_ACTION_RECOVERY_20260916.md` (reused, not rerun). Artifacts:
`results/grain_gptpro_foundation_v3/F10/decision_r1/` (no model calls) and
`F10/derivative_triage_r1/` (failed; see below). Label revised after Fable
seminar 02 (ZERO_DOMINANCE -> ZERO_REFERENCE_RETAINED; numbers unchanged).

## Decision 1 (F10-05): zero reference retained

Paired squared-error contrasts on the PV01 application effects (72 persona
pairs x 2 queries; 2000 pair and 2000 task-then-pair bootstrap draws; the
task-level interval is primary with 3 source / 6 application tasks):

- Neither the Euclidean nor the J-weighted (action) PV01 geometry beats the zero
  reference on the donor-effect or full-contrast target at either level.
- Euclidean vs zero on the donor target is HIGHER_LOSS at both levels
  (E +3.25e-4, N +1.5e-5). Weighted vs zero is pair-level HIGHER_LOSS but
  task-level INCONCLUSIVE; all full-contrast rows are task-level INCONCLUSIVE.
- The action metric beats Euclidean on the donor target at both levels
  (E -2.97e-4 [task -5.50e-4, -7.8e-5]; N -1.4e-5).

Reading: action weighting is the better *relative* geometry but provides no
deployable gain over predicting zero. The old PV01 "LIMITED_LOCAL_GAIN" label is
superseded for deployment decisions. This is a finite persona-pair result, not a
verdict on all GRAIN vector methods.

## Decision 2 (F10-01): cross-fit shrinkage, never in-sample

The frozen grid c in {0,.25,.5,.75,1} chosen by in-sample source loss picks c=1
for both queries (the fit is scored on its own rows). Leave-one-source-task-out
cross-fitting picks c=.25 (E) and c=0 (N). Cross-fit improves on in-sample for E
(-.004339 [task -.009464, -.000782], SUPPORTED) and equals zero for N; versus zero
it is not better (E +.000669, task INCONCLUSIVE). Production rule: select
shrinkage by cross-fitting over source clusters; keep zero as the reference when
no cross-fitted improvement exists.

## F10-04 derivative triage: failed run retained, decision requested

A pre-declared triage (six original cases; FP64/FP32 CPU analytic+FD at
h=1e-2/1e-3/1e-4, MPS FP32 replay; frozen classification rules; <=100 forwards)
executed both CPU arms (~79 local forwards) but crashed on the first MPS forward
(float64 cast of an MPS tensor) before writing any numbers. Static review found
no text/query mismatch between the stored gradients and FD evaluations (both at
the pair's low text). The six cases therefore remain UNKNOWN. A corrected r2
(per-arm saves) is prepared but not run because it would exceed the declared
forward budget; a bounded supervisor decision is requested.

## Still open (not required for the two decisions above)

- c x PV01-native shrinkage (source native interventions/weights were never saved).
- Discrete four-edit signatures on the native reader with an audience edit (the
  96-text bank has no audience edit).
- Rotation/unit mapping test for the geometry module.

## F10-04 derivative triage r2 (supervisor-authorized repair, 2026-09-26 02:34)

Output: `results/grain_gptpro_foundation_v3/F10/derivative_triage_r2`.

### Usage

| run | forwards | backward | notes |
|---|---|---|---|
| r2 | 90 / 90 cap | 6 / 6 | 0 failed calls; 310 s; cap enforced before each call |
| r1 | ~79 (estimated) | — | failed; numbers lost |
| cumulative | ~169 | — | not <= 100 |

The frozen r1 manifest (SHA verified), cases, steps, rules and model were
reused.

### Result

- **Frozen-rule outcome:** FORMULA_ERROR in 6/6 cases, retained as the
  protocol outcome.
- **Post-hoc diagnostic** (saved arms only; no relabelling): the installed
  Qwen2 code casts to float32 inside RMSNorm, rotary and softmax. The nominal
  FP64 arm is therefore not end-to-end FP64, and its finite-difference error
  grows as h shrinks (median ratio 71 from h=1e-2 to 1e-4).
- **Analytic gradients:** they agree across precisions (<= 3.3e-5 relative).
- **Nominal-FP64 finite difference at the original step:** it meets the
  original tolerance in 6/6 cases.
- **MPS FP32:** it fails in 6/6 and reproduces the original values within
  .63%.

### Conclusion

The data do not support a formula error. They are consistent with the
original failures being FP32 finite-difference precision failures on MPS.
That reading is not certified, because the frozen rules assumed an end-to-end
FP64 arm. Science status is INCONCLUSIVE; no third rerun.

The F10 decision, zero reference retained, is unchanged. The F10-05 review
card is built (`F10/review_r1/F10-05`, 22 checks).
