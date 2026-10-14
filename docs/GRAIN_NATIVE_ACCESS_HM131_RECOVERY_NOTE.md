# HM131 mechanical mask-check recovery

Frozen before further collection, 2026-10-11. r1 is FAILED and is never resumed.
Its 54 committed captures (48 full-access, 4 no-ops, 2 all-off) are retained.
Attempt55/batch54 completed its forward but failed BEFORE array commit at the
all-off guard. The recorded error is `All-off native judgment retained a history
dependency`; it conflates whole-mask identity with question-mask identity.

Whole support includes body rows, whose history padding varies. They SHOULD
differ: body permissions are deliberately untouched. All-off SUFFIX rows should
be identical. A synthetic counterexample reproduces this false rejection.
The proposed recovery does not change native inputs, masks, model, tolerance,
targets or the scientific intervention. Compare suffix permissions, plus the
same actual readout tolerance. Full per-row mask auditing remains unchanged.

Create r2 with the original 54 committed units copied after per-unit hash
validation, with r1's code/status/input/capture hashes in its source manifest.
The exact post-forward guard failure is distinguished from an uncertain live
attempt: the old process is terminal and no model job remains. Reacquire that
one LOST known-failed unit once in the new packet, then the remaining fixed
design units. This is a declared one-forward execution deviation, not a hidden
rerun or model-budget renewal. Total expected actual forwards389, committed
unique batches388. Original cumulative1500s cap includes r1's199.52906728890957s,
both loads and the failed attempt. Any new failure stops recovery; no third
attempt or cap extension. Old sealed source is not modified.

No new accuracy target, coefficient, source fit, numerical precision grid,
API, private data, training, Claude, Git, automation or lockbox operation.
