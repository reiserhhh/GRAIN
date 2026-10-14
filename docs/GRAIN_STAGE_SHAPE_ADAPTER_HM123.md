# HM123 input-layer guard repair, not scientific precision optimization

2026-10-10. The sealed primary failed in its input check, before any current
block recurrence: completed_units=0, pending=[0,2], attempts=1, no batch files.
Its status, inputs and six sealed code/spec/plan files are retained unchanged.

The historical cache has 17 layers, one per completed block 0--16. The weight
list has 18 groups: 17 complete blocks and the final current K/V projection.
That projection has no historical cache. The sealed constructor mistakenly
requires 18 history layers but reads only indices 0--16.

The independent adapter strictly admits 17 actual history layers and appends
one unused zero placeholder solely to satisfy the frozen admission check.
It is not a captured, imputed or scored layer. The loop never reads it.
Tests compare the constant schedule to the independent supported constructor
and change the unused placeholder to huge and nonfinite values: all actual
outputs must remain exactly equal. Missing actual layers are not imputed.

The child directory is stage_mass_action_r1/shape_adapter_r1. Schedules and
pair addresses are byte-identical to the primary; source cues, dose, cut,
49 arms, all 288 public histories and six targets are unchanged. The child
records primary code/input/status hashes, its own code hashes, and the failed
elapsed 0.06596758298110217 seconds. Its remaining numeric allowance is
599.9340324170189 seconds, not a new 600-second grant. No started scientific
batch is replayed, no full model capture or generated answer is called.

This repair yields no mechanism finding by itself. Scientific interpretation
must use the original HM123 stage interaction and selectivity questions.
Preserve HM122's failed target-selectivity result and every HM123 failure.
No private data/API/training/Claude/Git/automation/lockbox or old queue.
