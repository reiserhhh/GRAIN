# NEXT63: does the global read-in state change BODY evidence sensitivity?

2026-09-11. Planned, not executed. One bounded native experiment after NEXT62,
not another regression/alpha/bandwidth sweep. The previous additive reader has
identically zero BODY-by-last mixed derivatives. We must test that dependence
in the native target before deciding to learn conditional local potentials.

## Fixed source-only intervention

Use the same four source authors and two queries (neuroticism, confirmation_seeking)
from NEXT60/62. Never open new8 native targets in this step. No raw text, humanY,
MEPS, new factor discovery, release or lockbox changes. Use the exact H16 state,
BODY positions, RoPE/mask, fixed model weights and original logit contrast L.

For each source i/query q, compute last_ref as the arithmetic mean of the last
H16 state of the OTHER three source authors under the SAME query. Keep every
BODY/other retained token unchanged and set only the last state to last_ref.
One intervention magnitude eta=1, no donor/dose selection. Do not edit query
words, move token positions, or replace all contextual states simultaneously.

Run the frozen suffix at this changed H16, obtain L' and the BODY gradient g'.
Original L/g are reused from audited caches. One unchanged first-source/query
suffix replay checks the cached score/gradient binding; mismatch stops the run.
Planned9 suffix forward/backward calls:8 interventions+1 unchanged replay.
Maximum10 calls allows only one technical retry, never a new condition. Maximum
120s measured inference time; model loading separate with300s wall cap. Assess
remaining cost after the first complete pair; preserve partial outputs if overcap.

## Outputs and deciding the next mechanism

Report all8 source/query pairs in PRIVATE tables; public only query/cohort totals:

    delta_L=L'-L,
    delta_g=g'-g,
    delta_tau_f=sum_t <delta_g_t,D_ft>, f=1..4,
    field_change_ratio=||delta_g||/||g||,
    source_rank8_change=(delta_g U)U^T.

D_f is the original fixed GRAIN direction at the same BODY addresses, not a
new direction recomputed from the intervention. Decompose projected/outside-U
gradient change, retaining signs/covariances; rank8 explanatory capacity is a
coordinate diagnostic, not an optimally useful action subspace.

In the additive NEXT62 model, BODY ghat and all4tauhat are exactly invariant to
the same last-state replacement. Compute this numerical zero witness too. This
does not compare human questionnaire accuracy or train a new readout on4points.

Decisions use the existence and size of the observed response, not a psychology
r>.5 requirement. If native BODY/action changes clearly exceed unchanged replay
discrepancy (report both;10x numerical discrepancy is a computational resolution
check, not a significance test), develop ONE same-scalar token-by-context reader
next. Rank8 retained vs outside-U change determines whether its input also needs
an explicitly source-fitted context coordinate. No escalating this test to new8
just for a significance stamp. If effect is tiny, retain the negative result and
inspect the remaining retained/query positions before adding context parameters.

The intervention answers a local H16 read-in dependency, not a causal statement
about human wording or proof of a unique reasoning circuit. A mean source last
state need not lie on the natural input manifold; report its displacement norm.
No outcome-dependent nearest donor or smaller intervention is allowed.

## Artifacts and completion

New small context-reading module/runner/tests, frozen plan+code first; private
`results/grain_vector_mechanisms/logic_context_reading_interaction_v1/` and public
`CONTEXT_READING_INTERACTION_REPORT_V1.md`. Tests: replacement affects onlylast,
source exclusion, finite mixed-partial witness in a toy conditional potential,
old additive zero and fixedD. Check one native replay, exact saved metric replay,
call/time accounting, source membership and no current-target add-back.

Close after this one paired source intervention. Update theory/atlas/parent and
local commit. The next step must change the complete inference input/function;
do not remain in source capacity diagnostics or use the interaction as a reason
to start a long model run.
