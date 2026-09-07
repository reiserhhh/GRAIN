# P1U: target-change versus other-person-change in the cue panel

2026-09-07. Planned cache-only next step; not yet calculated. Decision: does P1T's
larger role gap reflect target-specific evidence or partly the unchanged target
being re-evaluated when another actor changes? No new neural forwards or fitting.

Use ALL256 native P1T rows:128 waiting-to-avoidance matched comparisons (4 contexts
x2 orders x2 roles x2 target people x2 criteria x2 codes). Preserve code-aligned
and raw logit changes. When the target is the negative actor, own behavior changes
and the other actor stays fixed (self_change); when it is the initiating actor,
own behavior stays fixed and the other changes (other_change). These are within
synthetic world comparisons, not person-level psychological effects.

For each fixed setting/person/query/order/code, pair the two role assignments:

    increase_in_role_gap = delta_other_change - delta_self_change.

Check this exact identity independently against all64 gap increases from P1T's
saved role-pair table. Report signed means, RMS and sign counts for each path,
per criterion/order/code and by setting; report the signed cross term if squared
gap increases are decomposed. Other/self RMS ratios are finite response ratios,
not independent information fractions or automatically pathological leakage.
Social context can legitimately influence interpretation; source attribution must
be explicit before calling the effect target-specific. Do not use one small signed
mean as proof of invariance or ignore offsetting individual effects.

Full128 comparisons/64 identities are required, including all waiting and coding
failures. CPU cap60seconds, no new text/labels/model/Claude/API, no choice of
favorable target or code. Save paired rows, per-path summaries, gap identities,
source/output hashes and an independent raw-logit audit. Tests include target
invariance, cross-person spillover and cancellation witnesses.

If other-change contributes materially, the next intervention must distinguish
target-bound encoding from relational scene information. If target-change clearly
dominates across both codes, test the existing target-anchor route before searching
new layers or ranks. Either branch keeps the human-personality validity boundary
and old queues paused. The measured outgoing-description contrast is not a Big5
ground truth, even if every authored pair is ordered as expected.
