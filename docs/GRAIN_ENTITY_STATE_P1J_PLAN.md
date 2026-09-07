# Next P1J: common-prefix entity state and criterion-conditioned reading

2026-09-07. Planned, not executed. Motivation: audited P1I found substantial
person-name-state effects in trait attribution, not stable criterion-word superiority.

## Remaining distinction

The name state precedes the criterion, but previous task prefixes differ. A strong
name patch might be prepared specifically by that prefix. Force identical text up
to the queried entity, then vary only the later criterion. Causal attention predicts
exactly identical entity states across criteria; verify this as instrumentation,
not as a new psychological law. The experiment is what different later queries do
with the same evidence-conditioned state.

## Frozen-material proposal

Reuse four authored two-person records (P1F/H/I), but give BOTH named people their
own questions. Four settings x two role assignments x two queried entities x three
criteria =48 prompts. Use a literal shared prefix such as `Question: For Alex, `,
followed by event, outgoing-attribution or unchanged-room criterion. Use identical
A=Yes/B=No answer convention for all three; this changes the previous event coding
and must be reported, not compared as a pure performance improvement.

Expected factual/trait contrast is positive for the author of the initiating action
and reversed for the other person. The trait label is still only authored behavior
direction, not measured personality. Venue is an unchanged-fact control. Keep native
answer preference, paired margin changes and both people, even if a query fails.

At block16 compare same-entity opposite-evidence patch at N, full Q and full EQD.
Do not patch a different lexical name into N: that would mix changing entity identity
with changing its behavioral evidence. Store the N-state equality across criteria,
source donor differences, all scalar outputs, and parent/material identities.

Planned48 baselines +144 patches +4 controls =196 forwards, cap300/600s/9GiB.
Finalize exact clauses and token accounting in a new collector before any neural
results. Verify that templates really have identical input-token prefix through N;
no marker tricks or selecting a successful name/layer. No fit/no private data.

## Collector freeze details

All prompts use256 execution tokens (right padding and attention mask), MPS FP32,
to avoid prefix comparisons with different GEMM shapes. Actual unpadded positions
remain authoritative. State tolerance1e-5 is numerical, not psychological. N donors
for every criterion come from the event-query cache of the same entity/opposite
evidence after verifying shared N states. Q/EQD donors match the criterion. Report
intended behavior direction separately from restoration of a possibly wrong donor.

## Decision

If a common entity state affects factual and tendency queries in the correct
person-specific direction, carry forward a reusable evidence-indexing candidate;
not a complete mediator or personality vector. Weak/wrong native binding limits
the interpretation. Nonzero unchanged-venue effects limit specificity. Full Q and
EQD reveal information not captured by the single entity state. After full audit,
choose between richer evidence composition controls and a compact vector operator;
do not open large author queues merely because this synthetic packet is positive.
