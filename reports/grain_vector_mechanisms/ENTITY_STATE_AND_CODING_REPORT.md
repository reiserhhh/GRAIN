# Entity-state reuse and criterion/coding specificity

2026-09-07. Working mechanism report; not human personality accuracy.

## P1J result: reuse observed, selectivity fails

Frozen in dcc2eec, P1J completed48 native cases and144 interventions,196 forwards
in87.28seconds. Four authored records, two role assignments, both queried people,
and three later questions. All use the identical prefix through the queried name
and A=Yes/B=No. Model: cached Qwen2.5-1.5B-Instruct, MPS FP32, block16, fixed256
execution length. No training, API, new private text or human labels.

The16 common-prefix groups have exactly identical name-token states across later
criteria (max difference0). This follows causal attention and is an instrument
check, not a novel theory. Interventions deliberately use the same opposite-role
event-cache name state for all later criteria, without replacing the person's name.

| Criterion | Native correct preference | Positive role pairs | Native role gap | N restoration error | Q restoration error |
|---|---:|---:|---:|---:|---:|
| Initiating behavior |14/16|8/8|2.3593|.462636|.015925|
| Outgoing attribution |12/16|7/8|2.1296|.606232|.019643|
| Room open (unchanged) |14/16|8/8|3.0047|.347142|.034785|

Restoration error is squared error relative to leaving the recipient unchanged:
0=donor output restored;1=no change. It is NOT correlation, accuracy, explained
personality variance, or a uniquely identified mediation fraction. FullEQD error0.
N effects follow authored behavior direction in16/16 event and16/16 trait cases;
trait donor-restoration direction is only14/16 because some native donor ordering
is wrong. Good semantic direction and faithful donor restoration are distinct.

The negative control is load-bearing: room status never changes, yet its role
contrast is larger than the event contrast and its N shift RMS1.2867 is comparable
to event1.2731/trait1.2554. The model seems to let person evidence contaminate an
irrelevant question. `For NAME, was the room open?` may itself encourage carryover;
this is a prompt-family limitation, not evidence that all LLM fact reasoning fails.
Criterion-specific entity evidence cannot be accepted from this packet alone.

The48 rows are not independent people; two queried entities share each record and
role-swapped prompts reuse the same actions. Four authored settings support finite
mechanism examples only. No claim of population significance is made.

## Independent replay

`scripts/audit_grain_entity_state.py` independently checks artifact/source hashes,
parent material and role signs, actual tokenizer offsets, N-state prefix equality,
common event-cache donor provenance, native pairs and all scalar summaries.
Max raw arithmetic7.11e-15, metric reconstruction4.44e-16, full restoration0,
common-state difference0. No new neural forwards in the audit.

Artifacts: `results/grain_vector_mechanisms/entity_state_r1/`; audit receipt
`entity_state_audit_r1/summary.json`. Run identity:
`17a2b61adbce0771797ef60972da1a4bd40630b431230d33e13fdc4ec3b23ab5`.
The collector source snapshot predates audit-test additions and remains immutable.

## Mathematical next distinction

A shared state n=F(prefix,evidence,entity) can precede the criterion while later
computation s=R(question,answer_code,n,other_states) remains highly conditional.
P1J observes an intervention on n, not a sufficient statistic or universal readout.
P1K therefore crosses criterion polarity with answer coding, preserving the same n.
Its exact four-cell role-contrast expansion separates constant, polarity, code and
polarity-times-code components. Only the last transforms as a criterion-and-code
equivariant role response, and even that does not guarantee psychological validity.

P1K is planned/frozen separately in `docs/GRAIN_CRITERION_CODING_P1K_PLAN.md`;
do not treat its expected contrasts as results until completion and independent audit.

## Relation to prior methods

Entity-attribute binding vectors have prior causal-intervention evidence in other
models/tasks ([Feng & Steinhardt,2024](https://arxiv.org/abs/2310.17191)). Our
contextualized name-state effect does not identify that particular binding-ID
subspace, nor a personality axis. This is methodological alignment, not a novelty
claim. Patching metrics and corruptions can change localization conclusions
([Zhang & Nanda,2024](https://arxiv.org/abs/2309.16042)); retaining the unchanged
venue, incorrect native answers and raw logits prevents an easy false-positive
story. A successful natural-donor patch can reproduce a model's own mistake.
