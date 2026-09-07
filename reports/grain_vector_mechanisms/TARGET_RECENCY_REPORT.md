# P1M: query target and action-record order

2026-09-07. Complete and independently audited. Frozen in8a89fd0.112 native cases,
224 patches,340 forwards in129.41seconds; Qwen2.5-1.5B-Instruct, block16, MPS FP32,
padded256. No training/new real data. All original48 prompts replay (CSV error0),
common-prefix states0, full donor restoration0.

## Results that change the interpretation

| Question target / order | Native role-gap RMS | Anchor-patch role-gap RMS | Native expected preference |
|---|---:|---:|---:|
| Person-prefixed venue / original |3.0445|1.2722|14/16|
| Explicit room / original |.7080|.0237|8/8|
| Person-prefixed venue / reversed |3.1155|1.3159|14/16|
| Explicit room / reversed |.6978|.0387|8/8|

Changing the question target sharply reduces unrelated behavior-role effects on
the room-anchor route. All8 setting/order comparisons have a smaller object effect.
The old person-room query is pragmatically ambiguous; do not generalize its
code-following leakage to every query or every LLM mechanism. The change involves
wording and a different contextual anchor, not an isolated lexical-token intervention.

Residual native venue effects are not zero: RMS about.70 despite mean signed gap
about.007. Mean cancellation must not be advertised as invariance. Correct binary
preference8/8 also does not mean the margin ignores irrelevant material. This panel
has only open rooms; it cannot yet show that the room anchor uses relevant truth
rather than ignoring both kinds of information.

Record reversal preserves all8/8 behavioral native role directions; native expected
preference changes14/16 to15/16. Attribution directions7/8 to8/8, expected preference
12/16 in both. Event mean role contrasts2.3593/2.4002, attribution2.1296/2.1942.
Anchor role directions remain8/8 in both tasks/orders. This weakens a pure last-
record explanation, not all position/salience explanations. Magnitudes vary across
individual settings and should not be collapsed into an order-invariance claim.

## Audit and artifacts

Independent `audit_grain_target_recency.py` checks immutable source/output hashes,
actual action text ownership, exact original/reversed material, tokenizer offsets,
all48 prefix groups (including16 object groups), donor target/order, and all168
role contrasts/84 order contrasts/24 venue comparisons. Max raw arithmetic7.11e-15,
metric4.44e-16, full/parent/prefix errors0; no new neural forwards.

Data: `results/grain_vector_mechanisms/target_recency_r1/`; audit receipt initially
`target_recency_audit_r1/summary.json`; the style-only `next()` audit revision is
replayed in `target_recency_audit_r2/summary.json` with identical metrics. Identity:
`82f51877c204cab10ab28901c3e4fac7c71e4c3d0fc1f966d43345e7a00dbf38`.
Four authored settings; repeated names and two orders are not independent people.

## Next discriminating packet

P1N crosses role assignment and room truth. It asks whether a target-bound state
responds to relevant evidence while rejecting unrelated evidence, then decomposes
joint effects into state-composition departure and downstream nonadditivity using
a declared artificial additive anchor. See GRAIN_ATTRIBUTE_ROUTING_P1N_PLAN.md.
This develops the query-conditioned vector mechanism, not a human personality score.
