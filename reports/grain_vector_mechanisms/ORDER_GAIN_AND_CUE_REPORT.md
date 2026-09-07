# P1R/S/T: order, conditional gain and cue diagnosticity

2026-09-07. All three packets COMPLETE with independent audits. The new native
cue panel changes the next action: inspect target-specific versus other-person
influence before another subspace sweep. No new human data, training, model or API.

## P1R: order matters, but is not the whole fixed-direction limitation

Frozen in5c4cea6. Reverse the two complete action lines in all64 P1P person cases,
retaining words/ownership/questions/directions.64 natives+192 patches+12 raw-logit
controls=268 forwards in106.49s. All64 projected relevant effects retain intended
signs. The unchanged descriptive gate requires projected error/natural RMS<=.5
AND complementary RMS/natural RMS<=.5 within a setting.

| Criterion | Original settings passing | Reversed passing | Original projected normalized error | Reversed error |
|---|---:|---:|---:|---:|
| Event |2/4|1/4|.54374|.47310|
| Behavioral outgoing attribution |2/4|2/4|.49326|.42198|

Cooking passes both orders/criteria. Books loses the event gate under reversal;
museum/photography still fail the combined gate because the complementary response
remains appreciable even where projected error improves. Thus neither a simple
last-record explanation nor universal rank-one sufficiency is supported. Wording
and scenario remain confounded; do not isolate a wording effect from this panel.

Native event fact preferences improve21/32 to24/32. Attribution preferences align
with the authored initiation/reservation proxy16/32 to18/32, with Yes responses
32/32 to30/32. The latter is NOT measured personality accuracy: waiting is not a
logical or psychometric definition of being reserved. P1T directly probes that gap.

Independent audit: exact paired text reversal/token layouts,32 common prefixes,
fixed direction digest, all192 patch vectors/effects and30 paired summaries.
Metric replay9.33e-15; FP32 patch replay1.87e-6; raw identity/full/original-repeat
controls replay0, common-prefix0. Unlike P1P, all12 controls now include raw A/B
logits and source/position provenance. No extra neural audit calls.

## P1S: geometric amplitude alone does not explain functional order effects

Frozen in80e651a. Cache-only64 pairs,0.467seconds. Let a_o be projection amplitude
along the same fixed direction and y_o its observed logit shift at order o.
The finite gain g_o=y_o/a_o is well-defined for all rows (minimum|a|3.67).

    y1-y0 = ((g1+g0)/2)*(a1-a0) + ((a1+a0)/2)*(g1-g0).

This exact symmetric accounting uses both observed outcomes, not a predictive
model or two independently isolated causal mechanisms. Gain includes the baseline
anchor, other recipient states and the remaining network; it is not a derivative.

| Criterion | Order-change SSE | Amplitude-term SSE | Gain-term SSE | Signed cross term |
|---|---:|---:|---:|---:|
| Event |1.546342|1.371900|1.249809|-1.075367|
| Attribution |.910721|1.300504|1.209726|-1.599509|

Opposing amplitude/gain changes partly conceal each other. An original-only
predictor y_pred=g_original*a_reversed has SSE1.251580 for event (versus unchanged
response1.546342), but1.194514 for attribution (versus.910721). Hence amplitude
rescaling is not a general solution even for this paired transformation. Very
large per-setting error ratios sometimes have tiny unchanged-error denominators;
absolute SSE is retained. No settings or coefficients were selected after testing.

Complement/readout cancellation also survives reversal: event projected-error
SSE1.693173=2.130920+.690479-1.128226; attribution1.167365=1.362360+.705356-.900352.
Independent raw-array, longdouble product-grid and dot-product replay8.31e-13.
The identities are standard finite algebra, not new psychological laws.

## P1T: clearer evidence improves ordinal attribution, but not code invariance

Frozen inae57eb3. Room=open only. Retain all waiting records, or replace ONLY the
noninitiator's behavior with explicitly declining/refusing conversation and staying
silent. No personality adjective is inserted into evidence. Cross both roles,
people, action orders, existing criteria and answer codes:256 prompts+4 repeats,
260 forwards in121.43s. All32 retained canonical P1P prompts replay exactly.

Code-aligned margin=code_sign*(logit_A-logit_B). The following are descriptions of
authored behavioral contrasts, not human Big5 labels or calibrated accuracy.

| Attribution contrast / order | Positive role gaps: standard / reversed code | Mean role gap: standard / reversed code | Authored-contrast preference: standard / reversed code |
|---|---:|---:|---:|
| Waiting / original |8/8 /4/8|1.15518 /-.01798|8/16 /8/16|
| Waiting / reversed |8/8 /4/8|1.17437 /-.02918|8/16 /8/16|
| Avoidance / original |8/8 /8/8|2.93434 /.44457|14/16 /11/16|
| Avoidance / reversed |8/8 /8/8|2.67444 /.42919|13/16 /10/16|

Clearer behavior increases attribution role ordering under BOTH codes and orders.
Thus the old poor threshold preference was not sufficient evidence of missing
behavioral information; material ambiguity contributed. This does not retrospectively
rescue the old approximation gate or validate an outgoing human trait. These are
four authored contexts, not independent people or population replications.

Output code remains a large limitation. Avoidance attribution agrees in semantic
preference across codes for only9/16 original-order and7/16 reversed-order cases,
although ordering is consistently positive. Waiting agreement is4/16 in each order.
Ordinal ordering, categorical threshold preference and code invariance are distinct.
For event facts, avoidance canonical-code preferences match16/16 and15/16, but
reversed-code11/16 and10/16. Hence clear cues do not fix general instruction/code use.

Independent audit checks all changed negative lines, unchanged positive lines,
both orders/maps, actual token anchors,64 common-prefix states,128 role pairs,
128 code pairs,80 summaries and raw controls/replays. Metric7.33e-15, raw3.55e-15,
prefix/repeat/parent replay0. No extra model calls.

## Next mechanism distinction, not another optimization sweep

P1T strengthens one actor's negative evidence. For queries about that actor, own
behavior changes; for queries about the other actor, OWN behavior is unchanged.
An improved role gap may therefore mix target-cue use with scene-relative contrast.
P1U will compare those two paths from the existing native cache before assigning
the improvement to a target-specific evidence channel. No additional forwards.
The larger goal remains active; author/group and psychological-validity requirements
are not waived by a better synthetic contrast.

## Artifacts

All under `results/grain_vector_mechanisms/`:
`direction_order_r1/` and `direction_order_audit_r1/summary.json`;
`order_gain_r1/` and `order_gain_audit_r1/summary.json`;
`cue_diagnosticity_r1/` and `cue_diagnosticity_audit_r1/summary.json`.
They retain materials, raw logits, states/patches, controls, complete per-setting
tables, immutable source snapshots and output hashes. Neural identities:
P1R `0a2ce61498adcb7760ee92577e77c49f8d2897e0375e018c707cefe3553a6b74`;
P1T `178e2d4867f47e19cafe4c6eaf4ca8be439b7031fb2a0dcc5cf16cb859092296`.
This turn adds528 forwards/227.92seconds total; P1S and all audits use zero forwards.
Old long queues and the released GRAIN scorer are unchanged.
