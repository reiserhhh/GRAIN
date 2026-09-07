# P1N: target-conditioned attribute selection and vector composition

2026-09-07. Complete and independently audited. Frozen in8f0673f.80 native prompts,
560 interventions,644 forwards in286.55seconds. Same cached Qwen2.5-1.5B-Instruct,
block16/MPS FP32/padded256. All40 original open-room baselines replay; common state
and fullEQD restore errors0 (independent CSV audit). No real-person data or training.

## Selection is not simply ignoring the input

Behavioral assignment B and room truth R are independently crossed. Both named
people receive behavior/attribution questions; the room question explicitly targets
the room. Every anchor donor changes B, R, or both in a matched same-target record.

| Question / replacement | Behavioral swap B: shift RMS | Room truth R: shift RMS | Relevant-direction fraction |
|---|---:|---:|---:|
| Event / single name anchor |1.26045|.06795|32/32 B|
| Outgoing attribution / same name anchor |1.24325|.07953|32/32 B|
| Room / room anchor |.03628|.25456|16/16 R|
| Event / fullEQD |2.68647|.25380|32/32 B|
| Outgoing attribution / fullEQD |2.76834|.45076|26/32 B|
| Room / fullEQD |.50902|8.20795|16/16 R|

Single-anchor relevant/irrelevant RMS ratios are about18.5 for event,15.6 for
attribution and7.0 for room. These are finite logit-response ratios, not personality
accuracy or information fractions. The room state responds consistently to its own
truth, so its low B response in P1M was not complete insensitivity. Both native
room truth classes have8/8 expected preference. Event preference13/16 closed and
14/16 open; attribution12/16 each. Weak native role ordering can coexist with a
correctly directed anchor intervention, as previously found in P1J.

The room anchor alone is NOT a sufficient donor-output replacement in this protocol:
its R shift.255 is far smaller than full-state8.208. Other states/interactions
matter for this intervention; this does not identify a mediation fraction or prove
the anchor lacks decodable room information. FullEQD is a deterministic restoration control, not evidence that all
its positions form the unique circuit. Off-target effects remain nonzero, and old
negative-polarity/code failures are not retested or waived by this canonical panel.

The intervention swaps the two people's roles rather than independently adding a
trait to one person. The room clause is a narrated condition, not a real change in
social opportunity. Outgoing attribution is authored behavioral plausibility, not
measured Big5. Four settings are not80 independent humans.

## Exact joint-effect decomposition

For each recipient, retain its other block states and let f be the remaining model.
Natural anchors are n0,nB,nR,nBR. The one artificial anchor is nAdd=nB+nR-n0:

    I = f(nBR)-f(nB)-f(nR)+f(n0)
      = [f(nBR)-f(nAdd)] + [f(nAdd)-f(nB)-f(nR)+f(n0)]
      = state_departure + readout_interaction.

The first term measures departure from additive state composition along this
reference path; the second tests the remaining network's finite nonadditivity.
They are not uniquely identifiable causal percentages, and nAdd may be off-manifold.
Changing the anchor or reference changes the decomposition. No gradient was fitted.

| Question | Total interaction SSE | State-departure SSE | Readout SSE | Signed cross term |
|---|---:|---:|---:|---:|
| Event (32 cases) |.436746|.689464|.122477|-.375196|
| Attribution (32) |.496650|.653142|.119673|-.276165|
| Room (16) |.073230|.044627|.007949|+.020654|

Thus near-additive aggregate responses can hide two larger opposing departures.
For event, .689464+.122477-.375196=.436746. Treating the parts as nonnegative
information shares would be wrong. Full native interactions have SSE4.92399,
2.97552,5.40872 respectively; the small anchor does not reproduce the whole
network's mixed response. No personality synergy claim is made.

## Audit and artifacts

Independent `audit_grain_attribute_routing.py` reconstructs the crossed materials,
real tokenizer offsets, all80 donors/worlds and common prefixes, raw logit effects,
80 native contrasts and80 interaction identities. Scalar/identity replay error
8.88e-16, raw arithmetic7.11e-15, full/parent/prefix0. Independently recomputed float64
norms differ from saved float32 norms by at most5.33e-6 (declared1e-4 bound). Source
and output hashes verified; all audits use zero new forwards.

Artifacts: `results/grain_vector_mechanisms/attribute_routing_r1/` includes native/
effect tables, 21-cell effect summary, all interaction energies, anchor geometry,
donor map, activations, immutable source snapshot/hashes. Audit receipt:
`attribute_routing_audit_r1/summary.json`. Identity:
`4e307e5e9b84a70245cd0d96d9b3ef42b0fe66fde8a3e7d0e2757f6ad31d132d`.

## Theoretical position and next step

This is finite support for `evidence -> target-indexed state -> conditional reading`,
not proof that the name state is an abstract personality concept. Entity-attribute
binding has independent prior intervention evidence in other settings
([Feng & Steinhardt,2024](https://arxiv.org/abs/2310.17191)); our raw anchor swaps
do not identify their binding-ID subspace. Causal abstraction/DAS explicitly tests
alignments between high-level variables and distributed representations
([Geiger et al.,2024](https://proceedings.mlr.press/v236/geiger24a.html)). Our finite
two-attribute contrasts are a precursor to such a claim, not an implemented DAS
analysis or a demonstrated faithful psychological abstraction.

Next P1O should use the saved four-world vectors only to distinguish weak encoding
from weak downstream sensitivity. Decompose each anchor's four states into mean,
B, R and BR vector contrasts; compare vector displacement to functional displacement.
No additional neural calls are justified for this immediate question. Establish the
target-conditioned geometry before any real-author/group extension.
