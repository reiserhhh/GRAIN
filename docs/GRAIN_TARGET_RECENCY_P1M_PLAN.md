# Next P1M: query target and record-order control

2026-09-07. Frozen in8a89fd0, now complete and independently audited. Original
prospective protocol retained below; results in TARGET_RECENCY_REPORT.md.

## Why this changes a decision

P1J/K show strong behavior-role contamination of `For NAME, was the room open?`.
That is an awkward pragmatic target: the question explicitly foregrounds a person
while ostensibly asking about a room. Before generalizing the failure, compare an
explicit room target. Separately reverse the two action lines: a name-state effect
could partly encode most-recent behavior rather than robust person ownership.
P1L vector predictability does not remove either alternative.

## Minimal complete panel

Keep all4 authored settings and both role assignments. Use two action-line orders:
original and exact reversal, retaining the event/room header. Retain the original
P1J three questions for both named people, canonical A=Yes/B=No only. Add one
explicit object question per setting/role/order: `Question: For the room, was the
room open during the event? Return A for Yes and B for No.` Do not duplicate this
same room prompt for both people. Cases:4x2x2x(2x3+1)=112.

For person-prefixed queries keep the common name-state source from the event
question of the same person/opposite evidence/order. For the object question use
the single ` room` token within `For the room` as anchor, with opposite evidence
but identical room truth. The original room fact always says open. FullEQD is a
matched same-query donor control. Expected112 baseline+224 patches+4 controls=340
forwards, max400/600seconds/9GiB MPS, FP32/padded256, fixedblock16. No code/polarity
rescue search: the P1K failures remain recorded, and this panel does not retest them.

Tokenize/align every donor before forward, hash material/model/source, snapshot
all outputs, verify original48 canonical baselines against P1J. Save same-prefix
checks and full restore. If record reversal changes token counts, use a declared
alignment rule or refuse before neural results; do not silently slice by offset.

## Quantities and interpretation

Save all raw candidate margins and native expected preferences, each person/setting
role contrast, anchor-patch contrasts and full restoration errors. For venue the
expected evidence effect is zero. Compare absolute/RMS venue contrast with and
without person foregrounding at setting/order level, not a duplicated-person n.
For behavioral questions compare correct role direction under action-line reversal;
record absolute accuracy separately. Never normalize by a near-zero venue gap
without showing raw logit effects. Factorial record-order contrasts describe this
finite material, not a unique causal attribution to recency or grammatical target.

If room effects collapse with explicit targeting, localize the specificity failure
to this prompt family rather than declaring a generic yes-only mechanism. If role
effects reverse with action order, treat recency as a major alternative. If both
controls survive, next test fresh compositional events/operators; do not yet expand
to real authors/groups. No new private data, labels, model downloads, training or API.

## Exact collector accounting

`run_grain_target_recency.py` derives112 distinct prompts from the immutable P1J
material, preserving every original person query byte-for-byte in the original
order. Room is an object, not a third psychological subject: its signed contrast
uses original record-level sign only as an indexing convention; report magnitude.
Anchor patches are always within the same line order and named target. No state
is swapped across changed prompt lengths or a different lexical entity. The source
event-cache anchor is shared across the person's three criteria. Object anchors
have no cross-criterion shared-state claim. One person and one object case receive
self/restore controls. Order-reversal token layouts must match before any forward;
failure is retained without silently trying a different wording or model.
