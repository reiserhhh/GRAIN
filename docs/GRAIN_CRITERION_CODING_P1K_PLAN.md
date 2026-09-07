# P1K: separate criterion reading from answer-code carryover

2026-09-07. Frozen in8156864 before results; now complete and independently audited.
See ENTITY_STATE_AND_CODING_REPORT.md. Original protocol retained below.
Independent continuation
after audited P1J. P1J's common entity state affects event and outgoing-attribution
questions, but also the unchanged room question. Its venue role gap is3.0047,
larger than event2.3593. This prevents a selective-evidence interpretation.

## Decision and complete unit

Can a shared entity state be read with opposite semantic criteria AND opposite
answer conventions? Or does its intervention mainly shift the same answer token,
or Yes regardless of the question? Cross both manipulations, not one at a time.

Use all four existing authored records, both evidence role assignments, both named
entities, three task families, two question polarities, and two answer codes:
4 x2 x2 x3 x2 x2 =192 cases. No new real text/labels. No model fitting or search.

Positive/negative clauses:
- event: initiate rather than wait / wait rather than initiate the conversation;
- trait: seem outgoing rather than reserved / reserved rather than outgoing;
- venue: room open / room closed during the event (all records say open).

Use the identical `Question: For NAME, ` prefix. Code c=+1 is A=Yes/B=No;
c=-1 is B=Yes/A=No. All candidates are one token. At block16, patch the same
entity's opposite-role N state, always drawn from the positive-event/c=+1 cache.
Full EQD same-question donor restoration is the instrument control, not a semantic
success. Do not add Q again: its large restoration is already established in P1J.

192 native +384 patches +4 self/restore =580 forwards. Cap650 calls/600seconds/
9GiB MPS, padded execution length256, FP32. Preserve partial results on cap but
interpret only a complete factorial. No optional expansion if a cell fails.

## Metrics frozen before results

Let s be logit(A)-logit(B), r the queried person's initiating-role sign, p the
criterion polarity, c the code. Behavioral expected sign is r*p*c; unchanged
venue sign is p*c. Report native preference separately from role contrasts.

For each setting/entity/task, g(p,c)=s(r=+1)-s(r=-1) is the native role contrast.
For patches, use g(p,c)=mean_r[-r*(s_patch-s_native)]. Expand the complete2x2:

    g(p,c) = b0 + bp*p + bc*c + bpc*p*c
    bS = mean_(p,c)[g(p,c) * product(signs in S)].

This is an exact finite contrast, not a latent factor or an additive causal share.
A question-and-code-equivariant role effect loads bpc; a fixed-token effect b0;
an always-Yes role effect bc. These labels describe response transformations,
not uniquely identified internal mechanisms. Venue role effects should be near0
at every polarity/code. Report coefficients, squared contrast shares, all cells,
both people, expected-direction fractions and donor-restoration errors.

Antonym questions are not guaranteed exact psychological complements; trait truth
remains authored directional plausibility. Four settings are the sampling units,
not192 people. No significance promotion from repeated synthetic rows.

## Continuation rules

If binding is code/criterion-sensitive, use an explicitly conditional readout and
retain nonselective channels; do not claim a context-free personality direction.
If behavior-specific parity survives, next decompose vectors using cached states
and held-setting controls before collecting more. Compare full-state donor effects
to N rather than attributing native failures exclusively to N. A result cannot
authorize large source queues, a new model, or Claude usage automatically.

Prior alignment: entity/attribute binding has causal-intervention precedents
([Feng & Steinhardt,2024](https://arxiv.org/abs/2310.17191)); our N patch alone
does not identify their binding-ID subspace. Patching depends on metric and
corruption design ([Zhang & Nanda,2024](https://arxiv.org/abs/2309.16042)), hence
raw logits, irrelevant questions, paired swaps and full restoration all remain.
