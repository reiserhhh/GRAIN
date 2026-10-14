# Event fields, presentation response and author composition

2026-10-09. Construction companion to the separately frozen HM72 experiment.
This file is not a replacement for that acquisition plan or its primary results.

## Distinguish observation units before learning a score

Let e be a declared behavioral configuration with event/actor identities, b a
wording bank, and o in {+1,-1} its presentation orientation. A fixed causal
reader supplies an event-indexed field H_b(e,o), rather than only a scalar
rating or a vector at whichever event happened to be last. Each field contains
contextual vector observations aligned to K1/K2/K3. It need not be invariant
to order: earlier context may genuinely change a later event representation.

This alignment retains a relation between vector and observed event. It does
not infer event identity, parse ambiguous natural reports, prove that a vector
contains a trait, or assume an equivariant neural architecture. In HM72 the
body prefix ends before any target question. There is no personal truth or
trained psychological scoring head in this observation.

## A two-component response that loses neither part

Define

    E_b(e) = [H_b(e,+1)+H_b(e,-1)]/2
    O_b(e) = [H_b(e,+1)-H_b(e,-1)]/2
    H_b(e,o) = E_b(e)+o O_b(e).

Swapping the reference presentation preserves E and changes O's sign. Keeping
both exactly preserves the two observations. E is not automatically signal;
O is not automatically noise. Artificial reversal probes a reader/material
response; a real author's voluntary order would be a separate observed choice.
The standard C2 projection and orthogonal energy identity are prior mathematics.
Calling E invariant is algebra, not evidence that it transports across wording.
Transport of E and O must therefore be tested separately on new observations.

## Author choice enters by a different function

If P_u(e) describes an author's event composition and
b_u(e)=P_u(+1|e)-P_u(-1|e) their conditional presentation composition, then

    Theta_u,b = sum_e P_u(e) [E_b(e)+b_u(e) O_b(e)].

This is ordinary conditional expectation with a lossless observation split.
It explains why choosing one canonical order, or averaging all orders and
discarding O, can erase an author's real presentation information. Conversely,
observing different orientations with the same P_u does not by itself imply
different behavior or personality. The two accounts are mathematically separate.
There is no interpretation of b_u as an identified psychological preference
without recording opportunities, task instructions and actual voluntary choices.

For source/new fields, their geometry keeps four pairing terms:

    <E_s+a O_s, E_n+b O_n>
      = <E_s,E_n> + b<E_s,O_n> + a<O_s,E_n> + ab<O_s,O_n>.

A stable order-averaged component and an unstable order interaction are thus
different transport objects. Averaging is not a license to delete the cross
terms. This formula supplies an interface for future choice/condition/state
analysis, not an assertion that all four terms have psychological meanings.

## Why a field is different from an unlabeled pool

Flattening the aligned (K1,K2,K3) matrix retains the slot relation. Pooling its
rows retains only their sum and cannot detect a cyclic reassignment of slots.
An improved pooled direction could therefore still be weaker as a relational
measurement. The empirical decision compares event-indexed endpoint/action/
sentence fields, their label-forgetting pool and the whole-account endpoint.
No source inverse, coordinate fit or layer selection is used.

HM72 applies all eight existing designed profile cases to each table and keeps
all seven Boolean coefficients. Same-group distributions, author distinctions
and dynamic/path responses remain separate. Designed-cell contrasts can test
the observation mechanism but are not additional human personality factors.

## Literature role and the next scientific decision

Deep Sets Section2 distinguishes invariant and equivariant outputs. Janossy
Pooling Definition2.1 constructs invariance by averaging sequence functions;
its canonical-order discussion warns that the selected order must be relevant
to the task. Those are established construction tools. HM72 does not implement
a trained Janossy network or the complete S3 orbit. The methodological question
is whether an explicitly identity-indexed observation transports and whether
discarding its order response discards meaningful author-composition channels.

OpenAI/math's fixed-observation/gauge/displacement viewpoint motivates keeping
the observation law separate from the object inferred through it. Harmonic
frames, elliptic PDEs, smooth bundles and the manuscript's extension theorem
are not established for this reader and are not claimed as applied results.

Positive transport would justify an event-field interface, not a perfected
personality scale. Failure would expose another representation gap, not demand
a better finite interpolation table. In both cases the next mechanism must
change the observation object or its semantic binding, not optimize accuracy.
Original18/100/44, natural evidence identity, personal response B_u and direct
psychological/behavioral interpretation remain in the full research scope.

## HM72 outcome and the information-opportunity correction

Actual capture is complete:960 body-prefix forwards atblock16, no later query
or upper block. Primary and two separately sealed result-after companions are
verified. Across wording at the same order, event-end fields have mean profile
cosine .663734/.647257, compared with body endpoints .610315/.601414. Across
opposite orders they fall to .303068/.305865; body endpoints .167288/.182504.
Correct event slots do not beat reversed position slots in the first cross-order
comparison (.303068 versus .307636). No universal event-interface promotion.

The observation object must include its causal information filtration. Write
H_b(z,o;e)=f_b,e(tokens up to e in order o), not a full-account measurement of e.
For actual prefix availability A(o,e) and Boolean character J on the complete
eight-cell design, J not subset A(o,e) forces coefficient_J(H_b(z,o;e))=0:
pair the cells in a missing factor; observations coincide and signs cancel.
This is known causal masking/character orthogonality, not a new theorem.

Actual prefix supports, in K1/K2/K3 order:

    forward: {x}, {x,y,g}, {x,y,g}
    reverse: {x,y,g}, {y,g}, {g}.

All60 bank/pair tables share that pattern. All2880 predicted unsupported
coefficients have norm at most2.823e-15. Charactersxy/xg/xyg are allowed only
at K2/K3 forward and K1 reverse: their event-slot supports are disjoint. All540
corresponding within-bank direction cosines have absolute value <=6.773e-16.
Their orthogonality therefore does not imply a missing interaction or changing
personality. It partly reflects observing different information opportunities.

Retain both E and O. Matching the available evidence is a separate requirement
from matching event identity; neither averaging nor a better finite chart
supplies an early token with future evidence. Next connect equal-information
content/ownership actions to the target-selective response object R_q, not a
cosine leaderboard. Author choice, conditional response and state still stay
separate; the present data do not estimate individualB_u or psychological truth.

The primary, cross-order and support results remain distinct. HM71's amplified
quadratic relation on60 score fibers is not HM72's raw first-moment direction
on240 cases; their numbers must not be advertised as an accuracy improvement.
Vaswani2017 Sections3.1/3.2.3 supply the causal-mask premise; Janossy/DeepSets
remain the established pooling ancestry. OpenAI harmonic/PDE extension is only
a problem-design analogy. Parent report: EVENT_FIELD_PRESENTATION_REPORT_20261009_ZH.md.
