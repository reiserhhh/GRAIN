# HM68: Ownership-Gated Cross-Target Evidence Composition

2026-10-09. Freeze before new local observations. This is a mechanism experiment,
not an observer repair, teacher-fidelity search or human personality benchmark.

## Decision

HM67 found context-dependent multi-target directions. Does that dependence track
evidence assigned to the queried person, generic co-occurring semantics, or the
final answer link? The decision changes whether GRAIN needs a joint content /
event-ownership observation law, beyond independent event eligibility gates.
We retain reader Tq, author evidence distribution Pu and personal response Bu as
different objects. A reader ownership effect is not evidence for human Bu.

## Complete Design and Bound

Six targets: extraversion, neuroticism, agreeableness, conscientiousness,
openness, anthropomorphism. All 30 ordered distinct focal/companion pairs,
both historical lexical families, both low/high focal and companion alternatives,
and both companion owners. Six neutral-companion controls per family use the
same full factorial. This is 576 new public authored bodies / 3456 readings,
all six queries on every body. No target selection, parameter fitting, generation,
neural training, API, private text, labels or psychological promotion threshold.
Collector bound: 3456 attempted forwards and 1500 collector seconds. Preserve
partial captures on failure; do not retry uncertain calls or launch a new grid.

Focal event K1 belongs to A. Companion event K2 belongs to A (g=+1) or B (g=-1).
An unchanged neutral K3 belongs to the opposite person. Thus A/B counts, complete
word/token bags, event count, format and query are identical between ownership
arms; only two actor-token positions exchange. K1/K2/K3 order is fixed. Own and
other arms are deliberately distinct evidence allocations, not equivalent facts.
The neutral companion has two nominally irrelevant alternatives (folder colour
or ribbon colour); it is a format/role control, not guaranteed psychologically
meaningless language. This is an existing-material follow-up with new readings,
not an independent population, model or blinded human replication.

## Group Projection, Not Accuracy Optimization

Let F(x,y,g) be one multi-target observation, x/y in {-1,+1}. The ownership swap
is the Z2 action g -> -g. Its even and odd projections are

    F_even=(F(x,y,+1)+F(x,y,-1))/2
    F_odd =(F(x,y,+1)-F(x,y,-1))/2.

For K_g=F(+,+,g)-F(+,-,g)-F(-,+,g)+F(-,-,g),

    D=K_+ - K_- = 8*c_{x y g},
    c_{x y g}=(1/8)*sum(x*y*g*F(x,y,g)).

This exactly isolates the ownership-odd mixed action in this finite design. It
does not prove that the model implemented correct subject binding: lexical / role
marker effects can also be odd. Subtract the matched focal neutral-companion D0
descriptively and retain D and D0 separately. Different behaviour coordinates
have no common physical unit, so D-D0 is a diagnostic, not a pure causal effect.

Native score, high/low log-odds, centred 10-candidate logits and question-free
block16 body endpoints are kept separately. Link controls remove only xy and
xyg Fourier coefficients from centred logits, retaining the other six terms,
then decode with the unchanged answer link. They allow ownership-dependent
single-evidence slopes; they test whether score interaction can arise without
pre-answer cross-behaviour interaction. Do not subtract layer energy fractions
as mediation or call a mixed derivative curvature / time noncommutativity.

## Fixed Readout and Acceptance

Use the existing local Qwen2.5-1.5B observer, same system / queries / candidate
alphabet and block16. Bind dependency code, material, prior HM67 verification,
model config, tokenizer and actual observer manifest; model weight files are not
individually hashed, which must remain explicit. Save each attempted call before
execution, raw logits / body / query endpoints after it, all stage arrays, all
30 pairs, neutral controls, target-wise interactions, and all finite spectra.

Tests cover swap parity, idempotence, exact 8*c identity, compatibility, Parseval,
neutral marker cancellation, ownership-independent semantic interaction, link-only
artefacts, and complete material/token proof. Four declared numeric worlds at
three seeds save actual arrays and separate the four explanations. Verification
rebuilds all raw stages and uses direct tensor differences and scalar summation,
not merely the same production helper. Same-agent verification is not independent
psychological replication. No population p-value from template rows.

Scientific outcomes: a qualified mechanism predicts target-directed own/other
mixed-action differences exceeding its neutral pattern before the answer link;
generic composition predicts similar own/other interactions; link-only predicts
effects after decoding without the corresponding centred-logit mixed terms.
Report mixtures and failed predictions as well as positive results. Stop after
this battery. Keep original18/100/44 scope open; no publication or lockbox action.

## Actual Primary Reading

Ronald de Wolf, A Brief Introduction to Fourier Analysis on the Boolean Cube
(2008), author PDF sections2.1--2.2 (pp1--3): character basis on Z2^n,
orthogonality, inversion, Parseval and +/-1 convention.
https://homepages.cwi.nl/~rdewolf/publ/other/fourier.pdf
The group projection / finite difference identities are established mathematics,
not a new theorem. Shapley-Taylor also studies feature interactions:
https://arxiv.org/abs/1902.05622 . Our question is the content-by-ownership,
multi-target measurement interface, not claiming discovery of interactions.

OpenAI fixed metric/connection introduction (previously read lines696--720)
motivates retaining vector-valued rather than scalar-only transport:
https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Determination-of-a-metric-and-a-unitary-connection-from-one-boundary-patch-October-5-2026/build/sections/01-introduction.tex
No PDE, manifold or connection uniqueness theorem is imported.
