# GRAIN: observation order, evidence composition and target-valued action

2026-10-13 JST. HM245 theory construction during a frozen new collection.
No new native outcomes are used in Sections1-6. Standard mathematical
identities are not claimed as original theorems. The collection is finite;
original18/100/44 remains OPEN.

## 1. A summary must preserve the action we intend to interpret

For qualified event sequences A and B, the current observation projection is

\[
 \phi(A,B)=(\bar h_A,\bar h_B,\text{declared unchanged metadata}).
\]

Let pi reverse only the seven interior events of B, leaving its first4 and
last1 events unchanged. For the fixed focal variants A_S and A_C:

\[
 \Delta_q(B)=F_q(A_S,B)-F_q(A_C,B),\quad
 I_q=\Delta_q(B)-\Delta_q(\pi B).
\]

The new observation keeps the same event vectors and both phi values. If
I_q differs from the actual instrument discrepancy, a deterministic
mean-only reader cannot reproduce both observed actions. Increasing precision,
changing its rank or replacing its nonlinear link cannot restore an input
distinction that phi has removed. This is a finite factorization obstruction,
not a universal statement that mean vectors cannot predict human traits.

For any common proposed action d on the two equal projected inputs:

\[
 \max(|d-\Delta_q(B)|,|d-\Delta_q(\pi B)|)\ge |I_q|/2.
\]

The midpoint attains this two-action minimax bound. It is in native action
units, NOT the error of human questionnaire prediction. A bound on individual
endpoint predictions would be a different quantity. No mechanism discovery
is made merely by verifying the triangle inequality.

## 2. Local evidence features and contextual action are different objects

Suppose a reader is an additive sum of arbitrary event-window potentials
with width at most5. Pure background terms cancel in the S/C action. Windows
crossing the A/B boundary see at most the first4 background events, which
are unchanged. All focal terms are also unchanged by pi. Therefore I_q=0.

This is a testable restriction on one declared model family. A nonlinear
outer link, posterior mixture, recurrent process or long-range positional
reader need not satisfy it. We are not establishing a general neural
receptive field lower bound or excluding all Markov mechanisms. The windows
here count comments, not tokens.

The point is constructive: assigning a static weight to a local event is
different from assigning a value to that event in the full evidence account.
HM244 already distinguishes profile reweighting from within-profile response;
HM245 asks whether the observed account itself needs an ordered relation.

## 3. Reuse the established path algebra, do not invent new factor names

The old frozen GRAIN15 projection gives event coordinates z_i. Build a
cumulative path with increments z_i, retaining every original increment.
This is a chosen representation of evidence composition, NOT literal
psychological time, physical motion or identified individual dynamics.

The existing Chen implementation computes signature levels S1,S2,S3 without
new fitting or normalization. For concatenation A*B:

\[
 S_2(A*B)=A_2+A_1\otimes B_1+B_2,
\]
\[
 S_3(A*B)=A_3+A_2\otimes B_1+A_1\otimes B_2+B_3.
\]

Since the companion permutation preserves B1, focal/order mixed contrasts
of S1 and S2 vanish. At depth3 they equal

\[
 I_{S_3}=(A_{S,1}-A_{C,1})\otimes(B_2-(\pi B)_2).
\]

This supplies a specific reusable input: a focal evidence difference coupled
to a background ordered-pair difference. Background area alone can change
while adding the same offset to both arms; it is not by itself a linear
explanation of the focal action. A nonlinear function of level2 can also
interact, so depth3 is not universally necessary. A nonzero tensor is not
proof that the native model uses it; target contraction remains to be
source-identified, not fitted from these native endpoints.

[Chevyrev/Kormilitzin2016,updated2025](https://arxiv.org/abs/1603.03788)
provides the signature method's mathematical context. V6 already implemented
this tool. We add a qualified composition contract, not a new signature theorem.

## 4. Do not confuse presentation order with an author's actual path

There are two different observation regimes:

| Regime | What order means | Appropriate next object |
| --- | --- | --- |
| Attested conversation/event history | Documented succession may convey response, correction and change | Retain order, participants, occasions and opportunity alongside vectors |
| Unordered comment collection | Display order is chosen by the measurement process | Study order sensitivity as an observation effect; do not name it an author trait |

HM245's order is the old declared input order, not confirmed human chronology.
A positive I therefore proves a model-action/input-contract distinction. It
does not prove that artificial reordering changed a person's psychology or
that preserving every accidental ordering will improve personality measurement.

For the single involution pi, one can retain both finite response accounts:

\[
 F_q^{+}(X)=[F_q(X)+F_q(\pi X)]/2,
 \qquad F_q^{-}(X)=[F_q(X)-F_q(\pi X)]/2.
\]

F+ is invariant and F- changes sign under this pi, by construction. The
corresponding action's odd component is I/2. This is ordinary two-element
group averaging, already part of the project's earlier symmetry reasoning.
It is NOT a new neutral personality score or invariance under every possible
ordering. Retain the sensitive component as an observation-process account
rather than deleting comment contents and calling them noise.

The focal edit and background permutation commute on raw disjoint inputs.
Nonzero I is a mixed response, NOT their operator commutator. Tensor
concatenation's noncommutative algebra does not imply quantum measurement.

This changes a measurement-design decision before any accuracy optimization.
If the requested target describes the same author from an unordered collection,
the presentation operation changes the observing process, not the author's
recorded target. A native-sensitive component then belongs first to the
observation account. If the target describes an attested sequence or response
to a preceding event, succession is part of the target evidence instead. One
formula can index these regimes; one unqualified universal score cannot
silently equate them. Whether a retained ordered component acquires psychological
meaning is a source-outcome question, not a consequence of model sensitivity.

### 4.1 A target-qualified criterion for forgetting an order

Let U be the retained event collection (including event identity, ownership,
choice and condition information), S its order, C the declared observation
regime, and Y the source-supported target. Under a specified source law,
forgetting S preserves the full target posterior precisely when

\[
 P(Y\mid U,S,C)=P(Y\mid U,C),
 \quad\text{equivalently}\quad Y\perp S\mid(U,C).
\]

This is ordinary statistical sufficiency relative to Y, not sufficiency for
every possible psychological target. A randomized presentation mechanism
Q(S|U,C) which does not use Y satisfies the criterion when it is the only
source of S. Attested event order is not such a mechanism by default. Its
chronology, response opportunities and conditioning history may carry target
information. Nor is presentation randomization a reason to exchange events
between authors or discard their chosen topics.

Without that conditional independence, the exact marginalization law is

\[
 P(Y\mid U,C)=\sum_s P(Y\mid U,s,C)P(s\mid U,C).
\]

This expression uses a declared source order law. Averaging native logits,
averaging their probabilities, or assigning a uniform law to all permutations
are different operations. None identifies Y merely by producing an invariant
number. HM245's two-element average F+ is an observation-process account,
not automatically this posterior. Conversely, native order sensitivity is
not evidence that the true source target depends on display order.

The `Removing sibling orders` argument in the pinned OpenAI/math three-state
manuscript (foundations1-53) uses exactly the needed kind of condition: ordered
representatives have a root-independent conditional law. We can reuse this
criterion without importing its tree construction or reconstruction theorem.
[Deep Sets2017](https://arxiv.org/abs/1703.06114) supplies established set-model
machinery once invariance is the correct design; it does not establish that
invariance is appropriate for every author target.

Accordingly, the next source-supported compiler must specify Y, reference
population, C, the provenance of S, and which event relations it retains.
Its observation contract must distinguish display permutation from actual
succession before selecting a set or path reader. This adds a measurement
construction requirement, not an accuracy threshold or a new theorem.

## 5. A native probe and a psychological anchor are not the same target

The finite test compares one frozen model's F_q under controlled evidence
operations. A future source-supported measurement must separately declare
what Y_q is: questionnaire score, observed behavior, state or another
construct-supported outcome. Its source population and observation regime
must be part of the interface, not silently replaced by the native answer.

A needed next compiler uses qualified event evidence and observed source
target associations to identify its target action on new evidence. It may
retain a neural encoder. A high-dimensional fit to these24 groups could
reproduce arbitrary endpoints (HM151), so no such fit or accuracy grid is
performed here. The new cold interface currently returns ordered evidence
accounts, explicitly no personality target score.

This preserves the psychometric spine: comparing people under a declared
reference and opportunity, retaining choice/conditional response/state.
Older stable author structures are not invalidated by this model-order test.
The result identifies what a particular inference interface discards, not
what every previous GRAIN factor must mean.

## 6. Which mathematics actually supports this step

[Geiger etal2025 Sections2.2-2.4](https://arxiv.org/html/2301.04709v4)
motivates intervention-preserving abstractions rather than mere unperturbed
answer agreement. HM245 is not a full causal abstraction of native attention.
[Heimersheim/Nanda2024](https://arxiv.org/html/2404.15255v1)
motivates distinguishing native readout effects from interpretations of
their internal carriers. The account split is not unique causal mediation.

Pinned OpenAI/math adc7f124 three-state foundations98-125 derives posterior
sufficiency within its declared experiment and conditional law. A generic
embedding mean does not inherit that sufficiency. This is a research-design
lesson, not transplantation of its reconstruction threshold to psychology.
Standard factorization, Chen algebra and finite symmetry are the mathematics
actually implemented. The prospective contribution is a reusable relation
between a measurement's observation unit and its target-valued evidence
operations; it must be established by the finite new observations, not by
the prestige or novelty of the source mathematics.

During collection, also read the pinned OpenAI/math manuscript
`From-partial-permutation-information-to-Fourier-bounds` foundations1-130.
It develops subgroup-coset laws, common mass trimming and Fourier bounds for
convolutions on S_n under explicit asymptotic hypotheses. Those hypotheses
are not established for human text or the one fixed HM245 involution. No
mixing bound, uniform author law, trimming operation or removal of choice
information is imported. The useful separation is between observing a
quotient and observing the full account; our exact pair-orbit accounting uses
only elementary C2 symmetry. A single tested reversal does not identify all
S7-sensitive modes. Do not launch a permutation sweep to fill that gap unless
it changes a source-supported measurement construction.

There are also different kinds of relabeling: permuting presentation slots
of one author's fixed events is not exchanging events between authors.
Uniformizing the latter can destroy author-choice structure. Observation
equivalence must therefore be declared per operation and regime, not inferred
from the fact that both operations can be written as permutations.

## 7. Outcome and next construction

The once-only collection completed all50 batches/250 inquiries in874.296289
model seconds. All24 registered groups from12 old completion-prefix authors
are observed;48 old own-context batches are reused. The original HM243
incomplete24-author experiment remains unchanged, not retrospectively complete.
The collector exited0, PID94728 is absent, and its lock is gone. Analysis,
verification and seal checking each ran once successfully.

| Target query | Mixed log-odds RMS | Action sign reversals /24 | Mean-only action minimax RMS |
| --- | ---: | ---: | ---: |
| E | .040370 | 1 | .020185 |
| N | .025919 | 0 | .012960 |
| A | .032886 | 1 | .016443 |
| C | .033913 | 1 | .016957 |
| O | .037899 | 1 | .018950 |

Every one of the120 target actions exceeds the registered empirical
instrument-discrepancy account1.441107e-5. This account is not a population
confidence interval or a general worst-case reconstruction bound. In a
separate post-outcome conservative readback, log-sum-exp is1-Lipschitz in
the infinity norm, so two such terms per log odds and four corners give
the head-reconstruction bound8e=2.882213e-5, with e=3.602766e-6.119/120
actions exceed this bound;author2/view1/A at2.508585e-5 is retained as small.
This head-only bound does not certify every possible source of measurement
drift. Both identity replay action and logit discrepancies were exactly0.
No registered threshold or sealed analysis was edited to obtain that count.

The observed difference therefore cannot be reproduced by a deterministic
reader of the fixed two means alone on this finite support. The restricted
additive event-window family in Section2 also cannot reproduce the nonzero
actions. This does not exclude nonlinear local aggregation, layout/recency,
coherence or another long-range implementation. Native action is a model
observable, not a human trait change or a personality-prediction error.

The old Chen construction also closes numerically:levels1/2 mixed maxima
4.44e-16/1.33e-15;level3 factorization error1.42e-15. Its mixed tensor RMS
is.003936 in the frozen GRAIN coordinate units, not a psychological effect.
Independent ordered sums match to6.66e-16, alternate native actions to
1.07e-14, and a separate cold process matches exactly. The representation
and reader accounts retain signed cross-products;their RMS values must not
be added or converted to information fractions. Native-account readback
discrepancy4.32e-6 is recorded alongside the direct raw-logit actions.
32 focused tests passed. This is same-agent alternate verification, not
blind independent replication.

The cold numeric interface now retains addressed event order and its
contextual interaction, but deliberately returns target_score=null. We
have not fitted a signature contraction to these outcomes or identified
the native model with Chen algebra. In particular, the new evidence does
not make a display-sensitive component an author trait.

The next mainline construction is source-supported target valuation under
an explicit observation regime. Inventory the existing same-event meaning
bridge(HM173/177/233) and attested path laws(HM235-237) first. Preserve event
identity, ownership, selection, condition and actual succession; distinguish
them from display ordering. Determine which source-qualified event relation
can change a new author's declared target score or evidence action without
using native answers to choose coefficients. This is a construction problem,
not another mean-loss demonstration or an agreement/accuracy contest.

Close this permutation family. No new permutation, signature-depth, layer,
kernel, prior, rank or precision sweep is justified by this outcome. Original
18/100/44 remains OPEN;Goal ACTIVE/nojob. The full result and all signed
actions are retained in `results/grain_math_response_v1/event_order_action_r1/`.
