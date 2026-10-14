# Observable author anchors without a recovered latent law

2026-10-11. HM160 finite construction; not global GRAIN completion.
Canonical experiment: results/grain_math_response_v1/observable_author_anchor_r1.
The six primary scientific files remain sealed. This document follows the results.

## 1. Contribution and prior art

This packet adds a source-observed endpoint constructor to GRAIN's vector/event
interface. It does not discover a new conditional-probability theorem, solve
general natural-language decoding, recover an unknown psychological trait, or
improve an existing reader's prediction accuracy. It separates three objects:

- F: the rule that qualifies an event and turns it into a vector observation;
- ell: a conditional behavior or diagnostic law, which may be unknown;
- G: a source population reference, which may also be unknown.

HM159 constructed author functionals with G unknown but ell stipulated. HM142
learned an anonymous three-class law with imposed class count and independence.
F15/R08 already cover predictive states and observational equivalence. HM160
instead constructs an observable endpoint function without recovering ell or G
as latent objects. These earlier methods are not rediscovered here.

The probability construction below is standard. Relevant prior art includes
latent-coordinate reparameterization in nonparametric IRT, prediction of
observables, and the distinction between observational and imposed-condition
regimes. [He (2023)](https://arxiv.org/html/2310.20165v1) discusses equivalent
latent parameterizations when the latent distribution is transformed. Its
asymptotic IRT theorem is NOT applied: this experiment does not establish its
unidimensionality, local-independence, derivative or long-assessment conditions.
[Dawid (2020)](https://arxiv.org/pdf/2004.12493) supplies the regime-based causal
distinction. The publisher summary of
[Dawid (1984)](https://academic.oup.com/jrsssa/article/147/2/278/7106293) motivates
observable prediction rather than an unanchored parameter interpretation; its
full proofs were not text-read in this packet.

OpenAI/math's metric/connection inverse-problem introduction, pinned at
adc7f1241b42e322a6451854ab7e4b4c146bf78a, was reread as an inspiration to distinguish
coordinate choices from observation invariants. No PDE, gauge-identification or
geometric theorem from that manuscript is transferred to this model.

## 2. Observable scoring function

Let H be a qualified past event, A a registered endpoint opportunity, and Y a
binary expression observed in a DIFFERENT event of the same source author.
Register a source population and collection regime before construction. For
randomized A with positive support, define

\[
\psi_q(h)=P(Y=1\mid H=h,A=q),\qquad
b_q=\sum_h P(H=h)\psi_q(h),\qquad
d_q(h)=\psi_q(h)-b_q.
\]

The constructor uses observed joint counts, not individual latent truth,
generator response functions, class labels, an LLM teacher answer, application
outcomes, or application-batch normalization. The registered source reference
b_q remains fixed for cold application.

Under the randomized collection design this is a future expression distribution
in the specified opportunity, for authors with the observed history. It is not
an exact individual probability or a guaranteed stable trait. Under natural
self-selection the same conditional formula is retained, but the result is an
ASSOCIATIONAL_CONDITION_PROFILE. It is not silently reported as P(Y | do(A=q)).
Choice remains information; it is not automatically removed as a confound.

Two conditions can produce two legitimate, differently ordered endpoint
profiles. The opportunity is part of the target meaning, not a nuisance to be
averaged away before the measurement object is defined.

## 3. Coordinate equivalence and a relative-mean witness

Let latent p have distribution G and event kernel K(. | p). For an invertible
monotone g, take lambda=g(p), transformed population g#G, and
K'(e | lambda)=K(e | g^{-1}(lambda)). Every observable joint event law is
unchanged. This is a direct change of variable, not finite-sample approximation.

Numerical latent means are not generally invariants of this equivalence:

\[
E[p\mid H]-E[p]\quad\hbox{and}\quad
E[g(p)\mid H]-E[g(p)]
\]

can have different signs. Pointwise order under a monotone transformation is
preserved; this is NOT a reversal of the true individuals' ranks.

HM160 uses p in {1/20,9/20,19/20}, masses {1/2,3/10,1/5}. Historical C and R are
conditionally independent Bern(p); address is 2C+(C xor R). For H=(C=1,R=0),

| Coordinate | Source mean | Conditional mean | Difference |
| --- | ---: | ---: | ---: |
| p | 7/20 = .350000 | 349/860 = .4058139535 | 12/215 = +.0558139535 |
| lambda=p^2 | 97/400 = .242500 | 3787/17200 = .2201744186 | -24/1075 = -.0223255814 |

The entire history/endpoint law is preserved after the inverse-kernel change;
paired generated records have zero differences. A prior midrank coordinate has
mean .5 and conditional mean 251/430=.5837209302 in both parameterizations. That
ordinal invariant is a diagnostic, not recovered or scored by the constructor.

Endpoint q0 has response probability p; endpoint q1 has p^2. Each specified
physical endpoint retains its own psi under reparameterization. These are two
different behavioral quantities, NOT two interchangeable names for one trait.
An observable endpoint therefore anchors a function, not all latent coordinates.

## 4. Selection equivalence is a different ambiguity

Two author types, p=.25/.75, each have mass .5. Natural condition selection is
A=type. Consider these type-by-imposed-condition response probabilities:

\[
W_0=\begin{pmatrix}.2&.3\\.7&.8\end{pmatrix},\qquad
W_1=\begin{pmatrix}.2&.9\\.1&.8\end{pmatrix}.
\]

The actual natural observations see only W[0,0]=.2 and W[1,1]=.8. The entire
joint observed law (H,A,Y) is identical in W0 and W1. In contrast, randomizing
A exposes the off-diagonal responses. For H=(C=0,R=0):

| World | q0 probability/reference/difference | q1 probability/reference/difference |
| --- | --- | --- |
| W0 randomized | .25 / .45 / -.20 | .35 / .55 / -.20 |
| W1 randomized | .19 / .15 / +.04 | .89 / .85 / +.04 |

No precision increase on this natural joint law selects between these imposed
responses without additional linking assumptions or observations. This is
standard causal-identification logic, not a new causal theorem. It complements
the existing F03 selection work; it does not invalidate natural choice scoring.

## 5. Positive construction and actual execution

Three fixed seeds 160/161/162 each produce 12,000 source authors and 4,000 new
application authors. Source input has one history and one randomized endpoint;
application input has only a history, with two separate endpoint events held
out for evaluation. All are simulated author instances, not human participants.

The 108,000 finite vector events use the first four public-source GRAIN15
prototypes from relative_choice_r1/dictionary.npz. Exact dictionary decoding is
the observation rule in this world. Arbitrary natural embeddings are rejected;
this does not remove the historical neural encoder dependency.

Source bundles contain only version, regime, target IDs, history mass,
probabilities, reference and support. Oracle p/lambda files and application
outcomes are separate and excluded from construction. Source/application IDs
are disjoint. No fit, model/seed selection, API or new neural forward was run.

The exact population Brier advantage over the SAME fixed reference is

\[
E[(Y-b_q)^2]-E[(Y-\psi_q(H))^2]
=\operatorname{Var}_H(\psi_q(H)).
\]

It equals .0835260361 for q0 and .0789368771 for q1 in this declared world.
All six actual application checks are positive: .081091-.090950 for q0 and
.072394-.087022 for q1. These are endpoint checks, not psychological accuracy
and not an optimization target. Null endpoints independent of H have exact
relative profiles and gains zero.

A separate cold CLI demonstration using seed160 source counts and only the
address3 history vector returns:

- q0: probability .3762057878, fixed reference .3427274275, relative +.0334783603;
- q1: probability .2078488372, fixed reference .2437250035, relative -.0358761663.

These empirical values are retained as observed; they are not replaced by the
exact population values, recalibrated against outcomes, or selected as a winner.

Unsupported history-condition cells return UNOBSERVED_HISTORY_CONDITION. A
reference missing positive-history support is unavailable, not fabricated by
renormalization, zero filling or an outcome-tuned prior.

Primary execution took .137366 seconds of finite-array CPU. The same-agent
alternate implementation recounted 36,000 source and 12,000 application authors,
24,000 cold scalars and five exact tables, maximum difference 1.11e-16. It is
alternate arithmetic, NOT independent blind validation. Sixteen new tests and
29 focused tests pass; those counts are engineering evidence, not theoretical
completion. Original 18 protocols / 100 cards / 44 computational questions
remain OPEN.

## 6. Mainline implication, not another precision branch

The change is to the construction contract: author comparison is a function of
qualified evidence, an explicit endpoint/opportunity, a collection regime and
a frozen source reference. It need not begin with a uniquely recovered latent
personality coordinate. Conversely, an anonymous vector axis or a good reader
fit does not supply that endpoint meaning for free.

The next useful question is the bridge between previously observed qualified
vector actions and source-supported behavioral anchors on the SAME evidence
units. Inventory existing HM142/143/145/147/148 and C2-C4 results first. State
what construction this would enable and two explanations an experiment can
separate before collecting anything. Keep neural target response, persona
assignment and observed author behavior as distinct possible anchors.

Do not repeat this table compiler, reparameterization witness or causal alias
under more seeds/classes/degrees. Do not tune Brier/RMSE, reader heads or
latent naming as a substitute for that bridge. Psychological interpretation
will require meaningful anchor evidence; present future-expression scores are
positive finite construction evidence, not renamed Big5/emotion measurements.
