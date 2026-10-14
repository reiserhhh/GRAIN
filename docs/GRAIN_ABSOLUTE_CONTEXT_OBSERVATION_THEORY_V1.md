# Qualified author evidence and background-conditioned observation

2026-10-13 JST. HM230 finite mechanism discriminator. This note was started
while the already sealed collection was running. It does not add hypotheses,
alter scientific code, or turn post-outcome descriptions into preregistration.
Collection, analysis and verification subsequently completed; section8 records
the result without changing the frozen plan. Original18 protocols/100 cards/
44 computation questions remain OPEN.

## 1. The construction problem

HM228 changed the observation of the SAME author histories from an absolute
digit judgment to a two-author A/B comparison. Its E/N target contact changed
from -.349749 to +.749677, with +.693144 already in the raw odd comparisons.
That is a change in the observer's score organization, not an accuracy gain.
The task also changed context, wording, answer alphabet and decoding together.

HM230 asks whether background alone produces such a change while retaining
the original absolute task. It is not an attempt to tune a model until it
reproduces the previous contact or obtains a larger questionnaire correlation.

## 2. Separate evidence from its observation

Let X_u be the complete addressed twelve-event history of author u. Let b
be an observation background, q a fixed target question, and A the fixed ten
digit candidates. The deployed model induces a candidate-conditioned law:

```text
K(q,b;k | X_u) = exp(z_k(X_u,q,b)) / sum_(j in A) exp(z_j(X_u,q,b)).

S(q,b | X_u) = sum_(k=0..9) (k/9) K(q,b;k | X_u).
Y(q,b | X_u) = log sum_(k=5..9) exp(z_k)
              - log sum_(k=0..4) exp(z_k).
```

The full-vocabulary candidate mass and greedy-token compliance are separate
observables and remain available. A conditional digit score must not conceal
that the model gave substantial mass to non-digit vocabulary.

The HM229 SOURCE law F_u is another object: a frozen SOURCE-supported joint
distribution over the five external anchor coordinates, computed using own
event vectors only. In HM230 F_u is unchanged BY CONTRACT. This is not an
empirical proof of author invariance and does not identify a latent true trait.
Nor is K a human posterior merely because the query has a personality name.

The qualified record must carry both, rather than silently equate them:

```text
target author and event IDs -> SOURCE reference and own-evidence F_u
target evidence + quotation occurrences + query + readout -> K(q,b | X_u)
```

Historical E5 encoding and the native Qwen observer are neural. These laws
are not a completely model-free raw-text personality system.

## 3. Three frozen background contracts

All24 public PANDORA authors have all12 cue-masked comments retained. For
each author, observe all E/N/A/C/O questions under:

- alone: the exact original prompt, checked at token level;
- duplicate: prepend the same twelve own comments as background quotations;
- other: prepend the twelve comments of fixed partner (u+1)%24 in the SAME
  quotation format.

The original SYSTEM, target body, excerpt order, question and digit alphabet
remain fixed. Both quotation conditions explicitly say background only, do
not assess it. They do not describe the quotations as a new target writer.
Lengths and target positions still change. Other-minus-duplicate therefore
does not isolate a uniquely semantic mechanism or eliminate all position effects.

An event ID and an occurrence address are different. Quoting an event twice
creates another observer occurrence, not another independent human event.
There are24 target authors and288 unique target events, not360 people.

## 4. Signed finite-panel accounts

For every target, define:

```text
B_u = Y(q,alone | X_u)
D_u = Y(q,duplicate | X_u) - B_u
O_u = Y(q,other | X_u) - Y(q,duplicate | X_u)

Y(q,other | X_u) = B_u + D_u + O_u.
```

Let c_0,c_1,c_2 be these three five-target vectors centered over the SAME
24 authors. For i,j in0..2 retain:

```text
C_ij = (1/24) sum_u c_i(u) c_j(u)^T.
Cov(Y_other) = sum_(i,j=0..2) C_ij.
```

All9 matrices, including negative cross terms, are necessary. They are signed
statistical accounts, not independent psychological information fractions.
Repeat the account on expected-digit scores without mixing the two scales.
The contact matrix divides each covariance by its two finite-panel standard
deviations. A changed E/N contact can involve both its numerator and variances;
it is not itself a new factor or a pointwise ordering reversal.

The999 whole-row resamples retain all modes and targets together. Background
authors recur across rows through the cyclic assignment, so the resulting
bands are descriptive conditional panel bands, not independent-author
population confidence intervals or significance tests.

## 5. What this experiment can change

If absolute background does not reproduce the previous target alignment,
background alone is not sufficient on this panel; the comparative operation
remains a separate construction problem. Do not chase prompt variants until
the old result reappears.

If duplicate and other both substantially change target organization, the
observer contract must account for added context/opportunity even when the
target evidence set is unchanged. Do not remove that account as noise.

If other differs from duplicate, retain the conditional background action.
Its relation to SOURCE-supported author evidence is the next question; it is
not already proof of a particular psychological reference or semantic mechanism.

Whichever outcome occurs, an interpretable construction has distinct target
evidence, observer conditions, SOURCE reference and target functional. Mere
faithful reconstruction of K does not establish the meaning of F_u or S.

## 6. Prior art and mathematical inspiration

The distinction between relevant cues and their detection/utilization is
already part of Funder's Realistic Accuracy Model and our existing
[cue-utilization note](GRAIN_CUE_UTILIZATION_AND_CONTEXTUAL_INFERENCE_V1.md).
HM230 extends its concrete qualified observer interface; it does not discover
that distinction for the first time.

[Lost in the Middle](https://arxiv.org/html/2307.03172v3) already establishes
position/length sensitivity in other LLM tasks. [LLM non-transitivity](https://arxiv.org/html/2502.14074)
already studies comparative observation problems. Ordinary softmax, covariance,
and conditional empirical laws are standard tools, not new mathematics.

OpenAI/math pinned adc7f124 Gaussian streaming section08 conditions on the
current state before applying a fresh-block bound. Its fresh independence,
Gaussian rows, exact labels and bounded conditional prior are not established
here. The free-group cocycle note lets earlier prefixes transport subsequent
increments; text does not thereby become an invertible free-group action.
These are design inspirations ONLY, not imported personality guarantees.

## 7. Scientific and computational completion are different

Canonical output is `results/grain_math_response_v1/absolute_context_observation_r1`.
The finite plan is [HM230](GRAIN_ABSOLUTE_CONTEXT_OBSERVATION_PLAN_HM230.md).
The separate stages are `prepare`, `collect`, `analyze`, `verify`, `check` in
`scripts/run_grain_absolute_context_observation_hm230.py`. Collection is one
shot under the1500 model-second cap; no implicit retry or scope shrink.

Verification independently reconstructs candidate laws, signed accounts,
target contacts, exact target-token binding and SOURCE references. It is
same-agent alternate arithmetic, not blind external replication. Passing it
establishes correct computation, not completion of the full research objective.

No accuracy/head/layer/rank/bandwidth grid follows automatically. A subsequent
packet must add a new mainline construction or distinguish a substantive
competing mechanism using existing observations wherever possible.

## 8. Completed result and resulting construction decision

All72 batches/360 observations completed in1126.660756 model seconds under
the1500 cap. All360 greedy outputs are digits, and full-vocabulary digit mass
is at least.999226. No observations, negatives or cross terms were discarded.

| Background | E/N log-odds contact | E/N expected-score contact |
| --- | ---: | ---: |
| alone | -.349747 | -.320312 |
| duplicate | -.392491 | -.384054 |
| other | -.207953 | -.267535 |

The primary other-minus-duplicate contact change is+.184538 with descriptive
conditional row band[-.163006,.518613]. Duplicate-minus-alone is-.042744 with
band[-.407472,.304941]. These are NOT population CIs or equivalence tests.
All three point contacts remain negative. N has negative log-odds contacts
with all four other targets in the alone and other arms. Added background
therefore does NOT reproduce HM228's all-positive comparative organization
on this registered panel. It does not uniquely isolate which comparative
wording, alphabet, readout or relational operation caused that organization.

Background still changes levels. Duplicate-minus-alone mean expected scores
are E/N/A/C/O:+.061121/+.036162/+.031312/+.028477/+.049652. Their positive
author counts are23/19/17/22/22 of24. Other-minus-duplicate means are
-.026316/-.058406/-.004144/+.004656/-.026155, with5/1/12/13/5 positive.
These are observer responses, not extra human evidence or increased accuracy.
Repetition changes the observer although the registered unique event set
does not change. Length, opportunity and content remain possible mechanisms.

The log-odds E/N covariance account, with row/column components B,D,O, is:

```text
[[-.033323782, -.020122262, +.022802580],
 [-.035444188, +.001430237, +.028357875],
 [+.042508362, -.024121456, -.007621123]]
sum = -.025533757
```

The signed cross terms are material, not nuisance to remove. Geometry and
level action must be separate measurements; one cannot infer meaningful
target transport from stronger scores or a large leading singular direction.

All360 candidate laws/450 cross accounts and token/event/SOURCE bindings
pass alternate verification. Head reconstruction max3.249624e-6; remaining
arithmetic max3.552714e-15. Original-alone rereading differs by RMS2.564496e-6,
max2.335096e-5, not an optimized hidden-state equivalence gate. All scientific
seals,38 focused tests and3 Python lint/format checks pass. Same-agent alternate
arithmetic is not blind replication. Process51759 and its lock are gone.

Decision: freeze background/precision experiments. Keep the qualified observer
record as a usable interface. The next mainline problem is a SOURCE-supported
comparative target functional: how absolute and relational observations retain
or change target meaning on the SAME addressed author evidence. Do not fit a
transport merely to recover the previous EN contact; do not erase comparison
cycles or presentation by projection. Inventory existing HM171/173/221/223/
228/229 interfaces before defining any genuinely new finite construction.
