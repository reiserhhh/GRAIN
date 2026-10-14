# Expression correspondence and actor-qualified observations

2026-10-10. HM115 finite results; not a new mathematical theorem or a completed
personality instrument. This document is a result interpretation, separate from
the sealed HM115 design/code/material/results.

## 1. The question, rather than a precision target

HM114 retained the same behavior clauses while changing opportunity wording.
HM105 changed behavior clauses and the current carrier together. HM115 fixes
the carrier, compares same-event rewording with a different behavioral instance,
and gives both named persons the same opportunity. It asks which observation
survives these changes, not how to imitate a target score more accurately.

Six declared behavior families are E, N, A, C, O and anthropomorphism. These are
material-design addresses, not established psychological equivalence classes.
The different instances can express different facets, states or topics. They
are not human labels, validated scale items or six newly discovered factors.

## 2. A standard finite decomposition

Let h_f(c,b,o) be an observed vector at the fixed current carrier, where c is
opportunity, b is the declared behavior polarity and o is whether A or B acts.
Each sign is -1 or +1. For each family and expression bank:

\[
 F_f(c)=\frac{h_f(c,+1,+1)-h_f(c,-1,+1)}2,\qquad
 O_f(c)=\frac{h_f(c,+1,-1)-h_f(c,-1,-1)}2,
\]
\[
 S_f(c)=\frac{F_f(c)+O_f(c)}2,\qquad
 D_f(c)=\frac{F_f(c)-O_f(c)}2.
\]

Thus F=S+D and O=S-D. S is even under actor assignment, while D is the odd
behavior-by-assignment interaction. A nonzero D does not identify a unique
internal binding mechanism: name positions and recorder roles also change.
Condition equality removes one previous opportunity confound, not every role
or position explanation.

The full binary character expansion remains:

\[
 h(c,b,o)=\sum_{J\subseteq\{c,b,o\}}\widehat h_J\chi_J(c,b,o),
 \quad \widehat h_J=\tfrac18\sum_{c,b,o}h(c,b,o)\chi_J(c,b,o).
\]

In particular, S(c)=h_hat_b+c*h_hat_cb and
D(c)=h_hat_bo+c*h_hat_cbo. All eight coefficients are retained, including
condition origins and contextual complements. This is ordinary factorial/
character mathematics, not a newly invented formula.

## 3. Direction, magnitude and identity are different properties

For a source expression a and a declared corresponding expression a', define
the descriptive native-coordinate cosine between their nonzero fields. Compare
the mean over six families/two conditions with every one of the 720 family
permutations. A high absolute cosine need not mean a distinctive family identity:
all families can move along a broadly shared direction.

HM115 uses native K, native V, token-mean V, and two fixed six-target readouts
(uniform and query-weighted). No reader, whitening, rotation, rank, metric or
prediction model was fitted. Native Euclidean angles are descriptive, not an
identified causal inner product. The permutation rank is a finite design
description, not a human-population p-value.

The exact author observation remains an integral over qualified evidence:

\[
 s_u=\sum_{f,c,b}P_u(f,c,b)\,h_f(c,b,\text{own}).
\]

For the declared paired numerical profiles with family mass 1/6, equal
condition mass and polarity expectations +/-a:

\[
 \Delta s_{u,f}=\frac{2a}{6}\,\mathbb E_c[F_f(c)],\qquad a=0.4.
\]

This identity connects an observation field to an author composition. It does
not estimate a real person's psychological trait. Choice distributions, state,
origins and expression complements remain available rather than being erased.

## 4. What the executed data add

Canonical results: results/grain_math_response_v1/action_expression_r1.
288 actual local prefixes, 28.571654 model seconds, 288 attempts, no retries;
75 model comparisons plus four lexical controls; 450 numerical author contrasts.

- With different behavioral instances, native token-mean-V assignment-odd
  fields have mean cosines .281704/.288494 against permutation means
  .072177/.088515. Declared identity ranks are 1/2 of 720. Native K/V also
  retain odd correspondence (K .253825/.275883, V .252952/.278081).
- Query-weighted odd observations have mean cosines .501309/.377267, identity
  ranks 3/12. This is partial expression correspondence in a fixed observation,
  not a new personality correlation or native answer intervention.
- The role-blind behavior-token count control has new-instance cosine .041783
  and rank128; same-event rewording cosine -.000643 and rank307. Direct count
  overlap is insufficient to describe the observed native correspondences.
  This does not exclude pretrained lexical semantics or template/role effects.
- Higher readout cosine does not monotonically improve identity specificity.
  For same-event rewording, own mean-V cosines .400861/.411790 rank1/1;
  weighted own cosines .539117/.493017 rank43/30. Uniform odd cosines can also
  be high while ranks are72/99. Optimizing only cosine can favor generic effects.
- Correspondence is heterogeneous. The E and C designed families retain strong
  weighted odd direction across both opportunity wordings, N is intermediate;
  A/O vary by condition, and anthropomorphism is weak/near-zero or reversed.
  Those addresses and failures remain, rather than being renamed or tuned away.

The finite contribution is an executable separation between expression content,
actor-assignment interaction and target-sensitive observation, with author
integrals retained. It is not discovery of generic semantic binding, a unique
personality vector, a universal transport map or human measurement validity.

## 5. Prior art and the limits of mathematical borrowing

Park, Choe & Veitch (ICML2024), *The Linear Representation Hypothesis and the
Geometry of Large Language Models*, distinguishes context/output directions,
probing/steering and a causal inner product. Its assumptions are not supplied by
an arbitrary Euclidean cosine. HM115 does not identify that causal metric.
https://proceedings.mlr.press/v235/park24c.html
https://arxiv.org/html/2311.03658v2

Feng, Russell & Steinhardt (2024/ICLR2025), *Monitoring Latent World States in
Language Models with Propositional Probes*, separates content from binding and
uses mixed Hessians plus interchange interventions. Its sections3--4.2 motivate
separating an assignment contrast from causal identification of binding. HM115
does not run its Hessian algorithm or claim D is that binding subspace.
https://arxiv.org/html/2406.19501v1

OpenAI/math, pinned adc7f1241b42e322a6451854ab7e4b4c146bf78a,
*Determination of a metric and a unitary connection from one boundary patch*,
06-matrix.tex lines71--109, supplies a finite complementary-observation and
range/right-inverse design analogy. Harmonic jets, smooth bundles and the PDE
hypotheses are not established for language. The preprint is not a performance
guarantee or a proof of the GRAIN mechanism.
https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Determination-of-a-metric-and-a-unitary-connection-from-one-boundary-patch-October-5-2026/build/sections/06-matrix.tex

## 6. The next mechanism decision, not executed here

Ask whether a source-derived assignment-sensitive vector action changes the
qualified author's target judgement on new instances while preserving the
other person's content. Competing explanations are a transported actor-qualified
action versus a role/template or generic polarity effect. First inventory
HM60/HM68/HM73/HM100/HM106/HM107 and the existing binding interventions; do not
repeat their existence tests or reset the HM107 budget.

Any new finite plan must fix source-only construction, both author readouts,
matched content/role controls and all six families before new calls. Inspect
causal selectivity, not just a larger cosine or a more accurate old score. A
positive result carries a qualified observation operation into GRAIN's author
composition/conditional response; a nonspecific result retains a content
observation and separates attribution. No new packet is started in this closeout.

## 7. Verified scope

All 288 captures, fixed current positions, 144 owner token-bag groups, signed
effects and numeric integrals were replayed. Maximum alternate error is
1.443290e-15; 26 focused tests (12 new) and three Python lint/format checks pass.
This is same-agent arithmetic verification, not independent reproduction.
No API, private text, neural training, Git mutation, automation or lockbox use.
Upstream encoding still uses pretrained Qwen; later analysis is numerical only.
Original18 protocols/100cards/44 computational questions remain OPEN.
