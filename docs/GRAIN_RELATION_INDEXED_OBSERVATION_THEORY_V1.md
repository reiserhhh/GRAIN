# Relation-indexed observations and author evidence composition

HM91, 2026-10-10 JST. Finite model evidence, not a new general theorem or a
completed personality instrument. The frozen experiment is
`results/grain_math_response_v1/relation_scope_r1/`; the precollection plan is
`docs/GRAIN_RELATION_SCOPE_PLAN_HM91.md`.

## 1. What this changes

HM90 retained connection/content pairing. HM91 asks whether the relation of a
statement to its described behavior can then be handled by one common scalar.
It also asks whether a vector observer detects changed relation/behavior/subject
pairing when all separate marginals and the token-bag distribution agree.

The answers in this fixed battery are: a common scalar is insufficient, and the
joint composition is visible, although its author-law contrasts are small.
This is a change to the required observation object, not an accuracy upgrade.
Preference, intention, imagined actions and reports are retained as observations;
they are not automatically treated as noise or as actual completed behavior.

## 2. Fixed observation

There are 1,152 fresh designed bodies: six behavior families, two reporting
frames, six relations and 16 independent content/subject corners. Target names
label designed behavior prototypes, not validated personality dimensions.
Application uses a frozen Qwen2.5-1.5B encoder through blocks 0--16 and layer-17
Q/K/V, source query seeds and source behavior-calibrated value covectors. No
application-specific question, upper-layer answer, generation or new fit runs.
All 12 heads remain. Generic assessment instructions and neural encoding remain.

For mode m, target reader t and the designated subject A, the raw observation is

```math
F_{m,t}(e,A) = \sum_{h,i} p_{m,t,h,i} H_{m,h,i,A} c_{m,t,h,i}.
```

Here p is the frozen-query body selection law, H is the fixed per-head one-step
named-subject gate, and c is a source covector applied to the contextual body
value. Subject/event anchors are design-provided; A is not an identified real
text author. Full, conditional, retained-mass and matched-behavior readouts are
separate quantities. The matched span is only a component of the full readout.

## 3. Common-scalar obstruction

Let u collect all nonconstant four-bit Walsh coefficients of asserted-event
observations and v those of another relation, across all six families and
readers, within one fixed frame and field. Each vector has 540 coordinates.

```math
a = \frac{u^T v}{u^T u},\qquad
\eta = \frac{\|v-a u\|^2}{\|v\|^2}
     = \frac{\|u\wedge v\|^2}{\|u\|^2\|v\|^2}.
```

This is a descriptive geometric projection, not a trained predictor. Constants
are removed, so an affine offset cannot explain these remaining directions.
Paired full-raw eta ranges from 0.178835 to 0.516958 across the ten nonreference
frame/relation conditions. Matched-behavior eta ranges from 0.230819 to 0.756510.
These are fractions of fixed response-coordinate energy, not psychological
information or prediction error. This excludes one common scalar action for
this battery, not every possible relation operator or a target-specific rule.

## 4. Equal marginals do not determine an author observation

Let s=(x,y,g,h) in {-1,+1}^4 and z_s=(xg+yh)/2. With six modes, set
w=(0,1,-1,0,0,0), selecting inner versus outer negation, and epsilon=0.5:

```math
P_{\pm}(m,s) = \frac{1\pm\epsilon w_m z_s}{96}.
\Theta_{\pm,t} = \sum_{m,s} P_{\pm}(m,s) F_{m,t}(s).
```

Both laws have identical complete-corner and mode marginals. Therefore every
additive a(m)+b(s) observation has zero contrast. Inner and outer negation have
identical token bags for each corner, so bag-only observations also agree.
In contrast,

```math
\Theta_{+,t}-\Theta_{-,t}
 = \frac{\epsilon}{6}\mathbb E_s[
 (xg+yh)(F_{\mathrm{inner},t}(s)-F_{\mathrm{outer},t}(s))].
```

All 12 matched full-raw contrasts are nonzero, with absolute values
0.000032659--0.001048854 in source vector units; RMS is 0.000550322.
This is a finite constructed evidence-law contrast, not 12 people, r, AUC,
psychological truth or evidence that one sign is the correct trait direction.
The necessary information in this comparison is joint relation/content/subject
structure; separately precise marginals still cannot recover it.

## 5. Contextual accounts are not causal interventions

For exactly matched behavior tokens, write p=p0+dp, H=H0+dH and c=c0+dc.
The difference of p H c is exactly the sum of seven terms:

```math
dp H0 c0 + p0 dH c0 + p0 H0 dc
+ dp dH c0 + dp H0 dc + p0 dH dc + dp dH dc.
```

The collected identity error is at most 4.164e-17. Signed terms can cancel;
they are an algebraic account, not separately identified neural mediators.
Moving negation changes order, grammar and positions, not pure semantic scope
alone. This experiment does not identify a unique causal semantic path.

Under the author laws, matched-behavior RMS is 0.000111911 and remaining-token
RMS is 0.000460513. Full/core signs differ in 4/12 conditions. The remainder
contains contextual, source, subject and grammatical observations, not only
personality and not only noise. Full energy includes the core/remainder cross
term; their RMS ratio must not be reported as an information percentage.

## 6. Mathematical ancestry and next decision

Scope/source-relative event factuality and entity binding have direct precedents:
[Sauri and Pustejovsky (2012), sections 2--3](https://aclanthology.org/J12-2002.pdf)
and [Feng and Steinhardt (2024), sections 2--3](https://arxiv.org/html/2310.17191v2).
Walsh contrasts, Gram minors, conditional laws and product expansion are mature
mathematics. This packet does not discover negation scope or binding itself.

OpenAI/math's pinned metric/connection manuscript, section `06-matrix.tex`,
lines 1--145, inspires the use of fixed vector-response families and retained
matrix observations. Its harmonic-jet/PDE/uniqueness results are not transferred
to language. That design inspiration is not a new proof of personality inference.

The next scientific question is how relation-indexed observations compose
across evidence units, including target-shared and target-specific response.
Do not replace this with head/layer/strength selection or classifier accuracy.
Natural author linkage, psychological interpretation, the historical GRAIN
factor interface and individual dynamic operators remain distinct open work.

## 7. Execution and verification

1,152 actual captures used 104.034 model seconds. The final six-module focused
suite passed 37 tests, with two Swig deprecation warnings; Ruff and format
checks passed for the five new Python files. Primary source/raw/output
hashes and separate verifier/interpretation hashes were checked without writing
to the seals. The alternate verifier checks 137,808 values, with maximum error
2.843e-14; it is same-agent alternate arithmetic, not an independent researcher.
The sealed primary summary keeps its original awaiting-review status;
`verification.json` records VERIFIED_FINITE_HM91_NOT_GLOBAL. No new API, private
data, neural training, Git mutation, Claude, automation or lockbox operation.
Original 18 protocols/100 cards/44 computational questions remain open.
