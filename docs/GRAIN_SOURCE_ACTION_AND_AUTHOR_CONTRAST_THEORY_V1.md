# Source actions, target observations and author contrasts

2026-10-10. HM116 mathematical construction, separate from its frozen design.
Collection ended BUDGET_STOP:464/484 batches,5568/5808 endpoints. Committed
partial evidence is read back;16 random and4 no-op batches remain missing.
This document does not claim a complete protocol or psychological validity.

## 1. Why a vector correspondence is not yet a measurement operation

HM115 observed correspondence between assignment-sensitive value differences
across different behavioral instances. An angle alone does not say which
person's judgement changes, which target changes, or whether every judgement
merely shifts together. Those are different observations of the same direction.

The object tested in HM116 is therefore a finite source action evaluated on
new evidence, with person and target retained. This is not a new projection
formula or a search for a more accurate teacher approximation.

## 2. Construct the action before seeing application judgements

For source family f and opportunity c, define physical current-token value
arrays V_f(c,b,o), with behavior polarity b and actor assignment o in {-1,+1}.
Both named persons have the same opportunity. Source arrays use the original
behavioral instance and first wording only. The source assignment interaction
is

\[
 D_f(c)=\frac14\sum_{b,o}bo\,V_f(c,b,o).
\]

The assignment-even behavior component is

\[
 S_f(c)=\frac14\sum_{b,o}b\,V_f(c,b,o).
\]

The controlled application changes selected physical value rows by

\[
 \delta V=-boX_f(c),
\]

where X is D, a head-norm-matched S, a fixed cyclically wrong-family D, or a
fixed random direction. Baseline has no edit. The D magnitude is not tuned.
Control normalization uses source arrays only, never new vectors or answers.
The action is not guaranteed to remove the new instance's entire interaction.
It changes neither keys nor weights; it is an off-natural-manifold surgery.

Every arm has the same bo sign pattern. Finding a bo effect alone is therefore
insufficient evidence of semantic or actor selectivity. Content and actor
assignment remain separate accounts rather than one being discarded as noise.

## 3. Measure the finite effect, not an assumed linear derivative

Let N_{p,q}^a(c,b,o) be a native observation of the new instance, where p is the
person queried (A/B), q is one of the six target addresses, and a is the action.
The observation includes the full inherited pretrained model and actual
target suffix. Scores, grouped log odds, candidate mass and entropy use
different units and are never averaged into one purported psychological score.

Define the actual effect E_{p,q}^a=N_{p,q}^0-N_{p,q}^a and its conditional
assignment interaction

\[
 \beta_{p,q}^a(c)=\frac14\sum_{b,o}bo\,E_{p,q}^a(c,b,o).
\]

The actor-differential and actor-common accounts are

\[
 T_{f,q}^a(c)=\frac{\beta_{A,q}^a(c)-\beta_{B,q}^a(c)}2,
 \qquad G_{f,q}^a(c)=\frac{\beta_{A,q}^a(c)+\beta_{B,q}^a(c)}2.
\]

T and G are finite responses, not information percentages. A large common
shift can coexist with a small actor distinction. Signed matched-target and
off-target cells must be inspected, rather than only maximizing total norm.
Both conditions, both wordings, every family and all eight characters remain.

For a differentiable reader an infinitesimal edit could be related to a
Jacobian. HM116 does not assume that this finite action is infinitesimal or
that a first-order approximation equals its effect. It records actual native
judgements instead. No query metric, derivative or reader is fitted.

## 4. The author integral is exact on the declared finite support

Use two positive numerical evidence laws, conditional on family f:

\[
 P_\pm(b,o\mid c,f)=\frac{1\pm abo}{4},\qquad a=0.4,
\]

with equal opportunity mass 1/2 and family mass 1/6. Their contribution to
person p's target observation is

\[
 \Delta s_{p,q}^{a,f}
 =\frac16\sum_{c,b,o}\frac12
    [P_+(b,o\mid c,f)-P_-(b,o\mid c,f)]N_{p,q}^a(c,b,o)
 =\frac{2a}{6}\widehat N_{p,q,bo}^{a,f},
\]

where the last coefficient averages the two conditions. The change in this
author contrast caused by the action is exactly (2a/6) times the corresponding
effect coefficient. This connects a local intervention to an author evidence
composition without a predictor or a population fitted from application truth.

The formula is classical expectation/character orthogonality. The laws are
designed accounts, not real participants, personality distributions or sampling
confidence intervals. Their purpose is to check what the observation preserves
when evidence composition changes. Unknown natural evidence qualification is
not solved by specifying these laws.

## 5. What the result will decide

Source D with selective matched-target actor effects beyond the same-strength
controls would supply a finite transported operation for this evidence design.
Predominantly common or similarly controlled effects would instead locate a
gap between directional correspondence and qualified target operation. A
condition-specific result calls for a conditional observation, not a universal
vector or a strength/layer/rank search to hide the difference.

The current carrier refers to A in all application prefixes. B is queried from
the same carrier, not from a symmetric B-specific source. Thus the comparison
measures actual queried-person contrast, not arbitrary carrier equivariance.
Direct history routes and other internal representations remain available.
A small local effect cannot rule out redundant routes or information elsewhere.

This separates four claims: direction correspondence; effect of a specified
model intervention; author evidence integration; human psychological meaning.
The first three can be studied here. Names of the six design families do not
establish the fourth, and new native judgements are not true Big Five labels.

## 6. Prior art and the mathematical borrowing boundary

- Feng, Russell & Steinhardt (2024), sections4.1--4.2: mixed-Hessian binding
  identification and interchange interventions. HM116 does not implement that
  algorithm or claim D is a uniquely identified binding subspace.
  https://arxiv.org/html/2406.19501v1
- Heimersheim & Nanda (2024), sections2.2--2.4: scoped activation/path patching,
  and differing consequences of necessity/sufficiency interventions. This
  informs interpretation of a specified local surgery, not human measurement.
  https://arxiv.org/html/2404.15255v1
- Feng et al. (2026), PERSONA, sections2.2--2.4: contrastive persona vectors,
  generation steering and context-dependent composition already exist. HM116
  instead tests source behavioral evidence and queried-person observation.
  https://arxiv.org/html/2602.15669v1
- OpenAI/math, pinned adc7f1241b42e322a6451854ab7e4b4c146bf78a,
  metric/connection manuscript, 06-matrix.tex71--109: complementary fixed
  observations and a range constraint inspire the experimental separation.
  No harmonic-jet, PDE, smooth-bundle or right-inverse theorem is transferred.
  https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Determination-of-a-metric-and-a-unitary-connection-from-one-boundary-patch-October-5-2026/build/sections/06-matrix.tex

These prior tools are not claimed as original mathematics. Any contribution
must be the demonstrated evidence/action/person/target construction and its
measurement consequences, not familiar mathematics under new vocabulary.

## 7. Reproduction and permissions

Runner: scripts/run_grain_actor_action_hm116.py; stages prepare/collect/analyze/
verify. Canonical root: results/grain_math_response_v1/actor_action_r1. Each
batch contains all twelve suffixes, retains actual attention permissions, and
records an uncertain pending call before execution. Model work, including load
and failures, has a fixed cumulative cap. No terminal retry or cap renewal.

No API, private/MEPS reading, neural training, Claude, Git, automation, lockbox
or previous queue. Frozen HM115 and stopped HM107 remain unchanged. The
original18 protocols/100cards/44 questions remain open. Completed collection
and empirical interpretation must be evidenced separately from this document.

## 8. Result and the next scientific decision

All four baseline/directed/even/cyclic arms are complete on both wordings.
The score-unit actor-differential RMS is .000610/.000619 for directed source D,
and .000632/.000615 for the wrong-family matched-norm control. Total strength
does not distinguish their target routing: matched-target minus off-target is
+.000134/+.000184 for D versus -.000126/-.000094 for the wrong family.
The complete first-wording random arm has RMS.000093 and selectivity-.000009.
Do not generalize that control to the two missing second-wording families.
Common action RMS exceeds directed actor-differential RMS by3.56/3.70.
All24 directed matched-target condition cells have positive signed effects,
but the action also shifts off-target/person-common judgements.

Retain a conditional person/target action account, not a pure trait vector or
a universal strength objective. The next question is the composition of common
displacement and qualified person/target routing in one observation law, not a
larger edit, a better teacher fit or another similarity projection. This is a
question registration, not an automatic new experiment in closeout.

Canonical partial_readback contains all58 complete eight-cell groups and keeps
missing cells as NaN. The no-op gate remains unknown. Same-shape keys/pre-edit
values agree exactly; edit realization error2.38e-7; readout/alternate character
replay errors0; positive-law integral error2.78e-17. These checks validate
available arithmetic, not missing evidence. Terminal status and all frozen
source/code/input seals are preserved; no cap renewal.
