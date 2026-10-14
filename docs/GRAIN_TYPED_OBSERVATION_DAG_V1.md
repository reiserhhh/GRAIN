# GRAIN Typed Observation DAG V1

Date: 2026-10-08. HM-07 is a finite observation-interface packet, not completion
of the psychometric theory or a new personality-accuracy benchmark.

## 1. Construction and provenance

Raw text is retained. A frozen UD parser produces addressed words. The adapter
then constructs separate nodes for mode/source operators, main propositions,
object modifiers, noncore adjuncts and explicit state-conditioned propositions.
Each predicate retains its raw token path, action/object spans, actor, conceiver,
mode chain and local node identity. Context nodes do not inherit the physical
event identity or participant of the containing main proposition. Unresolved
roles, nested modes and lower-scope negation remain explicit.

This is a linguistic adapter with finite tested rules, not a universal semantic
parser. xcomp control, acl object modification and advcl context are distinct.
The [UD relation definitions](https://universaldependencies.org/u/dep/advcl.html)
and [modal dependency formulation](https://aclanthology.org/2021.acl-long.122.pdf)
motivate the separation; no published modal parser was reproduced.

The main-proposition stream is scored; context nodes are retained for binding
and later conditional use, not separately counted as author actions. Retaining
context in a graph does NOT guarantee that the relevance function has used it.

## 2. Purpose-dependent units

For a routed event, the unchanged HM-03 maps yield
\[
z_n=(k_{Q,n},p_n,o_n,a_n).
\]
Here k is target relevance; p is signed expression; o is a learned owner
coordinate; a is the older actual/hypothetical coordinate. The new mode type
is observed separately, not reconstructed from a alone.

Set \(g(v)=\operatorname{clip}((v+1)/2,0,1)\). For purpose q,
\[
m_{Q,q,n}=\mathbf1\{M_n=q\}g(k_{Q,n})g(o_{q,n}),\qquad
v_{Q,q,n}=m_{Q,q,n}t_q(p_n).
\]
For state-expression exposure, t is 1; otherwise t(p)=p. State exposure is not
emotional intensity, valence, clinical state or verified experience. Signed
coordinates may exceed [-1,1]; they are not Pearson r.

In the typed arm, the owner coordinate is +1/-1 from observed grammatical roles:

- Actual/hypothetical: the actor is the author.
- Desire/plan: both conceiver and actor are the author.
- State expression: the experiencer/conceiver is the author.
- Desire-about-other: conceiver is the author and actor is another person.

The last account preserves "I want Tom to act" without claiming that the author
acted, or discarding the expression as irrelevant to every psychological goal.
Collective "we", unknown participants and passives with an unknown actor are
not silently assigned to one author.

Continuous opportunity-normalized contributions and qualified profiles differ.
Eligibility is the frozen rule k>0, owner>0 and matching mode. Continuous
contributions can be nonzero below the hard profile gate.

## 3. Author aggregation and missingness

Replay supplied event versions first. For active unique observations in
condition c, aggregate only purpose-qualified profiles:
\[
s_{u,Q,q}(c)=\frac1{N_{u,Q,q,c}}\sum_{n\in E_{u,Q,q,c}}v_{Q,q,n}.
\]
If N=0, the endpoint is unobserved, not zero. The finite experiment uses a
fixed two-condition reference \(\pi_+=.5\):
\[
L=(1-\pi_+)s(-1)+\pi_+s(+1),\qquad
B=\{s(+1)-s(-1)\}/2.
\]
Both endpoints are required. Condition-choice proportions are retained
separately; they are not centered away as noise.

The 320 programs in HM-07 are designed text programs, not human participants.
Their conditions are wording/context variants, not psychological changes.
Parser completeness and score availability must be reported separately:
r2 parses all 320 programs, but scores 288. E5 coordination desire and plan
each lack one condition in all 16 programs.

A revision invalidates both the main and dependent context snapshots.
Aliases add no independent event; retraction removes the active observation.
These checks use supplied event/version identities, not inferred real-world
event resolution.

## 4. Why aggregation can irreversibly lose roles

For finite linear maps S (compression) and F (desired readout),
\[
F=AS\text{ for some }A\quad\Longleftrightarrow\quad
\ker S\subseteq\ker F.
\]
Necessity follows by applying AS to a null vector. For sufficiency define
A(Sx)=Fx on the image of S; kernel inclusion makes this well-defined, then
extend A linearly to the remaining coordinates.

The packet tests S=(1,1), F=(1,0), d=(1,-1). Then Sd=0 but Fd=1.
Untyped pooling cannot recover a main/context assignment it has destroyed.
This is a standard linear identifiability fact, not a proof about all nonlinear
representations or personality.

## 5. Coordinate transport is not fixed-mask invariance

For invertible C, let y=C^-1 x, P'=C^-1 P C and d'=d C. Then
\[
d'P'y=dPx.
\]
Both projection and dual must move. Keeping the old mask can change the score
even though the represented object is unchanged.

The executed transport probe uses functional coordinates
x=(p_1,...,p_N,k_1,...,k_N). It does NOT embed actual context-node vectors.
The runner comment about a "context channel" is too broad: the second channel
is relevance. The scope is a same-reader finite-coordinate identity,
not E5-to-BGE or semantic transport.

The prescribed-subspace transfer in the pinned
[OpenAI math repository](https://github.com/openai/math) supplied a preservation
strategy. Its Schur-complement construction, infinite-dimensional hypotheses
and full Ulam-stability theorem were not implemented or verified here.

## 6. Conditional error decomposition

For the complete crossed design of action family, mode, object style, polarity
and actor, define f=k-k* under the explicitly uniform product law.
For each subset A of design factors:
\[
f_\varnothing=\mathbb E f,\qquad
f_A=\mathbb E[f\mid X_A]-\sum_{B\subsetneq A}f_B.
\]
The components reconstruct f, are pairwise orthogonal, and satisfy
\(\operatorname{Var}(f)=\sum_{A\ne\varnothing}\|f_A\|_2^2\).
This is functional ANOVA, not classical factor analysis or a new theorem.
[Owen (2003), section 2](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A13n11.pdf)
provides the standard product-space formulation.

The diagnostic is result-after and changes no model or threshold. It measures
error in the designed bank, not natural-population variance. Natural tokens are
correlated, so orthogonality cannot be assumed there. Totals for "involves mode"
and "involves object style" can overlap.

Automatic coordination slots have 14.2%-17.1% mode-involving error variance;
declared complete slots reduce it to numerical zero. The frozen relevance
function uses action/object inputs, not mode. All 1,280 single-interface
comparisons retain identical action inputs; 140 have changed object inputs.
Identical inputs never have a nonzero relevance difference. Thus the apparent
mode effect is localized to the observation boundary, not intrinsic failure
of an unchanged function on identical input.

## 7. Empirical state and next construction

- Primary observation_dag_r1 preserves the failed when interface:
  560/720 DAG-complete versus 608/720 flat-complete.
- observation_dag_r2 accepts when as advmod as well as mark. It reuses the same
  real UD words and vectors, reaching 704/720; this is a result-after repair,
  not another independent 720-text replication.
- 1,992 nodes include 576 retained context nodes. Sixteen unresolved scope
  constructions remain relation objects.
- The declared-reference arm has zero missed expected single observations.
  The automatic arm misses 17, all target-relevance failures with changed
  object boundaries; the declared reference is an oracle construction.
- All 24 supplied-version programs and numerical readout checks pass.
- A separate 64-sentence licensed GUM companion yields 39 matched main heads
  out of 41 in each structural view, and 37 agreeing roles among the 39.
  It has no verified author identities, psychological labels or author scores.

Next HM-08 must compare grammar-indexed object/context views, not merely fix
another threshold. Preserve strict objects, candidate adjuncts and ambiguity;
test whether context changes target action units without becoming duplicate
author evidence. Freeze new materials and source/application roles before new
readouts. Source-unit uncertainty and licensed multi-event author construction
remain explicit later requirements.

## 8. Reproduction

Run from the implementation repository with the parent virtual environment:

~~~bash
PY="/Volumes/mobile3/projects/project persona/.venv/bin/python"
$PY scripts/verify_grain_math_observation_dag.py \
  --packet results/grain_math_response_v1/observation_dag_r1 \
  --out results/grain_math_response_v1/observation_dag_verification_r1
$PY scripts/verify_grain_math_observation_dag.py
$PY scripts/analyze_grain_math_dag_units.py
$PY -m pytest -q tests/test_grain_math_observation_dag.py \
  tests/test_grain_math_conditional_variance.py
~~~

Do not rerun the current changed adapter into frozen r1. Primary r1 code is
retained in its 00/code snapshot; r2's snapshot contains the repair.
A fresh execution requires a distinct output directory. Licensed GUM text is
CC BY-NC-SA 4.0; its pinned README/LICENSE and attribution are retained locally.
