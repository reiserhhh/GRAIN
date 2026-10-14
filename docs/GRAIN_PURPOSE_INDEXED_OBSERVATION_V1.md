# GRAIN purpose-indexed observations: HM-06

2026-10-08. A finite mechanism extension, not a finished personality scale.
Canonical results: `results/grain_math_response_v1/purpose_accounts_r1`;
alternate-formula verification: `purpose_accounts_verification_r1`.
HM-03 source vector coefficients, tensor kernels, and midpoint were untouched.

## 1. Observation object

An event mention has target-related content, a source/actor, polarity, a report
mode, a context, and a supplied version identity. These are separate inputs.
Reports of action, hypothetical action, desire, plan, and state expression are
five different observation purposes, not five already validated personality
traits. A hypothetical is not an intention; an intention is not verified action.
State here means the presence of a target-related state expression. It does NOT
score anxiousness, valence, intensity, a persistent state, or a trait.

The scoped observer accepts real frozen Stanza UD words plus the original text.
Governors `want/wish`, `plan/intend`, `imagine`, and `feel`, and a root `would`,
determine report mode. Actual k/p/o/a coordinates come from frozen E5/BGE maps;
action/object relevance comes from the HM-03 source-only product kernel.
The mode inventory is explicit linguistic knowledge, not a discovery of an LLM
attention circuit or a purely vector-only mode decoder.

Event identity/version/condition metadata are supplied independently of role
labels. Evaluation modes never enter the automatic observer. Declared-mode
readouts are a separate structural reference, not the automatic score.

## 2. Operators and units

Let `g(v)=clip((v+1)/2,0,1)`, `m_e` be observed mode, and
`Q` one of the five purposes. Existing relevance and ownership coordinates are
`k_e,o_e`; `p_e` is source-calibrated expression polarity. Define

\[
q_Q(e)=g(k_e)g(o_e)\mathbf1\{m_e=Q\},\qquad
v_Q(e)=q_Q(e)b_Q(e),
\]

where `b_Q=p_e` for the four action-expression modes and `b_state=1`.
Bounded gates are NOT posterior probabilities. The negative desire pole is an
expressed denial of wanting; it is not the stronger assertion of wanting NOT
to act. Lower-clause negation is separately flagged in this scoped version.

The document contribution unit is `sum(v_Q)/number_of_declared_events`.
The saved soft numerator/mass are `sum(v_Q)` and `sum(q_Q)`. The runner's
author/event profile instead uses the explicitly hard-qualified events:

\[
I_Q(e)=\mathbf1\{k_e>0,o_e>0,m_e=Q\},\qquad
s_{u,Q,c}=\frac{\sum_{e\in E_{u,c}}I_Q(e)v_Q(e)}
{\sum_{e\in E_{u,c}}I_Q(e)}.
\]

No denominator means no observed profile, not a zero-valued person. The
soft-mass-normalized ratio is a possible DIFFERENT estimand, not silently
substituted for the runner's hard profile. A zero bookkeeping contribution
for an excluded event does not make that event a neutral observed author.

With a fixed external reference mass `nu(c=+1)=.5`, the conditional level is
`.5*s_- + .5*s_+`, and the endpoint contrast is `(s_+ - s_-)/2`.
Both endpoints must be observed. Choice proportions and counts remain saved.
This packet's author-program endpoints are TWO WORDING STYLES, not manipulated
psychological conditions or longitudinal state changes.

## 3. Why one binary actuality variable is insufficient

For declared ideal modes, the compression from the five mode coordinates to
`(constant,actual-versus-rest)` has rank 2 and a three-dimensional kernel.
Desire and hypothetical basis vectors have identical compressed coordinates
but different desire-account values. Therefore even a nonlinear function of
THAT compression cannot recover the desire account for both events. This is a
finite identifiability witness, not a theorem that all embeddings lose modes.
Indeed the real source-trained a coordinate sometimes classifies new intentions
as actual. That is a report-mode unit shift, not evidence of performed actions.

The richer report-mode observation is required BEFORE aggregation. A more
complex author regression cannot distinguish inputs that an earlier observation
map has already made identical. Standard direct sums/conditional integration
provide the mathematical form; they are not claimed as new mathematics.

## 4. Version history before qualification

Let `R(H)` retain the latest active declared version per event and let
`Pi_Q` route a purpose. The order is `Pi_Q(R(H))`, not `R(Pi_Q(H))`.
An actual report revised to hypothetical must remove its old actual record.
Filtering first can discard the revision and leave obsolete actual evidence.
Aliases retain the active original; revisions require the current same-event
parent; retract removes it. New IDs/contexts cannot fabricate independent events.
History r1/r2 correction counterexamples remain as earlier evidence.

## 5. Executed outcome and retained failures

736 public authored documents: 640 single-mode, 64 two-event mixtures, and 32
scope challenges. 736 actual Stanza calls; 547 new text inputs per frozen reader
(1,094 total); zero source refits/API/private text/neural training.

Automatic complete coverage: 634/640 singles and 60/64 mixtures. In complete
singles the five-purpose observability has zero false positives/negatives across
both readers/targets. Unit RMSE .0380-.1316, versus .1637-.3297 under the wrong
universal actual gate. These ranges aggregate different purpose units; consult
the per-cell CSV for paired comparisons. All 20 fixed-family bootstrap lower
bounds of MSE improvement are positive; this is not a human-population CI.

All observed head labels agree with their record mode, but that headline alone
would be misleading: extra object-modifier heads contaminate 10 documents.
The scorer detects record multiplicity and emits incomplete/null profiles.
32 explicit scope challenges remain unresolved (belief/other-object control,
nested imagination/desire, and lower-clause negation). No result-driven rescue.

40 version programs produce 400 reader/purpose checks; 320 controlled author
profile readouts retain fixed reference and coverage. Desire author-profile
RMSE reaches .4864 despite correct routing, showing that routing and continuous
unit calibration are different mechanisms. These are controlled program authors,
not additional human participants or psychological validation.

Alternate formulas verify 32,480 event rows, 29,440 document/purpose readouts,
400 version checks, and 2,506 source/application hashes. Maximum coordinate
error 1.259e-13. This is same-agent formula verification, not a blind human audit.

## 6. Next dependency

HM-07: develop a target-independent observation DAG separating mode-governor,
main proposition, and object modifiers; preserve every node, but give it its
own proposition identity rather than duplicate the outer action event. Freeze
rules before fresh material evaluation and keep this r1 unchanged. Then connect
the typed account interface to licensed natural event records and narrow targets.
Do not spend the next packet on a/p thresholds or alpha/rank tuning.

Sources read: [UD xcomp](https://universaldependencies.org/u/dep/xcomp.html),
[Yao et al. 2021](https://aclanthology.org/2021.acl-long.122.pdf) sections 1-2,
and [ModaFact](https://aclanthology.org/2025.coling-main.425.pdf) modality/factuality
definitions. This is a scoped interface inspired by those distinctions, not a
reproduction of their neural parsers or datasets.
