# GRAIN Joint Context Operator V1

Date: 2026-10-08. HM-08 implements a finite observation-law comparison and an
exact expansion of the existing HM-03 tensor scorer. No source model is refitted.

## Observation Law Before Vector Action

The unchanged raw text is parsed by the frozen Stanza reader. The HM-07 typed
graph is retained. A target-independent adapter constructs an object-view set
O(x), alongside an addressed action A(x) and separate actor/conceiver/mode slots.
Its members are strict object spans and legal raw-span attachment candidates.

A direct, nonfinite, unmarked adjunct with no explicit subject or auxiliary and
no comma boundary may supply a candidate. This is a finite linguistic rule, not
proof that every such participle modifies the object. Context nodes retain their
own identity and do not become additional author events. Finite marked adjuncts,
explicit other actors and comma-boundary contexts remain in the graph but cannot
extend the legal object view under this rule. A naive all-context arm deliberately
violates that distinction and acts as a negative control.

The longest legal raw span is chosen by its address, never by its target score.
A second conservative arm qualifies only when every legal candidate has k > 0.
The finite envelope [min k, max k] is a sensitivity range, NOT a probability,
95% confidence interval, or guaranteed partial-identification bound on truth.
It is invariant to candidate enumeration, not to arbitrary word permutations.

## Exact Contextual Response

The frozen source-centered tensor kernel is

\[
K_j(a,o)=(1+a^T A_j)(1+o^T O_j),\qquad
f_Q(a,o)=(K-\overline K_{row}-\overline K_{source,col}
  +\overline K_{source})d_Q+b_Q.
\]

Let m be the number of source pairs and
e = d - (sum d / m) 1. This keeps row centering exact, including nonzero
floating-point or deliberately nonzero sums of dual coefficients. Expanding:

\[
f_Q(a,o)=c_Q+c_{A,Q}^Ta+c_{O,Q}^To+a^T T_Q o,
\]

where c_A = A^T e, c_O = O^T e,
T = A^T diag(e) O, and
c = sum e - mean_col^T d + grand_mean sum d + b.
The implementation contracts the low-rank source factors and never has to
materialize a dense 1024 by 1024 matrix.

For an actual observed boundary change Delta o = o_joint - o_strict:

\[
\Delta f_Q=c_{O,Q}^T\Delta o+a^T T_Q\Delta o.
\]

This is an EXACT finite difference of this fixed model, not a first-order
approximation, and not proof that text semantics or human traits are bilinear.
The first term is an object-main effect; the second is action-conditioned
context response. Their RMS values are correlated and cannot be summed as
independent information/variance fractions.

With a fixed alternative action and strict/joint object views, the four-corner
contrast removes every additive term:

\[
f(a_1,o_1)-f(a_1,o_0)-f(a_0,o_1)+f(a_0,o_0)
=(a_1-a_0)^T T_Q(o_1-o_0).
\]

This supplies a measurable target-indexed response operator in embedding space.
It is the OBSERVER's response to object information, not yet the individual's
psychological response operator. Alternate formulations match actual cached
vectors to about 1e-13. No neural calls or fits are needed for compilation.

## Measurement After Meaning

After candidate construction, the existing source function produces k/p/o/a.
Typed actor/conceiver roles provide the purpose-specific +1/-1 owner coordinate.
The current gate g(k)=clip((k+1)/2,0,1) yields a soft contribution, while k > 0
defines hard observability; they are different readouts. Modes keep actual,
hypothetical, desire, plan, state exposure and desires about others separate.
Context is not assigned an extra author event merely because it contributes to k.

Author profiles aggregate only qualified, unique addressed events at a fixed
two-condition reference. A missing endpoint stays null. HM-08's 16 programs per
reader/target/account are software-designed expression programs, NOT people.
The two conditions are reduced versus finite construction frames, with designed
polarity changes; they are not psychological situations or clinical time points.

## Actual Evidence and Open Boundary

Canonical results: results/grain_math_response_v1/joint_context_r1.
Verification and exact-response companion: joint_context_verification_r1.
2656 cold public constructions, 2656 actual parses, 5362 frozen encoder inputs;
zero new source fits, API calls, private reads, or neural training.

2528 documents have one complete supplied event. The 128 remaining state-plus-
other-adjunct documents retain two state-condition heads for one supplied record;
they remain unscored, not rescued by knowing the intended design event. All 8192
nodes survive. There are 208 legal attachment alternatives and 27 gate disagreements.

E5 relevance errors on the complete single-event contrast set go 27 -> 0 under
legal views, while BGE is already 0. In the reported account after role routing,
E5 own-event omissions go 9 -> 0; legal views introduce no hard false positives
in this finite set. Naive all-context joins introduce 32/30 own-account false
positives for E5/BGE. A relevance fix is not exact continuous-unit calibration.

The envelope still records all 27 ambiguities; the conservative arm retains the
strict-view omissions rather than declaring a confidence bound. Coordinate,
purpose and program-endpoint checks are same-agent alternate calculations, not
independent human review. This packet does not close the original full program.

## Mathematical Sources and Scope

[Deep Sets](https://papers.nips.cc/paper_files/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf)
sections 2-3 give finite/countable permutation-invariant set-function structure
and conditioning. Here only candidate-enumeration invariance is used; ordered
text, roles and observation identities are not exchangeable sets.

[UD advcl](https://universaldependencies.org/u/dep/advcl.html) defines a noncore
predicate modifier, distinct from an object modifier; it motivates context
retention without event inheritance. It does not validate our candidate rule.

[OpenAI math](https://github.com/openai/math)'s pinned latest-anchor trace-
reconstruction manuscript, part-1.tex lines 1-210, separates exact blocks, loose
edges, anchored differences and phase products. This motivates preserving
addresses and testing interactions before compression. Its independent deletion
channel and sample bound are NOT applied to correlated language or personality.
Only this section was read here; no whole-repository or full-proof claim is made.

## Reproduction

Use the Project Persona virtual environment, cwd this implementation repository:

~~~sh
"/Volumes/mobile3/projects/project persona/.venv/bin/python" scripts/run_grain_math_joint_context.py --device mps --out results/grain_math_response_v1/joint_context_replication_r1
"/Volumes/mobile3/projects/project persona/.venv/bin/python" scripts/verify_grain_math_joint_context.py --out results/grain_math_response_v1/joint_context_replication_r1 --dest results/grain_math_response_v1/joint_context_replication_verification_r1
"/Volumes/mobile3/projects/project persona/.venv/bin/python" -m pytest -q tests/test_grain_math_joint_context.py tests/test_grain_math_context_operator.py
~~~

The frozen local Stanza/E5/BGE caches and HM-03 source artifacts must exist.
Completed output directories refuse reruns. CPU parsing and MPS embedding are
supported; no download or new model training is silently substituted.
