# HM32: Locate the Context Modulation Before Compiling It

Status: planned, not executed. Follow-up to verified HM31 and its H-only bound.
Question: how much identical-H modulation travels through the post-layer16
query endpoint Q, and how much remains in the other receiver states/layout?
Do not add a rank/regularization/prompt/accuracy grid.

## Fixed Eight-Corner Design

Use every HM31 pair/wording/query, same source/validation/application roles,
same fixed observer, exact token addresses, alphabet, two directions and .01
relative physical radius. Separate c=(R,Q), where R is every non-property,
non-endpoint state AND receiver layout. H and Q donors are independently chosen
from context sides0/1; R is the receiver side0/1. Define

    F_ijk = F(H_i,R_j,Q_k;q), g_ijk = d_H F(H_i,R_j,Q_k;q).

When k=j, inherit all192 HM31 references/768 endpoints, without replay. Only
k!=j corners are new:192 references/backwards and768 endpoints, bounded by
960 new forwards/192 backwards/1800 cumulative modelseconds. Combined48 cubes,
384 references/768 actions/1536 endpoints. No new fits or neural weight updates.
Acquire sources first, checkpoint, then held; preserve all frozen HM30 fields.

Q means the exact last query-token hidden state, not the entire question suffix.
R still contains earlier query states, other text states and token geometry.
A Q-only result cannot establish that all contextual information lies in Q;
remaining R effects may belong to earlier query nodes as well as text/context.

## Exact Intervention Requirements

First verify the natural receiver H and Q against HM30 saved states. Replace
only the mapped four property states with H_i and last-token state with Q_k;
then make the post-hook state a gradient leaf. Preserve R bitwise relative to
the live natural receiver, record every address, h/q/receiver arrays and audit.
Validate identical endpoint token ID and exact task query. No swapping whole
sentences or assuming token positions are interchangeable.

Endpoints use identical donor-H actual plus/minus arrays for all j/k. Save
candidate logits, gradient readout, odd/even changes, call-by-call journal and
source checkpoint hashes. A missing uncertain in-flight call stops; no replay.
There is no new no-op arm; inherited same-context corners are observed references,
not invented stochastic noise estimates.

## Mathematical Analysis and Decision

Use Fourier characters of the finite experimental index group Z2^3, not a
group physically acting on natural hidden vectors. With bits ordered H,R,Q,
the masks000/001/010/011/100/101/110/111 give mean/Q/R/RQ/H/HQ/HR/HRQ:

    chi_m(s)=(-1)^(m dot s), c_m=(1/8) sum_s chi_m(s) x_s.
    (1/8) sum_s ||x_s||^2 = sum_m ||c_m||^2.

On the old natural diagonal Q=R, characters m and m XOR011 coincide. The
four evaluation points therefore have rank4, regardless of repeated readings;
the full eight corners have rank8 and separate these formerly aliased effects.
Verify this using product averaging J/(I-J), not the same Fourier implementation.
Analyze native level, property gradient/covector, task-gradient seminorm and
both physical directions' actual raw/unit actions. Different native levels
alone do not establish changed property sensitivity.

At fixed H,Q compare R; at fixed H,R compare Q. Check exact common arrays and
deltas before contrasts. Source roles and all48 cubes are retained; do not select
favorable contexts or query directions. Raw vector and task-action units differ.

Apply the same conditional-variance bound to H-only versus (H,Q)-only action
laws on this finite cube. The HQ bound sums R/RQ/HR/HRQ energy; the HR bound
sums Q/RQ/HQ/HRQ energy. These are optimistic lower bounds on the declared
cube, not
held-fitted predictors. They determine whether Q materially removes the observed
collision; do not choose a tiny significance cutoff or claim psychological
validity from the intervention. A remaining R bound means Q alone is insufficient.

If Q accounts for the useful modulation, next construct a source-only action
law with explicit H/Q units and a cold source validation; if R remains important,
locate the next observed R coordinate before compilation. Do not hide R in a
high-dimensional fit and call that a mechanism. Multiple channels can survive.

## Acceptance and Original Scope

Synthetic8-corner known-Q, known-R, interaction-only, gradient-leaf and shared-
delta tests; freeze before calls; alternate candidate score reconstruction,
exact rational coordinate maps, product-projection and conditional-variance
identities; verify384 total references/1536 endpoints and960 actual NEW forwards.
Report576 decompositions/5184 fixed-other-coordinate contrasts/108 pooled rows.
Float32 differentiable readout vs float64 candidate reconstruction is recorded
separately with1e-6 arithmetic tolerance; alternate double-precision table
reconstruction must agree within1e-10, H/Q/R and actual delta checks exactly0.
Retain all negative effects and nonlinear remainders. No global theory completion
from a finite cube. H/Q extraction still uses a local neural encoder: successful
compilation would not by itself remove that encoder dependency.
Original18/100/44/natural authors/B_u/psychological interpretation remains open.
No API/private MEPS/Claude/automation/new training/commit/push/tag/lockbox/old queue.

## Mathematical Sources and Reading Scope

Yuval Filmus, Boolean Function Analysis lecture notes (2016), section1:
https://yuvalfilmus.cs.technion.ac.il/Courses/BooleanFunctionAnalysis/2015/LectureNotes.pdf
Characters, orthogonality and Parseval were read in the introductory section;
not all61 pages. The formula above is an exact finite-group identity.

OpenAI/math fixed commit adc7f1241b42e322a6451854ab7e4b4c146bf78a,
Polynomial-Hitting-Lists-for-Noncommutative-Rational-Formulas-September-24-2026,
full introduction reread. Its restricted-evaluation counterexample inspires
expanding the evaluation design, not transplanting its formula-complexity
theorem to a decoder softmax. Later division-algebra constructions are not
applied here. No claim of novel Fourier theory or of reading the full proof.
